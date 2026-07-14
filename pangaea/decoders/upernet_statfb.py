"""
SegMTUPerNetDecFb variant with a STATIC learned feedback parameter -- the
input-independent control for the decoder->encoder feedback experiments.

A zero-initialised nn.Parameter `fb_static`, shared across samples, is added
to the belief at EVERY frame of the recurrence (t=0 included):

    z_H_t = fold(z_H_{t-1}, P_t) + fb_static        (fold = z_H + P_t here)

Credit assignment is SAME-FRAME: frame t's loss backpropagates into the
fb_static injection made at frame t (with bptt_frames=1, ONLY that one --
earlier injections are behind the state detach). This differs from the
dynamic decfb path on purpose: dynamic feedback is computed FROM frame t's
logits so it can only be consumed at t+1, but a static prior depends on
nothing, so it rides an always-on hook (encoder.static_fb_fn, registered
once at init) instead of the pending_feedback mailbox. Injection is
therefore identical at train and eval, every frame.

It is not feedback in any dynamic sense -- it is a learned task prior on the
belief (prompt-tuning of the frozen recurrent backbone, cousin to the
pretrained zL_init) -- and that is exactly its purpose as a control: if this
matches the dynamic decfb numbers, the closed loop adds no information
beyond "any learned additive offset on z_H helps"; if it does nothing while
decfb helps, the decfb gain is genuinely content-dependent.

static_mode:
- "shared": one (1, num_patches, embed_dim) parameter, the same at every
  frame. Every frame's loss pulls on the same tensor (sequential per-frame
  updates under the stream trainer), so per-horizon preferences average out.
- "per_frame": a (T, num_patches, embed_dim) parameter, row t added at frame
  t -- a horizon-conditioned prompt. Rows below the trainer's supervision
  start (min_horizon) are still added but never receive gradient, so they
  stay at zero.

dynamic_feedback:
- False (default): pure control; the parent's fb_proj/fb_norm are deleted so
  the parameter count is honest, and make_feedback returns None (the mailbox
  stays empty).
- True: the static prior via the hook PLUS the parent's fb_proj(decoder
  output) via the mailbox -- a clean decomposition into prior + dynamics.

fb_scale multiplies both terms, matching the parent.

The parameter lives on the DECODER for the same reason fb_proj does: with
finetune=false the encoder's parameters are frozen and the optimizer is
built from decoder.parameters(); the encoder only holds a callable.

Caveats: fb_static's magnitude is unbounded (no softmax/LayerNorm upstream)
and it is re-added every frame, so a runaway prior would compound through
the recurrence -- watch its norm during training. And the hook only exists
in forward_stream: paths that use the encoder's plain forward() (e.g.
BeliefEvolutionTracker) run WITHOUT the prior.
"""

import torch
import torch.nn as nn

from pangaea.decoders.upernet_decfb import SegMTUPerNetDecFb


class SegMTUPerNetStatFb(SegMTUPerNetDecFb):
    def __init__(
        self,
        *args,
        static_mode: str = "shared",
        dynamic_feedback: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if static_mode not in ("shared", "per_frame"):
            raise ValueError(
                f"static_mode must be 'shared' or 'per_frame', got {static_mode!r}."
            )
        self.static_mode = static_mode
        self.dynamic_feedback = bool(dynamic_feedback)

        n_rows = 1 if static_mode == "shared" else int(self.multi_temporal)
        self.fb_static = nn.Parameter(
            torch.zeros(n_rows, self.encoder.num_patches, self.encoder.embed_dim)
        )

        if not self.dynamic_feedback:
            # Pure-control mode: drop the parent's dynamic adapter so it
            # neither trains nor pads the parameter count.
            del self.fb_proj
            del self.fb_norm

        if not hasattr(self.encoder, "static_fb_fn"):
            raise TypeError(
                "SegMTUPerNetStatFb needs an encoder with the static_fb_fn "
                "hook (RecursiveJEPAFeedbackDecFbEncoder); got "
                f"{type(self.encoder).__name__}."
            )
        self.encoder.static_fb_fn = self._static_fb

    def _static_fb(self, t: int) -> torch.Tensor:
        """The prior added at frame t, (1, num_patches, embed_dim);
        broadcasts over the batch inside the encoder."""
        if self.static_mode == "shared":
            row = self.fb_static
        else:
            # Row t = frame t; clamp in case a stream runs past T.
            row = self.fb_static[min(t, self.fb_static.shape[0] - 1)].unsqueeze(0)
        return self.fb_scale * row

    def make_feedback(
        self, logits: torch.Tensor, t: int | None = None
    ) -> torch.Tensor | None:
        """Mailbox feedback: only the dynamic term (the static prior travels
        through the always-on hook instead). None keeps the mailbox empty."""
        if not self.dynamic_feedback:
            return None
        return super().make_feedback(logits, t)
