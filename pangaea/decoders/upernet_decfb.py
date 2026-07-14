"""
SegMTUPerNet + a trainable decoder->encoder feedback adapter, for
RecursiveJEPAFeedbackDecFb-family encoders.

The adapter (`fb_proj`) turns the decoder's frame-t output into additive
belief tokens for the encoder's frame-t+1 fold. Two feedback sources:

- feedback_source="probs" (default): logits -> softmax -> adaptive-avg-pool
  to the encoder's patch grid -> linear (num_classes -> embed_dim). Compact
  (~8k params for 20 classes) but bottlenecked through class probabilities.
- feedback_source="feat": the pre-conv_seg decoder feature map (channels
  wide) -> adaptive-avg-pool to the patch grid -> LayerNorm -> linear
  (channels -> embed_dim). Richer signal (~200k params at channels=512);
  the LayerNorm bounds the unnormalised feature scale the way softmax
  bounds probabilities.

`fb_scale` multiplies the adapter output.

fb_proj is ZERO-INITIALISED in both modes, so step 0 of training is exactly
the no-feedback baseline; the loop only opens up where the task loss pushes.

The adapter lives on the DECODER (not the encoder) deliberately: with
finetune=false, SegUPerNet.__init__ sets requires_grad=False on every
encoder parameter, and run.py builds the optimizer from decoder.parameters()
-- so a decoder-side adapter is trainable out of the box while the backbone
stays frozen, with no changes to the freezing logic.

forward() runs the full closed loop (decode every frame, feed each frame's
output back into the next fold) so the evaluator scores EXACTLY the
behaviour the streaming trainer trains -- activation feedback needs no
labels, hence no train/test mismatch. Intended training path is
StreamDecodeFbSegTrainer (per-frame truncated updates); training this
decoder through plain SegTrainer's single end-of-sequence loss would reach
fb_proj only through the final frame's injection and hold every frame's
decode graph in memory, it works, but it is not the design point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.upernet import SegMTUPerNet


class SegMTUPerNetDecFb(SegMTUPerNet):
    def __init__(
        self,
        *args,
        feedback_source: str = "probs",
        temporal_pool: str = "none",
        fb_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if feedback_source not in ("probs", "feat"):
            raise ValueError(
                f"feedback_source must be 'probs' or 'feat', got {feedback_source!r}."
            )
        if temporal_pool not in ("none", "cum_mean"):
            raise ValueError(
                f"temporal_pool must be 'none' or 'cum_mean', got {temporal_pool!r}."
            )
        self.feedback_source = feedback_source
        self.fb_scale = float(fb_scale)

        in_dim = self.num_classes if feedback_source == "probs" else self.channels
        self.fb_norm = (
            nn.LayerNorm(in_dim) if feedback_source == "feat" else nn.Identity()
        )
        self.fb_proj = nn.Linear(in_dim, self.encoder.embed_dim)
        nn.init.zeros_(self.fb_proj.weight)
        nn.init.zeros_(self.fb_proj.bias)

        # stashed (detached) pre-conv_seg feature map from the most recent
        # _decode_taps call; consumed in "feat" mode.
        self._last_fb_feat: torch.Tensor | None = None

        # Temporal pool variable
        self.temporal_pool = temporal_pool
        self._pool_m: list[torch.Tensor] | None = None # Running mean per output layer
        self._pool_t = 0 

    def _fb_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Frame-t decoder output -> (B, num_patches, in_dim). 
        Gradient reaches the adapter from a LATER frame's loss only, 
        never back into the frame that produced the output."""
        g = self.encoder.grid_size
        if self.feedback_source == "probs":
            src = logits.softmax(dim=1)              # (B, K, H, W)
        else:
            if self._last_fb_feat is None:
                raise RuntimeError(
                    "feedback_source='feat' but no decode has stashed a "
                    "feature map -- decode via _decode_taps before "
                    "make_feedback."
                )
            src = self._last_fb_feat                 # (B, channels, h, w)
        tok = F.adaptive_avg_pool2d(src, (g, g))     # (B, in_dim, g, g)
        tok = tok.flatten(2).transpose(1, 2)         # (B, g*g, in_dim)
        return self.fb_norm(tok)

    

    def make_feedback(
        self, logits: torch.Tensor, t: int | None = None
    ) -> torch.Tensor:
        
        return self.fb_scale * self.fb_proj(self._fb_tokens(logits))

    def _pool_taps(self,taps):
        if self.temporal_pool != "cum_mean":
            return taps
        if self._pool_m is None:
            self._pool_m = list(taps)
        else:
            t = self._pool_t
            self._pool_m = [m.detach() *(t/(t+1))+ tap /(t+1) for m, tap in zip(self._pool_m,taps)] # I am handling the fact that we can define a taps from multiple layers
        self._pool_t += 1
        return self._pool_m

    def reset_pool(self):
        self._pool_m = None
        self._pool_t = 0


    def _decode_taps(self, taps: list[torch.Tensor], output_shape, t: int|None= None) -> torch.Tensor:
        """SegMTUPerNet.forward()'s tail (everything after the encoder call),
        applied to one frame's tap list. Stashes the pre-conv_seg feature
        (detached) for "feat"-mode feedback."""
        if t == 0:
            self.reset_pool()
        taps = self._pool_taps(taps)

        feat = self.neck(taps)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        self._last_fb_feat = feat.detach()
        output = self.conv_seg(feat)
        return F.interpolate(output, size=output_shape, mode="bilinear")

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        # Non-streaming encoder: plain SegMTUPerNet behaviour.
        if not hasattr(self.encoder, "forward_stream"):
            return super().forward(img, output_shape)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # Honor the encoder's eval_horizon (BeliefEvolutionTracker's
        # per-timestamp mIoU): stop the stream at frame h and return ITS
        # logits. For this decoder family the curve therefore measures the
        # real closed-loop readout at t (feedback included), not the
        # feedback-less replay the inherited encoder.forward() would run.

        horizon = getattr(self.encoder, "eval_horizon", None)

        logits = None
        for t, taps in self.encoder.forward_stream(img, min_horizon=0):
            logits = self._decode_taps(taps, output_shape,t=t)
            if horizon is not None and t >= horizon:
                break
            self.encoder.pending_feedback = self.make_feedback(logits.detach(), t=t)
        self.encoder.pending_feedback = None  # last frame's is never consumed
        self._last_fb_feat = None
        self.reset_pool()
        return logits
