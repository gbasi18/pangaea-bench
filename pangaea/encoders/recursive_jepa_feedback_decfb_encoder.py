###
# Decoder-feedback variant of the streaming recursive JEPA encoder.
#
# Extends RecursiveJEPAFeedbackStreamEncoder with an injection point for
# DYNAMIC decoder->encoder feedback: between frames, the caller (the decoder's
# streaming forward, or StreamDecodeFbSegTrainer) may set `pending_feedback`
# to a (B, num_patches, embed_dim) token tensor -- typically a small trainable
# adapter's projection of the decoder's frame-t prediction. The next frame's
# fold consumes it additively:
#
#     z_H_{t+1} = fold(z_H_t, P_{t+1}) + pending_feedback
#
# so the belief carried forward contains what the decoder read out at frame t.
# This is ACTIVATION-carried feedback: it needs no labels, so it runs
# identically at train and test time (unlike gradient-carried per-frame
# offsets, which would need targets at inference).
#
# Gradient policy (backbone stays FROZEN throughout):
# - Frames with no pending feedback run under torch.no_grad(), exactly like
#   the parent's forward_stream.
# - Frames WITH feedback run in the caller's ambient grad mode: the autograd
#   tape then spans this ONE frame, from the injected feedback through the
#   (frozen-weight) fold/outer-loop/taps to whatever the caller decodes. The
#   backbone's requires_grad=False means only the feedback path (i.e. the
#   decoder-side adapter that produced it) accumulates gradient.
# - z_H / z_L are detached at every bptt_frames-th frame boundary (after the
#   yield, i.e. after the caller has backpropped and freed that frame's
#   tape), so backprop is truncated to the window and values-only state
#   carries forward.
#
# pending_feedback is cleared at the start of every forward_stream() walk
# (never leak one sequence's feedback into the next) and after each
# consumption (used exactly once).
###

import torch

from pangaea.encoders.recursive_jepa_feedback_stream_encoder import (
    RecursiveJEPAFeedbackStreamEncoder,
)


class RecursiveJEPAFeedbackDecFbEncoder(RecursiveJEPAFeedbackStreamEncoder):
    """RecursiveJEPAFeedbackStreamEncoder + additive decoder-feedback tokens
    injected into the fold of the next frame (see module docstring)."""

    def __init__(self, *args, bptt_frames: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if bptt_frames < 1:
            raise ValueError(f"bptt_frames must be >= 1, got {bptt_frames}.")
        # How many frames the feedback credit-assignment window spans: state
        # is detached every bptt_frames frames (1 = per-frame truncation, the
        # original behaviour). The trainer must pass retain_graph=True on
        # non-boundary backwards (StreamDecodeFbSegTrainer._retain_graph).
        self.bptt_frames = int(bptt_frames)
        # Set by the caller between frames; consumed (and cleared) by the
        # next frame's fold. Shape (B, num_patches, embed_dim).
        self.pending_feedback: torch.Tensor | None = None
        # Optional ALWAYS-ON feedback hook: a callable t -> (1, num_patches,
        # embed_dim) tensor (or None), set once by the decoder at init (see
        # SegMTUPerNetStatFb). Unlike pending_feedback it is input-independent
        # and added at EVERY frame, t=0 included -- so the loss at frame t
        # trains exactly the tensor that was added at frame t (same-frame
        # credit), not the previous frame's injection. Deliberately a plain
        # callable, not a registered parameter: the tensor it returns must
        # stay owned by the decoder so it survives the encoder-freezing logic
        # and joins the optimizer via decoder.parameters().
        self.static_fb_fn = None

    def _frame_step(self, z_H, z_L, P_t, fb):
        """One frame of the recurrence: fold + optional additive feedback,
        full outer-cycle loop; returns the new states and per-cycle taps."""
        z_H = self._fold(z_H, P_t)
        if fb is not None:
            z_H = z_H + fb
        taps: list[torch.Tensor] = []
        for c in range(self.n_outer):
            ctx = self._inner_context(z_H, P_t)
            for _ in range(self.n_inner):
                z_L = self.model.L_block(z_L + ctx)
            if self.tap_source == "outer":
                z_H = self.model.H_block(z_H + z_L)
                taps.append(z_H)
            elif c < self.n_outer - 1:
                z_H = self.model.H_block(z_H + z_L)
            else:
                # LAST outer cycle, tap_source="layers": unroll H_block and
                # snapshot after each internal GatedBlock (parent forward()'s
                # final-frame semantics, applied per frame of the stream).
                h = z_H + z_L
                for blk in self.model.H_block.blocks:
                    h = blk(h)
                    taps.append(h)
                z_H = self.model.H_block.norm_f(h)
        return z_H, z_L, taps

    def forward_stream(self, image: dict[str, torch.Tensor], min_horizon: int = 0):
        x = image["optical"]                      # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        pos = self.model.pos_embed

        # Never start a sequence with feedback left over from a previous one.
        self.pending_feedback = None

        with torch.no_grad():
            z_L = self.model.zL_init.expand(B, self.num_patches, -1).to(x.device).contiguous()
        z_H = None

        for t in range(T):
            with torch.no_grad():
                P_t = self.model.patch_embed(x[:, :, t]) + pos

            fb = self.pending_feedback
            self.pending_feedback = None
            if self.static_fb_fn is not None:
                static = self.static_fb_fn(t)   # (1, L, D), broadcasts over B
                if static is not None:
                    fb = static if fb is None else fb + static

            if fb is None:
                with torch.no_grad():
                    z_H, z_L, taps = self._frame_step(z_H, z_L, P_t, None)
            else:
                z_H, z_L, taps = self._frame_step(z_H, z_L, P_t, fb)

            if t >= min_horizon:
                outputs: list[torch.Tensor] = []
                for layer in self.output_layers:
                    feat = self.model.final_norm(taps[layer])        # (B, L, D)
                    feat = feat.transpose(1, 2).reshape(
                        B, self.embed_dim, self.grid_size, self.grid_size
                    ).contiguous()                                    # (B, D, g, g)
                    outputs.append(feat)
                yield t, outputs

            if (t + 1) % self.bptt_frames == 0:
                z_H = z_H.detach()
                z_L = z_L.detach()
