"""
Stream-decode trainer with decoder->encoder feedback -- for
RecursiveJEPAFeedbackDecFb encoders paired with the SegMTUPerNetDecFb
decoder. Backbone stays FROZEN; only the decoder and its feedback adapter
(fb_proj) train.

Per frame t (inherited loop from StreamDecodeSegTrainer, plus the hook):
  1. the encoder folds frame t -- ADDING the feedback tokens computed from
     the decoder's frame t-1 prediction, if any;
  2. decode -> loss -> backward -> optimizer step. The backward reaches the
     decoder head (through this frame's readout) AND fb_proj (through the
     feedback injected at THIS frame, whose tape spans exactly one frame --
     the encoder detaches state at every frame boundary);
  3. this hook computes the NEXT frame's feedback from the just-updated
     adapter and the current prediction (detached: gradient never flows
     back into the frame that produced the logits).

So the adapter is trained by "did the feedback I injected help the NEXT
frame's prediction" -- one-step truncated credit assignment, no labels
needed at inference (the same closed loop runs inside
SegMTUPerNetDecFb.forward() at eval time).

Falls back to plain SegTrainer if the encoder has no forward_stream, and
degrades to exactly StreamDecodeSegTrainer if the decoder has no
make_feedback (feedback loop silently off).
"""

import torch

from pangaea.engine.trainer_stream_decode import StreamDecodeSegTrainer


class StreamDecodeFbSegTrainer(StreamDecodeSegTrainer):
    def _decode_from_taps(self, dec, taps, output_shape, t) -> torch.Tensor:
        # Prefer the decoder's own decode path: SegMTUPerNetDecFb stashes the
        # pre-conv_seg feature map there, which "feat"-mode feedback needs.
        decode = getattr(dec, "_decode_taps", None)
        if decode is not None:
            return decode(taps, output_shape,t)
        return super()._decode_from_taps(dec, taps, output_shape,t)

    def _after_frame_step(self, dec, encoder, t: int, logits_t: torch.Tensor) -> None:
        make_feedback = getattr(dec, "make_feedback", None)
        if make_feedback is None or not hasattr(encoder, "pending_feedback"):
            return
        # Recorded in the ambient (grad-enabled) training context: consumed
        # inside the encoder's next fold, so a later frame's loss can
        # backpropagate into the adapter. Logits detached: gradient never
        # flows back into the frame that produced them.
        encoder.pending_feedback = make_feedback(logits_t.detach(), t=t)

    def _retain_graph(self, encoder, t: int, T: int) -> bool:
        # With a k-frame credit window (encoder.bptt_frames > 1) the encoder
        # only detaches state at every k-th frame boundary, so backwards at
        # non-boundary frames must keep the window's tape alive for the next
        # frame's loss to traverse. Costs ~k frames of activation memory and
        # re-walks the shared segment once per frame in the window.
        k = int(getattr(encoder, "bptt_frames", 1))
        if k <= 1 or t >= T - 1:
            return False
        return (t + 1) % k != 0
