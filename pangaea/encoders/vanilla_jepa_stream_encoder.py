###
# Streaming wrapper for the vanilla I-JEPA encoder -- the FAIR no-memory
# counterpart to the recursive stream arms on per-frame-label tasks
# (DynamicEarthNetStream).
#
# forward_stream() yields (t, taps) with each frame encoded COMPLETELY
# INDEPENDENTLY (no state crosses frames). Because it exposes the same
# generator protocol as RecursiveJEPAFeedbackStreamEncoder, the identical
# trainer (StreamDecodeSegTrainer / *_labels supervision), identical decoder
# and identical evaluator run for both arms -- the ONLY difference between
# "vanilla" and "recursive" is what happens inside the encoder. That is the
# comparison contract.
#
# Deliberately NO `pending_feedback` attribute: the feedback trainer/decoder
# hooks check for it and stay inert, so pairing this encoder with a feedback
# decoder silently degrades to no-feedback instead of pretending to train a
# loop the encoder cannot consume.
#
# multi_temporal=True / multi_temporal_output=False are declared so that
# SegMTUPerNet treats it like the other stream encoders (single forward on
# the full series; the stream paths do the per-frame work). forward() on a
# (B, C, T, H, W) series returns the LAST frame's taps -- the same
# "final-frame readout" convention as the recursive family -- so any
# final-frame evaluator remains usable.
###

import torch

from pangaea.encoders.vanilla_jepa_encoder import VanillaJEPAEncoder


class VanillaJEPAStreamEncoder(VanillaJEPAEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the stream paths own time; SegMTUPerNet must not loop frames itself
        self.multi_temporal = True
        self.multi_temporal_output = False

    def _encode_frame(self, frame: torch.Tensor) -> list[torch.Tensor]:
        """One frame (B, C, H, W) -> tap list, via the parent's forward."""
        return VanillaJEPAEncoder.forward(self, {"optical": frame})

    def forward_stream(self, image: dict[str, torch.Tensor], min_horizon: int = 0):
        """Yields (t, taps) for t = min_horizon .. T-1; frames are encoded
        independently (no cross-frame state, no feedback consumption)."""
        x = image["optical"]                       # (B, C, T, H, W)
        T = x.shape[2]
        for t in range(T):
            if t < min_horizon:
                continue
            with torch.no_grad():
                taps = self._encode_frame(x[:, :, t])
            yield t, taps

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """(B, C, T, H, W) series -> LAST frame's taps (final-frame readout
        convention, matching the recursive stream family)."""
        x = image["optical"]
        if x.dim() == 4:                           # single frame: parent behaviour
            return VanillaJEPAEncoder.forward(self, image)
        return self._encode_frame(x[:, :, -1])
