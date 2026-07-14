###
# Trajectory-readout variants of the carry recursion: same recurrence as
# RecursiveJEPAFeedbackStreamEncoder / ...DtEncoder, but forward() returns the
# PER-FRAME tap trajectory stacked on a temporal axis, (B, D, T, g, g), and
# declares multi_temporal_output=True -- so the DECODER's multi_temporal_strategy
# (mean) does the aggregation instead of the final belief state being decoded.
#
# Purpose: the "recursion inside, pool outside" arm of the encoder ablation.
# Each per-frame tap has seen frames 0..t through the carry; averaging the
# trajectory measures what the carry adds under the SAME aggregation as the
# vanilla+mean (34.01) and reset+mean (33.15) arms. The Dt variant additionally
# uses the exp(-dt/tau) fold, isolating the value of real acquisition timing.
#
# Pair with decoder=seg_upernet_mt_mean and plain task=segmentation. Frozen
# backbone only (forward_stream runs under no_grad, like its parent).
###

import torch

from pangaea.encoders.recursive_jepa_feedback_stream_encoder import (
    RecursiveJEPAFeedbackStreamEncoder,
)
from pangaea.encoders.recursive_jepa_feedback_dt_encoder import (
    RecursiveJEPAFeedbackDtEncoder,
)


class RecursiveJEPAFeedbackTrajEncoder(RecursiveJEPAFeedbackStreamEncoder):
    """Carry recursion with per-frame trajectory output (constant fold)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the decoder now owns the temporal axis
        self.multi_temporal_output = True

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        frames = [taps for _, taps in self.forward_stream(image, min_horizon=0)]
        n_layers = len(frames[0])
        return [
            torch.stack([f[l] for f in frames], dim=2)  # (B, D, T, g, g)
            for l in range(n_layers)
        ]


class RecursiveJEPAFeedbackDtTrajEncoder(
    RecursiveJEPAFeedbackTrajEncoder, RecursiveJEPAFeedbackDtEncoder
):
    """Same trajectory readout, with the Δt-conditioned fold.

    MRO puts Traj's forward() (stacking) and Dt's _fold() (alpha-schedule
    cursor) in effect together; forward_stream calls self._fold exactly once
    per frame in temporal order, matching the cursor contract."""

    def forward_stream(self, image: dict[str, torch.Tensor], min_horizon: int = 0):
        # Every entry point that drives the recurrence must (re)build the
        # alpha schedule, else _fold reads a stale cursor (see dt encoder).
        self._setup_alpha_schedule(image)
        yield from super().forward_stream(image, min_horizon)
