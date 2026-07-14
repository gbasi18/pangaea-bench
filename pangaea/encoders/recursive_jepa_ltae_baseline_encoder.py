"""
Fairness baseline for the Recursive-JEPA temporal-fusion experiment.

Same backbone, same pretrained weights, same per-image spatial recurrence and
same outer-cycle feature pyramid as `RecursiveJEPAEncoder` — but the temporal
memory `z_L` is RESET every frame (no cross-frame threading). Each timestamp is
encoded independently, and time is fused downstream by the decoder's L-TAE
(`seg_upernet_mt_ltae`) instead of by the recursion.

So the ONLY difference vs `RecursiveJEPAEncoder` is the temporal-fusion
mechanism: recursion-as-temporal-merge (theirs) vs attention-as-temporal-merge
(this). Backbone, weights, band adaptation, spatial recursion depth, pyramid
tapping, decoder family, data, preprocessing and metrics are all held fixed —
which is exactly what makes this the control for "does temporal-fusion-by-
recursion help?".

Implementation note: this is presented to pangaea as a SINGLE-image encoder
(`multi_temporal=False`). pangaea's `SegMTUPerNet` therefore loops over the T
frames itself, calls this encoder once per frame, stacks the per-frame pyramids
to (B, C, T, H, W) and applies L-TAE (`upernet.py:342-364`). Each forward sees
one frame (B, C, H, W) and re-seeds `z_L` from `zL_init` — that re-seed is the
removal of temporal threading.
"""

from logging import Logger

import torch

from pangaea.encoders.recursive_jepa_feedback_encoder import RecursiveJEPAFeedbackEncoder


class RecursiveJEPALTAEBaselineEncoder(RecursiveJEPAFeedbackEncoder):
    """Per-frame Recursive-JEPA features; time fused by the decoder's L-TAE.

    Reuses `RecursiveJEPAFeedbackEncoder.__init__` / `load_encoder_weights`
    verbatim (backbone build, BEN->PASTIS stem slice, zL_init restore; original
    parent was the since-removed `RecursiveJEPAEncoder`). Only the temporal
    contract and the forward differ: no state is carried across frames, so none
    of the parent's feedback/fold machinery is ever used.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Present as single-temporal so the MT decoder handles time + L-TAE.
        # (Parent set multi_temporal=True for the recursion-fuses-time variant.)
        self.multi_temporal = False
        self.multi_temporal_output = False
        self.model_name = "RecursiveJEPA_LTAEbaseline"

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # The MT decoder slices one frame per call -> (B, C, H, W).
        x = image["optical"]
        B = x.shape[0]
        pos = self.model.pos_embed                                  # (1, L, D)

        # Temporal memory is re-seeded EVERY frame (no threading across time):
        # this is the single change that removes temporal-fusion-by-recursion.
        z_L = self.model.zL_init.expand(
            B, self.num_patches, -1
        ).to(x.device).contiguous()
        z_H = self.model.patch_embed(x) + pos

        # Same per-image spatial recurrence as the main encoder; the outer-cycle
        # snapshots become this frame's feature pyramid.
        taps: list[torch.Tensor] = []
        for _ in range(self.n_outer):
            for _ in range(self.n_inner):
                z_L = self.model.L_block(z_L + z_H)
            z_H = self.model.H_block(z_H + z_L)
            taps.append(z_H)

        outputs: list[torch.Tensor] = []
        for layer in self.output_layers:
            feat = self.model.final_norm(taps[layer])               # (B, L, D)
            feat = feat.transpose(1, 2).reshape(
                B, self.embed_dim, self.grid_size, self.grid_size
            ).contiguous()                                          # (B, D, 8, 8)
            outputs.append(feat)
        return outputs

    def load_encoder_weights(self, logger: Logger) -> None:
        super().load_encoder_weights(logger)
        logger.info(
            "RecursiveJEPA L-TAE baseline: z_L re-seeded per frame; temporal "
            "fusion delegated to the decoder L-TAE (n_outer=%d spatial cycles).",
            self.n_outer,
        )
