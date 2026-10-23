###
# S2-only PASTIS-HD variant that ALSO returns the acquisition dates.
#
# Identical to `pangaea.datasets.pastis_s2.PastisS2` in every way that affects
# the imagery (official fold splits, random-50 acquisition cap kept in
# chronological order, evenly-spaced `multi_temporal` subsampling, raw
# un-normalized outputs). The ONLY addition is that the per-frame Sentinel-2
# acquisition dates -- integer days since `self.reference_date`, as produced by
# `prepare_dates` -- are threaded through the exact same index operations as the
# frames and returned in `metadata["optical_dates"]`, aligned 1:1 with the T
# frames of `image["optical"]`.
#
# This is what lets a temporal aggregator (e.g. the recursive-JEPA feedback
# encoder) compute the real inter-acquisition gaps Δt for time-aware folding,
# instead of treating the irregularly-sampled series as if it were uniform.
###

import os

import numpy as np
import torch
from einops import rearrange

from pangaea.datasets.pastis import prepare_dates
from pangaea.datasets.pastis_s2 import PastisS2


class PastisS2Dates(PastisS2):
    """PastisS2 that additionally returns the S2 acquisition dates in metadata."""

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        line = self.meta_patch.iloc[i // (self.nb_split * self.nb_split)]
        name = line["ID_PATCH"]

        # --- semantic target: channel 0 of TARGET, void class = ignore_index ---
        label = torch.from_numpy(
            np.load(
                os.path.join(self.root_path, "ANNOTATIONS", "TARGET_" + str(name) + ".npy")
            )[0].astype(np.int32)
        )

        # --- Sentinel-2 time series: DATA_S2/S2_<id>.npy  [T, 10, H, W] ---
        s2 = torch.from_numpy(
            np.load(os.path.join(self.root_path, "DATA_S2", "S2_{}.npy".format(name)))
        )
        # integer days since self.reference_date, one per acquisition (shape (T,))
        dates = prepare_dates(line["dates-S2"], self.reference_date)

        # Random cap at 50 acquisitions drawn EXACTLY as upstream PANGAEA does
        # (randperm(N)[:50], unsorted), then the upstream frame selection on
        # that order. Only the selected T indices are sorted afterwards, so the
        # model sees the same frames as upstream, in chronological order (a
        # recurrent encoder needs time-ordered input; upstream feeds tile
        # t31tfj, N=61, in random order). Dates follow the same indices.
        N = len(s2)
        idx = torch.randperm(N)[:50] if N > 50 else torch.arange(N)

        if self.multi_temporal == 1:
            # Only the last frame of the capped order, as upstream. Keep a
            # length-1 T axis: the preprocessor requires (C, T, H, W).
            sel = idx[-1:]
        else:
            # evenly-spaced positions across the (capped) series
            sel = idx[torch.linspace(0, len(idx) - 1, self.multi_temporal,
                                     dtype=torch.long)].sort().values
        optical_ts = rearrange(s2[sel], "t c h w -> c t h w")
        dates = dates[sel]

        return {
            "image": {
                "optical": optical_ts.to(torch.float32),      # (C, T, H, W)
            },
            "target": label.to(torch.int64),                  # (H, W)
            "metadata": {
                # (T,) integer days since reference_date, aligned with optical's T axis
                "optical_dates": dates.to(torch.long),
            },
        }
