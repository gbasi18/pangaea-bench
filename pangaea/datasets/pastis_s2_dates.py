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

        # Random cap at 50 acquisitions, kept in chronological order so the
        # returned series -- and its dates -- stay time-ordered.
        N = len(s2)
        if N > 50:
            idx = torch.randperm(N)[:50].sort().values
            s2 = s2[idx]
            dates = dates[idx]

        optical_ts = rearrange(s2, "t c h w -> c t h w")

        if self.multi_temporal == 1:
            # only the last frame; keep its date as a length-1 series so the
            # contract (dates aligned with the temporal axis) still holds.
            optical_ts = optical_ts[:, -1]
            dates = dates[-1:]
        else:
            # evenly-spaced samples across the (capped) series -- apply the SAME
            # indices to the dates so image[t] <-> dates[t] stays aligned.
            optical_indexes = torch.linspace(
                0, optical_ts.shape[1] - 1, self.multi_temporal, dtype=torch.long
            )
            optical_ts = optical_ts[:, optical_indexes]
            dates = dates[optical_indexes]

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
