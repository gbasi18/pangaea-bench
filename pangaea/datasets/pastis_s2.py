###
# S2-only variant of the PANGAEA PASTIS-HD dataset.
#
# The stock `pangaea.datasets.pastis.Pastis` loads three modalities in
# __getitem__ (s2 + aerial SPOT6 + s1-asc) and crashes if DATA_SPOT / DATA_S1A
# are absent. This variant loads ONLY Sentinel-2, which is the relevant (and
# comparable) setting for an optical-only foundation model.
#
# Everything that affects comparability with the PANGAEA paper is kept
# identical to the original: official fold splits (train=1,2,3 / val=4 /
# test=5), the random-50 acquisition cap, the evenly-spaced `multi_temporal`
# subsampling, raw (un-normalized) outputs — normalization is applied by the
# pangaea preprocessor from the dataset config stats — and the same
# {"image": {...}, "target", "metadata"} contract. Only the SAR/aerial
# branches are dropped.
###

import os

import numpy as np
import torch
from einops import rearrange

from pangaea.datasets.pastis import Pastis, prepare_dates


class PastisS2(Pastis):
    """PASTIS-HD restricted to the Sentinel-2 optical time series."""

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # nb_split is 1 (set by the parent); keep the same indexing arithmetic
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
        dates = prepare_dates(line["dates-S2"], self.reference_date)

        # Random cap at 50 acquisitions (identical to the stock loader).
        N = len(s2)
        if N > 50:
            idx = torch.randperm(N)[:50].sort().values
            s2 = s2[idx]
            dates = dates[idx]

        optical_ts = rearrange(s2, "t c h w -> c t h w")

        if self.multi_temporal == 1:
            # only the last frame
            optical_ts = optical_ts[:, -1]
        else:
            # evenly-spaced samples across the (capped) series
            optical_indexes = torch.linspace(
                0, optical_ts.shape[1] - 1, self.multi_temporal, dtype=torch.long
            )
            optical_ts = optical_ts[:, optical_indexes]

        return {
            "image": {
                "optical": optical_ts.to(torch.float32),  # (C, T, H, W)
            },
            "target": label.to(torch.int64),              # (H, W)
            "metadata": {},
        }
