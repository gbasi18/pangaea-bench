###
# DynamicEarthNet, MONTH-STREAM formulation, Sentinel-2 modality.
#
# Data source: the TUM mediatum redistribution (labels + monthly Sentinel-2
# GeoTIFFs + splits.csv), NOT the pangaea/dynnet preprocessed npy tree that
# utae_dynamicen.py consumes. Layout under root_path:
#   splits.csv                      one row per (AOI, month): paths + flags
#   sentinel2/{aoi}/YYYY_MM.tif     12-band int16, 1024x1024 (S2 L2A minus
#                                   B10, resampled onto the 3m label grid)
#   labels/{aoi}_{zone}/Labels/Raster/.../*-YYYY_MM_01.tif
#                                   7-band uint8 one-hot (0/255); band order
#                                   [impervious, agriculture, forest&other,
#                                    wetlands, soil, water, snow&ice]
#
# Task: sample = one AOI's sequence of `multi_temporal` consecutive available
# months (one S2 image per month), target = the (T, H, W) stack of monthly
# class maps (argmax of the one-hot; snow&ice and unlabelled -> ignore_index).
# Land cover CHANGES month-to-month, so frame t must be decoded as the state
# at month t -- the formulation where a recurrent belief state has a
# structural job that permutation-invariant pooling cannot do.
#
# Splits: the public labels cover only DEN's 54 train AOIs (val/test were
# challenge-held-out) and the canonical UTAE-dynnet split files are no longer
# hosted, so this class carves a DETERMINISTIC AOI-level split from the
# sorted AOI names: last `test_aois` -> test, previous `val_aois` -> val,
# rest -> train. Reproducible, and identical across all encoder arms.
#
# metadata["optical_dates"] = days since 2018-01-01 per frame, same
# convention as PastisS2Dates -- dt encoders work unchanged.
###

import csv
import os
from datetime import datetime

import numpy as np
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset

# one-hot band 7 (index 6) is snow & ice: not among the 6 evaluated classes
_SNOW_BAND = 6


class DynamicEarthNetStream(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        val_aois: int = 8,
        test_aois: int = 8,
    ):
        """multi_temporal = number of MONTHS per sample."""
        super(DynamicEarthNetStream, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = root_path
        self.split = split
        self.val_aois = int(val_aois)
        self.test_aois = int(test_aois)
        self.reference_date = datetime(2018, 1, 1)

        self.set_files()

    def set_files(self):
        csv_path = os.path.join(self.root_path, "splits.csv")
        per_aoi: dict[str, list[tuple[str, str, str]]] = {}
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r["missing_s2"] == "True" or r["missing_label"] == "True":
                    continue
                aoi = r["s2_path"].split("/")[1]
                per_aoi.setdefault(aoi, []).append(
                    (r["year_month"], r["s2_path"], r["label_path"])
                )

        # deterministic AOI-level split from sorted names
        aois = sorted(per_aoi.keys())
        n_val, n_test = self.val_aois, self.test_aois
        split_aois = {
            "test": aois[-n_test:],
            "val": aois[-(n_test + n_val) : -n_test],
            "train": aois[: -(n_test + n_val)],
        }[self.split]

        K = self.multi_temporal
        self.sequences: list[dict] = []
        for aoi in split_aois:
            months = sorted(per_aoi[aoi], key=lambda m: m[0])  # "YYYY-MM" sorts
            for s in range(0, len(months) - K + 1, K):
                window = months[s : s + K]
                image_files, label_files, dates = [], [], []
                ok = True
                for ym, s2_path, label_path in window:
                    img = os.path.join(self.root_path, s2_path)
                    lbl = os.path.join(self.root_path, label_path)
                    if not (os.path.isfile(img) and os.path.isfile(lbl)):
                        ok = False
                        break
                    image_files.append(img)
                    label_files.append(lbl)
                    y, m = map(int, ym.split("-"))
                    dates.append((datetime(y, m, 1) - self.reference_date).days)
                if ok:
                    self.sequences.append(
                        {"images": image_files, "labels": label_files, "dates": dates}
                    )
        if not self.sequences:
            raise RuntimeError(
                f"DynamicEarthNetStream[{self.split}]: no complete {K}-month "
                f"windows found under {self.root_path}."
            )

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def _read_label(path: str) -> np.ndarray:
        """7-band 0/255 one-hot -> (H, W) int64 class map; snow&ice and
        unlabelled pixels -> -1."""
        with rasterio.open(path) as f:
            onehot = f.read()                          # (7, H, W) uint8
        label = onehot.argmax(axis=0).astype(np.int64)
        label[onehot.max(axis=0) == 0] = -1            # unlabelled
        label[label == _SNOW_BAND] = -1                # snow&ice: ignored
        return label

    def __getitem__(self, index):
        seq = self.sequences[index]

        frames = []
        for p in seq["images"]:
            with rasterio.open(p) as f:
                frames.append(f.read())                # (12, H, W) int16
        # keep int16 until after the (cheap) crop; NormalizeMeanStd promotes
        # to float on the cropped tensor, so per-sample memory stays low.
        images = torch.from_numpy(np.stack(frames, axis=1).astype(np.int16))
        # (C, T, H, W)

        target = torch.from_numpy(
            np.stack([self._read_label(p) for p in seq["labels"]], axis=0)
        )                                              # (T, H, W) int64

        return {
            "image": {"optical": images},
            "target": target,
            "metadata": {
                "optical_dates": torch.tensor(seq["dates"], dtype=torch.long),
            },
        }

    @staticmethod
    def download(self, silent=False):
        pass
