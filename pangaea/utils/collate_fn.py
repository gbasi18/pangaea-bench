from typing import Callable

import torch
import torch.nn.functional as F


def get_collate_fn(modalities: list[str]) -> Callable:
    def collate_fn(
        batch: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Collate function for torch DataLoader
        args:
            batch: list of dictionaries with keys 'image' and 'target'.
            'image' is a dictionary with keys corresponding to modalities and values being torch.Tensor
            of shape (C, H, W) for single images, (C, T, H, W) where T is the temporal dimension for
            time series data. 'target' is a torch.Tensor
        returns:
            dictionary with keys 'image' and 'target'
        """
        # compute the maximum temporal dimension
        T_max = 0
        for modality in modalities:
            for x in batch:
                # check if the image is a time series, i.e. has 4 dimensions
                if len(x["image"][modality].shape) == 4:
                    T_max = max(T_max, x["image"][modality].shape[1])
        # pad all images to the same temporal dimension
        for modality in modalities:
            for i, x in enumerate(batch):
                # check if the image is a time series, if yes then pad it
                # else do nothing
                if len(x["image"][modality].shape) == 4:
                    T = x["image"][modality].shape[1]
                    if T < T_max:
                        padding = (0, 0, 0, 0, 0, T_max - T)
                        batch[i]["image"][modality] = F.pad(
                            x["image"][modality], padding, "constant", 0
                        )

        # stack all images and targets
        image = {
            modality: torch.stack([x["image"][modality] for x in batch])
            for modality in modalities
        }
        # Bridge per-frame acquisition dates (kept in metadata so the per-sample
        # preprocessors leave them untouched) into the image dict AFTER
        # preprocessing, so they reach the encoder -- the only tensor channel the
        # model is handed. No-op for datasets that don't provide them. Shape (B, T),
        # aligned with the optical temporal axis (see PastisS2Dates).
        if batch and "optical_dates" in batch[0].get("metadata", {}):
            image["optical_dates"] = torch.stack(
                [x["metadata"]["optical_dates"] for x in batch]
            )

        return {
            "image": image,
            "target": torch.stack([x["target"] for x in batch]),
            "metadata": [sample["metadata"] for sample in batch]
        }

    return collate_fn
