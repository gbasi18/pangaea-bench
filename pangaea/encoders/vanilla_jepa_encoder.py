"""
PANGAEA encoder wrapper for a VANILLA I-JEPA ViT (Meta's I-JEPA), pretrained on
BigEarthNet — the cross-backbone baseline for the temporal-fusion study.

Role in the experiment:
  - This is a standard, non-recursive ViT trained with the plain I-JEPA
    objective. It is a per-FRAME image encoder (multi_temporal=False); time is
    fused downstream by the decoder's L-TAE (`seg_upernet_mt_ltae`) — exactly the
    same temporal head as the recursive L-TAE baseline. So pairing this with
    L-TAE isolates the BACKBONE (vanilla I-JEPA vs Recursive-JEPA), with the
    temporal-fusion mechanism held fixed.
  - NOTE the confound vs the within-backbone L-TAE baseline: this model has a
    DIFFERENT pretraining objective AND more backbone capacity (12 full
    transformer blocks vs the recursive model's two shared L/H blocks). Report it
    as a system-level comparison, not a controlled ablation of the recursion.

Feature pyramid: a plain ViT has no spatial hierarchy, so `output_layers` indexes
TRANSFORMER BLOCKS (e.g. [3,5,7,11] for the depth-12 vit_small). All taps are at
the patch-grid resolution; UPerNet's Feature2Pyramid rescales them to a pyramid,
identical to how the recursive encoder is consumed.

Band/size adaptation mirrors the recursive encoder: the 12-band BEN stem is
sliced to the 10 PASTIS optical bands, and the model is built at the PASTIS input
size so its sin-cos pos_embed matches the 8x8 grid (the pretrained pos_embed,
sized for the 224px grid, is therefore not loaded).
"""

import os
import sys
from logging import Logger
from pathlib import Path

import torch

from pangaea.encoders.base import Encoder


class VanillaJEPAEncoder(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: list[int],
        download_url: str,
        # --- I-JEPA repo + backbone (must match the pretrained checkpoint) ---
        repo_path: str,           # path to ijepa-main (the dir containing src/)
        model_name: str = "vit_small",
        patch_size: int = 16,
        in_chans: int = 10,
        # which encoder to read from the I-JEPA checkpoint: "target_encoder"
        # (EMA, the usual downstream choice) or "encoder" (context/online).
        weights_key: str = "target_encoder",
        # --- BEN(12-band) -> PASTIS(10-band) stem adaptation ---
        pretrain_in_chans: int = 12,
        pretrain_band_index: list[int] | None = None,
        num_frames: int = 6,
    ) -> None:
        super().__init__(
            model_name="VanillaIJEPA",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,         # per-frame extractor; L-TAE fuses time
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.repo_path = repo_path
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.weights_key = weights_key
        self.pretrain_in_chans = pretrain_in_chans
        self.pretrain_band_index = (
            list(pretrain_band_index)
            if pretrain_band_index is not None
            else [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        )

        # --- build the I-JEPA ViT (random init; weights loaded later) ---
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import src.models.vision_transformer as vit_mod

        model_fn = getattr(vit_mod, model_name)
        # img_size as a list (the ViT indexes img_size[0]); building at the PASTIS
        # input size makes pos_embed the 8x8 sin-cos grid directly (no interp).
        self.model = model_fn(
            img_size=[input_size], patch_size=patch_size, in_chans=in_chans
        )

        self.depth = len(self.model.blocks)
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        if max(self.output_layers) >= self.depth:
            raise ValueError(
                f"output_layers={self.output_layers} index transformer blocks, but "
                f"{model_name} has only depth={self.depth}. Use indices in "
                f"[0, {self.depth - 1}]."
            )

    def load_encoder_weights(self, logger: Logger) -> None:
        if not os.path.isfile(self.encoder_weights):
            raise FileNotFoundError(
                f"encoder_weights not found: {self.encoder_weights}. Point it at "
                f"an I-JEPA checkpoint (jepa-ep*.pth.tar)."
            )
        ck = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        if self.weights_key not in ck:
            raise KeyError(
                f"weights_key='{self.weights_key}' not in checkpoint "
                f"(keys: {list(ck.keys())})."
            )
        # strip the DDP 'module.' prefix.
        src = {k.replace("module.", "", 1): v for k, v in ck[self.weights_key].items()}

        model_sd = self.model.state_dict()
        new_sd: dict[str, torch.Tensor] = {}
        missing: dict[str, torch.Size] = {}
        incompatible: dict[str, tuple] = {}

        for name, param in model_sd.items():
            # pos_embed is a sin-cos buffer regenerated for the 8x8 PASTIS grid;
            # never load the (196-patch) pretrained one.
            if name == "pos_embed":
                continue
            if name not in src:
                missing[name] = param.shape
                continue
            w = src[name]
            # Adapt the 12-band BEN stem to the 10 PASTIS optical bands.
            if name == "patch_embed.proj.weight" and w.shape[1] != param.shape[1]:
                if (
                    w.shape[1] == self.pretrain_in_chans
                    and param.shape[1] == len(self.pretrain_band_index)
                ):
                    w = w[:, self.pretrain_band_index, :, :]
                    logger.info(
                        "VanillaIJEPA: sliced patch_embed stem %d->%d bands via "
                        "pretrain_band_index=%s (VERIFY this matches your BEN band "
                        "order).",
                        self.pretrain_in_chans,
                        len(self.pretrain_band_index),
                        self.pretrain_band_index,
                    )
                else:
                    incompatible[name] = (param.shape, w.shape)
                    continue
            if w.shape != param.shape:
                incompatible[name] = (param.shape, w.shape)
                continue
            new_sd[name] = w.clone()

        self.model.load_state_dict(new_sd, strict=False)
        self.parameters_warning(missing, incompatible, logger)
        logger.info(
            "VanillaIJEPA: loaded %d tensors from '%s' (depth=%d, taps=%s).",
            len(new_sd), self.weights_key, self.depth, self.output_layers,
        )

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # The MT decoder slices one frame per call -> (B, C, H, W).
        x = image["optical"]
        B = x.shape[0]

        x = self.model.patch_embed(x)                  # (B, N, D)
        x = x + self.model.pos_embed                   # 8x8 sin-cos (npatch == N)

        taps = set(self.output_layers)
        captured: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in taps:
                captured[i] = x

        outputs: list[torch.Tensor] = []
        for layer in self.output_layers:
            feat = captured[layer]                     # (B, N, D)
            feat = feat.transpose(1, 2).reshape(
                B, self.embed_dim, self.grid_size, self.grid_size
            ).contiguous()                             # (B, D, 8, 8)
            outputs.append(feat)
        return outputs

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
