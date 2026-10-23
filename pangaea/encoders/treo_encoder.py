###
# PANGAEA encoder wrapper for TREO — the Temporal-Recursive Earth-Observation
# encoder (Δt-conditioned recurrent state model, belief Y + reasoning Z, shared
# adaLN-Zero conditioned block, per-acquisition deep supervision).
#
# This is a fresh integration of the `treo` package (github: nested_mae_BEN/treo);
# it shares nothing with the recursive_jepa / vanilla_jepa encoders.
#
# Temporal contract:
#   - input image["optical"] is (B, C, T, H, W); the recurrence walks the T
#     acquisitions, propagating the belief with Δt conditioning and ingesting
#     each frame. Acquisition dates arrive via image["optical_dates"] (B, T),
#     integer days since the dataset reference_date (PastisS2Dates). Without
#     dates it falls back to a uniform cadence (index * default_dt_days).
#
# Readout (temporal_output):
#   "last"  -> the belief AFTER the final frame: (B, D, g, g) per output layer;
#              multi_temporal_output=False -> pair with seg_upernet_mt_none.
#   "stack" -> the belief after EVERY frame: (B, D, T, g, g);
#              multi_temporal_output=True  -> pair with seg_upernet_mt_mean.
#
# Two integration points handled here:
#   1. Band adaptation. TREO was pretrained on SSL4EO-S12's 12 bands; PASTIS
#      feeds 10. The pretrained patch-embed conv is sliced to the shared bands
#      (pretrain_band_index) at load time, keeping the pretrained weights.
#   2. Normalization. PANGAEA feeds z-scored inputs (NormalizeMeanStd), but TREO
#      was pretrained on reflectance x/10000. When input_denorm=True the z-score
#      is inverted with (denorm_mean, denorm_std) — which MUST equal the
#      dataset's data_mean/data_std — and rescaled to reflectance before the
#      encoder sees it.
#
# TREO produces a single belief per frame (no per-frame pyramid), so
# output_layers must be [0].
###

import os
import sys
from logging import Logger
from pathlib import Path

import torch

from pangaea.encoders.base import Encoder


class TREOEncoder(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: list[int],
        download_url: str,
        # path to the repo exposing the `treo` package (and `util`)
        repo_path: str,
        # backbone (must match the pretrained checkpoint)
        patch_size: int = 8,
        in_chans: int = 10,
        # readout
        temporal_output: str = "last",          # "last" | "stack"
        feature: str = "belief",                # "belief" | "reason" | "belief_reason"
        # truncated BPTT through time (finetune only; no-op when frozen)
        temporal_bp_frames: int = 4,
        num_frames: int = 6,
        # Early-season protocol: fit the decoder head on the first `train_prefix`
        # acquisitions only, then keep ingesting the rest at inference. None =
        # use the whole series (ordinary behaviour). See forward().
        train_prefix: int | None = None,
        default_dt_days: float = 30.0,          # fallback cadence when dates absent
        # TREO's day-of-year phase assumes days since 1970-01-01 (pretraining
        # dates); add the dataset's reference_date in those units.
        date_offset_days: float = 0.0,
        # band adaptation 12 -> 10
        pretrain_in_chans: int = 12,
        pretrain_band_index: list[int] | None = None,
        # invert PANGAEA's z-score back to pretraining reflectance
        input_denorm: bool = True,
        reflectance_scale: float = 10000.0,
        clip_max: float = 3.0,
        denorm_mean: list[float] | None = None,
        denorm_std: list[float] | None = None,
    ) -> None:
        if temporal_output not in ("last", "stack"):
            raise ValueError(f"temporal_output must be 'last'|'stack', got {temporal_output!r}")
        if feature not in ("belief", "reason", "belief_reason"):
            raise ValueError(f"feature must be 'belief'|'reason'|'belief_reason', got {feature!r}")

        super().__init__(
            model_name="TREO",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=True,
            multi_temporal_output=(temporal_output == "stack"),
            pyramid_output=False,
            download_url=download_url,
        )

        self.repo_path = repo_path
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.temporal_output = temporal_output
        self.feature = feature
        self.temporal_bp_frames = temporal_bp_frames
        if train_prefix is not None and train_prefix < 1:
            raise ValueError(f"train_prefix must be >= 1 or None, got {train_prefix}")
        self.train_prefix = train_prefix
        self.default_dt_days = float(default_dt_days)
        self.date_offset_days = float(date_offset_days)
        self.pretrain_in_chans = pretrain_in_chans
        self.pretrain_band_index = (
            list(pretrain_band_index) if pretrain_band_index is not None
            else [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        )

        # TREO has a single belief tap per frame.
        if list(output_layers) != [0]:
            raise ValueError(f"TREO exposes one tap per frame; output_layers must be [0], got {output_layers}")
        expected_dim = embed_dim * (2 if feature == "belief_reason" else 1)
        if any(d != expected_dim for d in self.output_dim):
            raise ValueError(
                f"feature={feature!r} produces {expected_dim}-dim features but "
                f"output_dim={self.output_dim}; set encoder.output_dim={expected_dim}.")

        # --- build the backbone from the checkpoint's own cfg ---
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from treo.model import TREO, TREOConfig
        from treo.model.embed import pool_validity  # noqa: F401 (kept for parity)

        if not os.path.isfile(encoder_weights):
            raise FileNotFoundError(f"encoder_weights not found: {encoder_weights}")
        ckpt = torch.load(encoder_weights, map_location="cpu", weights_only=False)
        if "cfg" not in ckpt:
            raise KeyError("checkpoint has no 'cfg'; expected a TREO pretraining checkpoint.")
        cfg = TREOConfig.from_dict(ckpt["cfg"])   # tolerant of pre-lean ckpts
        cfg.img_size = input_size          # regenerate pos-embed for this grid
        cfg.patch_size = patch_size
        cfg.in_chans = in_chans            # build a 10-channel patch embed directly
        self.cfg = cfg
        self._ckpt = ckpt                  # consumed by load_encoder_weights

        model = TREO(cfg)
        # pretraining-only heads — drop to save params / DDP fuss
        if hasattr(model, "teacher"):
            del model.teacher
        if hasattr(model, "head"):
            del model.head
        self.model = model

        self.grid_size = cfg.img_size // cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # input denormalization buffers
        self.input_denorm = input_denorm
        self.reflectance_scale = reflectance_scale
        self.clip_max = clip_max
        if input_denorm:
            if denorm_mean is None or denorm_std is None:
                raise ValueError("input_denorm=True requires denorm_mean and denorm_std "
                                 "(= the dataset's data_mean / data_std).")
            self.register_buffer("denorm_mean",
                                 torch.tensor(list(denorm_mean)).float().view(1, -1, 1, 1),
                                 persistent=False)
            self.register_buffer("denorm_std",
                                 torch.tensor(list(denorm_std)).float().view(1, -1, 1, 1),
                                 persistent=False)

    # -- weight loading --------------------------------------------------------
    def load_encoder_weights(self, logger: Logger) -> None:
        ckpt = self._ckpt
        src = ckpt["model"] if "model" in ckpt else ckpt
        model_sd = self.model.state_dict()
        new_sd: dict[str, torch.Tensor] = {}
        missing: dict[str, torch.Size] = {}
        incompatible: dict[str, tuple] = {}

        for name, param in model_sd.items():
            if name not in src:
                missing[name] = param.shape
                continue
            w = src[name]
            if name.endswith("patch_embed.proj.weight") and w.shape[1] != param.shape[1]:
                if (w.shape[1] == self.pretrain_in_chans
                        and param.shape[1] == len(self.pretrain_band_index)):
                    w = w[:, self.pretrain_band_index, :, :]
                    logger.info("TREO: sliced patch-embed stem %d->%d via "
                                "pretrain_band_index=%s.", self.pretrain_in_chans,
                                len(self.pretrain_band_index), self.pretrain_band_index)
                else:
                    incompatible[name] = (param.shape, w.shape)
                    continue
            if w.shape != param.shape:
                incompatible[name] = (param.shape, w.shape)
                continue
            new_sd[name] = w.clone()

        self.model.load_state_dict(new_sd, strict=False)
        self.parameters_warning(missing, incompatible, logger)
        logger.info("TREO: loaded %d tensors (dim=%d, patch=%d, "
                    "temporal_output=%s, feature=%s, input_denorm=%s).",
                    len(new_sd), self.cfg.dim, self.cfg.patch_size,
                    self.temporal_output, self.feature, self.input_denorm)
        del self._ckpt

    # -- helpers ---------------------------------------------------------------
    def _to_reflectance(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,T,H,W) z-scored -> reflectance/scale, clamped, to match pretraining."""
        if self.input_denorm:
            mean = self.denorm_mean.unsqueeze(2).to(x.dtype)   # (1,C,1,1,1)
            std = self.denorm_std.unsqueeze(2).to(x.dtype)
            x = (x * std + mean) / self.reflectance_scale
            x = x.clamp(0.0, self.clip_max)
        return x

    def _tokens(self, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if self.feature == "belief":
            return Y
        if self.feature == "reason":
            return Z
        return torch.cat([Y, Z], dim=-1)

    def _to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        return tokens.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size).contiguous()

    # -- forward ---------------------------------------------------------------
    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = image["optical"]
        if x.dim() == 4:                     # single frame -> add T axis
            x = x.unsqueeze(2)
        # Early-season protocol ("train the head at the start of the season, then
        # keep feeding images as they arrive"): while TRAINING, restrict the
        # recurrence to the first `train_prefix` acquisitions, so the decoder is
        # fitted on the belief that is actually available early on. Inference is
        # untouched -- the model is in eval mode then, and the horizon evaluator
        # drives forward_stream() -- so the SAME head can afterwards be probed at
        # every longer prefix to see whether extra images help or hurt it.
        # NB this is a prefix of the series, unlike dataset.multi_temporal=k,
        # which linspaces k frames across the WHOLE season (at k=2 that is the
        # first and last acquisition, a 13-month baseline -- not "early season").
        if self.train_prefix is not None and self.training:
            x = x[:, :, : self.train_prefix]
        B, C, T, H, W = x.shape
        x = self._to_reflectance(x)

        dates = image.get("optical_dates", None)
        if dates is None:
            dates = (torch.arange(T, device=x.device, dtype=torch.float32)
                     * self.default_dt_days).unsqueeze(0).expand(B, -1)
        else:
            # keep the dates aligned with the (possibly truncated) temporal axis
            dates = (dates[:, :T].to(device=x.device, dtype=torch.float32)
                     + self.date_offset_days)

        m = self.model
        N = self.num_patches
        Y = m.Y_init.expand(B, N, -1).contiguous()
        Z = m.Z_init.expand(B, N, -1).contiguous()
        v = x.new_ones(B, N, 1)              # PANGAEA gives no cloud mask -> all valid
        d_prev = dates[:, 0]
        keep_from = max(0, T - self.temporal_bp_frames)

        beliefs: list[torch.Tensor] = []
        for t in range(T):
            grad_on = torch.is_grad_enabled() and (t >= keep_from)
            with torch.set_grad_enabled(grad_on):
                if t == keep_from and keep_from > 0:
                    Y, Z = Y.detach(), Z.detach()
                d_t = dates[:, t]
                tau, a = m.time_cond(d_t, d_prev)
                Y, Z = m.recurrence.propagate(Y, Z, tau, a)
                U = m.frame_encoder(x[:, :, t])
                Y, Z = m.recurrence.incorporate(Y, Z, U, v, tau)
                d_prev = d_t
                if self.temporal_output == "stack":
                    beliefs.append(self._to_map(self._tokens(Y, Z)))

        if self.temporal_output == "last":
            return [self._to_map(self._tokens(Y, Z))]           # [(B,D,g,g)]
        return [torch.stack(beliefs, dim=2)]                     # [(B,D,T,g,g)]

    # -- streaming forward (per-frame-label tasks: DynamicEarthNetStream) -------
    def forward_stream(self, image: dict[str, torch.Tensor], min_horizon: int = 0):
        """Streaming protocol for StreamDecodeSegTrainer / StreamSegEvaluator:
        walk the Δt-conditioned recurrence ONCE and yield (t, taps) at every
        frame >= min_horizon, with `taps` in the exact one-frame format that
        forward() returns (a single [(B, D, g, g)] belief map, since TREO has
        one tap per frame).

        Semantics match forward(): fresh pretrained inits at stream start, one
        propagate+incorporate per acquisition with Δt conditioning from
        image["optical_dates"] (uniform cadence fallback), belief read AFTER the
        frame is incorporated so month t's prediction reflects month t's frame.

        The backbone always runs FROZEN here (per-frame no_grad, scoped to close
        BEFORE the yield — a no_grad left open across a yield would silently
        disable grad in the caller's decode/backward step). Only the decoder is
        trained on this path; matches the frozen-encoder stream arms.
        """
        x = image["optical"]
        if x.dim() == 4:                     # single frame -> add T axis
            x = x.unsqueeze(2)
        B, C, T, H, W = x.shape
        x = self._to_reflectance(x)

        dates = image.get("optical_dates", None)
        if dates is None:
            dates = (torch.arange(T, device=x.device, dtype=torch.float32)
                     * self.default_dt_days).unsqueeze(0).expand(B, -1)
        else:
            dates = dates.to(device=x.device, dtype=torch.float32) + self.date_offset_days

        m = self.model
        N = self.num_patches
        Y = m.Y_init.expand(B, N, -1).contiguous()
        Z = m.Z_init.expand(B, N, -1).contiguous()
        v = x.new_ones(B, N, 1)              # PANGAEA gives no cloud mask -> all valid
        d_prev = dates[:, 0]

        for t in range(T):
            with torch.no_grad():
                d_t = dates[:, t]
                tau, a = m.time_cond(d_t, d_prev)
                Y, Z = m.recurrence.propagate(Y, Z, tau, a)
                U = m.frame_encoder(x[:, :, t])
                Y, Z = m.recurrence.incorporate(Y, Z, U, v, tau)
                d_prev = d_t
                outputs = None
                if t >= min_horizon:
                    outputs = [self._to_map(self._tokens(Y, Z))]   # [(B,D,g,g)]
            if outputs is not None:
                yield t, outputs
