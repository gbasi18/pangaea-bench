"""
PANGAEA encoder wrapper for the Recursive (HRM-style) JEPA with an explicit
t-1 -> t FEEDBACK channel, adapted to PASTIS Sentinel-2 time series.

This encoder adds the outer belief `z_H` CARRIED ACROSS FRAMES. 
At timestamp t the running belief from t-1 is fed back
into the outer block together with the new frame's evidence, so the model gets
"the feedback of t-1 to t and can reason on what it should improve". The
recurrence's slow/outer loop is thereby unrolled over real time.

  z_L  (inner / fast)  : persistent worker state, threaded across all T frames.
  z_H  (outer / slow)  : persistent belief; z_H_t = H_block(z_H_{t-1} + P_t + z_L).

Per frame:
  1. inject the new frame's patches  P_t = patch_embed(frame_t) + pos
  2. fold them into the running belief:  z_H <- z_H_{t-1} + P_t   (t=0: z_H = P_0)
  3. run n_inner inner steps then ONE outer step -> z_H_t (the feedback for t+1)
  4. on the FINAL frame, run the full n_outer outer loop and snapshot every
     outer cycle -> the UPerNet feature pyramid (same output_layers semantics as
     the sibling encoder, so the two are drop-in comparable in the decoder).

OOD safety (verified against RecursiveJEPA_visible_only_REAL_normed.py):
  - StreamBlock applies RMSNorm (`norm_f`) after EVERY L_block/H_block call, so
    z_H/z_L stay at unit RMS — carrying z_H across frames does NOT blow up.
  - Feeding a previous H_block output back into H_block is exactly what the
    pretrained outer loop already does, so the feedback path is in-distribution.

`inner_inject` controls what the inner loop ingests with the new frame:
  - "belief"  (default, in-distribution): inner sees z_L + z_H, matching the
    pretrained coupling; the new frame reaches the inner loop via z_H.
  - "patches" (the user's literal description): inner sees z_L + P_t directly.

"""

import importlib
import os
import sys
from logging import Logger
from pathlib import Path

import torch
import torch.nn.functional as F

from pangaea.encoders.base import Encoder


# arch flavour -> module defining `RecursiveJEPA_VisibleOnly` and the
# `recursive_jepa_vo_{tiny,small,base}` factories. MUST match the --arch used at
# pretraining or the checkpoint will not load (gated/standard weights differ).
_ARCH_MODULES = {
    "gated": "RecursiveJEPA_visible_only_REAL",
    "standard": "RecursiveJEPA_visible_only_REAL_noGated",
    "gated_normed": "RecursiveJEPA_visible_only_REAL_normed",
}


class RecursiveJEPAFeedbackEncoder(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: list[int],
        download_url: str,
        # --- recursive backbone (must match the pretrained checkpoint) ---
        repo_path: str,
        buffer_weights: str | Path | None = None,
        model_name: str = "recursive_jepa_vo_small",
        arch: str = "gated_normed",
        patch_size: int = 16,
        in_chans: int = 10,
        n_inner_steps: int = 3,
        n_outer_steps: int = 4,
        # what the inner loop ingests with each new frame:
        #   "belief"  -> z_L + z_H  (in-distribution, recommended)
        #   "patches" -> z_L + P_t  (user's literal description)
        inner_inject: str = "belief",
        # how the new frame is folded into the running belief at t>=1:
        #   None         -> z_H <- z_H_{t-1} + P_t                  (plain sum, default)
        #   float in[0,1]-> z_H <- alpha*z_H_{t-1} + (1-alpha)*P_t  (convex combination)
        # For the convex case: alpha=0.5 is symmetric; alpha->1 keeps the prior
        # belief sticky, alpha->0 forgets it. Note no alpha reproduces the sum
        # (the sum keeps belief AND evidence at full weight relative to z_L).
        alpha: float | None = None,
        # number of outer steps to advance the belief on the intermediate
        # (non-final) frames. 1 matches the "one outer step per timestamp"
        # mental model; the final frame always runs the full n_outer loop to
        # build the pyramid.
        propagate_outer_steps: int = 1,
        # where the UPerNet feature pyramid is tapped from, on the FINAL frame:
        #   "outer"  -> snapshot z_H after each OUTER CYCLE (default; output_layers
        #              index into [0, n_outer_steps-1]).
        #   "layers" -> on the LAST outer cycle, unroll H_block into its
        #              GatedBlocks and snapshot after each internal layer
        #              (output_layers index into [0, n_layers_per_stream-1]).
        #              This mirrors how the vanilla JEPA encoder taps ViT blocks.
        tap_source: str = "outer",
        # --- BEN(12-band) -> PASTIS(10-band) stem adaptation ---
        pretrain_in_chans: int = 12,
        pretrain_band_index: list[int] | None = None,
        # --- truncated BPTT through time (keep grad on the last k frames) ---
        temporal_bp_frames: int = 2,
        num_frames: int = 6,
    ) -> None:
        super().__init__(
            model_name="RecursiveJEPAFeedback",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=True,          # ingests the (B,C,T,H,W) series
            multi_temporal_output=False,  # ...and merges time internally (z_L+z_H)
            pyramid_output=False,
            download_url=download_url,
        )

        if inner_inject not in ("belief", "patches"):
            raise ValueError(
                f"inner_inject must be 'belief' or 'patches', got {inner_inject!r}."
            )

        self.repo_path = repo_path
        self.arch = arch
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.pretrain_in_chans = pretrain_in_chans
        self.inner_inject = inner_inject
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be None or in [0, 1], got {alpha}.")
        self.alpha = None if alpha is None else float(alpha)
        self.propagate_outer_steps = propagate_outer_steps
        self.pretrain_band_index = (
            list(pretrain_band_index)
            if pretrain_band_index is not None
            else [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        )
        self.temporal_bp_frames = temporal_bp_frames
        self.buffer_weights = buffer_weights
        # Diagnostic: when set to an int h, forward decodes "as if the sequence
        # ended at frame h" (frame h is treated as final -> full outer loop +
        # pyramid snapshots; frames > h are skipped). Used by the
        # BeliefEvolutionTracker for per-timestamp mIoU. None => use all T frames.
        self.eval_horizon: int | None = None

        # --- build the pretrained backbone (random init; weights loaded later) ---
        if arch not in _ARCH_MODULES:
            raise ValueError(f"Unknown arch '{arch}'. Options: {list(_ARCH_MODULES)}")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        arch_mod = importlib.import_module(_ARCH_MODULES[arch])
        model_fn = getattr(arch_mod, model_name)
        self.model = model_fn(
            image_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            n_inner_steps=n_inner_steps,
            n_outer_steps=n_outer_steps,
        )
        # The predictor is pretraining-only; drop it (saves params / DDP fuss).
        if hasattr(self.model, "predictor"):
            del self.model.predictor

        self.n_inner = self.model.n_inner_steps
        self.n_outer = self.model.n_outer_steps
        self.num_patches = self.model.num_patches
        self.grid_size = int(self.num_patches ** 0.5)
        assert self.grid_size * self.grid_size == self.num_patches

        # number of internal transformer layers inside one H_block (StreamBlock).
        self.n_layers_per_stream = len(self.model.H_block.blocks)
        if tap_source not in ("outer", "layers"):
            raise ValueError(
                f"tap_source must be 'outer' or 'layers', got {tap_source!r}."
            )
        self.tap_source = tap_source
        # how many feature taps the pyramid can index into, given the source.
        self.n_taps = self.n_outer if tap_source == "outer" else self.n_layers_per_stream

        if max(self.output_layers) >= self.n_taps:
            unit = "outer H-cycles" if tap_source == "outer" else "H_block layers"
            raise ValueError(
                f"output_layers={self.output_layers} indexes the final-frame "
                f"{unit} (tap_source={tap_source!r}), but only {self.n_taps} are "
                f"available. Use indices in [0, {self.n_taps - 1}]."
            )

    # load_encoder_weights is identical in spirit to the sibling encoder: skip
    # the regenerated pos_embed, slice the patch_embed stem to the PASTIS bands,
    # and restore the zL_init temporal seed from the non-EMA checkpoint.
    
    def load_encoder_weights(self, logger: Logger) -> None:
        if not os.path.isfile(self.encoder_weights):
            raise FileNotFoundError(
                f"encoder_weights not found: {self.encoder_weights}. Point it at "
                f"the pretrained EMA checkpoint (weight_ema-*.pth)."
            )
        payload = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "shadow" in payload:
            src = payload["shadow"]
        elif isinstance(payload, dict) and "model" in payload:
            src = payload["model"]
        else:
            src = payload

        model_sd = self.model.state_dict()
        new_sd: dict[str, torch.Tensor] = {}
        missing: dict[str, torch.Size] = {}
        incompatible: dict[str, tuple] = {}

        for name, param in model_sd.items():
            if name.endswith("pos_embed"):
                continue  # sin-cos buffer regenerated for the 8x8 PASTIS grid
            if name not in src:
                missing[name] = param.shape
                continue
            w = src[name]
            if name == "patch_embed.proj.weight" and w.shape[1] != param.shape[1]:
                if (
                    w.shape[1] == self.pretrain_in_chans
                    and param.shape[1] == len(self.pretrain_band_index)
                ):
                    w = w[:, self.pretrain_band_index, :, :]
                    logger.info(
                        "RecursiveJEPAFeedback: sliced patch_embed stem %d->%d bands "
                        "via pretrain_band_index=%s (VERIFY vs your BEN band order).",
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

        if "zL_init" in missing and self.buffer_weights is not None:
            bpayload = torch.load(
                self.buffer_weights, map_location="cpu", weights_only=False
            )
            if isinstance(bpayload, dict) and "model" in bpayload:
                bsrc = bpayload["model"]
            elif isinstance(bpayload, dict) and "shadow" in bpayload:
                bsrc = bpayload["shadow"]
            else:
                bsrc = bpayload
                
            if "zL_init" in bsrc and bsrc["zL_init"].shape == model_sd["zL_init"].shape:
                new_sd["zL_init"] = bsrc["zL_init"].clone()
                missing.pop("zL_init")
                logger.info(
                    "RecursiveJEPAFeedback: restored zL_init temporal seed from "
                    "buffer_weights=%s.", self.buffer_weights,
                )
                
            else:
                logger.warning(
                    "RecursiveJEPAFeedback: buffer_weights=%s has no compatible "
                    "zL_init; keeping the random seed.", self.buffer_weights,
                )

        self.model.load_state_dict(new_sd, strict=False)
        if "zL_init" in missing:
            logger.warning(
                "RecursiveJEPAFeedback: zL_init not in checkpoint and no "
                "buffer_weights given; using a freshly-initialised random zL_init. "
                "Set encoder.buffer_weights to the non-EMA checkpoint to restore the "
                "temporal-memory seed the pretrained recurrence was tuned against."
            )
        self.parameters_warning(missing, incompatible, logger)
        logger.info(
            "RecursiveJEPAFeedback: loaded %d tensors (n_inner=%d, n_outer=%d, "
            "inner_inject=%s, fold=%s, tap_source=%s (n_taps=%d), propagate_outer=%d, "
            "temporal_bp_frames=%d).",
            len(new_sd), self.n_inner, self.n_outer, self.inner_inject,
            "sum" if self.alpha is None else f"convex(alpha={self.alpha:.3f})",
            self.tap_source, self.n_taps, self.propagate_outer_steps,
            self.temporal_bp_frames,
        )

    def _inner_context(self, z_H: torch.Tensor, P_t: torch.Tensor) -> torch.Tensor:
        """What the inner loop adds to z_L for the current frame."""
        return z_H if self.inner_inject == "belief" else P_t

    def _fold(self, z_H: torch.Tensor | None, P_t: torch.Tensor) -> torch.Tensor:
        """Fold the new frame's evidence P_t into the running belief z_H.
        t=0 (z_H is None) seeds the belief from the first frame. Otherwise:
          alpha is None -> plain sum:           z_H + P_t
          alpha in [0,1]-> convex combination:  alpha*z_H + (1-alpha)*P_t
        """
        if z_H is None:
            return P_t
        if self.alpha is None:
            return z_H + P_t
        return self.alpha * z_H + (1.0 - self.alpha) * P_t

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = image["optical"]                      # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        pos = self.model.pos_embed                # (1, L, D), 8x8 sin-cos

        # Persistent inner worker state: init ONCE per sequence.
        z_L = self.model.zL_init.expand(B, self.num_patches, -1).to(x.device).contiguous()
        z_H = None                                # persistent belief (feedback channel)

        # Truncated BPTT through time: keep the graph only on the last k frames.
        keep_from = max(0, T - self.temporal_bp_frames)
        final_taps: list[torch.Tensor] = []

        # `last` is the frame treated as final (full outer loop + pyramid
        # snapshots). Normally T-1; the diagnostic `eval_horizon` decodes the
        # prediction as if the sequence ended earlier.
        last = T - 1 if self.eval_horizon is None else min(int(self.eval_horizon), T - 1)

        for t in range(last + 1):
            frame_grad = torch.is_grad_enabled() and (t >= keep_from)
            with torch.set_grad_enabled(frame_grad):
                P_t = self.model.patch_embed(x[:, :, t]) + pos

                # Fold the new frame into the running belief (the t-1 -> t
                # feedback): plain sum or convex combination (see _fold). t=0
                # seeds the belief directly from the first frame, matching the
                # pretrained z_H initialisation (= patches).
                z_H = self._fold(z_H, P_t)

                if t < last:
                    # Advance the belief by `propagate_outer_steps` outer cycles.
                    for _ in range(self.propagate_outer_steps):
                        
                        ctx = self._inner_context(z_H, P_t)
                        for _ in range(self.n_inner):
                            z_L = self.model.L_block(z_L + ctx)
                        z_H = self.model.H_block(z_H + z_L)
                else:
                    # FINAL frame: full outer loop to advance the belief, then
                    # tap the UPerNet pyramid from either the outer cycles or the
                    # internal H_block layers of the last cycle (see tap_source).
                    for c in range(self.n_outer):
                        ctx = self._inner_context(z_H, P_t)
                        for _ in range(self.n_inner):
                            z_L = self.model.L_block(z_L + ctx)
                        if self.tap_source == "outer":
                            # snapshot z_H after each outer cycle (post norm_f).
                            z_H = self.model.H_block(z_H + z_L)
                            final_taps.append(z_H)
                        elif c < self.n_outer - 1:
                            # earlier cycles: run H_block whole, no taps yet.
                            z_H = self.model.H_block(z_H + z_L)
                        else:
                            # LAST outer cycle: unroll H_block into its GatedBlocks
                            # and snapshot after each internal layer (vanilla-style
                            # block taps). norm_f then finishes the block normally.
                            h = z_H + z_L
                            for blk in self.model.H_block.blocks:
                                h = blk(h)
                                final_taps.append(h)
                            z_H = self.model.H_block.norm_f(h)

        outputs: list[torch.Tensor] = []
        for layer in self.output_layers:
            feat = self.model.final_norm(final_taps[layer])      # (B, L, D)
            feat = feat.transpose(1, 2).reshape(
                B, self.embed_dim, self.grid_size, self.grid_size
            ).contiguous()                                       # (B, D, 8, 8)
            outputs.append(feat)
        return outputs

    @torch.no_grad()
    def belief_cosines(self, image: dict[str, torch.Tensor]) -> list[float]:
        """Diagnostic (no grad, no side effects): mean cosine similarity between
        consecutive END-OF-FRAME beliefs z_H, i.e. cos(z_H_{t-1}, z_H_t) for
        t=1..T-1, averaged over the batch and the patch tokens.

        Mirrors `forward`'s recurrence but collects the per-frame belief instead
        of the UPerNet pyramid. cos -> 1 means the belief stopped moving
        (converged / ignoring new frames); clearly < 1 means it integrates time.
        NB the final frame runs n_outer cycles vs propagate_outer_steps for the
        earlier frames, so the LAST transition mixes extra outer depth.
        
        """
        x = image["optical"]                      # (B, C, T, H, W)
        B, _, T, _, _ = x.shape
        pos = self.model.pos_embed
        z_L = self.model.zL_init.expand(B, self.num_patches, -1).to(x.device).contiguous()
        z_H = None

        beliefs: list[torch.Tensor] = []
        for t in range(T):
            P_t = self.model.patch_embed(x[:, :, t]) + pos
            z_H = self._fold(z_H, P_t)
            n_cycles = self.n_outer if t == T - 1 else self.propagate_outer_steps
            for _ in range(n_cycles):
                ctx = self._inner_context(z_H, P_t)
                for _ in range(self.n_inner):
                    z_L = self.model.L_block(z_L + ctx)
                z_H = self.model.H_block(z_H + z_L)
            beliefs.append(z_H.float())            # end-of-frame belief e_t

        cosines: list[float] = []
        for t in range(1, T):
            cos = F.cosine_similarity(beliefs[t], beliefs[t - 1], dim=-1)  # (B, L)
            cosines.append(cos.mean().item())
        return cosines

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
