"""
Diagnostic tracker for the Recursive-JEPA feedback encoder.

On a FIXED set of validation samples, every few epochs this records two things
and renders one figure each:

  1. belief_evolution.png  - cos(z_H_{t-1}, z_H_t) per frame transition: is the
     belief still moving across frames (integrating time) or has it converged?
  2. miou_per_timestamp.png - the segmentation mIoU obtained when the sequence is
     decoded "as if it ended at frame t" (encoder.eval_horizon = t), run through
     the REAL decoder head. Shows whether accumulating more frames actually
     improves the task, not just whether the belief changes.

Both are plotted with one colored line per epoch so you can see the trajectory
evolve over training. Active only for encoders exposing
`belief_cosines` + `eval_horizon` (RecursiveJEPAFeedback); no-op otherwise. The
trainer wires it via a small guarded hook (mirrors the `pop_aux_loss` pattern).
"""

from __future__ import annotations

import json
import os

import torch


class BeliefEvolutionTracker:
    def __init__(
        self,
        model,
        val_loader,
        device,
        exp_dir,
        n_samples: int = 4,
    ) -> None:
        # `model` is the (DDP-wrapped) decoder that owns the encoder. We call the
        # unwrapped module directly so a single rank can run forward safely.
        self.model_module = getattr(model, "module", model)
        self.encoder = self.model_module.encoder
        self.device = device
        self.exp_dir = str(exp_dir)
        self.n_samples = n_samples

        dataset = val_loader.dataset
        self.num_classes = int(dataset.num_classes)
        self.ignore_index = int(dataset.ignore_index)
        self.valid_classes = [c for c in range(self.num_classes) if c != self.ignore_index]

        self.cos_records: dict[int, list[float]] = {}
        self.miou_records: dict[int, list[float]] = {}

        # Grab `n_samples` FIXED, already-preprocessed val samples (with targets)
        # once, so every epoch is evaluated on exactly the same inputs.
        opt: list[torch.Tensor] = []
        tgt: list[torch.Tensor] = []
        dts: list[torch.Tensor] = []
        has_dates = True
        count = 0
        for data in val_loader:
            opt.append(data["image"]["optical"])
            tgt.append(data["target"])
            # carry per-frame dates when the dataset provides them, so the Δt-aware
            # encoder runs its real fold in the diagnostic instead of the constant
            # fallback. No-op for date-less datasets.
            if "optical_dates" in data["image"]:
                dts.append(data["image"]["optical_dates"])
            else:
                has_dates = False
            count += data["image"]["optical"].shape[0]
            if count >= n_samples:
                break
        self.sample = {"optical": torch.cat(opt, dim=0)[:n_samples].to(device)}
        if has_dates and dts:
            self.sample["optical_dates"] = torch.cat(dts, dim=0)[:n_samples].to(device)
        self.target = torch.cat(tgt, dim=0)[:n_samples].to(device)
        self.n_frames = self.sample["optical"].shape[2]

        # Val samples larger than the encoder's native input (e.g.
        # DynamicEarthNet's 1024px tiles vs input_size=128) would fail the
        # patch-embed size assert -- diagnose on a fixed CENTER CROP instead.
        in_size = getattr(self.encoder, "input_size", None)
        H, W = self.sample["optical"].shape[-2:]
        if in_size is not None and (H > in_size or W > in_size):
            top, left = (H - in_size) // 2, (W - in_size) // 2
            self.sample["optical"] = self.sample["optical"][
                ..., top : top + in_size, left : left + in_size
            ]
            self.target = self.target[
                ..., top : top + in_size, left : left + in_size
            ]

    def _miou(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        """mIoU over valid classes for (B, num_classes, H, W) logits vs target,
        matching pangaea's SegEvaluator (ignore_index excluded from the mean)."""
        pred = logits.argmax(dim=1)                    # (B, H, W)
        valid = target != self.ignore_index
        t = target[valid].view(-1)
        p = pred[valid].view(-1)
        n = self.num_classes
        cm = torch.bincount(t * n + p, minlength=n * n).reshape(n, n).float()
        inter = cm.diag()
        union = cm.sum(0) + cm.sum(1) - inter
        iou = inter / (union + 1e-6)
        return iou[self.valid_classes].mean().item() * 100.0

    @torch.no_grad()
    def record(self, epoch: int) -> None:
        epoch = int(epoch)
        was_training = self.model_module.training
        self.model_module.eval()                       # eval mode: SyncBN uses
        try:                                           # running stats (no sync)
            # 1) belief cosines across frames (uses all T frames).
            self.cos_records[epoch] = self.encoder.belief_cosines(self.sample)

            # 2) per-timestamp mIoU: decode as if the series ended at frame t.
            mious: list[float] = []
            for t in range(self.n_frames):
                self.encoder.eval_horizon = t
                logits = self.model_module(
                    self.sample, output_shape=self.target.shape[-2:]
                )
                # per-frame labels (B, T, H, W): horizon t is scored against
                # MONTH t's map; static labels (B, H, W) behave as before.
                tgt_t = self.target[:, t] if self.target.dim() == 4 else self.target
                mious.append(self._miou(logits, tgt_t))
            self.encoder.eval_horizon = None           # restore normal behaviour
            self.miou_records[epoch] = mious
        finally:
            self.encoder.eval_horizon = None
            if was_training:
                self.model_module.train()

    def _line_plot(self, records, x, xticklabels, ylabel, title, fname):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
        except Exception:
            return None

        epochs = sorted(records)
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(
            vmin=min(epochs), vmax=max(epochs) if len(epochs) > 1 else min(epochs) + 1
        )
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for ep in epochs:
            ax.plot(x, records[ep], marker="o", color=cmap(norm(ep)), linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="epoch")
        fig.tight_layout()
        path = os.path.join(self.exp_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)

        try:
            import wandb

            if wandb.run is not None:
                wandb.log({fname.replace(".png", ""): wandb.Image(path)}, commit=False)
        except Exception:
            pass
        return path

    def save_plot(self) -> str | None:
        if not self.cos_records:
            return None
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(os.path.join(self.exp_dir, "belief_evolution.json"), "w") as f:
            json.dump({"cosine": self.cos_records, "miou": self.miou_records}, f, indent=2)

        T = self.n_frames
        # cosine: one value per transition 1..T-1
        self._line_plot(
            self.cos_records,
            x=list(range(1, T)),
            xticklabels=[f"{t - 1}→{t}" for t in range(1, T)],
            ylabel=r"cosine$(z_H^{t-1},\, z_H^{t})$",
            title=(
                f"Belief (z_H) temporal evolution over training\n"
                f"(mean over {self.n_samples} fixed val samples; last transition "
                f"mixes final-frame outer depth)"
            ),
            fname="belief_evolution.png",
        )
        # mIoU: one value per timestamp 0..T-1 ("decoded using frames 0..t")
        miou_path = self._line_plot(
            self.miou_records,
            x=list(range(T)),
            xticklabels=[str(t) for t in range(T)],
            ylabel="mIoU (%)",
            title=(
                f"Per-timestamp mIoU over training\n"
                f"(decode as if sequence ended at frame t; "
                f"{self.n_samples} fixed val samples)"
            ),
            fname="miou_per_timestamp.png",
        )
        return miou_path or os.path.join(self.exp_dir, "belief_evolution.png")
