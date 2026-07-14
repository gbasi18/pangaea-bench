"""
All-horizons ("backprop the decoder at each outer step") SegTrainer for
RecursiveJEPAFeedback-family encoders, with the ENCODER KEPT FROZEN.

Under your actual configs (recursive_jepa_feedback*.yaml: n_outer_steps=1,
propagate_outer_steps=1), "each outer step" and "each frame" are the same
thing -- the encoder's `eval_horizon` diagnostic (RecursiveJEPAFeedbackEncoder
.forward: `last = eval_horizon`) already replays the recurrence exactly as if
frame h were final, full outer loop and pyramid taps included. So decoding at
every outer step reduces to: decode from every horizon h = 0..T-1, every
training step, and backpropagate the decoder from EACH of those readouts --
not just one sampled auxiliary horizon (trainer_horizon_aux.py) and not just
the true final one (SegTrainer).

Since finetune=false, this trains ONLY the decoder -- no path exists to the
encoder's frozen parameters regardless of how many horizons are decoded, or
how many outer steps they touch (see the discussion in-session: this is
explicitly NOT a mechanism for encoder feedback, only for teaching the
decoder to read out every accumulation point well).

Compute cost: T forward+backward passes through the (frozen-backbone,
trainable-decoder) model per training step, instead of 1. That's fine at
T=6, expensive at T=49 -- use `horizon_stride` to subsample horizons if T is
large (the true final horizon T-1 is always included regardless of stride).

Implementation note vs. trainer_horizon_aux.py: that trainer computed (up to)
2 horizons' losses and summed them before ONE .backward(), which is fine for
2 horizons but would require holding T decoder activation graphs in memory
at once here. Instead this trainer backpropagates after EACH horizon
(gradient accumulation: multiple .backward() calls, each adding into
.grad, before a single optimizer step), so peak memory is ~1 horizon's graph
regardless of T. Under DDP this also wraps all but the last horizon's
backward in `model.no_sync()` to skip the (otherwise wasted) all-reduce
until the final accumulation step -- a no-op for --nproc_per_node=1 runs but
required for correct/efficient multi-GPU behaviour later.

With `step_per_horizon=true` the accumulation is replaced by one optimizer
step per horizon: zero_grad -> forward -> backward -> step for each h, so
horizon h+1 is decoded with weights already updated on horizon h within the
same batch. Each step uses the horizon's full (unnormalised) loss, no_sync
is disabled (every backward must all-reduce under DDP), and the LR scheduler
still ticks once per batch so the schedule length is unchanged.
"""

import time
from contextlib import nullcontext

import torch

from pangaea.engine.trainer import SegTrainer


class AllHorizonsSegTrainer(SegTrainer):
    def __init__(
        self,
        *args,
        min_horizon: int = 1,
        horizon_stride: int = 1,
        log_horizon_losses: bool = True,
        step_per_horizon: bool = False,
        **kwargs,
    ):
        """
        Args:
            min_horizon (int): smallest horizon index decoded (0-indexed
                frame; horizon=1 means "at least frames 0 and 1
                accumulated"). Avoids training the decoder on degenerate
                1-frame beliefs. The true final horizon (T-1) is always
                included regardless of this or horizon_stride.
            horizon_stride (int): decode every `horizon_stride`-th horizon
                starting at min_horizon, instead of every single one.
                1 = every horizon (the literal "each outer step" reading);
                raise this for large T to bound the per-step compute cost.
            log_horizon_losses (bool): log each horizon's own loss to wandb
                as `train_loss_h{h}`, for diagnosing which horizons the
                decoder struggles with.
            step_per_horizon (bool): if True, run an optimizer step after
                EACH horizon's backward instead of accumulating gradients
                across all horizons into one step per batch. Horizon h+1 is
                then decoded with weights already updated on horizon h, and
                each step uses the horizon's full (unnormalised) loss. Note
                this multiplies optimizer steps per batch by n_horizons
                (the LR scheduler still steps once per batch, preserving the
                schedule length).
        """
        super().__init__(*args, **kwargs)
        if min_horizon < 0:
            raise ValueError(f"min_horizon must be >= 0, got {min_horizon}.")
        if horizon_stride < 1:
            raise ValueError(f"horizon_stride must be >= 1, got {horizon_stride}.")
        self.min_horizon = min_horizon
        self.horizon_stride = horizon_stride
        self.log_horizon_losses = log_horizon_losses
        self.step_per_horizon = step_per_horizon

    def _encoder(self):
        dec = getattr(self.model, "module", self.model)
        return getattr(dec, "encoder", None)

    def _horizons_for(self, T: int) -> list[int]:
        last = T - 1
        horizons = list(range(min(self.min_horizon, last), last, self.horizon_stride))
        horizons.append(last)  # always decode the true final horizon
        return horizons

    def train_one_epoch(self, epoch: int) -> None:
        encoder = self._encoder()
        if encoder is None or not hasattr(encoder, "eval_horizon"):
            # Not a RecursiveJEPAFeedback-family encoder: behave like SegTrainer.
            super().train_one_epoch(epoch)
            return

        self.model.train()
        end_time = time.time()
        try:
            for batch_idx, data in enumerate(self.train_loader):
                image, target = data["image"], data["target"]
                image = {
                    modality: value.to(self.device) for modality, value in image.items()
                }
                target = target.to(self.device)

                self.training_stats["data_time"].update(time.time() - end_time)

                T = image["optical"].shape[2]
                horizons = self._horizons_for(T)
                n_h = len(horizons)

                self.optimizer.zero_grad()
                total_loss_val = 0.0
                per_horizon_loss_val: dict[int, float] = {}
                logits_full = None

                for i, h in enumerate(horizons):
                    encoder.eval_horizon = None if h == T - 1 else h
                    last_horizon = i == n_h - 1
                    # When stepping per horizon every backward is immediately
                    # consumed by its own optimizer step, so DDP must
                    # all-reduce every time; no_sync only makes sense on the
                    # accumulate-then-step path.
                    sync_ctx = (
                        nullcontext()
                        if self.step_per_horizon
                        or last_horizon
                        or not hasattr(self.model, "no_sync")
                        else self.model.no_sync()
                    )
                    if self.step_per_horizon:
                        self.optimizer.zero_grad()
                    with sync_ctx:
                        with torch.autocast(
                            "cuda",
                            enabled=self.enable_mixed_precision,
                            dtype=self.precision,
                        ):
                            logits_h = self.model(image, output_shape=target.shape[-2:])
                            loss_h = self.compute_loss(logits_h, target)
                            _pop = getattr(encoder, "pop_aux_loss", None)
                            if _pop is not None:
                                _aux = _pop()
                                if _aux is not None:
                                    loss_h = loss_h + _aux
                            # accumulate mode: average across horizons so
                            # gradient scale (and thus the effective LR)
                            # doesn't depend on T. Per-horizon steps each
                            # consume one full loss, so no normalisation.
                            scaled_loss_h = (
                                loss_h if self.step_per_horizon else loss_h / n_h
                            )

                        if not torch.isfinite(scaled_loss_h):
                            raise FloatingPointError(
                                f"Rank {self.rank} got infinite/NaN loss at horizon "
                                f"{h} of batch {batch_idx}, epoch {epoch}!"
                            )

                        self.scaler.scale(scaled_loss_h).backward()

                    if self.step_per_horizon:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    # logged as the mean over horizons in both modes, so
                    # train_loss stays comparable across the two.
                    total_loss_val += loss_h.item() / n_h
                    per_horizon_loss_val[h] = loss_h.item()
                    if h == T - 1:
                        logits_full = logits_h.detach()

                if not self.step_per_horizon:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.training_stats["loss"].update(total_loss_val)
                with torch.no_grad():
                    # logged against the true-final-horizon logits, matching
                    # what the evaluator actually scores at test time.
                    self.compute_logging_metrics(logits_full, target)
                if (batch_idx + 1) % self.log_interval == 0:
                    self.log(batch_idx + 1, epoch)

                self.lr_scheduler.step()

                if self.use_wandb and self.rank == 0:
                    log_dict = {
                        "train_loss": total_loss_val,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    }
                    if self.log_horizon_losses:
                        for h, v in per_horizon_loss_val.items():
                            log_dict[f"train_loss_h{h}"] = v
                    self.wandb.log(
                        log_dict,
                        step=epoch * len(self.train_loader) + batch_idx,
                    )

                self.training_stats["batch_time"].update(time.time() - end_time)
                end_time = time.time()
        finally:
            # Never leave a truncated horizon set for the evaluator/belief
            # tracker right after this returns -- both expect the encoder's
            # default full-sequence behaviour (eval_horizon=None).
            encoder.eval_horizon = None
