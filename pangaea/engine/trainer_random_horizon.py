"""
Randomized-horizon SegTrainer for RecursiveJEPAFeedback-family encoders.

Motivation (see miou_per_timestamp.png diagnostics): during ordinary
training, the frozen encoder's belief is only ever read out through the
FULL n_outer loop / pyramid taps at the true final frame T-1
(RecursiveJEPAFeedbackEncoder.forward: `last = T-1` whenever
`eval_horizon` is None). The decoder therefore never receives segmentation
loss gradient for any other accumulation point along the recurrence, which
is why decoding "as if the sequence ended early" (encoder.eval_horizon = t,
t < T-1) looks close to the untrained floor at eval time and only becomes
useful once t reaches T-1 (the mIoU "cliff").

This trainer duplicates Trainer.train_one_epoch's loop but, before each
training forward pass, randomly sets encoder.eval_horizon to a horizon
short of T-1 (with probability 1 - p_full_horizon) so the decoder also
gets gradient at intermediate accumulation points. With probability
p_full_horizon the true full sequence (eval_horizon=None) is used instead,
matching ordinary training, so the metric PANGAEA actually reports
(full-sequence mIoU) doesn't regress.

No-op fallback: if the encoder doesn't expose `eval_horizon` (i.e. it's not
RecursiveJEPAFeedback/RecursiveJEPAFeedbackDt), this behaves exactly like
SegTrainer.
"""

import random
import time

import torch

from pangaea.engine.trainer import SegTrainer


class RandomHorizonSegTrainer(SegTrainer):
    def __init__(
        self,
        *args,
        min_horizon: int = 1,
        p_full_horizon: float = 0.5,
        log_horizon: bool = True,
        **kwargs,
    ):
        """
        Args:
            min_horizon (int): smallest horizon index eligible for random
                sampling (0-indexed frame; horizon=1 means "at least frames
                0 and 1 accumulated"). Avoids training the decoder on
                degenerate 1-frame beliefs.
            p_full_horizon (float): probability of using the true full
                sequence (eval_horizon=None) instead of a random short
                horizon, each training step.
            log_horizon (bool): log the sampled horizon to wandb as
                `train_horizon` for diagnostics.
        """
        super().__init__(*args, **kwargs)
        if not 0.0 <= p_full_horizon <= 1.0:
            raise ValueError(f"p_full_horizon must be in [0, 1], got {p_full_horizon}.")
        self.min_horizon = min_horizon
        self.p_full_horizon = p_full_horizon
        self.log_horizon = log_horizon

    def _encoder(self):
        dec = getattr(self.model, "module", self.model)
        return getattr(dec, "encoder", None)

    def _sample_horizon(self, T: int) -> int | None:
        """None => full sequence (t = T-1); otherwise an int horizon < T-1."""
        t_max = T - 1
        if t_max <= self.min_horizon or random.random() < self.p_full_horizon:
            return None
        return random.randint(self.min_horizon, t_max - 1)

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
                horizon = self._sample_horizon(T)
                encoder.eval_horizon = horizon

                with torch.autocast(
                    "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
                ):
                    logits = self.model(image, output_shape=target.shape[-2:])
                    loss = self.compute_loss(logits, target)
                    # Optional auxiliary loss stashed by the encoder (mirrors
                    # the base Trainer's pop_aux_loss hook).
                    _pop = getattr(encoder, "pop_aux_loss", None)
                    if _pop is not None:
                        _aux = _pop()
                        if _aux is not None:
                            loss = loss + _aux

                self.optimizer.zero_grad()

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Rank {self.rank} got infinite/NaN loss at batch "
                        f"{batch_idx} of epoch {epoch}!"
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.training_stats["loss"].update(loss.item())
                with torch.no_grad():
                    self.compute_logging_metrics(logits, target)
                if (batch_idx + 1) % self.log_interval == 0:
                    self.log(batch_idx + 1, epoch)

                self.lr_scheduler.step()

                if self.use_wandb and self.rank == 0:
                    log_dict = {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    }
                    if self.log_horizon:
                        log_dict["train_horizon"] = (T - 1) if horizon is None else horizon
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
