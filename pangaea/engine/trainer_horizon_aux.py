"""
Auxiliary-horizon SegTrainer for RecursiveJEPAFeedback-family encoders.

RandomHorizonSegTrainer (trainer_random_horizon.py) fixed the mIoU "cliff" by
giving the decoder gradient at intermediate horizons, but it did so by
REPLACING the full-horizon (t=T-1) step with a short-horizon step some
fraction of the time (1 - p_full_horizon). That halves the number of
optimizer steps that actually supervise the operating point PANGAEA scores
(t=T-1), which is the likely cause of the final-mIoU regression observed
after training with it (16.7 vs. a higher full-horizon-only baseline).

This trainer instead ADDS the short-horizon loss on top of the full-horizon
loss every step, rather than trading one off against the other:

    loss = loss(full-horizon logits, target)
         + aux_weight * loss(short-horizon logits, target)   [most steps]

so every optimizer step still fully supervises t=T-1 (matching ordinary
SegTrainer behaviour and preserving that training budget), while most steps
*also* push the decoder to interpret partial beliefs. This costs an extra
forward pass through the (frozen) encoder+decoder per step (the short
horizon changes control flow inside the recursive encoder, so the two
readouts can't share one forward call) -- roughly 2x the per-step compute of
plain SegTrainer, in exchange for not diluting the full-horizon signal.

`p_aux` (default 1.0) controls what fraction of steps pay that extra
forward-pass cost; `aux_weight` controls how strongly the auxiliary horizon
is weighted relative to the full-horizon loss.

No-op fallback: if the encoder doesn't expose `eval_horizon` (i.e. it's not
RecursiveJEPAFeedback/RecursiveJEPAFeedbackDt), this behaves exactly like
SegTrainer.
"""

import random
import time

import torch

from pangaea.engine.trainer import SegTrainer


class HorizonAuxSegTrainer(SegTrainer):
    def __init__(
        self,
        *args,
        min_horizon: int = 1,
        aux_weight: float = 0.3,
        p_aux: float = 1.0,
        log_horizon: bool = True,
        **kwargs,
    ):
        """
        Args:
            min_horizon (int): smallest horizon index eligible for the
                auxiliary short-horizon sample (0-indexed frame).
            aux_weight (float): weight of the auxiliary short-horizon loss
                relative to the (always-present) full-horizon loss.
            p_aux (float): probability, each training step, of paying for
                the extra short-horizon forward pass at all. 1.0 = every
                step gets an auxiliary loss; lower values trade calibration
                signal for training throughput.
            log_horizon (bool): log the sampled auxiliary horizon and its
                loss to wandb for diagnostics.
        """
        super().__init__(*args, **kwargs)
        if not 0.0 <= p_aux <= 1.0:
            raise ValueError(f"p_aux must be in [0, 1], got {p_aux}.")
        if aux_weight < 0.0:
            raise ValueError(f"aux_weight must be >= 0, got {aux_weight}.")
        self.min_horizon = min_horizon
        self.aux_weight = aux_weight
        self.p_aux = p_aux
        self.log_horizon = log_horizon

    def _encoder(self):
        dec = getattr(self.model, "module", self.model)
        return getattr(dec, "encoder", None)

    def _sample_aux_horizon(self, T: int) -> int | None:
        """A horizon strictly short of T-1, or None if T is too small."""
        t_max = T - 1
        if t_max <= self.min_horizon:
            return None
        return random.randint(self.min_horizon, t_max - 1)

    def _forward_with_aux(self, encoder, image, target):
        """One optimizer step's worth of loss: always the full-horizon loss,
        plus (most steps) an auxiliary short-horizon loss on top."""
        encoder.eval_horizon = None
        logits = self.model(image, output_shape=target.shape[-2:])
        loss = self.compute_loss(logits, target)
        _pop = getattr(encoder, "pop_aux_loss", None)
        if _pop is not None:
            _enc_aux = _pop()
            if _enc_aux is not None:
                loss = loss + _enc_aux

        T = image["optical"].shape[2]
        aux_horizon = None
        aux_loss_val = None
        if random.random() < self.p_aux:
            aux_horizon = self._sample_aux_horizon(T)
        if aux_horizon is not None:
            encoder.eval_horizon = aux_horizon
            aux_logits = self.model(image, output_shape=target.shape[-2:])
            aux_loss = self.compute_loss(aux_logits, target)
            _pop2 = getattr(encoder, "pop_aux_loss", None)
            if _pop2 is not None:
                _enc_aux2 = _pop2()
                if _enc_aux2 is not None:
                    aux_loss = aux_loss + _enc_aux2
            aux_loss_val = aux_loss.item()
            loss = loss + self.aux_weight * aux_loss

        return logits, loss, aux_horizon, aux_loss_val

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

                with torch.autocast(
                    "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
                ):
                    logits, loss, aux_horizon, aux_loss_val = self._forward_with_aux(
                        encoder, image, target
                    )

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
                    # logged against the full-horizon logits, matching what
                    # the evaluator actually scores at test time.
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
                        log_dict["train_aux_horizon"] = (
                            -1 if aux_horizon is None else aux_horizon
                        )
                        if aux_loss_val is not None:
                            log_dict["train_aux_loss"] = aux_loss_val
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
