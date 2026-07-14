"""
Stream-decode SegTrainer: "backprop the decoder at every frame, update, then
continue" -- for RecursiveJEPAFeedbackStream-family encoders, ENCODER KEPT
FROZEN throughout.

This replaces trainer_all_horizons.py's approach. That trainer decoded every
horizon via `encoder.eval_horizon = h` (a FULL, independent replay of frames
0..h from scratch for every h) and averaged all horizons' losses into ONE
combined gradient before a single optimizer step -- which (a) cost O(T^2)
total frame-compute across a batch, and (b) diluted the true final horizon's
gradient share down to 1/n_horizons, which measurably hurt full-sequence
mIoU (confirmed in-session: 15.3 vs. 17.5 for the equivalent horizon_aux
run).

This trainer instead walks the recurrence ONCE via encoder.forward_stream()
(O(T) total, see recursive_jepa_feedback_stream_encoder.py) and, at EACH
frame, decodes, computes the loss, and takes a COMPLETE, independent
optimizer step (zero_grad -> backward -> step) before moving to the next
frame -- like ordinary sequential/online SGD over the frames of one
sequence, rather than one averaged batch gradient. The true final horizon
(t=T-1) is always the LAST update applied to that sequence, at full
strength, so there's no dilution: every frame's update is a full step, not
a fraction of one.

Since the encoder is frozen (forward_stream always runs under torch.no_grad
internally), this can only ever train the decoder -- consistent with your
instruction to keep the encoder frozen. Nothing here changes that; more
frequent decoder updates cannot create a path to the encoder's parameters.

Implementation detail: this bypasses SegMTUPerNet.forward() (which bundles
"call the encoder" with "turn features into logits" in one call) and instead
calls the decoder's post-encoder pipeline directly (`neck`,
`_forward_feature`, `dropout`, `conv_seg`, then F.interpolate) -- all
existing public/protected methods on SegUPerNet/SegMTUPerNet, so upernet.py
is untouched. NB: this also means DDP's per-parameter backward hooks (set up
on the DDP-wrapped module's __call__) are bypassed the same way -- harmless
for the --nproc_per_node=1 runs used so far, but gradient sync would not
happen correctly across multiple GPUs without further work.

No-op fallback: if the encoder doesn't expose forward_stream (i.e. it's not
RecursiveJEPAFeedbackStream), this behaves exactly like SegTrainer.
"""

import time

import torch
import torch.nn.functional as F

from pangaea.engine.trainer import SegTrainer


class StreamDecodeSegTrainer(SegTrainer):
    def __init__(
        self,
        *args,
        min_horizon: int = 1,
        log_horizon_losses: bool = True,
        loss_horizons: str = "all",
        **kwargs,
    ):
        """
        Args:
            min_horizon (int): smallest frame index decoded+updated on
                (0-indexed; horizon=1 means "at least frames 0 and 1
                accumulated"). Avoids training the decoder on degenerate
                1-frame beliefs. The true final frame (T-1) is always
                included regardless of this.
            log_horizon_losses (bool): log each frame's own loss to wandb as
                `train_loss_h{t}`, for diagnosing which frames the decoder
                struggles with.
        """
        super().__init__(*args, **kwargs)
        if min_horizon < 0:
            raise ValueError(f"min_horizon must be >= 0, got {min_horizon}.")
        if loss_horizons not in ("all", "final"):
            raise ValueError(
                f"loss_horizons must be 'all' or 'final', got {loss_horizons!r}."
            )
        self.min_horizon = min_horizon
        self.log_horizon_losses = log_horizon_losses
        # 'all': backward + optimizer step at EVERY decoded frame (original
        # behaviour). 'final': every frame is still decoded (feedback and
        # per-horizon loss logging keep working) but only the true final
        # frame gets gradient + a step -- intermediate beliefs are free to be
        # informative rather than forced to look final.
        self.loss_horizons = loss_horizons

    def _decoder(self):
        return getattr(self.model, "module", self.model)

    def _encoder(self):
        return getattr(self._decoder(), "encoder", None)

    def _after_frame_step(self, dec, encoder, t: int, logits_t: torch.Tensor) -> None:
        """Hook called after each frame's optimizer step, before the stream
        advances to the next frame. No-op here; subclasses may use it to
        inject state into the encoder (see trainer_stream_decode_fb.py)."""

    def _retain_graph(self, encoder, t: int, T: int) -> bool:
        """Whether frame t's backward must keep its tape alive (multi-frame
        feedback credit windows). False here: per-frame truncation."""
        return False

    @staticmethod
    def _frame_target(target: torch.Tensor, t: int) -> torch.Tensor:
        """Per-frame supervision target. (B, H, W): one static label for the
        whole sequence (PASTIS) -- every frame is compared against it.
        (B, T, H, W): per-frame labels (DynamicEarthNetStream) -- frame t is
        compared against ITS month's map, so the belief must track change."""
        return target[:, t] if target.dim() == 4 else target

    def _decode_from_taps(self, dec, taps: list[torch.Tensor], output_shape,t) -> torch.Tensor:
        """Mirrors SegMTUPerNet.forward()'s tail (the part after `feats =
        self.encoder(img)`), applied directly to one frame's tap list."""
        feat = dec.neck(taps)
        feat = dec._forward_feature(feat)
        feat = dec.dropout(feat)
        output = dec.conv_seg(feat)
        output = F.interpolate(output, size=output_shape, mode="bilinear")
        return output

    def train_one_epoch(self, epoch: int) -> None:
        encoder = self._encoder()
        if encoder is None or not hasattr(encoder, "forward_stream"):
            super().train_one_epoch(epoch)
            return

        dec = self._decoder()
        self.model.train()
        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            image, target = data["image"], data["target"]
            image = {
                modality: value.to(self.device) for modality, value in image.items()
            }
            target = target.to(self.device)

            self.training_stats["data_time"].update(time.time() - end_time)

            output_shape = target.shape[-2:]
            last_logits = None
            last_loss_val = None
            per_horizon_loss_val: dict[int, float] = {}

            T = image["optical"].shape[2]
            # New sequence: clear any decoder-side temporal-pool accumulator.
            # Necessary because with min_horizon > 0 the decoder never sees
            # t == 0, so its own t-keyed reset cannot fire on this path.
            reset_pool = getattr(dec, "reset_pool", None)
            if reset_pool is not None:
                reset_pool()
            for t, taps in encoder.forward_stream(image, min_horizon=self.min_horizon):
                # 'final' mode: intermediate frames are still decoded (the
                # feedback hook and per-horizon logging need the logits) but
                # get no gradient and no optimizer step.
                supervised = self.loss_horizons == "all" or t == T - 1

                with torch.set_grad_enabled(supervised and torch.is_grad_enabled()):
                    with torch.autocast(
                        "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
                    ):
                        logits_t = self._decode_from_taps(dec, taps, output_shape,t)
                        loss_t = self.compute_loss(logits_t, self._frame_target(target, t))

                if not torch.isfinite(loss_t):
                    raise FloatingPointError(
                        f"Rank {self.rank} got infinite/NaN loss at frame {t} of "
                        f"batch {batch_idx}, epoch {epoch}!"
                    )

                if supervised:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss_t).backward(
                        retain_graph=self._retain_graph(encoder, t, T)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self._after_frame_step(dec, encoder, t, logits_t)

                per_horizon_loss_val[t] = loss_t.item()
                last_logits = logits_t.detach()
                last_loss_val = loss_t.item()

            self.training_stats["loss"].update(last_loss_val)
            with torch.no_grad():
                # logged against the true-final-frame logits, matching what
                # the evaluator actually scores at test time.
                self.compute_logging_metrics(last_logits, self._frame_target(target, T - 1))
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            # one lr-schedule step per BATCH (not per frame), so the total
            # number of scheduled steps over training matches every other
            # trainer in this codebase regardless of sequence length T.
            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                log_dict = {
                    "train_loss": last_loss_val,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    **{
                        f"train_{k}": v.avg
                        for k, v in self.training_metrics.items()
                    },
                }
                if self.log_horizon_losses:
                    for t, v in per_horizon_loss_val.items():
                        log_dict[f"train_loss_h{t}"] = v
                self.wandb.log(
                    log_dict,
                    step=epoch * len(self.train_loader) + batch_idx,
                )

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()
