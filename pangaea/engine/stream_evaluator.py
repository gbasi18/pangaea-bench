"""
Evaluator for per-frame-label streaming segmentation (DynamicEarthNetStream).

Differences from SegEvaluator:
- target is (B, T, H, W): frame t's prediction is scored against MONTH t's
  label map, for every t (not only a final-frame readout).
- decoding runs the same closed loop the stream trainers train: iterate
  encoder.forward_stream, decode each frame (via the decoder's own
  _decode_taps when it has one -- temporal pooling / feedback stashes
  included -- else the plain UPerNet tail), and, if the decoder exposes
  make_feedback AND the encoder consumes pending_feedback, inject it
  between frames. Encoders without the attribute (VanillaJEPAStreamEncoder)
  stay feedback-free by construction.
- large tiles are evaluated on a NON-OVERLAPPING grid of encoder-sized
  crops (1024 = 8 x 128): confusion matrices are additive over disjoint
  tiles, so this is exact, not an approximation. inference_mode is accepted
  for config compatibility but tiling is always used when needed.

Metrics: overall mIoU pooled over every (frame, pixel) -- the headline
number -- plus a per-month-index mIoU curve (val_mIoU_t{t} in wandb) and the
final-month mIoU.
"""

import os
import time

import torch
import torch.nn.functional as F

import wandb
from pangaea.engine.evaluator import SegEvaluator


class StreamSegEvaluator(SegEvaluator):
    def _decode_frame(self, dec, taps, output_shape, t):
        decode = getattr(dec, "_decode_taps", None)
        if decode is not None:
            return decode(taps, output_shape, t=t)
        feat = dec.neck(taps)
        feat = dec._forward_feature(feat)
        feat = dec.dropout(feat)                    # identity in eval mode
        output = dec.conv_seg(feat)
        return F.interpolate(output, size=output_shape, mode="bilinear")

    def _stream_logits(self, dec, encoder, image, output_shape):
        """Closed-loop decode of one (tiled) sequence: list of per-frame
        logits, feedback injected between frames when both sides support it."""
        feedback_on = hasattr(dec, "make_feedback") and hasattr(
            encoder, "pending_feedback"
        )
        reset_pool = getattr(dec, "reset_pool", None)
        if reset_pool is not None:
            reset_pool()

        logits = []
        for t, taps in encoder.forward_stream(image, min_horizon=0):
            logits_t = self._decode_frame(dec, taps, output_shape, t)
            logits.append(logits_t)
            if feedback_on:
                encoder.pending_feedback = dec.make_feedback(logits_t.detach(), t=t)
        if feedback_on:
            encoder.pending_feedback = None
        if reset_pool is not None:
            reset_pool()
        return logits

    @torch.no_grad()
    def evaluate(self, model, model_name="model", model_ckpt_path=None):
        t_start = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(
                model_ckpt_path, map_location=self.device, weights_only=False
            )
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)
            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        dec = model.module
        encoder = dec.encoder
        if not hasattr(encoder, "forward_stream"):
            raise ValueError(
                "StreamSegEvaluator needs an encoder with forward_stream(); "
                f"got {type(encoder).__name__}."
            )
        input_size = encoder.input_size

        conf_per_t: torch.Tensor | None = None  # (T, K, K), lazily sized

        tag = f"Evaluating {model_name} on {self.split} set"
        for data in self.val_loader:
            image = {k: v.to(self.device) for k, v in data["image"].items()}
            target = data["target"].to(self.device)          # (B, T, H, W)
            if target.dim() != 4:
                raise ValueError(
                    "StreamSegEvaluator expects per-frame targets (B, T, H, W); "
                    f"got {tuple(target.shape)}. Use SegEvaluator for static labels."
                )
            B, T, H, W = target.shape
            if conf_per_t is None:
                conf_per_t = torch.zeros(
                    (T, self.num_classes, self.num_classes), device=self.device
                )

            # non-overlapping tile grid (exact for confusion accumulation)
            if H % input_size != 0 or W % input_size != 0:
                raise ValueError(
                    f"Eval size ({H}x{W}) must be a multiple of the encoder "
                    f"input size ({input_size}) for exact tiled evaluation."
                )
            for hs in range(0, H, input_size):
                for ws in range(0, W, input_size):
                    tile = {
                        k: (
                            v[..., hs : hs + input_size, ws : ws + input_size]
                            if v.dim() == 5
                            else v  # e.g. optical_dates (B, T): not spatial
                        )
                        for k, v in image.items()
                    }
                    logits = self._stream_logits(
                        dec, encoder, tile, (input_size, input_size)
                    )
                    for t in range(T):
                        pred = torch.argmax(logits[t], dim=1)
                        tgt = target[:, t, hs : hs + input_size, ws : ws + input_size]
                        valid = tgt != self.ignore_index
                        p, g = pred[valid], tgt[valid]
                        count = torch.bincount(
                            p * self.num_classes + g,
                            minlength=self.num_classes**2,
                        )
                        conf_per_t[t] += count.view(
                            self.num_classes, self.num_classes
                        )

        torch.distributed.all_reduce(
            conf_per_t, op=torch.distributed.ReduceOp.SUM
        )

        # headline: every (frame, pixel) pooled; plus per-month curve
        conf_total = conf_per_t.sum(dim=0).cpu()
        metrics = self.compute_metrics(conf_total)
        per_t_miou = [
            self.compute_metrics(conf_per_t[t].cpu())["mIoU"]
            for t in range(conf_per_t.shape[0])
        ]
        metrics["mIoU_per_month"] = per_t_miou
        metrics["mIoU_final_month"] = per_t_miou[-1]

        self.log_metrics(metrics)
        self.logger.info(
            f"[{self.split}] per-month mIoU: "
            + ", ".join(f"t{t}={v:.2f}" for t, v in enumerate(per_t_miou))
        )
        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    **{f"{self.split}_mIoU_t{t}": v for t, v in enumerate(per_t_miou)},
                    f"{self.split}_mIoU_final_month": per_t_miou[-1],
                }
            )

        return metrics, time.time() - t_start

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)
