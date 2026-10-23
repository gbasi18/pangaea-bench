#!/usr/bin/env bash
# TREO (Temporal-Recursive EO encoder) downstream on PASTIS-HD, PANGAEA
# frozen-encoder protocol (finetune=false). dataset=pastis_s2_dates so the
# acquisition dates reach TREO's Δt conditioning. You launch it yourself.
#
# NOTE: inference_mode=whole is REQUIRED with dates — the sliding-window
# evaluator crops every image-dict entry as a 5-D tensor and chokes on the
# (B,T) optical_dates. PASTIS is 128x128 = encoder input_size, so 'whole' gives
# identical coverage anyway (matches the recursive_jepa_dt runs).
#
# Checkpoint: pass as $1, else defaults to the lean 100-epoch pilot. Override
# on the CLI or in configs/encoder/treo.yaml to point at another checkpoint.
set -e
cd "$(dirname "$0")"

CKPT="${1:-/home/gianfranco/projects/2025/Project_1/nested_mae_BEN/treo_out/small_full_20260727_101153/checkpoint-0099.pth}"

# last belief -> seg_upernet_mt_none
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  finetune=false \
  dataset=pastis_s2_dates \
  encoder=treo \
  encoder.encoder_weights="$CKPT" \
  decoder=seg_upernet_mt_none \
  task=segmentation \
  task.evaluator.inference_mode=whole \
  preprocessing=seg_default \
  criterion=cross_entropy \
  use_wandb=true

# ---- alternative: per-frame beliefs, mean-pooled by the decoder ----
# add:  encoder.temporal_output=stack  decoder=seg_upernet_mt_mean
