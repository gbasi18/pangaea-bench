#!/usr/bin/env bash
# TREO on PASTIS-HD, PANGAEA frozen-encoder protocol, at T=1 (single frame).
# This is the LEFT ENDPOINT of the mIoU-vs-T curve: identical to run_treo_pastis.sh
# in every way EXCEPT dataset.multi_temporal=1, so the ONLY difference from the
# T=6 run is the number of acquisitions fed to the recurrence. You launch it.
#
# At T=1 the dataset returns just the LAST acquisition (+ its date), TREO walks a
# single frame (propagate with Δt=0 -> incorporate one frame), and the belief is
# read out for seg_upernet_mt_none. inference_mode=whole is REQUIRED because the
# (B,T) optical_dates break the sliding-window evaluator; PASTIS is 128px =
# encoder input_size, so 'whole' gives identical coverage.
#
# Optional arg: a checkpoint path to probe (overrides encoder_weights in
# configs/encoder/treo.yaml). Handy for comparing the n_inner=1 vs n_inner=3
# pretrains at T=1:
#   ./run_treo_pastis_t1.sh /home/gianfranco/projects/2025/Project_1/nested_mae_BEN/treo_out/small_f10_20260723_133046/checkpoint-0099.pth
set -e
cd "$(dirname "$0")"

CKPT="${1:-}"
CKPT_OVERRIDE=()
if [[ -n "$CKPT" ]]; then
  CKPT_OVERRIDE=(encoder.encoder_weights="$CKPT")
fi

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29533 pangaea/run.py \
  --config-name=train \
  finetune=false \
  dataset=pastis_s2_dates \
  dataset.multi_temporal=1 \
  encoder=treo \
  "${CKPT_OVERRIDE[@]}" \
  decoder=seg_upernet_mt_none \
  task=segmentation \
  task.evaluator.inference_mode=whole \
  preprocessing=seg_default \
  criterion=cross_entropy \
  use_wandb=true
