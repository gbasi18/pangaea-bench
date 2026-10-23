#!/usr/bin/env bash
# TREO (Temporal-Recursive EO encoder) downstream on DynamicEarthNet, PANGAEA
# per-MONTH streaming protocol. FROZEN-encoder setting (finetune=false): the
# TREO belief is read after each month via TREOEncoder.forward_stream and only
# the UPerNet decoder is trained (StreamDecodeSegTrainer / StreamSegEvaluator).
# Same trainer/decoder/evaluator/task as the recursive_jepa & vanilla_jepa DEN
# arms -> directly comparable. You launch it yourself.
#
# Key wiring:
#   dataset=dynamicen_stream          12 monthly S2 frames, per-month labels,
#                                     supplies optical_dates (days since
#                                     2018-01-01) -> TREO's Δt conditioning.
#   encoder=treo_dynamicen            DEN-specific input-denorm stats.
#   decoder=seg_upernet_mt_none       decode one belief map per frame.
#   task=segmentation_stream_labels   per-month CE (min_horizon=0).
# The stream evaluator always scores on a non-overlapping 8x8 grid of 128px
# tiles (1024 = 8*128), so no inference_mode override is needed.
#
# Swap encoder_weights in configs/encoder/treo_dynamicen.yaml (or override on
# the CLI) to point at a newer TREO checkpoint.
set -e
cd "$(dirname "$0")"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29532 pangaea/run.py \
  --config-name=train \
  finetune=false \
  dataset=dynamicen_stream \
  encoder=treo_dynamicen \
  decoder=seg_upernet_mt_none \
  task=segmentation_stream_labels \
  preprocessing=seg_default \
  criterion=cross_entropy \
  use_wandb=true
