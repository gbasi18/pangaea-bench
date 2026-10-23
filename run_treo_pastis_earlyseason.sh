#!/usr/bin/env bash
# "Train the decoder head early in the season, then keep feeding images as they
# arrive, reusing the SAME head and TREO's recurrence to fold them in."
#
# Both arms share one dataset grid: dataset.multi_temporal=12, i.e. ~monthly
# acquisitions across the season. A prefix of length k is therefore literally
# "the season so far, at month k" -- unlike dataset.multi_temporal=k, which
# linspaces k frames across the WHOLE year (at k=2 that is the first and last
# acquisition, a 13-month baseline, NOT an early-season pair).
#
# Both arms are evaluated by HorizonSegEvaluator, which walks forward_stream
# once and reports mIoU after every acquisition -> `val_mIoU_T{k}` for k=1..12.
# That curve IS the answer to "does performance improve as more images arrive".
#
#   naive (arm 1)  encoder.train_prefix=K: the head sees ONLY the first K
#                  acquisitions during training, then is probed at every longer
#                  prefix. This is the protocol exactly as described -- train
#                  early, extend later, no retraining. Expect the belief to
#                  drift off the head's training distribution as k grows; the
#                  curve quantifies how fast.
#
#   allhz (arm 2)  the same single head, but trained at EVERY prefix
#                  (StreamDecodeSegTrainer, one optimizer step per horizon).
#                  Still ONE decoder used at all k -- it just learned to read
#                  beliefs of every age. This is the version that can actually
#                  improve as the season unfolds, and it is the control that
#                  says whether arm 1's decay is a readout artifact (arm 2 rises
#                  where arm 1 falls) or a real limit of the recurrence (both
#                  fall).
#
# Usage: ./run_treo_pastis_earlyseason.sh [naive|allhz|both] [K] [T] [epochs]
#   K       training prefix for the naive arm (default 3 = first ~3 months)
#   T       season grid size (default 12)
set -e
cd "$(dirname "$0")"

ARM="${1:-both}"
K="${2:-3}"
T="${3:-12}"
EPOCHS="${4:-80}"

VENV=/home/gianfranco/projects/2025/Project_1/nested_mae_BEN/.venv/bin
export PYTHONPATH=/home/gianfranco/projects/2025/Project_1/pangaea-bench-old

common=(
  --config-name=train
  finetune=false
  dataset=pastis_s2_dates
  dataset.multi_temporal="$T"
  encoder=treo
  decoder=seg_upernet_mt_none
  preprocessing=seg_default
  criterion=cross_entropy
  use_wandb=true
)

run_naive () {
  echo "=== arm 1 NAIVE: head trained on first $K of $T acquisitions, probed at every prefix ==="
  $VENV/torchrun --nnodes=1 --nproc_per_node=1 --master_port=29650 pangaea/run.py \
    "${common[@]}" \
    encoder.train_prefix="$K" \
    task=segmentation_horizon_eval \
    task.trainer.n_epochs="$EPOCHS"
}

run_allhz () {
  echo "=== arm 2 ALL-HORIZON: one head trained at every prefix of $T ==="
  $VENV/torchrun --nnodes=1 --nproc_per_node=1 --master_port=29651 pangaea/run.py \
    "${common[@]}" \
    task=segmentation_horizon_curve \
    task.trainer.n_epochs="$EPOCHS"
}

case "$ARM" in
  naive) run_naive ;;
  allhz) run_allhz ;;
  both)  run_naive; run_allhz ;;
  *) echo "unknown arm: $ARM (want naive|allhz|both)"; exit 1 ;;
esac
