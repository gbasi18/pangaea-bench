#!/usr/bin/env bash
# TREO on PASTIS-HD: mIoU as a function of HOW MANY ACQUISITIONS the recurrence
# has seen. Frozen encoder (finetune=false), decoder = seg_upernet_mt_none,
# dataset = pastis_s2_dates so real acquisition dates reach TREO's Dt conditioning.
#
# Two arms, same encoder, same decoder, same evaluator -> directly comparable:
#
#   calib  (task=segmentation_horizon_curve)  StreamDecodeSegTrainer decodes and
#          updates at EVERY horizon, so the readout is calibrated across all
#          belief ages. Its curve measures INFORMATION accumulated in the belief.
#
#   extrap (task=segmentation_horizon_eval)   ordinary SegTrainer: the decoder
#          only ever sees the full-T belief. Its curve measures whether that
#          readout EXTRAPOLATES down to shorter beliefs (the "mIoU cliff").
#
# Both report, per epoch, `val_mIoU_T{k}` for k = 1..T (see
# pangaea/engine/horizon_evaluator.py). The headline `mIoU` is the FINAL
# horizon, numerically identical to what SegEvaluator reports, so these runs
# stay comparable to every other PASTIS run in this repo.
#
# NOTE the two curves mean different things along the x axis:
#   - WITHIN a run, horizon k = the first k of T evenly-spaced acquisitions,
#     i.e. a growing PREFIX of the season (the causal/streaming question).
#   - ACROSS runs with different T, the final horizon = the whole season at
#     different sampling DENSITY (the sample-efficiency question). Use
#     run_treo_pastis_density.sh for that axis.
#
# Usage:  ./run_treo_pastis_horizon.sh [arm] [T] [epochs] [ckpt]
#   arm    calib | extrap | both     (default: both)
#   T      dataset.multi_temporal    (default: 12)
#   epochs                           (default: 80)
set -e
cd "$(dirname "$0")"

ARM="${1:-both}"
T="${2:-12}"
EPOCHS="${3:-80}"
CKPT="${4:-}"

VENV=/home/gianfranco/projects/2025/Project_1/nested_mae_BEN/.venv/bin
export PYTHONPATH=/home/gianfranco/projects/2025/Project_1/pangaea-bench-old

CKPT_OVERRIDE=()
[[ -n "$CKPT" ]] && CKPT_OVERRIDE=(encoder.encoder_weights="$CKPT")

run_arm () {
  local task="$1" port="$2"
  echo "=== TREO PASTIS horizon sweep | task=$task | T=$T | epochs=$EPOCHS ==="
  $VENV/torchrun --nnodes=1 --nproc_per_node=1 --master_port="$port" pangaea/run.py \
    --config-name=train \
    finetune=false \
    dataset=pastis_s2_dates \
    dataset.multi_temporal="$T" \
    encoder=treo \
    "${CKPT_OVERRIDE[@]}" \
    decoder=seg_upernet_mt_none \
    task="$task" \
    task.trainer.n_epochs="$EPOCHS" \
    preprocessing=seg_default \
    criterion=cross_entropy \
    use_wandb=true
}

case "$ARM" in
  calib)  run_arm segmentation_horizon_curve 29601 ;;
  extrap) run_arm segmentation_horizon_eval  29602 ;;
  both)   run_arm segmentation_horizon_curve 29601
          run_arm segmentation_horizon_eval  29602 ;;
  *) echo "unknown arm: $ARM (want calib|extrap|both)"; exit 1 ;;
esac
