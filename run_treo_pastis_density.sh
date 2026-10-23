#!/usr/bin/env bash
# TREO on PASTIS-HD: the SAMPLE-EFFICIENCY axis -- full-season coverage at
# increasing acquisition DENSITY. One training run per T; the number reported
# for each T is the final-horizon mIoU (identical readout to SegEvaluator).
#
# This complements run_treo_pastis_horizon.sh: there, the x axis is a growing
# prefix of one fixed series (season unfolding); here, every point sees the
# WHOLE season, just sampled with more frames (PastisS2Dates picks T frames by
# torch.linspace over the capped series, so span is constant and only density
# grows). T=1 is the exception: the dataset returns the LAST acquisition only.
#
# TREO was pretrained on 4-frame seasonal series (SSL4EO-S12), so T > 4 is
# genuine sequence-length extrapolation for the recurrence -- the point of the
# whole exercise. The default grid brackets that: 1 2 3 4 6 8 10 12.
#
# ARM (default `eval`): train and evaluate at the SAME frame count.
#   eval  -> ordinary SegTrainer. The decoder is trained on the belief after all
#            T frames, and the reported number is the readout after all T frames.
#            Train-on-T / val-on-T / test-on-T are matched by construction:
#            run.py instantiates the SAME cfg.dataset (carrying multi_temporal=T)
#            for all three splits, varying only `split`. This is the arm you want
#            for "how would my model do if it were trained with more frames".
#   curve -> StreamDecodeSegTrainer, which trains ONE decoder to read every
#            prefix 1..T at once. Do NOT use it for this sweep: it answers a
#            different question, and because that decoder is multi-tasking it
#            UNDERSTATES the T-frame endpoint (1-epoch smoke at T=6: 9.67 vs
#            12.12 for the specialist). Only useful if you want the within-run
#            prefix curve, which run_treo_pastis_horizon.sh already gives you.
#
# The evaluator reports every horizon either way, but the headline `mIoU` -- the
# checkpoint-selection metric -- is always the FINAL horizon, i.e. all T frames.
# The extra per-horizon numbers ride along as free diagnostics.
#
# CAVEAT at T=1: PastisS2Dates special-cases multi_temporal==1 to the LAST
# acquisition rather than linspace, so that point is "the final image of the
# season", not "one image from the middle". Fine as a left endpoint, but it is
# not the same kind of sample as T>=2.
#
# Usage: ./run_treo_pastis_density.sh [eval|curve] [epochs] "[T list]"
set -e
cd "$(dirname "$0")"

ARM="${1:-eval}"
EPOCHS="${2:-80}"
TS="${3:-1 2 3 4 6 8 10 12}"

case "$ARM" in
  curve) TASK=segmentation_horizon_curve ;;
  eval)  TASK=segmentation_horizon_eval ;;
  *) echo "unknown arm: $ARM (want eval|curve)"; exit 1 ;;
esac

VENV=/home/gianfranco/projects/2025/Project_1/nested_mae_BEN/.venv/bin
export PYTHONPATH=/home/gianfranco/projects/2025/Project_1/pangaea-bench-old

# One T failing must NOT abort the remaining points -- an 8-run sweep is hours
# long and unattended, so a late OOM should cost one point, not the tail of the
# grid. Failures are collected and reported at the end.
PORT=29610
FAILED=""
for T in $TS; do
  echo "=== TREO PASTIS density sweep | T=$T | task=$TASK | epochs=$EPOCHS ==="
  if $VENV/torchrun --nnodes=1 --nproc_per_node=1 --master_port="$PORT" pangaea/run.py \
    --config-name=train \
    finetune=false \
    dataset=pastis_s2_dates \
    dataset.multi_temporal="$T" \
    encoder=treo \
    decoder=seg_upernet_mt_none \
    task="$TASK" \
    task.trainer.n_epochs="$EPOCHS" \
    preprocessing=seg_default \
    criterion=cross_entropy \
    use_wandb=true
  then
    echo "=== T=$T done ==="
  else
    echo "=== T=$T FAILED (exit $?) -- continuing with the rest of the grid ==="
    FAILED="$FAILED $T"
  fi
  PORT=$((PORT + 1))
done

if [[ -n "$FAILED" ]]; then
  echo "=== sweep finished WITH FAILURES at T:$FAILED ==="
  exit 1
fi
echo "=== sweep finished: all of [$TS] completed ==="
