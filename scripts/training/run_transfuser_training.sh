#!/usr/bin/env bash
set -euo pipefail

# python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
# agent=transfuser_agent \
# experiment_name=transfuser/default \
# dataloader.params.num_workers=8 \
# dataloader.params.batch_size=64 \
# scene_filter=navtrain \
# split=trainval \

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT=/path/to/NAVSIM devkit}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/transfuser_checkpoint.ckpt" >&2
  exit 1
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
agent=transfuser_agent \
"agent.checkpoint_path=$CHECKPOINT_PATH" \
experiment_name=eval/transfuser/default/ \
split=test \
scene_filter=navtest \
