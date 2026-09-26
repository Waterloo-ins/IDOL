#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT=/path/to/NAVSIM devkit}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/ego_mlp_checkpoint.ckpt" >&2
  exit 1
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
agent=ego_status_mlp_agent \
experiment_name=ego_mlp_agent \
split=test \
scene_filter=navtest \
"agent.checkpoint_path=$CHECKPOINT_PATH"
