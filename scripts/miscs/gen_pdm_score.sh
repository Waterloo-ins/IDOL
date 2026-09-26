#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$ROOT_DIR/dataset/metric_cache/trainval}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt" >&2
  exit 1
fi

python scripts/miscs/gen_multi_trajs_pdm_score.py \
agent=IDOL_agent \
"agent.checkpoint_path=$CHECKPOINT_PATH" \
agent.config._target_=navsim.agents.IDOL.configs.default.IDOLConfig \
experiment_name=eval/gen_data \
"metric_cache_path=$METRIC_CACHE_PATH" \
split=trainval \
scene_filter=navtrain \
