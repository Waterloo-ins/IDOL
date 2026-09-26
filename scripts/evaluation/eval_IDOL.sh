#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_NAME="${CONFIG_NAME:-default}"
START_EPOCH="${START_EPOCH:-29}"
END_EPOCH="${END_EPOCH:-10}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT_DIR/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"

CKPT_DIR="${CKPT_DIR:-}"

if [[ -z "$CKPT_DIR" ]]; then
  echo "[ERROR] Set CKPT_DIR=/path/to/lightning_logs/version_x/checkpoints" >&2
  exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[ERROR] checkpoint directory not found: $CKPT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"

for EPOCH in $(seq "$START_EPOCH" -1 "$END_EPOCH"); do
  CHECKPOINT_PATH="$(find "$CKPT_DIR" -name "*epoch=${EPOCH}-*" -print -quit)"
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "[WARN] No checkpoint found for epoch=${EPOCH}, skipping."
    continue
  fi

  echo "[INFO] Evaluating epoch=${EPOCH}: ${CHECKPOINT_PATH}"
  EVAL_EPOCH="$EPOCH" python ./navsim/planning/script/run_pdm_score.py \
    agent=IDOL_agent \
    "agent.checkpoint_path=${CHECKPOINT_PATH}" \
    agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig \
    experiment_name=eval/IDOL/${CONFIG_NAME}/ \
    split=test \
    scene_filter=navtest
done
