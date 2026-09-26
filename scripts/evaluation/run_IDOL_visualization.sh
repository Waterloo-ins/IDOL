#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_NAME="${CONFIG_NAME:-default}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
NUM_SCENES="${NUM_SCENES:-5}"
SEED="${SEED:-0}"
PLOT_TYPES="${PLOT_TYPES:-[bev_frame,bev_agent,bev_candidates,bev_lidar,cameras,cameras_annotations,cameras_full,front_view]}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT_DIR/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

python ./navsim/planning/script/run_visualization.py \
  agent=IDOL_agent \
  "agent.checkpoint_path=${CHECKPOINT_PATH}" \
  agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig \
  experiment_name=visualization/IDOL/${CONFIG_NAME}/ \
  split=test \
  scene_filter=navtest \
  num_scenes="$NUM_SCENES" \
  seed="$SEED" \
  plot_types="$PLOT_TYPES"

echo "[INFO] Figures saved under: ${NAVSIM_EXP_ROOT}/visualization/IDOL/${CONFIG_NAME}/"
