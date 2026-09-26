#!/usr/bin/env bash
set -euo pipefail

# Warmup evaluation script for IDOL checkpoint on current repo (v1-style config).
# Official v1 warmup uses: split=mini, scene_filter=warmup_test_e2e.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT_DIR/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"

CONFIG_NAME="${CONFIG_NAME:-default}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

SPLIT="${SPLIT:-mini}"
SCENE_FILTER="${SCENE_FILTER:-warmup_test_e2e}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_warmup_test_e2e}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval/IDOL/${CONFIG_NAME}/warmup_test_e2e}"
RUN_CACHE_FIRST="${RUN_CACHE_FIRST:-true}"

LOG_DIR="$OPENSCENE_DATA_ROOT/navsim_logs/$SPLIT"
SENSOR_DIR="$OPENSCENE_DATA_ROOT/sensor_blobs/$SPLIT"

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] checkpoint not found:"
  echo "  $CHECKPOINT_PATH"
  exit 1
fi

if [ ! -d "$LOG_DIR" ] || [ ! -d "$SENSOR_DIR" ]; then
  echo "[ERROR] Missing warmup v1 dataset directories:"
  echo "  - $LOG_DIR"
  echo "  - $SENSOR_DIR"
  echo
  echo "This repo is currently v1-style and expects warmup on mini split."
  echo "If you only downloaded NAVSIM v2 warmup_two_stage, this codebase cannot directly run it."
  echo "Please either:"
  echo "  1) download mini split for v1 warmup, or"
  echo "  2) switch to official NAVSIM v2 code for warmup_two_stage."
  exit 1
fi

if [ "$RUN_CACHE_FIRST" = "true" ]; then
  echo "Step 1/2: Building warmup metric cache..."
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  split="$SPLIT" \
  scene_filter="$SCENE_FILTER" \
  cache.cache_path="$CACHE_PATH" \
  scene_filter.frame_interval=1
else
  echo "Skip metric caching (RUN_CACHE_FIRST=false)"
fi

echo "Step 2/2: Running IDOL warmup evaluation..."
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
agent=IDOL_agent \
"agent.checkpoint_path=$CHECKPOINT_PATH" \
agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig \
experiment_name="$EXPERIMENT_NAME" \
split="$SPLIT" \
scene_filter="$SCENE_FILTER" \
metric_cache_path="$CACHE_PATH"

echo "Done. Results under:"
echo "  $NAVSIM_EXP_ROOT/$EXPERIMENT_NAME"
