#!/usr/bin/env bash
set -euo pipefail

# Warmup metric cache script for current repo (v1-style config: split + scene_filter).
# Official v1 warmup uses: split=mini, scene_filter=warmup_test_e2e.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT_DIR/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"

SPLIT="${SPLIT:-mini}"
SCENE_FILTER="${SCENE_FILTER:-warmup_test_e2e}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_warmup_test_e2e}"
FRAME_INTERVAL="${FRAME_INTERVAL:-1}"

LOG_DIR="$OPENSCENE_DATA_ROOT/navsim_logs/$SPLIT"
SENSOR_DIR="$OPENSCENE_DATA_ROOT/sensor_blobs/$SPLIT"

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

echo "Running metric caching with:"
echo "  split=$SPLIT"
echo "  scene_filter=$SCENE_FILTER"
echo "  cache.cache_path=$CACHE_PATH"
echo "  scene_filter.frame_interval=$FRAME_INTERVAL"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
split="$SPLIT" \
scene_filter="$SCENE_FILTER" \
cache.cache_path="$CACHE_PATH" \
scene_filter.frame_interval="$FRAME_INTERVAL"

echo "Metric cache done. Metadata:"
echo "  $CACHE_PATH/metadata/metric_cache_metadata_node_0.csv"
