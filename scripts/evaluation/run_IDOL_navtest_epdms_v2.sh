#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_DEVKIT_ROOT="${IDOL_NAVSIM_V2_ROOT:-$ROOT_DIR/navsim-v2-IDOL}"
export OPENSCENE_DATA_ROOT="${IDOL_OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export NAVSIM_EXP_ROOT="${IDOL_NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/navsim_matplotlib_${USER:-user}}"

resolve_max_workers() {
  local requested="${1:-auto}"
  if [[ "$requested" != "auto" ]]; then
    echo "$requested"
    return
  fi

  local cpu_count
  cpu_count="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 1)"

  local mem_available_kb
  mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
  if [[ -z "$mem_available_kb" ]]; then
    mem_available_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  fi

  local mem_per_worker_gb="${EPDMS_MEM_PER_WORKER_GB:-10}"
  local max_auto_workers="${EPDMS_MAX_AUTO_WORKERS:-8}"
  local cpu_divisor="${EPDMS_CPU_WORKER_DIVISOR:-4}"
  local cpu_cap=$(((cpu_count + cpu_divisor - 1) / cpu_divisor))
  local mem_cap=1

  if [[ "$mem_available_kb" =~ ^[0-9]+$ ]] && [[ "$mem_available_kb" -gt 0 ]]; then
    mem_cap=$((mem_available_kb / (mem_per_worker_gb * 1024 * 1024)))
  fi

  ((cpu_cap < 1)) && cpu_cap=1
  ((mem_cap < 1)) && mem_cap=1

  local resolved="$cpu_cap"
  ((mem_cap < resolved)) && resolved="$mem_cap"
  ((max_auto_workers < resolved)) && resolved="$max_auto_workers"
  ((resolved < 1)) && resolved=1

  echo "$resolved"
}

CONFIG_NAME="${CONFIG_NAME:-default}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CACHE_PATH="${CACHE_PATH:-$ROOT_DIR/exp/metric_cache_navtest_epdms}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-IDOL/default/navtest_epdms/single}"
WORKER="${WORKER:-single_machine_thread_pool}"
MAX_WORKERS="${MAX_WORKERS:-}"
if [[ "$MAX_WORKERS" == "auto" ]]; then
  MAX_WORKERS="$(resolve_max_workers "$MAX_WORKERS")"
fi
VERBOSE="${VERBOSE:-true}"

if [[ ! -d "$NAVSIM_DEVKIT_ROOT/navsim" ]]; then
  echo "[ERROR] NAVSIM v2 devkit not found: $NAVSIM_DEVKIT_ROOT" >&2
  exit 1
fi

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ ! -d "$OPENSCENE_DATA_ROOT/navsim_logs/test" ]]; then
  echo "[ERROR] navtest logs not found: $OPENSCENE_DATA_ROOT/navsim_logs/test" >&2
  exit 1
fi

if [[ ! -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test" ]]; then
  echo "[ERROR] navtest sensor blobs not found: $OPENSCENE_DATA_ROOT/sensor_blobs/test" >&2
  exit 1
fi

if [[ ! -d "$CACHE_PATH" ]] || ! find "$CACHE_PATH" -name metric_cache.pkl -print -quit | grep -q .; then
  echo "[ERROR] EPDMS metric cache not found under: $CACHE_PATH" >&2
  echo "Run scripts/evaluation/run_metric_caching_navtest_epdms_v2.sh first, or set CACHE_PATH to an existing v2 navtest cache." >&2
  exit 1
fi

mkdir -p "$NAVSIM_EXP_ROOT" "$MPLCONFIGDIR"
echo "[INFO] worker=$WORKER max_workers=${MAX_WORKERS:-hydra_default} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"

cd "$NAVSIM_DEVKIT_ROOT"

PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}" \
python - "$CACHE_PATH" <<'PY'
import lzma
import pickle
import sys
from pathlib import Path

cache_root = Path(sys.argv[1])
cache_file = next(cache_root.glob("*/*/*/metric_cache.pkl"), None)
if cache_file is None:
    raise SystemExit(f"[ERROR] No metric_cache.pkl found under: {cache_root}")

with lzma.open(cache_file, "rb") as f:
    metric_cache = pickle.load(f)

required_fields = [
    "log_name",
    "timepoint",
    "scene_type",
    "human_trajectory",
    "past_human_trajectory",
    "past_detections_tracks",
    "current_tracked_objects",
    "future_tracked_objects",
    "map_parameters",
]
missing_fields = [field for field in required_fields if not hasattr(metric_cache, field)]
if missing_fields:
    raise SystemExit(
        "[ERROR] CACHE_PATH points to a v1/PDMS cache, but EPDMS needs a v2 cache. "
        f"Missing fields: {', '.join(missing_fields)}. "
        "Run scripts/evaluation/run_metric_caching_navtest_epdms_v2.sh first."
    )
PY

overrides=(
  "train_test_split=navtest"
  "agent=IDOL_agent"
  "agent.checkpoint_path=\"$CHECKPOINT_PATH\""
  "agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig"
  "experiment_name=$EXPERIMENT_NAME"
  "metric_cache_path=\"$CACHE_PATH\""
  "verbose=$VERBOSE"
)

if [[ -n "$WORKER" ]]; then
  overrides+=("worker=$WORKER")
fi

if [[ "$WORKER" == "single_machine_thread_pool" && -n "$MAX_WORKERS" ]]; then
  overrides+=("worker.max_workers=$MAX_WORKERS")
fi

if [[ -n "${MAX_SCENES:-}" ]]; then
  overrides+=("train_test_split.scene_filter.max_scenes=$MAX_SCENES")
fi

PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}" \
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" "${overrides[@]}"

latest_csv="$(find "$NAVSIM_EXP_ROOT/$EXPERIMENT_NAME" -name '*.csv' 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$latest_csv" ]]; then
  echo "[INFO] Latest navtest EPDMS csv: $latest_csv"
fi
