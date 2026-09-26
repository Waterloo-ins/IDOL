#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_DEVKIT_ROOT="${IDOL_NAVSIM_V2_ROOT:-$ROOT_DIR/navsim-v2-IDOL}"
export OPENSCENE_DATA_ROOT="${IDOL_OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export NAVSIM_EXP_ROOT="${IDOL_NAVSIM_EXP_ROOT:-$ROOT_DIR/exp_navsim_v2}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/navsim_matplotlib_${USER:-user}}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
LTF_CHECKPOINT_PATH="${LTF_CHECKPOINT_PATH:-}"
LATENT_INIT="${LATENT_INIT:-ltf}"
CACHE_PATH="${CACHE_PATH:-$ROOT_DIR/exp/metric_cache_navhard_two_stage}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-IDOL_navhard_epdms_ltf}"
MAX_WORKERS="${MAX_WORKERS:-4}"
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

if [[ "$LATENT_INIT" == "ltf" && -z "$LTF_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set LTF_CHECKPOINT_PATH=/path/to/ltf_checkpoint.ckpt when LATENT_INIT=ltf" >&2
  exit 1
fi

if [[ "$LATENT_INIT" == "ltf" && ! -f "$LTF_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] LTF checkpoint not found: $LTF_CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ ! -d "$OPENSCENE_DATA_ROOT/navsim_logs/test" ]]; then
  echo "[ERROR] original navhard/test logs not found: $OPENSCENE_DATA_ROOT/navsim_logs/test" >&2
  exit 1
fi

if [[ ! -d "$OPENSCENE_DATA_ROOT/sensor_blobs/test" ]]; then
  echo "[ERROR] original navhard/test sensor blobs not found: $OPENSCENE_DATA_ROOT/sensor_blobs/test" >&2
  exit 1
fi

if [[ ! -d "$SYNTHETIC_SENSOR_PATH" ]]; then
  echo "[ERROR] synthetic sensor path not found: $SYNTHETIC_SENSOR_PATH" >&2
  exit 1
fi

if [[ ! -d "$SYNTHETIC_SCENES_PATH" ]]; then
  echo "[ERROR] synthetic scenes path not found: $SYNTHETIC_SCENES_PATH" >&2
  exit 1
fi

if ! find "$CACHE_PATH" -name metric_cache.pkl -print -quit | grep -q .; then
  echo "[ERROR] metric cache not found under: $CACHE_PATH" >&2
  echo "Run metric caching first, or set CACHE_PATH to an existing navhard cache." >&2
  exit 1
fi

mkdir -p "$NAVSIM_EXP_ROOT"
mkdir -p "$MPLCONFIGDIR"
cd "$NAVSIM_DEVKIT_ROOT"

overrides=(
  "train_test_split=navhard_two_stage"
  "agent=IDOL_agent"
  "agent.checkpoint_path=\"$CHECKPOINT_PATH\""
  "+agent.config.latent=true"
  "+agent.config.latent_init=$LATENT_INIT"
  "+agent.config.latent_ckpt_path=$LTF_CHECKPOINT_PATH"
  "+agent.config.sensor_frame_idx=3"
  "worker=single_machine_thread_pool"
  "worker.max_workers=$MAX_WORKERS"
  "experiment_name=$EXPERIMENT_NAME"
  "metric_cache_path=\"$CACHE_PATH\""
  "synthetic_sensor_path=\"$SYNTHETIC_SENSOR_PATH\""
  "synthetic_scenes_path=\"$SYNTHETIC_SCENES_PATH\""
  "verbose=$VERBOSE"
)

if [[ -n "${MAX_SCENES:-}" ]]; then
  overrides+=("train_test_split.scene_filter.max_scenes=$MAX_SCENES")
fi

PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}" \
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" "${overrides[@]}"

latest_csv="$(find "$NAVSIM_EXP_ROOT/$EXPERIMENT_NAME" -name '*.csv' | sort | tail -n 1)"
if [[ -n "$latest_csv" ]]; then
  echo "[INFO] Latest EPDMS csv: $latest_csv"
fi
