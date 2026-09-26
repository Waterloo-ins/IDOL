#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_DEVKIT_ROOT="${IDOL_NAVSIM_V2_ROOT:-$ROOT_DIR/navsim-v2-IDOL}"
export OPENSCENE_DATA_ROOT="${IDOL_OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export NAVSIM_EXP_ROOT="${IDOL_NAVSIM_EXP_ROOT:-$ROOT_DIR/exp/submission/IDOL/default}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/navsim_matplotlib_${USER:-user}}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
LTF_CHECKPOINT_PATH="${LTF_CHECKPOINT_PATH:-}"
LATENT_INIT="${LATENT_INIT:-ltf}"
PRIVATE_ROOT="${PRIVATE_ROOT:-$OPENSCENE_DATA_ROOT/private_test_hard_two_stage}"
ORIGINAL_SENSOR_PATH="${ORIGINAL_SENSOR_PATH:-$PRIVATE_ROOT/sensor_blobs}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-$PRIVATE_ROOT/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-$PRIVATE_ROOT/synthetic_scene_pickles}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-private_test_hard_two_stage_ltf_v2}"

TEAM_NAME="${TEAM_NAME:-IDOL}"
AUTHORS="${AUTHORS:-MUST_SET}"
EMAIL="${EMAIL:-MUST_SET}"
INSTITUTION="${INSTITUTION:-MUST_SET}"
COUNTRY="${COUNTRY:-MUST_SET}"

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

if [[ ! -d "$PRIVATE_ROOT/openscene_meta_datas" ]]; then
  echo "[ERROR] private test metadata not found: $PRIVATE_ROOT/openscene_meta_datas" >&2
  exit 1
fi

if [[ ! -d "$ORIGINAL_SENSOR_PATH" ]]; then
  echo "[ERROR] private test sensor blobs not found: $ORIGINAL_SENSOR_PATH" >&2
  exit 1
fi

if [[ ! -d "$SYNTHETIC_SCENES_PATH" ]]; then
  echo "[ERROR] private synthetic_scene_pickles not found: $SYNTHETIC_SCENES_PATH" >&2
  echo "This is required for NAVSIM v2 two-stage submission generation." >&2
  exit 1
fi

mkdir -p "$NAVSIM_EXP_ROOT" "$MPLCONFIGDIR"
cd "$NAVSIM_DEVKIT_ROOT"

PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}" \
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py" \
  train_test_split=private_test_hard_two_stage \
  agent=IDOL_agent \
  agent.checkpoint_path="$CHECKPOINT_PATH" \
  +agent.config.latent=true \
  +agent.config.latent_init="$LATENT_INIT" \
  +agent.config.latent_ckpt_path="$LTF_CHECKPOINT_PATH" \
  +agent.config.sensor_frame_idx=3 \
  experiment_name="$EXPERIMENT_NAME" \
  navsim_log_path="$PRIVATE_ROOT/openscene_meta_datas" \
  original_sensor_path="$ORIGINAL_SENSOR_PATH" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
  synthetic_scenes_path="$SYNTHETIC_SCENES_PATH" \
  team_name="$TEAM_NAME" \
  authors="$AUTHORS" \
  email="$EMAIL" \
  institution="$INSTITUTION" \
  country="$COUNTRY"

latest_submission="$(find "$NAVSIM_EXP_ROOT/$EXPERIMENT_NAME" -name submission.pkl | sort | tail -n 1)"
if [[ -n "$latest_submission" ]]; then
  echo "[INFO] Official submission pickle: $latest_submission"
fi
