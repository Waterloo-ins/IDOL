#!/usr/bin/env bash
set -euo pipefail

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
LTF_CHECKPOINT_PATH="${LTF_CHECKPOINT_PATH:-}"
LATENT_INIT="${LATENT_INIT:-ltf}"
NAVHARD_ROOT="${NAVHARD_ROOT:-$ROOT_DIR/dataset/navhard_two_stage}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-submission/IDOL/${CONFIG_NAME}/navhard_two_stage_ltf}"

TEAM_NAME="${TEAM_NAME:-MUST_SET}"
AUTHORS="${AUTHORS:-MUST_SET}"
EMAIL="${EMAIL:-MUST_SET}"
INSTITUTION="${INSTITUTION:-MUST_SET}"
COUNTRY="${COUNTRY:-MUST_SET}"

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] checkpoint not found:"
  echo "  $CHECKPOINT_PATH"
  exit 1
fi

if [[ "$LATENT_INIT" == "ltf" && -z "$LTF_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set LTF_CHECKPOINT_PATH=/path/to/ltf_checkpoint.ckpt when LATENT_INIT=ltf"
  exit 1
fi

if [[ "$LATENT_INIT" == "ltf" && ! -f "$LTF_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] LTF checkpoint not found:"
  echo "  $LTF_CHECKPOINT_PATH"
  exit 1
fi

if [ ! -d "$NAVHARD_ROOT/openscene_meta_datas" ] || [ ! -d "$NAVHARD_ROOT/sensor_blobs" ]; then
  echo "[ERROR] navhard_two_stage directories not found under:"
  echo "  $NAVHARD_ROOT"
  exit 1
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py" \
agent=IDOL_agent \
"agent.checkpoint_path=\"$CHECKPOINT_PATH\"" \
agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig \
+agent.config.latent=true \
+agent.config.latent_init="$LATENT_INIT" \
+agent.config.latent_ckpt_path="$LTF_CHECKPOINT_PATH" \
+agent.config.sensor_frame_idx=3 \
split=navhard_two_stage \
scene_filter=all_scenes \
scene_filter.num_history_frames=5 \
scene_filter.num_future_frames=0 \
scene_filter.frame_interval=1 \
scene_filter.has_route=true \
navsim_log_path="$NAVHARD_ROOT/openscene_meta_datas" \
sensor_blobs_path="$NAVHARD_ROOT/sensor_blobs" \
experiment_name="$EXPERIMENT_NAME" \
team_name="$TEAM_NAME" \
authors="$AUTHORS" \
email="$EMAIL" \
institution="$INSTITUTION" \
country="$COUNTRY"

echo "Submission written under:"
echo "  $NAVSIM_EXP_ROOT/$EXPERIMENT_NAME/submission.pkl"
