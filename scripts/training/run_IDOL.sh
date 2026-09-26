#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_NAME="default"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-IDOL/default}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT_DIR/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"

cd "$ROOT_DIR"

python ./navsim/planning/script/run_training.py \
  agent=IDOL_agent \
  agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig \
  experiment_name="$EXPERIMENT_NAME" \
  scene_filter=navtrain \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  ++agent.config.max_epochs="$MAX_EPOCHS" \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  trainer.params.check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH" \
  trainer.params.devices="$NUM_GPUS" \
  trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  split=trainval \
  "$@"
