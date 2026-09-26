#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT_DIR/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$ROOT_DIR/dataset}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/navsim_matplotlib_${USER:-user}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export CUDA_VISIBLE_DEVICES
NUM_GPUS="${NUM_GPUS:-4}"

CONFIG_NAME="${CONFIG_NAME:-default}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-IDOL_LTF/${CONFIG_NAME}}"

LTF_CHECKPOINT_PATH="${LTF_CHECKPOINT_PATH:-}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-}"

# Default is full IDOL-LTF training. Set RESUME_FROM_IDOL=true only for a finetuning ablation.
RESUME_FROM_IDOL="${RESUME_FROM_IDOL:-false}"
LATENT_INIT="${LATENT_INIT:-ltf}"
LATENT_FREEZE="${LATENT_FREEZE:-false}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
ACCELERATOR="${ACCELERATOR:-gpu}"
PRECISION="${PRECISION:-16-mixed}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
PIN_MEMORY="${PIN_MEMORY:-false}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
RESUME_TRAINER_CKPT_PATH="${RESUME_TRAINER_CKPT_PATH:-}"

if [[ "$LATENT_INIT" == "ltf" && -z "$LTF_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set LTF_CHECKPOINT_PATH=/path/to/ltf_checkpoint.ckpt when LATENT_INIT=ltf" >&2
  exit 1
fi

if [[ "$LATENT_INIT" == "ltf" && ! -f "$LTF_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] LTF checkpoint not found: $LTF_CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ "$RESUME_FROM_IDOL" == "true" && -z "$BASE_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Set BASE_CHECKPOINT_PATH=/path/to/idol_checkpoint.ckpt when RESUME_FROM_IDOL=true" >&2
  exit 1
fi

if [[ "$RESUME_FROM_IDOL" == "true" && ! -f "$BASE_CHECKPOINT_PATH" ]]; then
  echo "[ERROR] IDOL base checkpoint not found: $BASE_CHECKPOINT_PATH" >&2
  exit 1
fi

agent_args=(
  "agent=IDOL_agent"
  "agent.config._target_=navsim.agents.IDOL.configs.${CONFIG_NAME}.IDOLConfig"
  "+agent.config.latent=true"
  "+agent.config.latent_init=$LATENT_INIT"
  "+agent.config.latent_ckpt_path=$LTF_CHECKPOINT_PATH"
  "+agent.config.latent_freeze=$LATENT_FREEZE"
  "+agent.config.sensor_frame_idx=3"
)

if [[ "$RESUME_FROM_IDOL" == "true" ]]; then
  agent_args+=(
    "agent.checkpoint_path=\"$BASE_CHECKPOINT_PATH\""
    "+agent.resume_from_checkpoint=true"
  )
fi

trainer_resume_args=()
if [[ -n "$RESUME_TRAINER_CKPT_PATH" ]]; then
  if [[ ! -f "$RESUME_TRAINER_CKPT_PATH" ]]; then
    echo "[ERROR] trainer checkpoint not found: $RESUME_TRAINER_CKPT_PATH" >&2
    exit 1
  fi
  trainer_resume_args+=("+resume_trainer_ckpt_path=$RESUME_TRAINER_CKPT_PATH")
fi

cd "$ROOT_DIR"

python "$ROOT_DIR/navsim/planning/script/run_training.py" \
  "${agent_args[@]}" \
  "${trainer_resume_args[@]}" \
  "experiment_name=$EXPERIMENT_NAME" \
  "scene_filter=navtrain" \
  "dataloader.params.batch_size=$BATCH_SIZE" \
  "dataloader.params.num_workers=$NUM_WORKERS" \
  "dataloader.params.pin_memory=$PIN_MEMORY" \
  "dataloader.params.prefetch_factor=$PREFETCH_FACTOR" \
  "trainer.params.max_epochs=$MAX_EPOCHS" \
  "trainer.params.check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH" \
  "trainer.params.accelerator=$ACCELERATOR" \
  "trainer.params.devices=$NUM_GPUS" \
  "trainer.params.precision=$PRECISION" \
  "trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES" \
  "trainer.params.fast_dev_run=$FAST_DEV_RUN" \
  "trainer.params.limit_train_batches=$LIMIT_TRAIN_BATCHES" \
  "trainer.params.limit_val_batches=$LIMIT_VAL_BATCHES" \
  "split=trainval"
