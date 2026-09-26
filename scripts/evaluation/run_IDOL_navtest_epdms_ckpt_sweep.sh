#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SINGLE_EVAL_SCRIPT="${SINGLE_EVAL_SCRIPT:-$ROOT_DIR/scripts/evaluation/run_IDOL_navtest_epdms_v2.sh}"

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

RUN_LABEL="${RUN_LABEL:-custom_ckpt_sweep}"
CKPT_DIR="${CKPT_DIR:-}"
START_EPOCH="${START_EPOCH:-29}"
END_EPOCH="${END_EPOCH:-17}"

export IDOL_NAVSIM_EXP_ROOT="${IDOL_NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
CACHE_PATH="${CACHE_PATH:-$ROOT_DIR/exp/metric_cache_navtest_epdms}"
EVAL_SCORE_DIR="${EVAL_SCORE_DIR:-$ROOT_DIR/exp/IDOL/default/eval_score}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-IDOL/default/navtest_epdms/${RUN_LABEL}}"

MAX_WORKERS="${MAX_WORKERS:-}"
if [[ "$MAX_WORKERS" == "auto" ]]; then
  MAX_WORKERS="$(resolve_max_workers "$MAX_WORKERS")"
fi
VERBOSE="${VERBOSE:-true}"

mkdir -p "$EVAL_SCORE_DIR"

ALL_SUMMARY_CSV="$EVAL_SCORE_DIR/${RUN_LABEL}_navtest_epdms_all_epochs.csv"
rm -f "$ALL_SUMMARY_CSV"

if [[ -z "$CKPT_DIR" ]]; then
  echo "[ERROR] Set CKPT_DIR=/path/to/lightning_logs/version_x/checkpoints" >&2
  exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[ERROR] checkpoint directory not found: $CKPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$SINGLE_EVAL_SCRIPT" ]]; then
  echo "[ERROR] single eval script not found: $SINGLE_EVAL_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$CACHE_PATH" ]] || ! find "$CACHE_PATH" -name metric_cache.pkl -print -quit | grep -q .; then
  echo "[ERROR] EPDMS metric cache not found under: $CACHE_PATH" >&2
  echo "Run scripts/evaluation/run_metric_caching_navtest_epdms_v2.sh first, or set CACHE_PATH to an existing v2 navtest cache." >&2
  exit 1
fi

mapfile -t EPOCH_CKPTS < <(
python - "$CKPT_DIR" "$START_EPOCH" "$END_EPOCH" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
start_epoch = int(sys.argv[2])
end_epoch = int(sys.argv[3])

by_epoch = {}
for ckpt in ckpt_dir.glob("*.ckpt"):
    if ckpt.name == "last.ckpt":
        continue
    match = re.search(r"epoch=(?:epoch=)?(\d+)", ckpt.name)
    if match is None:
        continue
    epoch = int(match.group(1))
    if not (end_epoch <= epoch <= start_epoch):
        continue
    priority = 0 if "epoch=epoch=" in ckpt.name else 1
    by_epoch.setdefault(epoch, []).append((priority, ckpt.name, ckpt))

for epoch in range(start_epoch, end_epoch - 1, -1):
    candidates = by_epoch.get(epoch)
    if not candidates:
        continue
    ckpt = sorted(candidates)[0][2]
    print(f"{epoch}\t{ckpt}")
PY
)

if [[ "${#EPOCH_CKPTS[@]}" -eq 0 ]]; then
  echo "[ERROR] no checkpoint found in $CKPT_DIR for epoch range [$END_EPOCH, $START_EPOCH]" >&2
  exit 1
fi

first_epoch="${EPOCH_CKPTS[0]%%$'\t'*}"
if [[ "$first_epoch" != "$START_EPOCH" ]]; then
  echo "[ERROR] start checkpoint containing epoch=$START_EPOCH was not found under $CKPT_DIR" >&2
  exit 1
fi

echo "[INFO] Found ${#EPOCH_CKPTS[@]} checkpoints. Text results will be saved to: $EVAL_SCORE_DIR"
echo "[INFO] Full CSV outputs will be saved under: $IDOL_NAVSIM_EXP_ROOT/$EXPERIMENT_PREFIX"
echo "[INFO] worker=single_machine_thread_pool max_workers=${MAX_WORKERS:-hydra_default} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"

for item in "${EPOCH_CKPTS[@]}"; do
  epoch="${item%%$'\t'*}"
  ckpt_path="${item#*$'\t'}"
  epoch_label="$(printf "%02d" "$epoch")"
  experiment_name="$EXPERIMENT_PREFIX/epoch_${epoch_label}"
  log_path="$EVAL_SCORE_DIR/${RUN_LABEL}_navtest_epdms_epoch_${epoch_label}.log"

  echo "[INFO] Evaluating navtest EPDMS epoch=$epoch_label"
  echo "[INFO] Checkpoint: $ckpt_path"

  set +e
  eval_env=(
    "CHECKPOINT_PATH=$ckpt_path"
    "CACHE_PATH=$CACHE_PATH"
    "EXPERIMENT_NAME=$experiment_name"
    "VERBOSE=$VERBOSE"
  )
  if [[ -n "$MAX_WORKERS" ]]; then
    eval_env+=("MAX_WORKERS=$MAX_WORKERS")
  fi
  env "${eval_env[@]}" bash "$SINGLE_EVAL_SCRIPT" 2>&1 | tee "$log_path"
  eval_status="${PIPESTATUS[0]}"
  set -e

  if [[ "$eval_status" -ne 0 ]]; then
    echo "[ERROR] Evaluation failed for epoch=$epoch_label. See log: $log_path" >&2
    exit "$eval_status"
  fi

  latest_csv="$(find "$IDOL_NAVSIM_EXP_ROOT/$experiment_name" -name '*.csv' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$latest_csv" ]]; then
    echo "[ERROR] No navtest EPDMS csv found for epoch=$epoch_label under $IDOL_NAVSIM_EXP_ROOT/$experiment_name" >&2
    exit 1
  fi

  text_path="$EVAL_SCORE_DIR/${RUN_LABEL}_eval_epoch_${epoch}_epdms.txt"
  python - "$latest_csv" "$ALL_SUMMARY_CSV" "$text_path" "$epoch" "$ckpt_path" <<'PY'
import sys
from pathlib import Path

import pandas as pd

source_csv = Path(sys.argv[1])
all_summary_csv = Path(sys.argv[2])
text_path = Path(sys.argv[3])
epoch = int(sys.argv[4])
checkpoint_path = Path(sys.argv[5])

df = pd.read_csv(source_csv)
if "token" not in df.columns:
    raise SystemExit(f"[ERROR] token column not found in {source_csv}")

average = df[df["token"].astype(str).eq("average_all_frames")].copy()
if average.empty:
    raise SystemExit(f"[ERROR] average_all_frames row not found in {source_csv}")

average_row = average.iloc[-1]
scenario_rows = df[~df["token"].astype(str).eq("average_all_frames")].copy()
valid_mask = scenario_rows["valid"].astype(str).str.lower().isin(["true", "1"])
num_successful = int(valid_mask.sum())
num_failed = int(len(scenario_rows) - num_successful)

metrics = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
]

summary_row = {
    "epoch": epoch,
    "checkpoint": checkpoint_path.name,
    "checkpoint_path": str(checkpoint_path),
    "source_csv": str(source_csv),
    "successful_scenarios": num_successful,
    "failed_scenarios": num_failed,
    "EPDMS": average_row.get("score"),
}
for metric in metrics:
    if metric in average_row.index:
        summary_row[metric] = average_row.get(metric)

pd.DataFrame([summary_row]).to_csv(
    all_summary_csv,
    mode="a",
    header=not all_summary_csv.exists(),
    index=False,
)

lines = [
    "Finished running navtest EPDMS evaluation.",
    f"    Epoch: {epoch}.",
    f"    Checkpoint: {checkpoint_path}.",
    f"    Number of successful scenarios: {num_successful}.",
    f"    Number of failed scenarios: {num_failed}.",
    f"    Final EPDMS score: {average_row.get('score')}.",
    f"    Results are stored in: {source_csv}.",
]
for metric in metrics:
    if metric in average_row.index:
        lines.append(f"    {metric}: {average_row.get(metric)}.")

text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[INFO] epoch={epoch:02d} navtest EPDMS={average_row.get('score')}")
print(f"[INFO] text summary saved to: {text_path}")
PY

done

echo "[INFO] Finished navtest EPDMS checkpoint sweep."
echo "[INFO] All epoch EPDMS summary: $ALL_SUMMARY_CSV"
