#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SINGLE_EVAL_SCRIPT="${SINGLE_EVAL_SCRIPT:-$ROOT_DIR/scripts/evaluation/run_IDOL_navhard_epdms_v2.sh}"

CKPT_DIR="${CKPT_DIR:-}"
START_EPOCH="${START_EPOCH:-29}"
END_EPOCH="${END_EPOCH:-0}"
APPEND_RESULTS="${APPEND_RESULTS:-false}"

EVAL_SCORE_DIR="${EVAL_SCORE_DIR:-$ROOT_DIR/exp/IDOL_LTF/default_resume_epoch_save/eval_score}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-IDOL_LTF/default_resume_epoch_save_navhard_epdms}"
export IDOL_NAVSIM_EXP_ROOT="${IDOL_NAVSIM_EXP_ROOT:-$ROOT_DIR/exp_navsim_v2}"

MAX_WORKERS="${MAX_WORKERS:-4}"
VERBOSE="${VERBOSE:-true}"

mkdir -p "$EVAL_SCORE_DIR"

ALL_SUMMARY_CSV="$EVAL_SCORE_DIR/all_checkpoints_epdms_summary.csv"
ALL_COMBINED_CSV="$EVAL_SCORE_DIR/all_checkpoints_epdms_combined.csv"
PAPER_METRICS_CSV="$EVAL_SCORE_DIR/all_checkpoints_epdms_paper_metrics.csv"

case "${APPEND_RESULTS,,}" in
  true|1|yes|y)
    append_results=true
    ;;
  *)
    append_results=false
    rm -f "$ALL_SUMMARY_CSV" "$ALL_COMBINED_CSV" "$PAPER_METRICS_CSV"
    ;;
esac

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
    # Prefer the periodic epoch checkpoint over the best checkpoint if both exist.
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

echo "[INFO] Found ${#EPOCH_CKPTS[@]} checkpoints. Results will be saved to: $EVAL_SCORE_DIR"

for item in "${EPOCH_CKPTS[@]}"; do
  epoch="${item%%$'\t'*}"
  ckpt_path="${item#*$'\t'}"
  epoch_label="$(printf "%02d" "$epoch")"
  experiment_name="$EXPERIMENT_PREFIX/epoch_${epoch_label}"
  log_path="$EVAL_SCORE_DIR/epoch_${epoch_label}.log"

  echo "[INFO] Evaluating epoch=$epoch_label"
  echo "[INFO] Checkpoint: $ckpt_path"

  set +e
  CHECKPOINT_PATH="$ckpt_path" \
  EXPERIMENT_NAME="$experiment_name" \
  MAX_WORKERS="$MAX_WORKERS" \
  VERBOSE="$VERBOSE" \
  bash "$SINGLE_EVAL_SCRIPT" 2>&1 | tee "$log_path"
  eval_status="${PIPESTATUS[0]}"
  set -e

  if [[ "$eval_status" -ne 0 ]]; then
    echo "[ERROR] Evaluation failed for epoch=$epoch_label. See log: $log_path" >&2
    exit "$eval_status"
  fi

  latest_csv="$(find "$IDOL_NAVSIM_EXP_ROOT/$experiment_name" -name '*.csv' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$latest_csv" ]]; then
    echo "[ERROR] No EPDMS csv found for epoch=$epoch_label under $IDOL_NAVSIM_EXP_ROOT/$experiment_name" >&2
    exit 1
  fi

  epoch_summary_csv="$EVAL_SCORE_DIR/epoch_${epoch_label}_epdms_summary.csv"
  python - "$latest_csv" "$epoch_summary_csv" "$ALL_SUMMARY_CSV" "$ALL_COMBINED_CSV" "$PAPER_METRICS_CSV" "$epoch" "$ckpt_path" "$append_results" <<'PY'
import sys
from pathlib import Path

import pandas as pd

source_csv = Path(sys.argv[1])
epoch_summary_csv = Path(sys.argv[2])
all_summary_csv = Path(sys.argv[3])
all_combined_csv = Path(sys.argv[4])
paper_metrics_csv = Path(sys.argv[5])
epoch = int(sys.argv[6])
checkpoint_path = sys.argv[7]
append_results = sys.argv[8].lower() in {"true", "1", "yes", "y"}


def remove_epoch_rows(csv_path: Path, epoch: int) -> None:
    if not append_results or not csv_path.exists():
        return
    existing = pd.read_csv(csv_path)
    if "epoch" not in existing.columns:
        return
    existing = existing[existing["epoch"].astype(str) != str(epoch)]
    if existing.empty:
        csv_path.unlink()
    else:
        existing.to_csv(csv_path, index=False)

df = pd.read_csv(source_csv)
if "token" not in df.columns:
    raise SystemExit(f"[ERROR] token column not found in {source_csv}")

summary = df[df["token"].astype(str).str.startswith("extended_pdm_score")].copy()
if summary.empty:
    raise SystemExit(f"[ERROR] extended_pdm_score rows not found in {source_csv}")

summary.insert(0, "epoch", epoch)
summary.insert(1, "checkpoint_path", checkpoint_path)
summary.insert(2, "source_csv", str(source_csv))
summary.to_csv(epoch_summary_csv, index=False)

combined = summary[summary["token"].astype(str).eq("extended_pdm_score_combined")].copy()
if combined.empty:
    raise SystemExit(f"[ERROR] extended_pdm_score_combined row not found in {source_csv}")

for csv_path in (all_summary_csv, all_combined_csv, paper_metrics_csv):
    remove_epoch_rows(csv_path, epoch)

summary.to_csv(all_summary_csv, mode="a", header=not all_summary_csv.exists(), index=False)
combined.to_csv(all_combined_csv, mode="a", header=not all_combined_csv.exists(), index=False)

metric_names = [
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
metric_aliases = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "DDC",
    "traffic_light_compliance": "TLC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "lane_keeping": "LK",
    "history_comfort": "HC",
    "two_frame_extended_comfort": "EC",
}

combined_row = combined.iloc[0]
paper_row = {
    "epoch": epoch,
    "checkpoint": Path(checkpoint_path).name,
    "checkpoint_path": checkpoint_path,
    "source_csv": str(source_csv),
    "EPDMS": combined_row.get("score"),
}
for metric in metric_names:
    paper_row[f"combined_{metric_aliases[metric]}"] = combined_row.get(f"{metric}_combined")
for stage in ("stage_one", "stage_two"):
    for metric in metric_names:
        paper_row[f"{stage}_{metric_aliases[metric]}"] = combined_row.get(f"{metric}_{stage}")

pd.DataFrame([paper_row]).to_csv(
    paper_metrics_csv,
    mode="a",
    header=not paper_metrics_csv.exists(),
    index=False,
)

score = combined["score"].iloc[0] if "score" in combined.columns else "N/A"
print(f"[INFO] epoch={epoch:02d} extended_pdm_score_combined score={score}")
print(f"[INFO] epoch summary saved to: {epoch_summary_csv}")
PY

done

echo "[INFO] Finished checkpoint sweep."
echo "[INFO] All EPDMS summary rows: $ALL_SUMMARY_CSV"
echo "[INFO] Combined EPDMS rows only: $ALL_COMBINED_CSV"
echo "[INFO] Paper-friendly EPDMS metrics: $PAPER_METRICS_CSV"
