#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/featurize/work/anima}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/envs/anima-train-v2}"
SOCIALBENCH_DIR="${SOCIALBENCH_DIR:-$ROOT/data/raw/github/SocialBench}"
CEVAL_DIR="${CEVAL_DIR:-$ROOT/data/raw/hf_dataset/ceval__chinese_language_and_literature}"
CEVAL_DATASET="${CEVAL_DATASET:-ceval/ceval-exam}"
CEVAL_CONFIG="${CEVAL_CONFIG:-chinese_language_and_literature}"
export CEVAL_DIR CEVAL_DATASET CEVAL_CONFIG

source "$ROOT/env.sh"
PY="$TRAIN_ENV/bin/python"

mkdir -p "$(dirname "$SOCIALBENCH_DIR")" "$CEVAL_DIR"

echo "[download] SocialBench"
if [ ! -d "$SOCIALBENCH_DIR/.git" ]; then
  rm -rf "$SOCIALBENCH_DIR"
  git clone --depth 1 https://github.com/X-PLUG/SocialBench "$SOCIALBENCH_DIR"
else
  git -C "$SOCIALBENCH_DIR" fetch --depth 1 origin main || true
fi
git -C "$SOCIALBENCH_DIR" rev-parse HEAD | tee "$SOCIALBENCH_DIR/.snapshot_revision"

echo "[download] C-Eval (labeled val slice across subjects)"
"$PY" - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

try:
    from datasets import get_dataset_config_names
except Exception:  # pragma: no cover - very old datasets
    get_dataset_config_names = None

dataset_name = os.environ.get("CEVAL_DATASET", "ceval/ceval-exam")
primary_config = os.environ.get("CEVAL_CONFIG", "chinese_language_and_literature")
out_dir = Path(os.environ.get("CEVAL_DIR", "/home/featurize/work/anima/data/raw/hf_dataset/ceval__chinese_language_and_literature"))
out_dir.mkdir(parents=True, exist_ok=True)
target = int(os.environ.get("CEVAL_TARGET_ROWS", "120"))
min_rows = int(os.environ.get("CEVAL_MIN_ROWS", "40"))

# Only the val/dev splits carry public answer labels; C-Eval test labels are
# hidden, so test is excluded (the MCQ scorer needs a gold label). We accumulate
# a subject-diverse, labeled slice large enough that even the un-finetuned Base
# arm clears the per-arm reportability floor. This is a general-capability
# regression canary, so broader subject coverage is a stronger guardrail. The
# old single-subject (n=23) slice was too small: Base's MCQ format-compliance
# rate dropped it below the >=20 parseable-row floor (a known base-model trait,
# cf. "SFT stabilizes output format" -- see docs/report/FORMAL_V2_RESULT_INTERPRETATION.md).
all_configs = []
if get_dataset_config_names is not None:
    try:
        all_configs = list(get_dataset_config_names(dataset_name))
    except Exception:  # noqa: BLE001 - fall back to a fixed list
        all_configs = []
if not all_configs:
    all_configs = [
        primary_config, "high_school_chinese", "modern_chinese",
        "high_school_history", "middle_school_history", "logic",
        "high_school_geography", "ideological_and_moral_cultivation",
    ]
configs = [primary_config] + [c for c in all_configs if c != primary_config]

collected: list[tuple[str, list[dict]]] = []
total = 0
errors: list[str] = []
for config in dict.fromkeys(configs):
    if total >= target:
        break
    for split in ("val", "dev"):
        try:
            ds = load_dataset(dataset_name, config, split=split)
        except Exception as exc:  # noqa: BLE001 - surfaced in summary
            errors.append(f"{config}/{split}: {exc}")
            continue
        rows = []
        for index, row in enumerate(ds):
            item = dict(row)
            if not str(item.get("answer", "")).strip():
                continue  # need a gold label to score the MCQ
            item.setdefault("id", f"ceval_{config}_{split}_{index:06d}")
            item["source"] = "C-Eval"
            item["split"] = split
            item["ceval_config"] = config
            rows.append(item)
        if not rows:
            continue
        collected.append((f"{config}_{split}.jsonl", rows))
        total += len(rows)
        break  # one labeled split (val preferred) per subject is enough

if total < min_rows:
    raise SystemExit(
        "C-Eval labeled rows %d below minimum %d (existing slice left untouched); errors: %s"
        % (total, min_rows, "; ".join(errors[-10:]))
    )

# Success: write a clean slice (drop any stale per-subject files first).
for stale in out_dir.glob("*.jsonl"):
    stale.unlink()
for filename, rows in collected:
    path = out_dir / filename
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

summary = {
    "stage": "download_formal_v2_ceval",
    "dataset": dataset_name,
    "total_labeled_rows": total,
    "subjects": [{"file": name, "rows": len(rows)} for name, rows in collected],
}
(out_dir / ".snapshot_revision").write_text(
    json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

echo "SOCIALBENCH_DIR=$SOCIALBENCH_DIR"
echo "CEVAL_DIR=$CEVAL_DIR"
