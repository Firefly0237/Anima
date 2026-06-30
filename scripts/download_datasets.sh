#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/featurize/work/anima}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/envs/anima-train-v2}"
ROLEBENCH_DIR="$ROOT/data/raw/hf_snapshot/ZenMoore__RoleBench"

source "$ROOT/env.sh"
PY="$TRAIN_ENV/bin/python"
mkdir -p "$ROLEBENCH_DIR"

"$PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ.get("ROOT", "/home/featurize/work/anima")) / "data/raw/hf_snapshot/ZenMoore__RoleBench"
requested_revision = os.environ.get("ROLEBENCH_REVISION", "main")
api = HfApi()
info = api.dataset_info("ZenMoore/RoleBench", revision=requested_revision)
resolved_revision = info.sha
snapshot_download(
    repo_id="ZenMoore/RoleBench",
    repo_type="dataset",
    local_dir=str(root),
    local_dir_use_symlinks=False,
    revision=resolved_revision,
    allow_patterns=[
        "rolebench-zh/role_specific/rolegpt_baseline.jsonl",
        "README*",
        "LICENSE*",
    ],
)
(root / ".snapshot_revision").write_text(resolved_revision + "\n", encoding="utf-8")
print(f"ROLEBENCH_DIR={root}")
print(f"ROLEBENCH_REVISION={resolved_revision}")
PY

RAW="$(find "$ROLEBENCH_DIR" -path '*rolebench-zh*role_specific*rolegpt_baseline.jsonl' -print -quit)"
if [ -z "$RAW" ]; then
  echo "RoleBench role-specific file not found under $ROLEBENCH_DIR" >&2
  exit 2
fi
echo "ROLEBENCH_ROLE_ZH=$RAW"
