#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/home/featurize/work/Anima}"
ROOT="${ROOT:-/home/featurize/work/anima}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/envs/anima-train-v2}"
RM_ENV="${RM_ENV:-$ROOT/envs/anima-rm-eval}"
FORMAL_V1_RUN="${FORMAL_V1_RUN:-$ROOT/formal_v1/formal_v1_rolebench_20260628T155402Z}"
RUN_ID="${RUN_ID:-formal_v2_eval_first_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$ROOT/formal_v2/$RUN_ID"
MAX_SOCIALBENCH_RECORDS="${MAX_SOCIALBENCH_RECORDS:-80}"
MAX_CEVAL_RECORDS="${MAX_CEVAL_RECORDS:-80}"
MIN_SOCIALBENCH_RECORDS="${MIN_SOCIALBENCH_RECORDS:-20}"
MIN_CEVAL_RECORDS="${MIN_CEVAL_RECORDS:-20}"
MIN_CHARRM_ROWS="${MIN_CHARRM_ROWS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
FORMAL_V2_ALLOW_RM_DOWNLOAD="${FORMAL_V2_ALLOW_RM_DOWNLOAD:-0}"

mkdir -p "$RUN_ROOT"/{logs,data,seeds,rollouts,scores,aggregate,artifacts}

ART="$ROOT/backups/${RUN_ID}.tar.gz"
finish() {
  status=$?
  mkdir -p "$ROOT/backups" "$RUN_ROOT/artifacts"
  printf '{"run_id":"%s","exit_status":%s}\n' "$RUN_ID" "$status" > "$RUN_ROOT/artifacts/EXIT_STATUS.json" || true
  if [ -d "$RUN_ROOT" ]; then
    tar -czf "$ART" -C "$ROOT/formal_v2" "$RUN_ID" || true
    if [ -f "$ART" ]; then
      sha256sum "$ART" | tee "$ART.sha256" || true
      ls -lh "$ART" "$ART.sha256" || true
      echo "DOWNLOAD=$ART"
      echo "DOWNLOAD_SHA256=$ART.sha256"
    fi
  fi
  exit "$status"
}
trap finish EXIT

cd "$CODE"

echo "[0] setup env"
bash scripts/featurize_setup.sh 2>&1 | tee "$RUN_ROOT/logs/00_featurize_setup.log"
source "$ROOT/env.sh"
PY="$TRAIN_ENV/bin/python"
export PYTHONPATH="$CODE/src:${PYTHONPATH:-}"

echo "[1] formal-v1 artifact gate"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/01_formal_v1_gate.log"
import json
from pathlib import Path

run = Path("$FORMAL_V1_RUN")
required = [
    run / "artifacts/REPORTABILITY_VERDICT.json",
    run / "data/source_ledger.json",
    run / "data/reward_train.jsonl",
    run / "models/sft_rolebench/adapter_model.safetensors",
    run / "models/dpo_rolebench/adapter_model.safetensors",
    run / "models/grpo_rolebench/adapter_model.safetensors",
]
for arm in ["base", "sft", "dpo", "grpo"]:
    required.append(run / f"eval/{arm}_heldout_rollouts.jsonl")
    required.append(run / f"eval/{arm}_heldout_scores.jsonl")
    required.append(run / f"eval/{arm}_heldout_scores.summary.json")
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(json.dumps({
        "stage": "formal_v1_gate",
        "status": "missing_formal_v1_artifacts",
        "formal_v1_run": str(run),
        "missing": missing,
        "remedy": "restore the local formal-v1 tar.gz to the server or rerun formal-v1 deliberately; do not start new training inside formal-v2",
    }, ensure_ascii=False, sort_keys=True))
verdict = json.loads((run / "artifacts/REPORTABILITY_VERDICT.json").read_text(encoding="utf-8"))
if verdict.get("status") != "PASS":
    raise SystemExit(json.dumps({"stage": "formal_v1_gate", "status": "formal_v1_not_pass", "verdict": verdict}, ensure_ascii=False, sort_keys=True))
print(json.dumps({"stage": "formal_v1_gate", "status": "PASS", "formal_v1_run": str(run)}, ensure_ascii=False, sort_keys=True))
PY

echo "[2] code validation"
"$PY" -m compileall -q src tests scripts 2>&1 | tee "$RUN_ROOT/logs/02_compileall.log"
"$PY" -m pytest \
  tests/test_generate_rollouts.py \
  tests/test_parsing.py \
  tests/test_rewards_reference.py \
  tests/test_source_ledger.py \
  2>&1 | tee "$RUN_ROOT/logs/03_pytest.log"

echo "[3] claim boundary"
"$PY" -m anima.utils.claim_boundary \
  --output-dir "$RUN_ROOT/artifacts" \
  --run-name "$RUN_ID" \
  --stage formal_v2_eval_first \
  --notes "Eval-first formal-v2: reuse formal-v1 adapters; no new SFT/DPO/GRPO training."
cat > "$RUN_ROOT/artifacts/CLAIM_BOUNDARY.txt" <<'EOF'
Formal-v2 is eval-first.
It reuses formal-v1 adapters and does not train broader adapters.
Output-contract health is not a quality metric.
BaichuanCharRM is scalar-only external diagnostic unless full CharacterEval metrics are implemented.
Do not claim GRPO superiority unless logged formal-v2 metrics support it.
EOF

echo "[4] download eval sources"
ROOT="$ROOT" TRAIN_ENV="$TRAIN_ENV" bash scripts/download_formal_v2_eval_sources.sh 2>&1 | tee "$RUN_ROOT/logs/04_download_eval_sources.log"
SOCIALBENCH_DIR="${SOCIALBENCH_DIR:-$ROOT/data/raw/github/SocialBench}"
CEVAL_DIR="${CEVAL_DIR:-$ROOT/data/raw/hf_dataset/ceval__chinese_language_and_literature}"
SOCIAL_INPUT="$SOCIALBENCH_DIR/data"
if [ ! -d "$SOCIAL_INPUT" ]; then
  SOCIAL_INPUT="$SOCIALBENCH_DIR"
fi

echo "[5] source ledgers"
cp "$FORMAL_V1_RUN/data/source_ledger.json" "$RUN_ROOT/data/source_ledger.formal_v1.json"
cp "$FORMAL_V1_RUN/data/source_ledger.json" "$RUN_ROOT/data/source_ledger.json"
SOCIAL_REV="$(cat "$SOCIALBENCH_DIR/.snapshot_revision" 2>/dev/null || git -C "$SOCIALBENCH_DIR" rev-parse HEAD)"
CEVAL_REV="$(cat "$CEVAL_DIR/.snapshot_revision" 2>/dev/null || echo ceval_downloaded_dataset_snapshot)"
"$PY" -m anima.data.source_ledger \
  --output "$RUN_ROOT/data/source_ledger.json" \
  --source-id SocialBench \
  --name "SocialBench / RoleInteract" \
  --url "https://github.com/X-PLUG/SocialBench" \
  --snapshot-or-commit "$SOCIAL_REV" \
  --license-name "research-benchmark-no-spdx-observed" \
  --license-url-or-path "https://github.com/X-PLUG/SocialBench" \
  --redistribution "manifest_only" \
  --project-use "formal-v2 eval-only MCQ social/role reasoning" \
  --public-artifact-policy "do_not_redistribute_raw_rows" \
  --notes "Eval-only; report aggregate metrics, manifests, and checksums." \
  --path "$SOCIAL_INPUT" \
  2>&1 | tee "$RUN_ROOT/logs/05_source_ledger_socialbench.log"
"$PY" -m anima.data.source_ledger \
  --output "$RUN_ROOT/data/source_ledger.json" \
  --source-id CEval \
  --name "C-Eval Chinese regression slice" \
  --url "https://huggingface.co/datasets/ceval/ceval-exam" \
  --snapshot-or-commit "$CEVAL_REV" \
  --license-name "public-benchmark-upstream-terms" \
  --license-url-or-path "https://huggingface.co/datasets/ceval/ceval-exam" \
  --redistribution "manifest_only" \
  --project-use "formal-v2 eval-only Chinese capability regression canary" \
  --public-artifact-policy "do_not_redistribute_raw_rows" \
  --notes "Eval-only; not a role-play quality metric." \
  --path "$CEVAL_DIR" \
  2>&1 | tee "$RUN_ROOT/logs/06_source_ledger_ceval.log"

echo "[6] build eval seeds"
"$PY" -m anima.eval.build_w4_eval_slices \
  --output-dir "$RUN_ROOT/seeds" \
  --socialbench-input "$SOCIAL_INPUT" \
  --ceval-input "$CEVAL_DIR" \
  --train-jsonl "$FORMAL_V1_RUN/data/reward_train.jsonl" \
  --max-socialbench "$MAX_SOCIALBENCH_RECORDS" \
  --max-ceval "$MAX_CEVAL_RECORDS" \
  --min-socialbench "$MIN_SOCIALBENCH_RECORDS" \
  --min-ceval "$MIN_CEVAL_RECORDS" \
  2>&1 | tee "$RUN_ROOT/logs/07_build_eval_slices.log"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/08_manifest_gate.log"
import json
from pathlib import Path
summary = json.loads(Path("$RUN_ROOT/seeds/slice_summary.json").read_text(encoding="utf-8"))
outputs = summary.get("outputs", {})
payload = {
    "stage": "formal_v2_manifest_gate",
    "socialbench_rows": outputs.get("socialbench", {}).get("rows", 0),
    "ceval_rows": outputs.get("ceval", {}).get("rows", 0),
    "min_socialbench": int("$MIN_SOCIALBENCH_RECORDS"),
    "min_ceval": int("$MIN_CEVAL_RECORDS"),
}
payload["status"] = "PASS" if payload["socialbench_rows"] >= payload["min_socialbench"] and payload["ceval_rows"] >= payload["min_ceval"] else "FAIL"
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
if payload["status"] != "PASS":
    raise SystemExit(11)
PY

roll_axis() {
  local axis="$1"
  local seed="$2"
  local scorer="$3"
  mkdir -p "$RUN_ROOT/rollouts/$axis" "$RUN_ROOT/scores/$axis"
  for ARM in base sft dpo grpo; do
    case "$ARM" in
      base) ADAPTER_ARGS=() ;;
      sft) ADAPTER_ARGS=(--adapter-path "$FORMAL_V1_RUN/models/sft_rolebench") ;;
      dpo) ADAPTER_ARGS=(--adapter-path "$FORMAL_V1_RUN/models/dpo_rolebench") ;;
      grpo) ADAPTER_ARGS=(--adapter-path "$FORMAL_V1_RUN/models/grpo_rolebench") ;;
    esac
    "$PY" -m anima.eval.generate_rollouts \
      --input-jsonl "$seed" \
      --output-jsonl "$RUN_ROOT/rollouts/$axis/${ARM}_${axis}.jsonl" \
      --arm "$ARM" \
      --axis "formal_v2_${axis}" \
      "${ADAPTER_ARGS[@]}" \
      --max-records "$( [ "$axis" = socialbench ] && echo "$MAX_SOCIALBENCH_RECORDS" || echo "$MAX_CEVAL_RECORDS" )" \
      --prompt-format chat_template \
      --temperature 0.0 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      2>&1 | tee "$RUN_ROOT/logs/rollout_${axis}_${ARM}.log"
    "$PY" -m "anima.eval.$scorer" \
      --input-jsonl "$RUN_ROOT/rollouts/$axis/${ARM}_${axis}.jsonl" \
      --output-jsonl "$RUN_ROOT/scores/$axis/${ARM}_${axis}.jsonl" \
      2>&1 | tee "$RUN_ROOT/logs/score_${axis}_${ARM}.log"
  done
}

echo "[7] SocialBench and C-Eval rollouts/scores"
roll_axis socialbench "$RUN_ROOT/seeds/socialbench_mcq.jsonl" run_socialbench
roll_axis ceval "$RUN_ROOT/seeds/ceval_mcq.jsonl" run_ceval

echo "[8] BaichuanCharRM scalar on formal-v1 heldout rollouts"
if [ ! -x "$RM_ENV/bin/python" ]; then
  echo "Missing RM env: $RM_ENV. Restore/build anima-rm-eval before formal-v2 quality scoring." >&2
  exit 12
fi
RM_PY="$RM_ENV/bin/python"
RM_MODEL="${BAICHUAN_CHARRM_PATH:-morecry/BaichuanCharRM}"
RM_DOWNLOAD_ARGS=()
if [ "$FORMAL_V2_ALLOW_RM_DOWNLOAD" = "1" ]; then
  RM_DOWNLOAD_ARGS=(--allow-download)
fi
mkdir -p "$RUN_ROOT/scores/charrm_scalar"
for ARM in base sft dpo grpo; do
  PYTHONPATH="$CODE/src:${PYTHONPATH:-}" "$RM_PY" -m anima.eval.run_charactereval \
    --input-jsonl "$FORMAL_V1_RUN/eval/${ARM}_heldout_rollouts.jsonl" \
    --output-jsonl "$RUN_ROOT/scores/charrm_scalar/${ARM}_charrm_scalar.jsonl" \
    --model-name "$RM_MODEL" \
    --trust-remote-code \
    --batch-size 1 \
    "${RM_DOWNLOAD_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/score_charrm_${ARM}.log"
done
RM_SNAPSHOT="$(find "$ROOT/models/transformers_cache" "$ROOT/hf_cache" -path '*models--morecry--BaichuanCharRM*snapshots*' -type d 2>/dev/null | sort | tail -n 1 || true)"
if [ -n "$RM_SNAPSHOT" ] && [ -d "$RM_SNAPSHOT" ]; then
  "$PY" -m anima.data.source_ledger \
    --output "$RUN_ROOT/data/source_ledger.json" \
    --source-id BaichuanCharRM \
    --name "BaichuanCharRM scalar scorer" \
    --url "https://huggingface.co/morecry/BaichuanCharRM" \
    --snapshot-or-commit "$(basename "$RM_SNAPSHOT")" \
    --license-name "upstream-huggingface-model-card" \
    --license-url-or-path "https://huggingface.co/morecry/BaichuanCharRM" \
    --redistribution "model_weights_not_redistributed" \
    --project-use "formal-v2 eval-only scalar role-play quality proxy" \
    --public-artifact-policy "do_not_commit_model_weights" \
    --notes "Scalar-only; not full 13-metric CharacterEval coverage." \
    --path "$RM_SNAPSHOT" \
    2>&1 | tee "$RUN_ROOT/logs/09_source_ledger_baichuan_charrm.log"
else
  echo "WARNING: could not locate local BaichuanCharRM snapshot for source ledger" | tee "$RUN_ROOT/logs/09_source_ledger_baichuan_charrm.log"
fi

echo "[9] aggregate"
"$PY" - <<PY
import json
from pathlib import Path
run = Path("$RUN_ROOT")
formal_v1 = Path("$FORMAL_V1_RUN")
config = {
    "run_name": "formal_v2_eval_first",
    "allow_missing_inputs": False,
    "baseline_arm": "SFT",
    "arms": ["Base", "SFT", "DPO", "GRPO"],
    "axes": [
        {
            "name": "heldout",
            "label": "RoleBench heldout rule replay continuity",
            "score_fields": ["heldout.score"],
            "status_fields": ["heldout.status"],
            "inputs": {
                "Base": [str(formal_v1 / "eval/base_heldout_scores.jsonl")],
                "SFT": [str(formal_v1 / "eval/sft_heldout_scores.jsonl")],
                "DPO": [str(formal_v1 / "eval/dpo_heldout_scores.jsonl")],
                "GRPO": [str(formal_v1 / "eval/grpo_heldout_scores.jsonl")],
            },
        },
        {
            "name": "charrm_scalar",
            "label": "BaichuanCharRM scalar on RoleBench heldout rollouts",
            "score_fields": ["charactereval.score"],
            "status_fields": ["charactereval.status"],
            "inputs": {
                "Base": [str(run / "scores/charrm_scalar/base_charrm_scalar.jsonl")],
                "SFT": [str(run / "scores/charrm_scalar/sft_charrm_scalar.jsonl")],
                "DPO": [str(run / "scores/charrm_scalar/dpo_charrm_scalar.jsonl")],
                "GRPO": [str(run / "scores/charrm_scalar/grpo_charrm_scalar.jsonl")],
            },
        },
        {
            "name": "socialbench",
            "label": "SocialBench MCQ",
            "score_fields": ["socialbench.accuracy"],
            "status_fields": ["socialbench.status"],
            "inputs": {
                "Base": [str(run / "scores/socialbench/base_socialbench.jsonl")],
                "SFT": [str(run / "scores/socialbench/sft_socialbench.jsonl")],
                "DPO": [str(run / "scores/socialbench/dpo_socialbench.jsonl")],
                "GRPO": [str(run / "scores/socialbench/grpo_socialbench.jsonl")],
            },
        },
        {
            "name": "ceval",
            "label": "C-Eval Chinese regression canary",
            "score_fields": ["ceval.accuracy"],
            "status_fields": ["ceval.status"],
            "inputs": {
                "Base": [str(run / "scores/ceval/base_ceval.jsonl")],
                "SFT": [str(run / "scores/ceval/sft_ceval.jsonl")],
                "DPO": [str(run / "scores/ceval/dpo_ceval.jsonl")],
                "GRPO": [str(run / "scores/ceval/grpo_ceval.jsonl")],
            },
        },
    ],
    "caveats": [
        "Formal-v2 is eval-first and reuses formal-v1 adapters; no new training is started.",
        "BaichuanCharRM is scalar-only on heldout rollouts, not full 13-metric CharacterEval.",
        "Output-contract health is a validity axis, not a quality metric.",
        "Do not claim GRPO superiority unless paired deltas support it.",
    ],
}
path = run / "aggregate/formal_v2_eval_config.json"
path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(path)
PY
"$PY" -m anima.eval.aggregate_results \
  --config "$RUN_ROOT/aggregate/formal_v2_eval_config.json" \
  --markdown-out "$RUN_ROOT/aggregate/formal_v2_results.md" \
  --json-out "$RUN_ROOT/aggregate/formal_v2_results.json" \
  --strict-missing \
  2>&1 | tee "$RUN_ROOT/logs/10_aggregate.log"

echo "[10] verdict"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/11_formal_v2_verdict.log"
import json
from pathlib import Path

run = Path("$RUN_ROOT")
formal_v1 = Path("$FORMAL_V1_RUN")
min_social = int("$MIN_SOCIALBENCH_RECORDS")
min_ceval = int("$MIN_CEVAL_RECORDS")
min_charrm = int("$MIN_CHARRM_ROWS")
arms = ["base", "sft", "dpo", "grpo"]

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

checks = {}
for arm in arms:
    heldout_summary = json.loads((formal_v1 / f"eval/{arm}_heldout_scores.summary.json").read_text(encoding="utf-8"))
    social_rows = read_jsonl(run / f"scores/socialbench/{arm}_socialbench.jsonl")
    ceval_rows = read_jsonl(run / f"scores/ceval/{arm}_ceval.jsonl")
    charrm_rows = read_jsonl(run / f"scores/charrm_scalar/{arm}_charrm_scalar.jsonl")
    checks[arm.upper()] = {
        "heldout_rows": heldout_summary.get("rows"),
        "heldout_contract_pass": heldout_summary.get("output_contract_status_counts", {}).get("pass", 0),
        "socialbench_rows": len(social_rows),
        "socialbench_ok": sum(1 for row in social_rows if row.get("socialbench", {}).get("status") == "ok"),
        "ceval_rows": len(ceval_rows),
        "ceval_ok": sum(1 for row in ceval_rows if row.get("ceval", {}).get("status") == "ok"),
        "charrm_rows": len(charrm_rows),
        "charrm_ok": sum(1 for row in charrm_rows if row.get("charactereval", {}).get("status") == "ok"),
    }

failures = []
for arm, row in checks.items():
    if row["socialbench_rows"] < min_social or row["socialbench_ok"] < min_social:
        failures.append(f"{arm}:socialbench")
    if row["ceval_rows"] < min_ceval or row["ceval_ok"] < min_ceval:
        failures.append(f"{arm}:ceval")
    if row["charrm_rows"] < min_charrm or row["charrm_ok"] < min_charrm:
        failures.append(f"{arm}:charrm")

verdict = {
    "stage": "formal_v2_eval_first_verdict",
    "run_id": "$RUN_ID",
    "formal_v1_run": str(formal_v1),
    "status": "PASS" if not failures else "FAIL",
    "failures": failures,
    "checks": checks,
    "allowed_claim": "Formal-v2 eval-first completed if status=PASS; this is a multi-axis evaluation of formal-v1 adapters, not new training.",
    "forbidden_claims": [
        "Do not claim GRPO superiority without supported deltas.",
        "Do not claim full CharacterEval 13-metric coverage from BaichuanCharRM scalar alone.",
        "Do not publish raw third-party rows.",
    ],
}
(run / "artifacts/REPORTABILITY_VERDICT.json").write_text(
    json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print(json.dumps(verdict, ensure_ascii=False, sort_keys=True))
if failures:
    raise SystemExit(13)
PY

echo "[DONE] formal-v2 eval-first completed"
