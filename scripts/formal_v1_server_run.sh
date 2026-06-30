#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/home/featurize/work/Anima}"
ROOT="${ROOT:-/home/featurize/work/anima}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/envs/anima-train-v2}"
RUN_ID="${RUN_ID:-formal_v1_rolebench_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$ROOT/formal_v1/$RUN_ID"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-96}"
MAX_HELDOUT_RECORDS="${MAX_HELDOUT_RECORDS:-96}"
MIN_TRAIN_RECORDS="${MIN_TRAIN_RECORDS:-32}"
MIN_HELDOUT_RECORDS="${MIN_HELDOUT_RECORDS:-16}"
MIN_SFT_OVERFIT_CONTRACT_PASS="${MIN_SFT_OVERFIT_CONTRACT_PASS:-4}"
MIN_TRAINED_HELDOUT_CONTRACT_RATE="${MIN_TRAINED_HELDOUT_CONTRACT_RATE:-0.80}"

mkdir -p "$RUN_ROOT"/{logs,data,models,eval,aggregate,artifacts}

ART="$ROOT/backups/${RUN_ID}.tar.gz"
finish() {
  status=$?
  mkdir -p "$ROOT/backups" "$RUN_ROOT/artifacts"
  printf '{"run_id":"%s","exit_status":%s}\n' "$RUN_ID" "$status" > "$RUN_ROOT/artifacts/EXIT_STATUS.json" || true
  if [ -d "$RUN_ROOT" ]; then
    tar -czf "$ART" -C "$ROOT/formal_v1" "$RUN_ID" || true
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
mkdir -p "$RUN_ROOT/artifacts/env_locks"
cp -f "$ROOT"/envs/locks/* "$RUN_ROOT/artifacts/env_locks/" 2>/dev/null || true

echo "[0b] API and tokenizer inspect"
"$PY" - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/00b_api_tokenizer_inspect.log"
import inspect, json
import torch, transformers, trl, peft, datasets
from transformers import AutoTokenizer
from trl import SFTTrainer, DPOTrainer, GRPOTrainer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
payload = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "peft": peft.__version__,
    "datasets": datasets.__version__,
    "qwen_has_chat_template": bool(getattr(tok, "chat_template", None)),
    "qwen_eos_token": tok.eos_token,
    "sft_trainer_signature": str(inspect.signature(SFTTrainer.__init__)),
    "dpo_trainer_signature": str(inspect.signature(DPOTrainer.__init__)),
    "grpo_trainer_signature": str(inspect.signature(GRPOTrainer.__init__)),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
assert payload["cuda_available"], "CUDA is required"
assert payload["qwen_has_chat_template"], "Qwen tokenizer chat_template is required"
PY

echo "[1] code validation"
"$PY" -m compileall -q src tests scripts 2>&1 | tee "$RUN_ROOT/logs/01_compileall.log"
"$PY" -m pytest \
  tests/test_train_chat_template.py \
  tests/test_generate_rollouts.py \
  tests/test_parsing.py \
  tests/test_rewards_reference.py \
  tests/test_build_reset_dataset.py \
  tests/test_build_formal_rolebench.py \
  tests/test_source_ledger.py \
  2>&1 | tee "$RUN_ROOT/logs/02_pytest.log"

echo "[2] claim boundary"
"$PY" -m anima.utils.claim_boundary \
  --output-dir "$RUN_ROOT/artifacts" \
  --run-name "$RUN_ID" \
  --stage formal_v1_rolebench \
  --notes "RoleBench source-disjoint formal-v1 gate. Not final multi-source leaderboard."

echo "[3] download public source"
bash scripts/download_datasets.sh 2>&1 | tee "$RUN_ROOT/logs/03_download_datasets.log"
ROLEBENCH_RAW="$(find "$ROOT/data/raw/hf_snapshot/ZenMoore__RoleBench" -path '*rolebench-zh*role_specific*rolegpt_baseline.jsonl' -print -quit)"
test -f "$ROLEBENCH_RAW"
ROLEBENCH_REVISION_RESOLVED="$(cat "$ROOT/data/raw/hf_snapshot/ZenMoore__RoleBench/.snapshot_revision")"

echo "[4] source ledger"
"$PY" -m anima.data.source_ledger \
  --output "$RUN_ROOT/data/source_ledger.json" \
  --source-id RoleBench \
  --name RoleBench \
  --url "https://huggingface.co/datasets/ZenMoore/RoleBench" \
  --snapshot-or-commit "$ROLEBENCH_REVISION_RESOLVED" \
  --license-name "Apache-2.0" \
  --license-url-or-path "https://huggingface.co/datasets/ZenMoore/RoleBench" \
  --redistribution "manifest_only" \
  --project-use "formal-v1 source-disjoint SFT/DPO/GRPO role-play gate" \
  --public-artifact-policy "do_not_redistribute_raw_rows" \
  --path "$ROLEBENCH_RAW" \
  2>&1 | tee "$RUN_ROOT/logs/04_source_ledger.log"

echo "[5] build source-disjoint RoleBench formal data"
"$PY" -m anima.data.build_formal_rolebench \
  --input "$ROLEBENCH_RAW" \
  --source-ledger "$RUN_ROOT/data/source_ledger.json" \
  --output-dir "$RUN_ROOT/data" \
  --source RoleBench \
  --id-prefix formal_rolebench \
  --heldout-fraction 0.5 \
  --max-train-records "$MAX_TRAIN_RECORDS" \
  --max-heldout-records "$MAX_HELDOUT_RECORDS" \
  --min-train-records "$MIN_TRAIN_RECORDS" \
  --min-heldout-records "$MIN_HELDOUT_RECORDS" \
  2>&1 | tee "$RUN_ROOT/logs/05_build_formal_rolebench.log"

echo "[6] data gate"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/06_data_gate.log"
import json, sys
from pathlib import Path
summary = json.loads(Path("$RUN_ROOT/data/formal_rolebench_summary.json").read_text(encoding="utf-8"))
train = int(summary["reward_rows"])
heldout = int(summary["heldout_rows"])
train_keys = set(summary["train_keys"])
heldout_keys = set(summary["heldout_keys"])
payload = {
    "stage": "formal_v1_data_gate",
    "train_rows": train,
    "heldout_rows": heldout,
    "train_keys": len(train_keys),
    "heldout_keys": len(heldout_keys),
    "disjoint": not bool(train_keys & heldout_keys),
    "label_source": summary.get("label_source"),
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
if train < int("$MIN_TRAIN_RECORDS") or heldout < int("$MIN_HELDOUT_RECORDS") or train_keys & heldout_keys:
    sys.exit(2)
PY

echo "[7] schema render audit"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/07_schema_render_audit.log"
import json
from pathlib import Path
from transformers import AutoTokenizer
from anima.train.common import build_sft_chat_rows, build_grpo_chat_rows, render_messages_with_chat_template
from anima.train.dpo import build_dpo_rows

reward = Path("$RUN_ROOT/data/reward_train.jsonl")
dpo = Path("$RUN_ROOT/data/dpo_train.jsonl")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
sft = build_sft_chat_rows([reward], max_records=1, require_gold_format=True).rows[0]
dpo_row = build_dpo_rows([dpo], max_records=1, dpo_schema="chat_template").rows[0]
grpo = build_grpo_chat_rows([reward], max_records=1, validate_schema=True).rows[0]
payload = {
    "stage": "formal_v1_schema_render_audit",
    "sft_prompt_rendered_prefix": render_messages_with_chat_template(sft["prompt"], tok, mode="formal")[:500],
    "sft_completion": sft["completion"],
    "dpo_prompt_rendered_prefix": render_messages_with_chat_template(dpo_row["prompt"], tok, mode="formal")[:500],
    "dpo_chosen": dpo_row["chosen"],
    "dpo_rejected": dpo_row["rejected"],
    "grpo_prompt_rendered_prefix": render_messages_with_chat_template(grpo["prompt"], tok, mode="formal")[:500],
}
out = Path("$RUN_ROOT/artifacts/schema_render_audit.json")
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({"stage": payload["stage"], "output": str(out)}, ensure_ascii=False, sort_keys=True))
PY

echo "[8] SFT overfit4 learnability gate"
"$PY" -m anima.train.sft \
  --config configs/official_reset_sft_overfit4.yaml \
  --train-jsonl "$RUN_ROOT/data/reward_train.jsonl" \
  --output-dir "$RUN_ROOT/models/sft_overfit4" \
  --max-records 4 \
  2>&1 | tee "$RUN_ROOT/logs/08_sft_overfit4_train.log"
"$PY" -m anima.eval.generate_rollouts \
  --input-jsonl "$RUN_ROOT/data/reward_train.jsonl" \
  --output-jsonl "$RUN_ROOT/eval/sft_overfit4_seen4_rollouts.jsonl" \
  --arm SFT_OVERFIT4 \
  --axis formal_v1_seen4 \
  --adapter-path "$RUN_ROOT/models/sft_overfit4" \
  --max-records 4 \
  --prompt-format chat_template \
  --temperature 0.0 \
  --max-new-tokens 256 \
  2>&1 | tee "$RUN_ROOT/logs/09_sft_overfit4_rollout.log"
"$PY" -m anima.eval.run_heldout \
  --input-jsonl "$RUN_ROOT/eval/sft_overfit4_seen4_rollouts.jsonl" \
  --output-jsonl "$RUN_ROOT/eval/sft_overfit4_seen4_scores.jsonl" \
  --summary-json "$RUN_ROOT/eval/sft_overfit4_seen4_scores.summary.json" \
  --require-gold \
  2>&1 | tee "$RUN_ROOT/logs/10_sft_overfit4_score.log"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/11_sft_overfit4_gate.log"
import json, sys
from pathlib import Path
summary = json.loads(Path("$RUN_ROOT/eval/sft_overfit4_seen4_scores.summary.json").read_text(encoding="utf-8"))
passed = int(summary.get("output_contract_status_counts", {}).get("pass", 0))
payload = {"stage": "sft_overfit4_gate", "contract_pass": passed, "required": int("$MIN_SFT_OVERFIT_CONTRACT_PASS")}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
if passed < int("$MIN_SFT_OVERFIT_CONTRACT_PASS"):
    sys.exit(3)
PY

echo "[12] train SFT/DPO/GRPO"
"$PY" -m anima.train.sft \
  --config configs/formal_v1_sft_rolebench.yaml \
  --train-jsonl "$RUN_ROOT/data/reward_train.jsonl" \
  --output-dir "$RUN_ROOT/models/sft_rolebench" \
  --max-records "$MAX_TRAIN_RECORDS" \
  2>&1 | tee "$RUN_ROOT/logs/12_sft_train.log"
"$PY" -m anima.utils.claim_boundary --output-dir "$RUN_ROOT/models/sft_rolebench" --run-name "$RUN_ID" --stage formal_v1_sft --notes "SFT formal-v1 RoleBench source-disjoint gate."

"$PY" -m anima.train.dpo \
  --config configs/formal_v1_dpo_rolebench.yaml \
  --dpo-jsonl "$RUN_ROOT/data/dpo_train.jsonl" \
  --sft-adapter "$RUN_ROOT/models/sft_rolebench" \
  --output-dir "$RUN_ROOT/models/dpo_rolebench" \
  --max-records "$MAX_TRAIN_RECORDS" \
  2>&1 | tee "$RUN_ROOT/logs/13_dpo_train.log"
"$PY" -m anima.utils.claim_boundary --output-dir "$RUN_ROOT/models/dpo_rolebench" --run-name "$RUN_ID" --stage formal_v1_dpo --notes "DPO comparison arm; reward-independent degraded pairs."

"$PY" -m anima.train.grpo \
  --config configs/formal_v1_grpo_rolebench.yaml \
  --reward-jsonl "$RUN_ROOT/data/reward_train.jsonl" \
  --sft-adapter "$RUN_ROOT/models/sft_rolebench" \
  --output-dir "$RUN_ROOT/models/grpo_rolebench" \
  --max-records "$MAX_TRAIN_RECORDS" \
  2>&1 | tee "$RUN_ROOT/logs/14_grpo_train.log"
"$PY" -m anima.utils.claim_boundary --output-dir "$RUN_ROOT/models/grpo_rolebench" --run-name "$RUN_ID" --stage formal_v1_grpo --notes "GRPO RLVR reproduction arm; rule rewards only."

echo "[15] source-disjoint heldout four-arm eval"
for ARM in base sft dpo grpo; do
  case "$ARM" in
    base) ADAPTER_ARGS=() ;;
    sft) ADAPTER_ARGS=(--adapter-path "$RUN_ROOT/models/sft_rolebench") ;;
    dpo) ADAPTER_ARGS=(--adapter-path "$RUN_ROOT/models/dpo_rolebench") ;;
    grpo) ADAPTER_ARGS=(--adapter-path "$RUN_ROOT/models/grpo_rolebench") ;;
  esac
  "$PY" -m anima.eval.generate_rollouts \
    --input-jsonl "$RUN_ROOT/data/heldout_roleplay.jsonl" \
    --output-jsonl "$RUN_ROOT/eval/${ARM}_heldout_rollouts.jsonl" \
    --arm "$ARM" \
    --axis formal_v1_rolebench_heldout \
    "${ADAPTER_ARGS[@]}" \
    --max-records "$MAX_HELDOUT_RECORDS" \
    --prompt-format chat_template \
    --temperature 0.0 \
    --max-new-tokens 256 \
    2>&1 | tee "$RUN_ROOT/logs/15_rollout_${ARM}.log"

  "$PY" -m anima.eval.run_heldout \
    --input-jsonl "$RUN_ROOT/eval/${ARM}_heldout_rollouts.jsonl" \
    --output-jsonl "$RUN_ROOT/eval/${ARM}_heldout_scores.jsonl" \
    --summary-json "$RUN_ROOT/eval/${ARM}_heldout_scores.summary.json" \
    --require-gold \
    2>&1 | tee "$RUN_ROOT/logs/16_score_${ARM}.log"
done

echo "[17] aggregate formal v1 heldout table"
"$PY" - <<PY
import json
from pathlib import Path
run = Path("$RUN_ROOT")
config = {
    "run_name": "formal_v1_rolebench_source_disjoint",
    "allow_missing_inputs": False,
    "baseline_arm": "SFT",
    "arms": ["Base", "SFT", "DPO", "GRPO"],
    "axes": [
        {
            "name": "heldout",
            "score_fields": ["heldout.score"],
            "status_fields": ["heldout.status"],
            "inputs": {
                "Base": [str(run / "eval/base_heldout_scores.jsonl")],
                "SFT": [str(run / "eval/sft_heldout_scores.jsonl")],
                "DPO": [str(run / "eval/dpo_heldout_scores.jsonl")],
                "GRPO": [str(run / "eval/grpo_heldout_scores.jsonl")],
            },
        }
    ],
    "caveats": [
        "RoleBench source-disjoint formal-v1 gate; not final multi-source leaderboard.",
        "Deterministic bootstrap labels are not human-gold Character-R1 quality data.",
        "Do not claim superiority without paired held-out deltas and reportability gates.",
    ],
}
path = run / "aggregate/formal_v1_eval_config.json"
path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(path)
PY
"$PY" -m anima.eval.aggregate_results \
  --config "$RUN_ROOT/aggregate/formal_v1_eval_config.json" \
  --markdown-out "$RUN_ROOT/aggregate/formal_v1_results.md" \
  --json-out "$RUN_ROOT/aggregate/formal_v1_results.json" \
  --strict-missing \
  2>&1 | tee "$RUN_ROOT/logs/17_aggregate.log"

echo "[18] formal v1 verdict"
"$PY" - <<PY 2>&1 | tee "$RUN_ROOT/logs/18_formal_v1_verdict.log"
import json, sys
from pathlib import Path
run = Path("$RUN_ROOT")
min_rate = float("$MIN_TRAINED_HELDOUT_CONTRACT_RATE")
rows = {}
for arm in ["base", "sft", "dpo", "grpo"]:
    summary = json.loads((run / f"eval/{arm}_heldout_scores.summary.json").read_text(encoding="utf-8"))
    total = int(summary.get("rows") or 0)
    passed = int(summary.get("output_contract_status_counts", {}).get("pass", 0))
    rows[arm.upper()] = {"rows": total, "contract_pass": passed, "contract_rate": passed / total if total else 0.0}
verdict = {
    "stage": "formal_v1_verdict",
    "run_id": "$RUN_ID",
    "min_trained_heldout_contract_rate": min_rate,
    "arms": rows,
    "allowed_claim": "RoleBench source-disjoint formal-v1 gate completed if status=PASS; not final multi-source leaderboard.",
    "forbidden_claims": [
        "Do not claim final GRPO superiority.",
        "Do not claim human-gold Character-R1 data.",
        "Do not publish raw third-party rows.",
    ],
}
trained_ok = all(rows[arm]["contract_rate"] >= min_rate for arm in ["SFT", "DPO", "GRPO"])
verdict["status"] = "PASS" if trained_ok else "FAIL"
(run / "artifacts/REPORTABILITY_VERDICT.json").write_text(
    json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print(json.dumps(verdict, ensure_ascii=False, sort_keys=True))
if not trained_ok:
    sys.exit(4)
PY

echo "[DONE] formal v1 RoleBench source-disjoint gate completed"
