"""Build a small official-reset dataset from public role-play rows.

This is a deterministic bootstrap builder. It does not call an API and does not
pretend to create human-gold Character-R1 data. It turns public role prompts and
reference answers into auditable SFT/GRPO/DPO records so the official Qwen/TRL
path can be rebuilt from a clean server.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from anima.data.build_dpo_pairs import build_pairs, write_jsonl as write_dpo_jsonl
from anima.data.build_reward_records import (
    collect_input_files,
    first_nonempty,
    iter_json_objects,
    normalize_conversations,
    normalize_text_value,
    write_jsonl,
)
from anima.data.schemas import validate_record


DEFAULT_FOCUS = ("Style", "Engagement")
BOOTSTRAP_LABEL_SOURCE = "deterministic_bootstrap_not_human_gold"


def build_reward_record(raw: Mapping[str, Any], *, index: int, source: str, id_prefix: str) -> dict[str, Any] | None:
    record = dict(raw)
    character = first_nonempty(record, ("character", "role", "name", "character_name"))
    if not character:
        return None
    source_work = first_nonempty(record, ("source_work", "work", "book", "movie"), f"{source}/{character}")
    conversations = normalize_conversations(record)
    if not conversations:
        return None
    reference_answer = first_nonempty(record, ("reference_answer", "answer", "response", "generated", "output"))
    if not reference_answer:
        return None
    profile = first_nonempty(
        record,
        ("profile", "persona", "description", "character_profile", "system_prompt"),
        (
            f"{character}角色卡：来自{source_work}。请保持该角色身份、称谓、语气和价值观，"
            "用中文回应，不要以普通助手口吻回答。"
        ),
    )
    rejected_answer = normalize_text_value(record.get("rejected_answer"))
    return {
        "id": str(record.get("id") or f"{id_prefix}_{index:06d}"),
        "character": character,
        "source_work": source_work,
        "character_cluster": int(record.get("character_cluster", -1) or -1),
        "profile": profile,
        "conversations": conversations,
        "gold_focus": list(record.get("gold_focus") or DEFAULT_FOCUS),
        "gold_focus_attr": normalize_text_value(record.get("gold_focus_attr"))
        or f"保持{character}的身份、语气和互动推进感。",
        "reference_answer": reference_answer,
        "rejected_answer": rejected_answer or None,
        "source": source,
        "split": "reward",
        "synth_meta": None
        if not rejected_answer
        else {
            "generator": "source_reset_builder",
            "prompt_id": "official_reset_deterministic_bootstrap_v1",
            "lore_sources": [source_work],
            "rejected_strategy": "provided_degraded_answer",
            "human_reviewed": False,
        },
        "label_source": BOOTSTRAP_LABEL_SOURCE,
        "claim_boundary": (
            "Heuristic bootstrap labels for official trainer-path smoke. "
            "Do not report as human-gold Character-R1 quality data."
        ),
    }


def build_records(
    inputs: Sequence[Path],
    *,
    source: str,
    id_prefix: str,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for path in collect_input_files(list(inputs)):
        if not path.exists():
            skipped["missing_input_file"] += 1
            continue
        for raw in iter_json_objects(path):
            row = build_reward_record(raw, index=len(rows), source=source, id_prefix=id_prefix)
            if row is None:
                skipped["unusable_row"] += 1
                continue
            errors = validate_record(row)
            if errors:
                skipped["schema_error"] += 1
                continue
            rows.append(row)
            if max_records is not None and len(rows) >= max_records:
                return rows, skipped
    return rows, skipped


def write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source_ledger(path: Path, *, source_id: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} has no entries list")
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("source_id") != source_id:
            continue
        license_info = entry.get("license")
        if not isinstance(license_info, dict):
            raise ValueError(f"{path} entry {source_id} has no license object")
        if str(license_info.get("name") or "").lower() in {"", "unknown", "none", "todo"}:
            raise ValueError(f"{path} entry {source_id} has unknown license")
        if int(entry.get("file_count") or 0) <= 0:
            raise ValueError(f"{path} entry {source_id} has no files")
        return entry
    raise ValueError(f"{path} has no ledger entry for source_id={source_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--reward-output", required=True, type=Path)
    parser.add_argument("--dpo-output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--source", default="RoleBench")
    parser.add_argument("--id-prefix", default="official_reset")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--allow-skips-for-smoke",
        action="store_true",
        help="Allow unusable rows to be skipped only for explicit reset smoke runs.",
    )
    args = parser.parse_args(argv)

    ledger_entry = validate_source_ledger(args.source_ledger, source_id=args.source)
    rows, skipped = build_records(
        args.input,
        source=args.source,
        id_prefix=args.id_prefix,
        max_records=args.max_records,
    )
    if not rows:
        raise SystemExit("No usable reset records were built")
    if skipped and not args.allow_skips_for_smoke:
        raise SystemExit(f"Reset dataset build skipped rows in strict mode: {dict(skipped)}")

    reward_count = write_jsonl(args.reward_output, rows)
    pairs, strategies = build_pairs(rows)
    dpo_count = write_dpo_jsonl(args.dpo_output, pairs)
    summary = {
        "stage": "build_reset_dataset",
        "source": args.source,
        "input": [str(path) for path in args.input],
        "reward_output": str(args.reward_output),
        "dpo_output": str(args.dpo_output),
        "reward_rows": reward_count,
        "dpo_rows": dpo_count,
        "skipped": dict(sorted(skipped.items())),
        "focus_policy": "deterministic_bootstrap",
        "default_focus": list(DEFAULT_FOCUS),
        "label_source": BOOTSTRAP_LABEL_SOURCE,
        "claim_boundary": (
            "Deterministic reset dataset for official trainer-path reproduction. "
            "Not a human-gold Character-R1 quality dataset."
        ),
        "source_ledger": {
            "path": str(args.source_ledger),
            "source_id": ledger_entry["source_id"],
            "snapshot_or_commit": ledger_entry.get("snapshot_or_commit"),
            "license": ledger_entry.get("license"),
        },
        "characters": len({row["character"] for row in rows}),
        "source_works": len({row["source_work"] for row in rows}),
        "rejected_strategies": dict(sorted(strategies.items())),
    }
    write_summary(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
