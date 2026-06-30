"""Build source-disjoint formal RoleBench splits.

This builder is stricter than the official-reset bootstrap: it creates a train
reward file, matched DPO pairs, and a held-out role-play file whose characters /
source works are disjoint from training. It is still not a human-gold
Character-R1 dataset; labels come from deterministic public RoleBench fields.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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
from anima.data.build_reset_dataset import BOOTSTRAP_LABEL_SOURCE, validate_source_ledger
from anima.data.schemas import validate_record


DEFAULT_FOCUS = ("Style", "Engagement")


def split_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('source_work')}::{row.get('character')}"


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_generated_answer(value: Any) -> str:
    if isinstance(value, list | tuple):
        for item in value:
            text = normalize_text_value(item)
            if text:
                return text
        return ""
    return normalize_text_value(value)


def normalize_rolebench_record(raw: Mapping[str, Any], *, index: int, source: str, id_prefix: str) -> dict[str, Any] | None:
    record = dict(raw)
    character = first_nonempty(record, ("character", "role", "name", "character_name"))
    if not character:
        return None
    source_work = first_nonempty(record, ("source_work", "work", "book", "movie"), f"{source}/{character}")
    conversations = normalize_conversations(record)
    if not conversations:
        return None
    reference_answer = first_nonempty(record, ("reference_answer", "answer", "response", "output"))
    if not reference_answer:
        reference_answer = first_generated_answer(record.get("generated"))
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
        "rejected_answer": None,
        "source": source,
        "split": "reward",
        "synth_meta": None,
        "label_source": BOOTSTRAP_LABEL_SOURCE,
        "reference_answer_source": "public_rolebench_field_or_first_generated",
        "claim_boundary": (
            "Deterministic RoleBench formal-v1 labels for official reproduction. "
            "Not human-gold Character-R1 quality data."
        ),
    }


def build_rows(
    inputs: Sequence[Path],
    *,
    source: str,
    id_prefix: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for path in collect_input_files(list(inputs)):
        if not path.exists():
            skipped["missing_input_file"] += 1
            continue
        for raw in iter_json_objects(path):
            row = normalize_rolebench_record(raw, index=len(rows), source=source, id_prefix=id_prefix)
            if row is None:
                skipped["unusable_row"] += 1
                continue
            errors = validate_record(row)
            if errors:
                skipped["schema_error"] += 1
                continue
            rows.append(row)
    return rows, skipped


def partition_by_key(
    rows: Sequence[dict[str, Any]],
    *,
    heldout_fraction: float,
) -> tuple[set[str], set[str]]:
    keys = sorted({split_key(row) for row in rows})
    if len(keys) < 2:
        raise ValueError("source-disjoint split needs at least two distinct character/source_work keys")
    ranked = sorted(keys, key=stable_digest)
    heldout_count = max(1, round(len(ranked) * heldout_fraction))
    heldout_count = min(heldout_count, len(ranked) - 1)
    heldout_keys = set(ranked[:heldout_count])
    train_keys = set(ranked[heldout_count:])
    return train_keys, heldout_keys


def cap_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_records: int,
    max_per_character: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        character = str(row.get("character") or "")
        if max_per_character > 0 and counts[character] >= max_per_character:
            continue
        out.append(dict(row))
        counts[character] += 1
        if max_records > 0 and len(out) >= max_records:
            break
    return out


def to_heldout_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "id": f"formal_rolebench_heldout_{index:06d}",
        "axis": "heldout",
        "source": row["source"],
        "split": "eval_heldout",
        "character": row["character"],
        "source_work": row["source_work"],
        "profile": row["profile"],
        "conversations": row["conversations"],
        "gold_focus": row["gold_focus"],
        "gold_focus_attr": row["gold_focus_attr"],
        "reference_answer": row["reference_answer"],
        "label_source": row.get("label_source"),
        "reference_answer_source": row.get("reference_answer_source"),
        "reportability_status": "reportable",
        "reportability_reasons": [],
    }


def write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="RoleBench")
    parser.add_argument("--id-prefix", default="formal_rolebench")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-records", type=int, default=256)
    parser.add_argument("--max-heldout-records", type=int, default=128)
    parser.add_argument("--max-train-per-character", type=int, default=128)
    parser.add_argument("--max-heldout-per-character", type=int, default=128)
    parser.add_argument("--min-train-records", type=int, default=32)
    parser.add_argument("--min-heldout-records", type=int, default=16)
    parser.add_argument("--allow-small-formal-v1", action="store_true")
    args = parser.parse_args(argv)

    ledger_entry = validate_source_ledger(args.source_ledger, source_id=args.source)
    rows, skipped = build_rows(args.input, source=args.source, id_prefix=args.id_prefix)
    if skipped:
        raise SystemExit(f"formal RoleBench build skipped rows: {dict(skipped)}")
    if not rows:
        raise SystemExit("formal RoleBench build produced no rows")

    train_keys, heldout_keys = partition_by_key(rows, heldout_fraction=args.heldout_fraction)
    train_rows = cap_rows(
        (row for row in rows if split_key(row) in train_keys),
        max_records=args.max_train_records,
        max_per_character=args.max_train_per_character,
    )
    heldout_base = cap_rows(
        (row for row in rows if split_key(row) in heldout_keys),
        max_records=args.max_heldout_records,
        max_per_character=args.max_heldout_per_character,
    )
    heldout_rows = [to_heldout_row(row, index=index) for index, row in enumerate(heldout_base)]
    if not args.allow_small_formal_v1 and (
        len(train_rows) < args.min_train_records or len(heldout_rows) < args.min_heldout_records
    ):
        raise SystemExit(
            "formal RoleBench split below minimums: "
            f"train={len(train_rows)} min={args.min_train_records}, "
            f"heldout={len(heldout_rows)} min={args.min_heldout_records}"
        )

    pairs, strategies = build_pairs(train_rows)
    output_dir = args.output_dir
    reward_path = output_dir / "reward_train.jsonl"
    dpo_path = output_dir / "dpo_train.jsonl"
    heldout_path = output_dir / "heldout_roleplay.jsonl"
    reward_count = write_jsonl(reward_path, train_rows)
    dpo_count = write_dpo_jsonl(dpo_path, pairs)
    heldout_count = write_jsonl(heldout_path, heldout_rows)
    payload = {
        "stage": "build_formal_rolebench",
        "source": args.source,
        "input": [str(path) for path in args.input],
        "source_ledger": {
            "path": str(args.source_ledger),
            "source_id": ledger_entry["source_id"],
            "snapshot_or_commit": ledger_entry.get("snapshot_or_commit"),
            "license": ledger_entry.get("license"),
        },
        "reward_output": str(reward_path),
        "dpo_output": str(dpo_path),
        "heldout_output": str(heldout_path),
        "reward_rows": reward_count,
        "dpo_rows": dpo_count,
        "heldout_rows": heldout_count,
        "train_keys": sorted(train_keys),
        "heldout_keys": sorted(heldout_keys),
        "train_characters": len({row["character"] for row in train_rows}),
        "heldout_characters": len({row["character"] for row in heldout_rows}),
        "rejected_strategies": dict(sorted(strategies.items())),
        "label_source": BOOTSTRAP_LABEL_SOURCE,
        "claim_boundary": (
            "Formal-v1 public RoleBench split for official reproduction. "
            "It proves source-disjoint trainer/eval plumbing, not human-gold model quality."
        ),
    }
    write_summary(output_dir / "formal_rolebench_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
