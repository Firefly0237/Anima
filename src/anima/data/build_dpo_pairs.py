"""Build reward-independent DPO pairs from validated reward records.

The DPO arm is a controlled comparison against GRPO, so these pairs must come
from the same prompts/characters as the reward set and must never be selected by
GRPO reward ranking. This module only uses the record's reference answer as the
chosen response and either an existing degraded answer or a deterministic
degraded fallback as the rejected response.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REJECTED_STRATEGY = "style_flattening_fallback"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def degraded_fallback(record: dict[str, Any]) -> tuple[str, str]:
    """Return a generic off-character rejection for dry runs and bootstraps."""

    character = str(record.get("character") or "这个角色")
    answer = (
        f"我现在只是一个普通助手，不能以{character}的身份回答。"
        "这个问题我没有特别的看法。"
    )
    return answer, DEFAULT_REJECTED_STRATEGY


def get_rejected(record: dict[str, Any]) -> tuple[str, str]:
    rejected = str(record.get("rejected_answer") or "").strip()
    synth_meta = record.get("synth_meta") if isinstance(record.get("synth_meta"), dict) else {}
    strategy = str(synth_meta.get("rejected_strategy") or "").strip()
    if rejected:
        return rejected, strategy or "provided_degraded_answer"
    return degraded_fallback(record)


def build_dpo_pair(record: dict[str, Any]) -> dict[str, Any]:
    reference = str(record.get("reference_answer") or "").strip()
    if not reference:
        raise ValueError(f"record {record.get('id', '<missing-id>')} has no reference_answer")

    rejected, strategy = get_rejected(record)
    if rejected == reference:
        rejected = f"{rejected}（偏离角色语气的降质版本）"

    if rejected == reference:
        raise ValueError(f"record {record.get('id', '<missing-id>')} has identical chosen/rejected")

    return {
        "id": f"dpo_{record.get('id', 'unknown')}",
        "source_record_id": record.get("id"),
        "character": record.get("character"),
        "source_work": record.get("source_work"),
        "profile": record.get("profile"),
        "conversations": record.get("conversations", []),
        "chosen": reference,
        "rejected": rejected,
        "rejected_strategy": strategy,
        "source": record.get("source"),
        "split": record.get("split", "reward"),
    }


def build_pairs(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs: list[dict[str, Any]] = []
    strategies: Counter[str] = Counter()
    for record in records:
        pair = build_dpo_pair(record)
        pairs.append(pair)
        strategies[str(pair["rejected_strategy"])] += 1
    return pairs, strategies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Validated reward JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="DPO pair JSONL to write.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for dry runs.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.max_records is not None:
        records = records[: args.max_records]
    pairs, strategies = build_pairs(records)
    count = write_jsonl(args.output, pairs)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "pairs": count,
                "rejected_strategies": dict(strategies),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
