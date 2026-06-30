"""Held-out rollout scorer for Character-R1 style completions.

This scorer is intentionally local and deterministic: it reads rollout JSONL,
preserves every input field, and adds a ``heldout`` object with format/length
metrics plus optional reward components when gold fields are present.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from anima.rewards.combine import REWARD_WEIGHTS, component_rewards
from anima.rewards.format_reward import format_score
from anima.rewards.parsing import completion_to_text, output_contract_report


RESPONSE_FIELDS = ("response", "completion", "model_response", "completion_raw")
GOLD_FIELDS = ("gold_focus", "gold_focus_attr", "reference_answer")
COMPONENT_NAMES = ("focus", "attribute", "reference", "format")


def load_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            yield line_no, record


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def score_record(record: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    response_field, response_value = first_present(record, RESPONSE_FIELDS)
    response_text = completion_to_text(response_value)
    missing_gold = [field for field in GOLD_FIELDS if not has_value(record.get(field))]

    heldout: dict[str, Any] = {
        "status": "ok",
        "score": None,
        "format_score": format_score(response_value),
        "output_contract": output_contract_report(response_value),
        "length_chars": len(response_text),
        "response_field": response_field,
        "gold_available": not missing_gold,
        "missing_gold_fields": missing_gold,
    }

    if response_field is None:
        heldout["status"] = "missing_response"
        return heldout

    if dry_run:
        heldout["status"] = "dry_run"
        return heldout

    if missing_gold:
        heldout["status"] = "missing_gold"
        return heldout

    try:
        raw_components = component_rewards(
            response_value,
            gold_focus=record.get("gold_focus"),
            gold_focus_attr=record.get("gold_focus_attr"),
            reference_answer=record.get("reference_answer"),
        )
        components = {name: float(values[0]) for name, values in raw_components.items()}
        heldout["components"] = components
        heldout["reward_weights"] = {
            name: weight for name, weight in zip(COMPONENT_NAMES, REWARD_WEIGHTS)
        }
        heldout["score"] = float(
            sum(
                weight * components[name]
                for weight, name in zip(REWARD_WEIGHTS, COMPONENT_NAMES)
            )
        )
    except Exception as exc:  # Keep row-level scorer failures inspectable.
        heldout["status"] = "error"
        heldout["error"] = str(exc)

    return heldout


def score_rows(rows: Iterable[Mapping[str, Any]], *, dry_run: bool = False) -> Iterator[dict[str, Any]]:
    for record in rows:
        row = dict(record)
        row["heldout"] = score_record(record, dry_run=dry_run)
        yield row


def first_present(record: Mapping[str, Any], names: Iterable[str]) -> tuple[str | None, Any]:
    for name in names:
        value = record.get(name)
        if has_value(value):
            return name, value
    return None, ""


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score held-out rollout JSONL locally.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--require-gold",
        action="store_true",
        help="fail if any row lacks gold_focus/gold_focus_attr/reference_answer or score is missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write format/length scaffold rows without recomputing gold rewards",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    input_rows = [row for _line_no, row in load_jsonl(args.input_jsonl)]
    scored_rows = list(score_rows(input_rows, dry_run=args.dry_run))
    count = dump_jsonl(args.output_jsonl, scored_rows)
    summary = summarize(scored_rows, input_jsonl=args.input_jsonl, output_jsonl=args.output_jsonl)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    mode = "dry-run validated" if args.dry_run else "scored"
    print(f"{mode} {count} rows; wrote {args.output_jsonl}")
    if args.require_gold and (
        summary["score_count"] != summary["rows"]
        or any(status != "ok" for status in summary["status_counts"])
    ):
        print(json.dumps({"error": "require_gold_failed", **summary}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


def summarize(
    rows: Iterable[Mapping[str, Any]],
    *,
    input_jsonl: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    materialized = list(rows)
    status_counts: Counter[str] = Counter()
    output_contract_status_counts: Counter[str] = Counter()
    output_contract_issue_counts: Counter[str] = Counter()
    score_count = 0
    for row in materialized:
        heldout = row.get("heldout")
        if not isinstance(heldout, Mapping):
            status_counts["missing_heldout"] += 1
            continue
        status_counts[str(heldout.get("status") or "missing_status")] += 1
        contract = heldout.get("output_contract")
        if isinstance(contract, Mapping):
            output_contract_status_counts[str(contract.get("status") or "missing_status")] += 1
            issues = contract.get("issues")
            if isinstance(issues, list):
                for issue in issues:
                    output_contract_issue_counts[str(issue)] += 1
        if isinstance(heldout.get("score"), int | float) and not isinstance(heldout.get("score"), bool):
            score_count += 1
    return {
        "stage": "run_heldout",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "rows": len(materialized),
        "score_count": score_count,
        "status_counts": dict(sorted(status_counts.items())),
        "output_contract_status_counts": dict(sorted(output_contract_status_counts.items())),
        "output_contract_issue_counts": dict(sorted(output_contract_issue_counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
