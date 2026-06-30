"""C-Eval/MCQ regression scorer. [W4]

Reads rollout/prediction JSONL rows, extracts an A/B/C/D choice from the model
response, compares it with the row's gold choice, and appends a ``ceval`` metric
object. This scorer is intentionally offline and deterministic: no model loads,
network calls, or benchmark downloads happen here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


MISSING = object()
CHOICES = ("A", "B", "C", "D")
DIGIT_TO_CHOICE = {"1": "A", "2": "B", "3": "C", "4": "D"}
RESPONSE_FIELDS = (
    "response",
    "prediction",
    "completion",
    "completion_raw",
    "model_response",
    "output",
    "generated",
)
TARGET_FIELDS = ("answer", "label", "gold", "target", "correct_answer")

CUE_PATTERN = re.compile(
    r"(?:"
    r"正确答案|最终答案|答案|答|选择|选项|我选|应选|"
    r"FINAL\s+ANSWER|ANSWER|OPTION|CHOICE"
    r")\s*(?:是|为|应为|:|：|-|=)?\s*[\(\[\{【]?\s*([A-D1-4])",
    re.IGNORECASE,
)
BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D1-4])\s*\}", re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([A-D1-4])\s*</answer>", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"(?<![A-Z0-9])([A-D1-4])(?![A-Z0-9])", re.IGNORECASE)


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            yield line_no, row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    response_raw = first_present(row, RESPONSE_FIELDS)
    target_raw = first_present(row, TARGET_FIELDS)
    prediction = extract_choice(response_raw)
    target = normalize_choice(target_raw)

    if target is None:
        status = "missing_target" if is_missing(target_raw) else "unparseable_target"
        correct: bool | None = None
        accuracy: float | None = None
    elif prediction is None:
        status = "missing_prediction" if is_missing(response_raw) else "unparseable_prediction"
        correct = False
        accuracy = 0.0
    else:
        correct = prediction == target
        accuracy = 1.0 if correct else 0.0
        status = "ok"

    out["ceval"] = {
        "prediction": prediction,
        "target": target,
        "correct": correct,
        "accuracy": accuracy,
        "status": status,
    }
    return out


def first_present(record: Mapping[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = get_path(record, field)
        if value is MISSING or value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return MISSING


def get_path(record: Mapping[str, Any], path: str) -> Any:
    if path in record:
        return record[path]

    current: Any = record
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return MISSING
    return current


def extract_choice(value: Any) -> str | None:
    if is_missing(value):
        return None
    return choice_from_text(str(value))


def normalize_choice(value: Any) -> str | None:
    if is_missing(value) or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if float(value).is_integer():
            return DIGIT_TO_CHOICE.get(str(int(value)))
        return None
    if isinstance(value, list | tuple):
        for item in value:
            choice = normalize_choice(item)
            if choice is not None:
                return choice
        return None
    if isinstance(value, Mapping):
        nested = first_present(value, TARGET_FIELDS)
        return normalize_choice(nested)
    return choice_from_text(str(value))


def choice_from_text(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        return None

    direct = token_to_choice(text)
    if direct is not None:
        return direct

    for pattern in (BOXED_PATTERN, ANSWER_TAG_PATTERN, CUE_PATTERN, TOKEN_PATTERN):
        matches = list(pattern.finditer(text))
        if matches:
            return token_to_choice(matches[-1].group(1))
    return None


def token_to_choice(value: str) -> str | None:
    token = unicodedata.normalize("NFKC", value).strip().upper()
    token = token.strip(" \t\r\n.。,:：;；!！?？()[]{}<>\"'`“”‘’【】")
    if token in CHOICES:
        return token
    return DIGIT_TO_CHOICE.get(token)


def is_missing(value: Any) -> bool:
    return value is MISSING


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    scored = 0
    correct = 0
    for row in rows:
        ceval = row.get("ceval")
        if not isinstance(ceval, Mapping):
            continue
        status_counts[str(ceval.get("status") or "missing_status")] += 1
        accuracy = ceval.get("accuracy")
        if isinstance(accuracy, int | float) and not isinstance(accuracy, bool):
            scored += 1
            if ceval.get("correct") is True:
                correct += 1

    return {
        "rows": len(rows),
        "scored": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else None,
        "status_counts": dict(sorted(status_counts.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score C-Eval/MCQ rollout JSONL accuracy.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="score rows and print summary without writing output")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    rows = [score_row(row) for _, row in load_jsonl(args.input_jsonl)]
    written = 0 if args.dry_run else write_jsonl(args.output_jsonl, rows)
    summary = {
        "stage": "run_ceval",
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "dry_run": bool(args.dry_run),
        "written": written,
        **summarize(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
