"""Deterministic SocialBench MCQ scorer. [W4]

This scorer is intentionally offline and model-free. It consumes rollout or
prediction JSONL, extracts a multiple-choice answer from each model output, and
writes the original rows with an added ``socialbench`` metric block.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


RESPONSE_FIELDS = ("response", "completion", "model_response", "prediction")
TARGET_FIELDS = ("answer", "label", "gold", "target", "correct_answer")
LETTERS = "ABCD"

CHOICE_CUE_RE = re.compile(
    r"""
    (?:final\s+answer|correct\s+answer|answer|option|choice|label|
       最终答案|正确答案|答案|选项|选择|我选|应选|选|答)
    \s*(?:is|as|=|:|：|-|为|是|应该是|应为)?
    \s*(?:option|choice|选项)?
    \s*[\(\[（【]?\s*([A-D1-4])\s*[\)\]）】]?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
LINE_START_CHOICE_RE = re.compile(
    r"(?m)^\s*[\(\[（【]?\s*([A-D1-4])\s*[\)\]）】\.、:：]?(?:\s|$)"
)
STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score and summarize rows without writing the output JSONL",
    )
    args = parser.parse_args(argv)

    rows = [score_record(row) for _, row in load_jsonl(args.input_jsonl)]
    summary = summarize(rows, input_jsonl=args.input_jsonl, output_jsonl=args.output_jsonl, dry_run=args.dry_run)

    if not args.dry_run:
        write_jsonl(args.output_jsonl, rows)

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    response_value = first_present(record, RESPONSE_FIELDS)
    target_value = first_present(record, TARGET_FIELDS)

    prediction = extract_choice(response_value)
    target = extract_choice(target_value)

    if target is None:
        status = "missing_target" if target_value is None else "unparseable_target"
        correct: bool | None = None
        accuracy: float | None = None
    elif response_value is None:
        status = "missing_prediction"
        correct = False
        accuracy = 0.0
    elif prediction is None:
        status = "unparseable_prediction"
        correct = False
        accuracy = 0.0
    else:
        status = "ok"
        correct = prediction == target
        accuracy = 1.0 if correct else 0.0

    row = dict(record)
    row["socialbench"] = {
        "prediction": prediction,
        "target": target,
        "correct": correct,
        "accuracy": accuracy,
        "status": status,
    }
    return row


def summarize(
    rows: Iterable[Mapping[str, Any]],
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    dry_run: bool,
) -> dict[str, Any]:
    materialized = list(rows)
    status_counts: Counter[str] = Counter()
    scored = 0
    correct = 0

    for row in materialized:
        socialbench = row.get("socialbench", {})
        if isinstance(socialbench, Mapping):
            status_counts[str(socialbench.get("status", "missing_status"))] += 1
            if socialbench.get("accuracy") is not None:
                scored += 1
            if socialbench.get("correct") is True:
                correct += 1

    accuracy = correct / scored if scored else None
    return {
        "stage": "run_socialbench",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "rows": len(materialized),
        "scored": scored,
        "correct": correct,
        "accuracy": accuracy,
        "status_counts": dict(sorted(status_counts.items())),
        "dry_run": dry_run,
    }


def first_present(record: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def extract_choice(value: Any) -> str | None:
    """Extract a canonical A/B/C/D option from free-form output or labels."""

    text = value_to_text(value)
    if not text:
        return None

    normalized = unicodedata.normalize("NFKC", text).strip()
    exact = choice_from_token(strip_wrappers(normalized))
    if exact is not None:
        return exact

    cued_matches = [
        (match.start(), choice)
        for match in CHOICE_CUE_RE.finditer(normalized)
        if (choice := choice_from_token(match.group(1))) is not None
    ]
    if cued_matches:
        return sorted(cued_matches)[-1][1]

    line_start_matches = [
        (match.start(), choice)
        for match in LINE_START_CHOICE_RE.finditer(normalized)
        if (choice := choice_from_token(match.group(1))) is not None
    ]
    if line_start_matches:
        return sorted(line_start_matches)[-1][1]

    letter_matches = [
        (match.start(), choice)
        for match in STANDALONE_LETTER_RE.finditer(normalized)
        if (choice := choice_from_token(match.group(1))) is not None
    ]
    if letter_matches:
        return sorted(letter_matches)[-1][1]

    return None


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in (
            "prediction",
            "answer",
            "label",
            "content",
            "text",
            "message",
            "completion",
            "response",
        ):
            nested = value.get(key)
            if nested not in (None, ""):
                return value_to_text(nested)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list | tuple):
        return " ".join(part for item in value if (part := value_to_text(item)))
    return str(value)


def strip_wrappers(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^[\s'\"`“”‘’\(\[\{（【]+", "", stripped)
    stripped = re.sub(r"[\s'\"`“”‘’\)\]\}）】\.。,:：;；!！?？]+$", "", stripped)
    return stripped.strip()


def choice_from_token(token: Any) -> str | None:
    normalized = strip_wrappers(unicodedata.normalize("NFKC", str(token))).upper()
    if normalized in LETTERS:
        return normalized
    if normalized in {"1", "2", "3", "4"}:
        return LETTERS[int(normalized) - 1]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
