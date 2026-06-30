"""Build W4 formal eval seed JSONL files.

This command normalizes already-downloaded benchmark/raw files into the single
rollout-seed contract consumed by ``anima.eval.generate_rollouts``. It performs
no downloads and no model calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from anima.data.build_reward_records import (
    first_nonempty,
    iter_json_objects,
    normalize_conversations,
    normalize_text_value,
)


DEFAULT_TRAIN_PATHS = (
    Path("/home/featurize/work/anima/data/w2_public_seed_dry_run/rolebench_zh_role_live_1000/synth.jsonl"),
    Path("/home/featurize/work/anima/data/dpo/rolebench_zh_role_dpo_239.jsonl"),
)
CHOICES = ("A", "B", "C", "D")
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet", ".csv", ".tsv"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-input", action="append", type=Path, default=[])
    parser.add_argument("--socialbench-input", action="append", type=Path, default=[])
    parser.add_argument("--ceval-input", action="append", type=Path, default=[])
    parser.add_argument("--train-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--max-heldout", type=int, default=40)
    parser.add_argument("--max-heldout-per-character", type=int, default=10)
    parser.add_argument("--max-socialbench", type=int, default=80)
    parser.add_argument("--max-ceval", type=int, default=80)
    parser.add_argument("--min-heldout", type=int, default=0)
    parser.add_argument("--min-socialbench", type=int, default=0)
    parser.add_argument("--min-ceval", type=int, default=0)
    parser.add_argument("--require-heldout-reference", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = load_seen(args.train_jsonl or list(DEFAULT_TRAIN_PATHS))
    summary: dict[str, Any] = {
        "stage": "build_w4_eval_slices",
        "output_dir": str(output_dir),
        "seen_characters": len(seen["characters"]),
        "seen_source_works": len(seen["source_works"]),
        "outputs": {},
        "claim_guardrail": (
            "Only rows with reportability_status=reportable may enter headline W4 tables; "
            "heldout rows are filtered against W2/W3 train/reward/DPO characters and source works."
        ),
    }

    if args.heldout_input:
        rows, axis_summary = build_heldout_rows(
            args.heldout_input,
            seen=seen,
            max_records=args.max_heldout,
            max_per_character=args.max_heldout_per_character,
            require_reference=args.require_heldout_reference,
        )
        path = output_dir / "heldout_roleplay.jsonl"
        write_jsonl(path, rows)
        summary["outputs"]["heldout"] = {**axis_summary, "path": str(path), "rows": len(rows)}
        require_minimum("heldout", rows, args.min_heldout)

    if args.socialbench_input:
        rows, axis_summary = build_mcq_rows(
            args.socialbench_input,
            axis="socialbench",
            id_prefix="w4_socialbench",
            max_records=args.max_socialbench,
        )
        path = output_dir / "socialbench_mcq.jsonl"
        write_jsonl(path, rows)
        summary["outputs"]["socialbench"] = {**axis_summary, "path": str(path), "rows": len(rows)}
        require_minimum("socialbench", rows, args.min_socialbench)

    if args.ceval_input:
        rows, axis_summary = build_mcq_rows(
            args.ceval_input,
            axis="ceval",
            id_prefix="w4_ceval",
            max_records=args.max_ceval,
        )
        path = output_dir / "ceval_mcq.jsonl"
        write_jsonl(path, rows)
        summary["outputs"]["ceval"] = {**axis_summary, "path": str(path), "rows": len(rows)}
        require_minimum("ceval", rows, args.min_ceval)

    (output_dir / "slice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_heldout_rows(
    inputs: Sequence[Path],
    *,
    seen: Mapping[str, set[str]],
    max_records: int,
    max_per_character: int,
    require_reference: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    input_files = collect_input_files(inputs)
    for raw_index, record in enumerate(iter_records(input_files)):
        normalized = normalize_heldout_record(record, raw_index)
        if normalized is None:
            skipped["unusable_heldout"] += 1
            continue
        character = normalized["character"]
        if max_per_character > 0 and character_counts[character] >= max_per_character:
            skipped["max_per_character"] += 1
            continue
        source_work = normalized["source_work"]
        leakage_reasons = leakage_reasons_for(character, source_work, seen)
        if leakage_reasons:
            skipped["leakage"] += 1
            continue
        if require_reference and not normalized.get("reference_answer"):
            skipped["missing_reference"] += 1
            continue
        normalized["reportability_status"] = "reportable"
        normalized["reportability_reasons"] = []
        rows.append(normalized)
        character_counts[character] += 1
        if len(rows) >= max_records:
            break
    return rows, {
        "input_files": [str(path) for path in input_files],
        "character_counts": dict(sorted(character_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "max_records": max_records,
        "max_per_character": max_per_character,
        "require_reference": require_reference,
    }


def normalize_heldout_record(record: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    raw = dict(record)
    character = first_nonempty(raw, ("character", "role", "character_name", "name"))
    if not character:
        return None

    source_label = infer_roleplay_source(raw)
    source_work = first_nonempty(raw, ("source_work", "work", "book", "movie"), f"{source_label}/{character}")
    conversations = normalize_conversations(raw)
    if not conversations:
        return None

    reference = first_nonempty(
        raw,
        ("reference_answer", "reference_response", "generated", "reference"),
    )
    row = {
        "id": f"w4_heldout_{index:06d}",
        "axis": "heldout",
        "source": str(raw.get("source") or source_label),
        "split": "eval_heldout",
        "character": character,
        "source_work": source_work,
        "raw_id": str(raw.get("id") or ""),
        "raw_path": str(raw.get("__raw_path") or ""),
        "profile": first_nonempty(
            raw,
            ("profile", "persona", "description", "character_profile", "system_prompt"),
            (
                f"{character}角色卡：来自{source_work}。请保持该角色身份、称谓、语气和价值观，"
                "用中文回应，不要以普通助手口吻回答。"
            ),
        ),
        "conversations": conversations,
    }
    if reference:
        row["reference_answer"] = reference
        row["gold_focus"] = raw.get("gold_focus") or ["Style", "Engagement"]
        row["gold_focus_attr"] = raw.get("gold_focus_attr") or f"保持{character}的身份、语气和互动推进感。"
    return row


def infer_roleplay_source(record: Mapping[str, Any]) -> str:
    path = str(record.get("__raw_path") or "")
    if "CharacterBench" in path:
        return "CharacterBench"
    if "CharacterEval" in path:
        return "CharacterEval"
    if "RoleBench" in path:
        return "RoleBench"
    return "heldout_roleplay"


def build_mcq_rows(
    inputs: Sequence[Path],
    *,
    axis: str,
    id_prefix: str,
    max_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    input_files = collect_input_files(inputs)
    for raw_index, record in enumerate(iter_records(input_files)):
        normalized = normalize_mcq_record(record, axis=axis, id_prefix=id_prefix, index=raw_index)
        if normalized is None:
            skipped["unusable_mcq"] += 1
            continue
        rows.append(normalized)
        if len(rows) >= max_records:
            break
    return rows, {
        "input_files": [str(path) for path in input_files],
        "skipped": dict(sorted(skipped.items())),
        "max_records": max_records,
    }


def normalize_mcq_record(
    record: Mapping[str, Any],
    *,
    axis: str,
    id_prefix: str,
    index: int,
) -> dict[str, Any] | None:
    raw = dict(record)
    question = first_nonempty(
        raw,
        ("question", "prompt", "query", "instruction", "input", "text", "sentence", "context"),
    )
    choices = extract_choices(raw)
    raw_answer = first_present_raw(raw, ("answer", "label", "gold", "target", "correct_answer", "answerKey"))
    answer = normalize_answer_label(raw_answer, choices, raw)
    if not question or len(choices) < 2 or not answer:
        return None
    policy_prompt = render_mcq_prompt(question, choices)
    row = {
        "id": str(raw.get("id") or raw.get("qid") or raw.get("question_id") or f"{id_prefix}_{index:06d}"),
        "axis": axis,
        "source": str(raw.get("source") or axis),
        "split": str(raw.get("split") or "eval"),
        "character": "通用答题者",
        "source_work": axis,
        "profile": "通用中文答题者。请按照题目要求回答，不要扮演角色。",
        "conversations": [{"role": "user", "content": policy_prompt}],
        "policy_prompt": policy_prompt,
        "question": question,
        "choices": choices,
        "answer": answer,
        "label": answer,
        "reportability_status": "reportable",
        "reportability_reasons": [],
    }
    raw_answer_text = normalize_text_value(raw_answer)
    if raw_answer_text and raw_answer_text != answer:
        row["raw_answer"] = raw_answer_text
    return row


def extract_choices(record: Mapping[str, Any]) -> dict[str, str]:
    direct: dict[str, str] = {}
    for letter in CHOICES:
        value = first_nonempty(
            dict(record),
            (letter, letter.lower(), f"option_{letter.lower()}", f"option{letter}", f"choice_{letter.lower()}"),
        )
        if value:
            direct[letter] = value
    if direct:
        return direct

    for key in ("choices", "options", "candidates"):
        value = record.get(key)
        if isinstance(value, Mapping):
            for letter in CHOICES:
                text = normalize_text_value(value.get(letter) or value.get(letter.lower()))
                if text:
                    direct[letter] = text
            if direct:
                return direct
        if isinstance(value, list | tuple):
            for letter, item in zip(CHOICES, value):
                text = normalize_text_value(item)
                if text:
                    direct[letter] = text
            if direct:
                return direct
    return {}


def first_present_raw(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record.get(key) not in (None, ""):
            return record.get(key)
    return None


def normalize_answer_label(value: Any, choices: Mapping[str, str], record: Mapping[str, Any]) -> str:
    text = normalize_text_value(value)
    if not text:
        return ""
    token = canonical_token(text)
    if token in CHOICES:
        return token
    if token in {"0", "1", "2", "3"} and has_sequence_choices(record):
        return CHOICES[int(token)]
    if token in {"1", "2", "3", "4"}:
        return CHOICES[int(token) - 1]

    normalized_text = canonical_answer_text(text)
    for letter, choice in choices.items():
        if canonical_answer_text(choice) == normalized_text:
            return letter
    return ""


def has_sequence_choices(record: Mapping[str, Any]) -> bool:
    for key in ("choices", "options", "candidates"):
        if isinstance(record.get(key), list | tuple):
            return True
    return False


def canonical_token(text: str) -> str:
    return text.strip().strip("'\"`“”‘’()[]{}（）【】.。,:：;；!！?？").upper()


def canonical_answer_text(text: str) -> str:
    return "".join(str(text).strip().split()).lower()


def render_mcq_prompt(question: str, choices: Mapping[str, str]) -> str:
    lines = ["请回答以下单项选择题。只输出一个大写字母 A/B/C/D，不要解释。", "", f"题目: {question}"]
    for letter in CHOICES:
        if letter in choices:
            lines.append(f"{letter}. {choices[letter]}")
    lines.append("答案:")
    return "\n".join(lines)


def load_seen(paths: Sequence[Path]) -> dict[str, set[str]]:
    seen = {"characters": set(), "source_works": set()}
    for path in collect_input_files(paths):
        if not path.exists():
            continue
        for record in iter_records([path]):
            character = first_nonempty(dict(record), ("character", "role", "name", "character_name"))
            source_work = first_nonempty(dict(record), ("source_work", "work", "book", "movie"))
            if character:
                seen["characters"].add(character)
            if source_work:
                seen["source_works"].add(source_work)
    return seen


def leakage_reasons_for(character: str, source_work: str, seen: Mapping[str, set[str]]) -> list[str]:
    reasons: list[str] = []
    if character in seen["characters"]:
        reasons.append("character_seen_in_train_reward_or_dpo")
    if source_work in seen["source_works"]:
        reasons.append("source_work_seen_in_train_reward_or_dpo")
    return reasons


def collect_input_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*") if child.suffix.lower() in SUPPORTED_SUFFIXES))
        elif path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def iter_records(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                yield from attach_raw_path(iter_json_recursive(path), path)
            elif suffix in {".jsonl", ".parquet"}:
                yield from attach_raw_path(iter_json_objects(path), path)
            elif suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    yield from attach_raw_path(csv.DictReader(handle, delimiter=delimiter), path)
        except Exception as exc:
            warning = {
                "stage": "build_w4_eval_slices",
                "warning": "skipped_unreadable_input_file",
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(json.dumps(warning, ensure_ascii=False, sort_keys=True), file=sys.stderr)


def attach_raw_path(records: Iterable[Mapping[str, Any]], path: Path) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        row = dict(record)
        row.setdefault("__raw_path", str(path))
        row.setdefault("__raw_index", index)
        yield row


def iter_json_recursive(path: Path) -> Iterator[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    yield from walk_json_records(value)


def walk_json_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from walk_json_records(item)
        return
    if not isinstance(value, Mapping):
        return
    if looks_like_record(value):
        yield dict(value)
        return
    for item in value.values():
        if isinstance(item, list | tuple | Mapping):
            yield from walk_json_records(item)


def looks_like_record(value: Mapping[str, Any]) -> bool:
    keys = set(value)
    mcq_keys = {"question", "prompt", "query", "instruction", "input", "answer", "label", "target"}
    role_keys = {"character", "role", "name", "profile", "persona", "conversations", "dialogue", "history"}
    character_bench_keys = {"character_name", "character_profile", "reference_response", "response_messages"}
    choice_keys = {"A", "B", "C", "D", "choices", "options", "candidates"}
    return (
        bool(keys & mcq_keys and keys & choice_keys)
        or bool(keys & role_keys and keys & mcq_keys)
        or bool({"character_name", "character_profile", "dialogue"} <= keys and keys & character_bench_keys)
    )


def require_minimum(axis: str, rows: Sequence[Mapping[str, Any]], minimum: int) -> None:
    if len(rows) < minimum:
        raise SystemExit(f"{axis} produced {len(rows)} rows, below required minimum {minimum}")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


if __name__ == "__main__":
    raise SystemExit(main())
