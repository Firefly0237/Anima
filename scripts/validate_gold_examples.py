"""Validate the W2 first-20 human gold examples.

This script is intentionally local and rule based. It validates the formal
human-authored gate file(s), and it should fail on the placeholder template.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anima.data.validators import (  # noqa: E402
    HSR_LOCAL_ONLY_CHARACTERS,
    VALID_SOURCES,
    VALID_SPLITS,
    validate_records,
)


DEFAULT_INPUT = REPO_ROOT / "tests" / "fixtures" / "gold_examples_template.jsonl"
DEFAULT_MIN_RECORDS = 20
HSR_SOURCES = {"HSR-canon", "HSR-synth"}
PLACEHOLDER_MARKERS = (
    "PLACEHOLDER",
    "DO_NOT_USE",
    "TEMPLATE",
    "TODO",
    "FILL_ME",
    "NOT_A_GOLD_RECORD",
    "占位",
    "待填写",
)


@dataclass(frozen=True)
class Location:
    path: Path
    line: int

    def render(self) -> str:
        try:
            rel = self.path.resolve().relative_to(REPO_ROOT)
            path_text = str(rel)
        except ValueError:
            path_text = str(self.path)
        return f"{path_text}:{self.line}"


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str
    location: Location | None = None
    record_id: str | None = None
    field: str | None = None

    def render(self) -> str:
        prefix_parts: list[str] = []
        if self.location is not None:
            prefix_parts.append(self.location.render())
        if self.record_id:
            prefix_parts.append(self.record_id)
        if self.field:
            prefix_parts.append(self.field)
        prefix = " | ".join(prefix_parts)
        if prefix:
            return f"- {prefix}: [{self.code}] {self.message}"
        return f"- [{self.code}] {self.message}"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate W2 first-20 human gold example JSONL files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "JSONL file(s) to validate. Defaults to the placeholder template, "
            "which should fail the gate."
        ),
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=DEFAULT_MIN_RECORDS,
        help=f"Minimum records required for the first-20 gate. Default: {DEFAULT_MIN_RECORDS}.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    paths = args.paths or [DEFAULT_INPUT]

    records, locations, load_issues = load_jsonl_files(paths)
    issues: list[GateIssue] = list(load_issues)

    if records:
        issues.extend(validate_with_project_rules(records, locations))
        issues.extend(validate_first20_gate(records, locations, paths, min_records=args.min_records))
    else:
        issues.append(GateIssue("gate.no_records", "No JSONL records were loaded."))

    print_report(records, issues, min_records=args.min_records)
    return 1 if issues else 0


def load_jsonl_files(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[Location], list[GateIssue]]:
    records: list[dict[str, Any]] = []
    locations: list[Location] = []
    issues: list[GateIssue] = []

    for raw_path in paths:
        path = resolve_path(raw_path)
        if not path.exists():
            issues.append(GateIssue("io.missing", f"File does not exist: {path}"))
            continue
        if not path.is_file():
            issues.append(GateIssue("io.not_file", f"Path is not a file: {path}"))
            continue

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                location = Location(path=path, line=line_number)
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    issues.append(
                        GateIssue(
                            "json.invalid",
                            f"Invalid JSON: {exc.msg}",
                            location=location,
                        )
                    )
                    continue
                if not isinstance(value, dict):
                    issues.append(
                        GateIssue(
                            "json.not_object",
                            "Each JSONL line must be a JSON object.",
                            location=location,
                        )
                    )
                    continue
                records.append(value)
                locations.append(location)

    return records, locations, issues


def validate_with_project_rules(
    records: list[dict[str, Any]],
    locations: list[Location],
) -> list[GateIssue]:
    report = validate_records(
        records,
        split_config=None,
        check_within_duplicates=True,
        check_cross_split_duplicates=True,
    )
    issues: list[GateIssue] = []

    for result, location in zip(report.results, locations, strict=True):
        for issue in result.issues:
            issues.append(
                GateIssue(
                    code=issue.code,
                    message=issue.message,
                    location=location,
                    record_id=issue.record_id,
                    field=issue.field,
                )
            )

    for issue in report.batch_issues:
        issues.append(
            GateIssue(
                code=issue.code,
                message=issue.message,
                record_id=issue.record_id,
                field=issue.field,
            )
        )

    return issues


def validate_first20_gate(
    records: list[dict[str, Any]],
    locations: list[Location],
    paths: Iterable[Path],
    *,
    min_records: int,
) -> list[GateIssue]:
    issues: list[GateIssue] = []

    if len(records) < min_records:
        issues.append(
            GateIssue(
                "gate.too_few_records",
                f"Loaded {len(records)} record(s); first-20 gate requires at least {min_records}.",
            )
        )

    for raw_path in paths:
        path = resolve_path(raw_path)
        if "template" in path.name.lower():
            issues.append(
                GateIssue(
                    "gate.template_input",
                    "Template files cannot pass the formal first-20 gate. Copy to a final gold file first.",
                )
            )

    for record, location in zip(records, locations, strict=True):
        record_id = str(record.get("id", "<missing-id>"))
        if not is_human_reviewed(record):
            issues.append(
                GateIssue(
                    "gate.human_reviewed_false",
                    "Set synth_meta.human_reviewed=true only after human review of this row.",
                    location=location,
                    record_id=record_id,
                    field="synth_meta.human_reviewed",
                )
            )

        placeholder_fields = find_placeholder_fields(record)
        for field_name in placeholder_fields:
            issues.append(
                GateIssue(
                    "gate.placeholder_marker",
                    "Placeholder/template text is still present.",
                    location=location,
                    record_id=record_id,
                    field=field_name,
                )
            )

        issues.extend(validate_source_split_local_only(record, location))

    return issues


def validate_source_split_local_only(record: Mapping[str, Any], location: Location) -> list[GateIssue]:
    issues: list[GateIssue] = []
    record_id = str(record.get("id", "<missing-id>"))
    source = record.get("source")
    split = record.get("split")

    if not isinstance(source, str) or source not in VALID_SOURCES:
        issues.append(
            GateIssue(
                "gate.source_invalid",
                f"source must be one of {sorted(VALID_SOURCES)}.",
                location=location,
                record_id=record_id,
                field="source",
            )
        )
    if not isinstance(split, str) or split not in VALID_SPLITS:
        issues.append(
            GateIssue(
                "gate.split_invalid",
                f"split must be one of {list(VALID_SPLITS)}.",
                location=location,
                record_id=record_id,
                field="split",
            )
        )

    if is_hsr_record(record):
        if not isinstance(source, str) or source not in HSR_SOURCES:
            issues.append(
                GateIssue(
                    "local_only.hsr_source",
                    "HSR / 三月七 rows must use source HSR-canon or HSR-synth.",
                    location=location,
                    record_id=record_id,
                    field="source",
                )
            )
        if split != "eval_heldout":
            issues.append(
                GateIssue(
                    "local_only.hsr_split",
                    "HSR / 三月七 rows must be eval_heldout only.",
                    location=location,
                    record_id=record_id,
                    field="split",
                )
            )
        if not str(record.get("id", "")).startswith("hsr_"):
            issues.append(
                GateIssue(
                    "local_only.hsr_id_prefix",
                    "HSR / 三月七 row ids must start with hsr_.",
                    location=location,
                    record_id=record_id,
                    field="id",
                )
            )
        source_work = str(record.get("source_work", ""))
        if "local-only" not in source_work.casefold() and "local_only" not in source_work.casefold():
            issues.append(
                GateIssue(
                    "local_only.source_work_marker",
                    "HSR / 三月七 source_work must visibly include local-only.",
                    location=location,
                    record_id=record_id,
                    field="source_work",
                )
            )
        if not is_under(location.path, REPO_ROOT / "work" / "data" / "hsr"):
            issues.append(
                GateIssue(
                    "local_only.path",
                    "Real HSR rows must be stored under work/data/hsr/. Repo fixtures may contain placeholders only.",
                    location=location,
                    record_id=record_id,
                )
            )

    return issues


def is_human_reviewed(record: Mapping[str, Any]) -> bool:
    synth_meta = record.get("synth_meta")
    if isinstance(synth_meta, Mapping) and synth_meta.get("human_reviewed") is True:
        return True
    return record.get("human_reviewed") is True


def find_placeholder_fields(record: Mapping[str, Any]) -> list[str]:
    fields: list[str] = []
    for field_name, value in iter_string_fields(record):
        upper_value = value.upper()
        if any(marker.upper() in upper_value for marker in PLACEHOLDER_MARKERS):
            fields.append(field_name)
    return fields


def iter_string_fields(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix or "<record>", value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_string_fields(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from iter_string_fields(child, child_prefix)


def is_hsr_record(record: Mapping[str, Any]) -> bool:
    character = str(record.get("character", ""))
    source = record.get("source")
    record_id = str(record.get("id", ""))
    source_work = str(record.get("source_work", ""))
    return (
        character in HSR_LOCAL_ONLY_CHARACTERS
        or (isinstance(source, str) and source in HSR_SOURCES)
        or record_id.startswith("hsr_")
        or "hsr" in source_work.casefold()
        or "星穹铁道" in source_work
    )


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def print_report(records: list[dict[str, Any]], issues: list[GateIssue], *, min_records: int) -> None:
    status = "PASS" if not issues else "FAIL"
    reviewed_count = sum(1 for record in records if is_human_reviewed(record))
    print(f"W2 first-20 gold gate: {status}")
    print(f"Records loaded: {len(records)} / required >= {min_records}")
    print(f"human_reviewed=true: {reviewed_count} / {len(records)}")

    if issues:
        print("\nBlocking issues:")
        for issue in issues:
            print(issue.render())
        print(
            "\nCurrent status: 未达 gate。"
            " 如果你正在检查模板，这是预期结果：模板只有 2-3 条占位记录，且 human_reviewed=false。"
        )
    else:
        print("\nGate passed mechanically. Human judgment remains the source of truth.")


if __name__ == "__main__":
    raise SystemExit(main())
