"""Build and check deterministic anti-leakage character splits.

The split unit is source_work wherever it is available: every character from the
same originating work is assigned to the same split. This keeps reported eval
free from reward/SFT works, not merely reward/SFT character names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from anima.data.validators import (
    HSR_LOCAL_ONLY_CHARACTERS,
    VALID_SPLITS,
    ValidationIssue,
    check_split_invariants,
    load_jsonl,
)


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("character_split.json")


@dataclass(frozen=True)
class SplitRatios:
    sft: float = 0.6
    reward: float = 0.2
    eval_heldout: float = 0.2

    def normalized(self) -> dict[str, float]:
        values = {"sft": self.sft, "reward": self.reward, "eval_heldout": self.eval_heldout}
        if any(value < 0 for value in values.values()):
            raise ValueError("split ratios must be non-negative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("at least one split ratio must be positive")
        return {split: value / total for split, value in values.items()}


def build_character_split(
    records: Sequence[Mapping[str, Any]],
    *,
    ratios: SplitRatios | None = None,
    respect_existing_split: bool = True,
    eval_only_characters: set[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic split JSON object from reward-data records."""

    record_list = list(records)
    eval_only = set(HSR_LOCAL_ONLY_CHARACTERS)
    if eval_only_characters:
        eval_only.update(eval_only_characters)

    split_to_characters: dict[str, set[str]] = {split: set() for split in VALID_SPLITS}
    source_work_by_character: dict[str, str] = {}

    if respect_existing_split and record_list and all(record.get("split") in VALID_SPLITS for record in record_list):
        for record in record_list:
            split = str(record["split"])
            character = str(record.get("character", "")).strip()
            source_work = str(record.get("source_work", "")).strip()
            if not character:
                continue
            if character in eval_only:
                split = "eval_heldout"
            split_to_characters[split].add(character)
            if source_work:
                source_work_by_character[character] = source_work
    else:
        groups = _group_by_source_work(record_list)
        reserved_groups: list[_SourceWorkGroup] = []
        assignable_groups: list[_SourceWorkGroup] = []
        for group in groups:
            if group.characters & eval_only or _looks_hsr_group(group):
                reserved_groups.append(group)
            else:
                assignable_groups.append(group)

        for group in reserved_groups:
            _add_group(split_to_characters, source_work_by_character, "eval_heldout", group)

        quotas = _split_quotas(
            len(assignable_groups),
            (ratios or SplitRatios()).normalized(),
        )
        ordered_groups = sorted(assignable_groups, key=lambda group: (_stable_hash(group.key), group.key))
        cursor = 0
        for split in VALID_SPLITS:
            for group in ordered_groups[cursor : cursor + quotas[split]]:
                _add_group(split_to_characters, source_work_by_character, split, group)
            cursor += quotas[split]

    split_config = {
        "sft": sorted(split_to_characters["sft"]),
        "reward": sorted(split_to_characters["reward"]),
        "eval_heldout": sorted(split_to_characters["eval_heldout"]),
        "_split_unit": "source_work",
        "_source_work_by_character": dict(sorted(source_work_by_character.items())),
        "_invariants": {
            "disjoint": [
                "sft∩reward=∅",
                "sft∩eval_heldout=∅",
                "reward∩eval_heldout=∅",
                "source_work is disjoint across all splits",
            ],
            "cross_split_dedup": (
                "Exact/normalized near-dup between reward and eval_heldout "
                "must report ZERO surviving collisions"
            ),
            "note": (
                "Split is by source_work where possible. "
                "三月七 is eval_heldout/local-only if present."
            ),
        },
    }
    return split_config


def check_character_split(
    split_config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Return invariant violations for a split JSON object."""

    return check_split_invariants(split_config, records)


def write_character_split(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    ratios: SplitRatios | None = None,
    respect_existing_split: bool = True,
) -> dict[str, Any]:
    split_config = build_character_split(
        records,
        ratios=ratios,
        respect_existing_split=respect_existing_split,
    )
    issues = check_character_split(split_config, records)
    if issues:
        rendered = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"split invariants failed: {rendered}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(split_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="validated reward/DPO JSONL records")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output split JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--sft", type=float, default=0.6, help="SFT split ratio")
    parser.add_argument("--reward", type=float, default=0.2, help="reward split ratio")
    parser.add_argument("--eval", type=float, default=0.2, help="eval heldout split ratio")
    parser.add_argument(
        "--ignore-existing-split",
        action="store_true",
        help="auto-assign source_work groups even if records already contain split fields",
    )
    args = parser.parse_args(argv)

    records = load_jsonl(args.input_jsonl)
    write_character_split(
        records,
        args.output,
        ratios=SplitRatios(sft=args.sft, reward=args.reward, eval_heldout=args.eval),
        respect_existing_split=not args.ignore_existing_split,
    )
    return 0


@dataclass(frozen=True)
class _SourceWorkGroup:
    key: str
    source_work: str
    characters: frozenset[str]
    sources: frozenset[str]


def _group_by_source_work(records: Sequence[Mapping[str, Any]]) -> list[_SourceWorkGroup]:
    grouped: dict[str, dict[str, set[str]]] = {}
    for record in records:
        character = str(record.get("character", "")).strip()
        if not character:
            continue
        source_work = str(record.get("source_work") or character).strip()
        source = str(record.get("source", "")).strip()
        bucket = grouped.setdefault(
            source_work,
            {"characters": set(), "sources": set()},
        )
        bucket["characters"].add(character)
        if source:
            bucket["sources"].add(source)

    return [
        _SourceWorkGroup(
            key=source_work,
            source_work=source_work,
            characters=frozenset(values["characters"]),
            sources=frozenset(values["sources"]),
        )
        for source_work, values in grouped.items()
    ]


def _split_quotas(group_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if group_count <= 0:
        return {split: 0 for split in VALID_SPLITS}

    raw = {split: group_count * ratios.get(split, 0.0) for split in VALID_SPLITS}
    quotas = {split: math.floor(value) for split, value in raw.items()}
    remaining = group_count - sum(quotas.values())
    by_remainder = sorted(
        VALID_SPLITS,
        key=lambda split: (raw[split] - quotas[split], ratios.get(split, 0.0), split),
        reverse=True,
    )
    for split in by_remainder[:remaining]:
        quotas[split] += 1
    return quotas


def _add_group(
    split_to_characters: dict[str, set[str]],
    source_work_by_character: dict[str, str],
    split: str,
    group: _SourceWorkGroup,
) -> None:
    split_to_characters[split].update(group.characters)
    for character in group.characters:
        source_work_by_character[character] = group.source_work


def _looks_hsr_group(group: _SourceWorkGroup) -> bool:
    return any(source.startswith("HSR-") for source in group.sources) or "HSR" in group.source_work


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


if __name__ == "__main__":
    raise SystemExit(main())
