"""Reward-record schema and JSONL helpers.

This module implements the frozen W2 reward-record contract from
docs/REWARD_DATA_GUIDE_march7th.md. It intentionally uses only the standard
library so validation can run before the training stack is installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

FOCUS_LABELS: tuple[str, ...] = (
    "Knowledge",
    "Style",
    "Worldview",
    "Emotion",
    "Empathetic",
    "Engagement",
    "Human_Like",
    "Extension",
    "Memory",
    "Safety",
)

VALID_SPLITS: tuple[str, ...] = ("sft", "reward", "eval_heldout")

VALID_SOURCES: tuple[str, ...] = (
    "CharacterBench",
    "RoleBench",
    "DeepSeek-synth",
    "HSR-canon",
    "HSR-synth",
)

VALID_ROLES: tuple[str, ...] = ("user", "assistant")

SYNTHETIC_SOURCES: frozenset[str] = frozenset({"DeepSeek-synth", "HSR-synth"})

FOCUS_LABEL_SET: frozenset[str] = frozenset(FOCUS_LABELS)
VALID_SPLIT_SET: frozenset[str] = frozenset(VALID_SPLITS)
VALID_SOURCE_SET: frozenset[str] = frozenset(VALID_SOURCES)
VALID_ROLE_SET: frozenset[str] = frozenset(VALID_ROLES)


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant turn in the policy-visible dialogue context."""

    role: str
    content: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConversationTurn":
        return cls(role=_coerce_str(value.get("role")), content=_coerce_str(value.get("content")))


@dataclass(frozen=True)
class SynthMeta:
    """Reproducibility metadata for synthesized gold or DPO fields."""

    generator: str
    prompt_id: str
    lore_sources: tuple[str, ...] = field(default_factory=tuple)
    rejected_strategy: str | None = None
    human_reviewed: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SynthMeta":
        lore_sources = value.get("lore_sources", ())
        if not isinstance(lore_sources, list | tuple):
            lore_sources = ()
        return cls(
            generator=_coerce_str(value.get("generator")),
            prompt_id=_coerce_str(value.get("prompt_id")),
            lore_sources=tuple(str(item) for item in lore_sources),
            rejected_strategy=_optional_str(value.get("rejected_strategy")),
            human_reviewed=value.get("human_reviewed", False),
        )


@dataclass(frozen=True)
class RewardRecord:
    """A single JSONL reward/DPO record.

    Gold fields are reward-only targets and must never be inserted into the
    policy prompt.
    """

    id: str
    character: str
    source_work: str
    character_cluster: int
    profile: str
    conversations: tuple[ConversationTurn, ...]
    gold_focus: tuple[str, ...]
    gold_focus_attr: str
    reference_answer: str
    rejected_answer: str | None
    source: str
    split: str
    synth_meta: SynthMeta | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RewardRecord":
        conversations = value.get("conversations", ())
        if not isinstance(conversations, list | tuple):
            conversations = ()

        gold_focus = value.get("gold_focus", ())
        if not isinstance(gold_focus, list | tuple):
            gold_focus = ()

        synth_meta = value.get("synth_meta")
        parsed_synth_meta = None
        if isinstance(synth_meta, Mapping):
            parsed_synth_meta = SynthMeta.from_mapping(synth_meta)

        return cls(
            id=_coerce_str(value.get("id")),
            character=_coerce_str(value.get("character")),
            source_work=_coerce_str(value.get("source_work")),
            character_cluster=_coerce_int(value.get("character_cluster")),
            profile=_coerce_str(value.get("profile")),
            conversations=tuple(ConversationTurn.from_mapping(turn) for turn in conversations),
            gold_focus=tuple(str(label) for label in gold_focus),
            gold_focus_attr=_coerce_str(value.get("gold_focus_attr")),
            reference_answer=_coerce_str(value.get("reference_answer")),
            rejected_answer=_optional_str(value.get("rejected_answer")),
            source=_coerce_str(value.get("source")),
            split=_coerce_str(value.get("split")),
            synth_meta=parsed_synth_meta,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["conversations"] = [asdict(turn) for turn in self.conversations]
        data["gold_focus"] = list(self.gold_focus)
        if self.synth_meta is not None:
            data["synth_meta"] = asdict(self.synth_meta)
            data["synth_meta"]["lore_sources"] = list(self.synth_meta.lore_sources)
        else:
            data["synth_meta"] = None
        return data


class SchemaValidationError(ValueError):
    """Raised when a reward record violates the frozen schema."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_record(record: RewardRecord | Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for one reward record.

    The helper checks the W2 frozen schema only. Rich leakage, dedup, and
    parse-round-trip checks belong in src/anima/data/validators.py.
    """

    if not isinstance(record, RewardRecord):
        record = RewardRecord.from_mapping(record)

    errors: list[str] = []

    _require_text(errors, record.id, "id")
    _require_text(errors, record.character, "character")
    _require_text(errors, record.source_work, "source_work")
    _require_text(errors, record.profile, "profile")
    _require_text(errors, record.gold_focus_attr, "gold_focus_attr")
    _require_text(errors, record.reference_answer, "reference_answer")

    if record.character_cluster < -1:
        errors.append("character_cluster must be -1 or a non-negative integer")

    if not record.conversations:
        errors.append("conversations must contain at least one turn")
    for index, turn in enumerate(record.conversations):
        if turn.role not in VALID_ROLE_SET:
            errors.append(f"conversations[{index}].role must be one of {VALID_ROLES}")
        if not turn.content.strip():
            errors.append(f"conversations[{index}].content must be non-empty")

    if not record.gold_focus:
        errors.append("gold_focus must contain at least one label")
    illegal_labels = [label for label in record.gold_focus if label not in FOCUS_LABEL_SET]
    if illegal_labels:
        errors.append(f"gold_focus contains illegal labels: {illegal_labels}")
    if len(set(record.gold_focus)) != len(record.gold_focus):
        errors.append("gold_focus must not contain duplicate labels")

    if record.source not in VALID_SOURCE_SET:
        errors.append(f"source must be one of {VALID_SOURCES}")
    if record.split not in VALID_SPLIT_SET:
        errors.append(f"split must be one of {VALID_SPLITS}")

    if record.source in SYNTHETIC_SOURCES and record.synth_meta is None:
        errors.append("synth_meta is required for synthetic sources")

    if record.synth_meta is not None:
        _require_text(errors, record.synth_meta.generator, "synth_meta.generator")
        _require_text(errors, record.synth_meta.prompt_id, "synth_meta.prompt_id")
        if not isinstance(record.synth_meta.human_reviewed, bool):
            errors.append("synth_meta.human_reviewed must be a boolean")

    if record.rejected_answer is not None:
        if not record.rejected_answer.strip():
            errors.append("rejected_answer must be non-empty when present")
        if record.rejected_answer == record.reference_answer:
            errors.append("rejected_answer must differ from reference_answer")
        if record.synth_meta is None:
            errors.append("synth_meta is required when rejected_answer is present")
        elif not record.synth_meta.rejected_strategy:
            errors.append("synth_meta.rejected_strategy is required when rejected_answer is present")

    return errors


def require_valid_record(record: RewardRecord | Mapping[str, Any]) -> RewardRecord:
    """Return a RewardRecord or raise SchemaValidationError."""

    parsed = record if isinstance(record, RewardRecord) else RewardRecord.from_mapping(record)
    errors = validate_record(parsed)
    if errors:
        raise SchemaValidationError(errors)
    return parsed


def load_jsonl(path: str | Path, *, validate: bool = True) -> list[RewardRecord]:
    """Load reward records from JSONL."""

    records: list[RewardRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError([f"line {line_number}: invalid JSON: {exc.msg}"]) from exc
            if not isinstance(payload, Mapping):
                raise SchemaValidationError([f"line {line_number}: record must be a JSON object"])
            record = RewardRecord.from_mapping(payload)
            if validate:
                errors = validate_record(record)
                if errors:
                    raise SchemaValidationError([f"line {line_number}: {error}" for error in errors])
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[RewardRecord | Mapping[str, Any]]) -> None:
    """Write reward records to JSONL after schema validation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            parsed = require_valid_record(record)
            handle.write(json.dumps(parsed.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _require_text(errors: list[str], value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name} must be a non-empty string")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -2
