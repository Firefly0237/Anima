"""Reward-record validation, leakage checks, and lightweight dedup helpers.

The W2 data gate is deliberately rule based: no embeddings, no remote calls, and
no silent coercion. A record either satisfies the frozen reward-data contract or
returns concrete issues that the data builder can log and reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from anima.data.schemas import (
    FOCUS_LABELS as SCHEMA_FOCUS_LABELS,
    VALID_ROLES as SCHEMA_VALID_ROLES,
    VALID_SOURCES as SCHEMA_VALID_SOURCES,
    VALID_SPLITS as SCHEMA_VALID_SPLITS,
)

FOCUS_LABELS: frozenset[str] = frozenset(SCHEMA_FOCUS_LABELS)
VALID_SPLITS: tuple[str, ...] = SCHEMA_VALID_SPLITS
VALID_SOURCES: frozenset[str] = frozenset(SCHEMA_VALID_SOURCES)
VALID_ROLES: frozenset[str] = frozenset(SCHEMA_VALID_ROLES)
SYNTHETIC_SOURCES: frozenset[str] = frozenset({"DeepSeek-synth", "HSR-synth"})
HSR_LOCAL_ONLY_CHARACTERS: frozenset[str] = frozenset({"三月七", "March 7th"})

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "character",
    "source_work",
    "character_cluster",
    "profile",
    "conversations",
    "gold_focus",
    "gold_focus_attr",
    "reference_answer",
    "rejected_answer",
    "source",
    "split",
    "synth_meta",
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_FOCUS_RE = re.compile(r"<focus>\s*(?P<value>.*?)\s*</focus>", re.IGNORECASE | re.DOTALL)
_FOCUS_ATTR_RE = re.compile(
    r"<focus_attr>\s*(?P<value>.*?)\s*</focus_attr>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{(?P<value>.*?)\}", re.DOTALL)
_FOCUS_SPLIT_RE = re.compile(r"[,，、;/；\s]+")


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable validation failure."""

    code: str
    field: str
    message: str
    record_id: str | None = None


@dataclass(frozen=True)
class RecordValidationResult:
    """Validation result for one reward/DPO record."""

    record_id: str | None
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class DuplicatePair:
    """A lightweight exact/normalized near-duplicate match."""

    left_id: str
    right_id: str
    similarity: float
    reason: str
    left_split: str | None = None
    right_split: str | None = None


@dataclass(frozen=True)
class BatchValidationReport:
    """Batch validation output, including leakage and duplicate reports."""

    results: tuple[RecordValidationResult, ...]
    batch_issues: tuple[ValidationIssue, ...] = ()
    near_duplicates: tuple[DuplicatePair, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            all(result.ok for result in self.results)
            and not self.batch_issues
            and not self.near_duplicates
        )

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        return tuple(result.record_id or "" for result in self.results if result.ok)

    @property
    def rejected_ids(self) -> tuple[str, ...]:
        return tuple(result.record_id or "" for result in self.results if not result.ok)


@dataclass(frozen=True)
class ParsedCompletion:
    """Minimal parser output used for validator round-trip checks."""

    focus: tuple[str, ...]
    focus_attr: str | None
    answer: str | None
    illegal_focus: tuple[str, ...] = ()
    well_formed: bool = False


def normalize_text(text: Any) -> str:
    """Normalize text for exact and near-duplicate checks."""

    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    kept: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in {"P", "S", "Z", "C"}:
            continue
        kept.append(char)
    return "".join(kept)


def has_cjk(text: Any) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def record_text(record: Mapping[str, Any]) -> str:
    """Return the prompt+answer surface used for duplicate checks."""

    parts: list[str] = []
    for key in ("profile", "gold_focus_attr", "reference_answer"):
        value = record.get(key)
        if isinstance(value, str):
            parts.append(value)

    conversations = record.get("conversations")
    if isinstance(conversations, Sequence) and not isinstance(conversations, (str, bytes)):
        for turn in conversations:
            if isinstance(turn, Mapping):
                parts.append(str(turn.get("role", "")))
                parts.append(str(turn.get("content", "")))
    return "\n".join(parts)


def char_ngrams(text: str, n: int = 5) -> frozenset[str]:
    normalized = normalize_text(text)
    if not normalized:
        return frozenset()
    if len(normalized) <= n:
        return frozenset(normalized)
    return frozenset(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def normalized_similarity(left: str, right: str, *, ngram_size: int = 5) -> tuple[float, str]:
    """Score two texts using exact-normalized match, containment, then n-grams."""

    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0, "empty"
    if left_norm == right_norm:
        return 1.0, "normalized_exact"

    shorter, longer = sorted((left_norm, right_norm), key=len)
    if len(shorter) >= 24 and shorter in longer:
        return len(shorter) / len(longer), "normalized_containment"

    score = jaccard_similarity(char_ngrams(left_norm, ngram_size), char_ngrams(right_norm, ngram_size))
    return score, f"char_{ngram_size}gram_jaccard"


def is_near_duplicate(
    left: str,
    right: str,
    *,
    threshold: float = 0.88,
    ngram_size: int = 5,
) -> bool:
    score, _ = normalized_similarity(left, right, ngram_size=ngram_size)
    return score >= threshold


def find_near_duplicates(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]] | None = None,
    *,
    threshold: float = 0.88,
    ngram_size: int = 5,
) -> list[DuplicatePair]:
    """Find exact/normalized near duplicates within one set or across two sets."""

    pairs: list[DuplicatePair] = []
    if right_records is None:
        iterable = itertools.combinations(enumerate(left_records), 2)
    else:
        iterable = (
            ((left_index, left), (right_index, right))
            for left_index, left in enumerate(left_records)
            for right_index, right in enumerate(right_records)
        )

    for (left_index, left), (right_index, right) in iterable:
        left_id = _record_id(left, fallback=f"left_{left_index}")
        right_id = _record_id(right, fallback=f"right_{right_index}")
        if left_id == right_id:
            continue
        score, reason = normalized_similarity(
            record_text(left),
            record_text(right),
            ngram_size=ngram_size,
        )
        if score >= threshold:
            pairs.append(
                DuplicatePair(
                    left_id=left_id,
                    right_id=right_id,
                    similarity=round(score, 6),
                    reason=reason,
                    left_split=_optional_str(left.get("split")),
                    right_split=_optional_str(right.get("split")),
                )
            )
    return pairs


def find_cross_split_near_duplicates(
    records: Sequence[Mapping[str, Any]],
    *,
    left_split: str = "reward",
    right_split: str = "eval_heldout",
    threshold: float = 0.88,
    ngram_size: int = 5,
) -> list[DuplicatePair]:
    left_records = [record for record in records if record.get("split") == left_split]
    right_records = [record for record in records if record.get("split") == right_split]
    return find_near_duplicates(
        left_records,
        right_records,
        threshold=threshold,
        ngram_size=ngram_size,
    )


def dedupe_records(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.88,
    ngram_size: int = 5,
) -> tuple[list[Mapping[str, Any]], list[DuplicatePair]]:
    """Keep the first occurrence of each near-duplicate cluster."""

    kept: list[Mapping[str, Any]] = []
    duplicates: list[DuplicatePair] = []
    for candidate in records:
        candidate_id = _record_id(candidate)
        duplicate_found = False
        for existing in kept:
            score, reason = normalized_similarity(
                record_text(existing),
                record_text(candidate),
                ngram_size=ngram_size,
            )
            if score >= threshold:
                duplicates.append(
                    DuplicatePair(
                        left_id=_record_id(existing),
                        right_id=candidate_id,
                        similarity=round(score, 6),
                        reason=reason,
                        left_split=_optional_str(existing.get("split")),
                        right_split=_optional_str(candidate.get("split")),
                    )
                )
                duplicate_found = True
                break
        if not duplicate_found:
            kept.append(candidate)
    return kept, duplicates


def parse_completion(completion: str) -> ParsedCompletion:
    """Parse the required Character-R1-style output format."""

    if not isinstance(completion, str):
        return ParsedCompletion(focus=(), focus_attr=None, answer=None)

    focus_values: list[str] = []
    illegal_focus: list[str] = []
    for match in _FOCUS_RE.finditer(completion):
        for label in _split_focus_labels(match.group("value")):
            if label in FOCUS_LABELS:
                focus_values.append(label)
            else:
                illegal_focus.append(label)

    attr_match = _FOCUS_ATTR_RE.search(completion)
    boxed_match = _BOXED_RE.search(completion)
    focus = tuple(dict.fromkeys(focus_values))
    focus_attr = attr_match.group("value").strip() if attr_match else None
    answer = boxed_match.group("value").strip() if boxed_match else None
    well_formed = (
        bool(_THINK_RE.search(completion))
        and bool(focus)
        and not illegal_focus
        and bool(focus_attr)
        and bool(answer)
    )
    return ParsedCompletion(
        focus=focus,
        focus_attr=focus_attr,
        answer=answer,
        illegal_focus=tuple(illegal_focus),
        well_formed=well_formed,
    )


def make_perfect_completion(record: Mapping[str, Any]) -> str:
    focus_text = ", ".join(str(label) for label in record.get("gold_focus", ()))
    attr_text = str(record.get("gold_focus_attr", ""))
    answer = str(record.get("reference_answer", ""))
    return (
        "<think>"
        f"<focus>{focus_text}</focus>"
        f"<focus_attr>{attr_text}</focus_attr>"
        "</think>\n"
        f"\\boxed{{{answer}}}"
    )


def focus_overlap(predicted: Iterable[str], gold: Iterable[str]) -> float:
    pred_set = {label for label in predicted if label in FOCUS_LABELS}
    gold_set = {label for label in gold if label in FOCUS_LABELS}
    return jaccard_similarity(pred_set, gold_set)


def validate_record(
    record: Mapping[str, Any],
    *,
    split_config: Mapping[str, Any] | None = None,
    check_cjk: bool = True,
) -> RecordValidationResult:
    """Validate one reward/DPO record without mutating it."""

    issues: list[ValidationIssue] = []
    if not isinstance(record, Mapping):
        return RecordValidationResult(
            record_id=None,
            issues=(
                ValidationIssue(
                    code="schema.record_type",
                    field="<record>",
                    message="record must be a mapping",
                ),
            ),
        )

    record_id = _optional_str(record.get("id"))

    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            issues.append(_issue("schema.missing", field_name, "required field is missing", record_id))

    _validate_scalar_fields(record, issues, record_id)
    _validate_conversations(record, issues, record_id)
    _validate_focus(record, issues, record_id)
    _validate_reference_and_dpo(record, issues, record_id, check_cjk=check_cjk)
    _validate_provenance(record, issues, record_id)
    _validate_split_membership(record, issues, record_id, split_config=split_config)
    _validate_round_trip(record, issues, record_id)

    return RecordValidationResult(record_id=record_id, issues=tuple(issues))


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    split_config: Mapping[str, Any] | None = None,
    check_within_duplicates: bool = False,
    check_cross_split_duplicates: bool = True,
    near_duplicate_threshold: float = 0.88,
) -> BatchValidationReport:
    """Validate a batch and report cross-record leakage/duplicate issues."""

    record_list = list(records)
    results = tuple(validate_record(record, split_config=split_config) for record in record_list)
    batch_issues: list[ValidationIssue] = []

    seen_ids: dict[str, int] = {}
    for index, record in enumerate(record_list):
        record_id = _record_id(record, fallback=f"row_{index}")
        if record_id in seen_ids:
            batch_issues.append(
                _issue(
                    "schema.duplicate_id",
                    "id",
                    f"duplicate id also seen at row {seen_ids[record_id]}",
                    record_id,
                )
            )
        else:
            seen_ids[record_id] = index

    batch_issues.extend(check_split_invariants(split_config or {}, record_list))

    near_duplicates: list[DuplicatePair] = []
    if check_within_duplicates:
        near_duplicates.extend(
            find_near_duplicates(
                record_list,
                threshold=near_duplicate_threshold,
            )
        )
    if check_cross_split_duplicates:
        near_duplicates.extend(
            find_cross_split_near_duplicates(
                record_list,
                threshold=near_duplicate_threshold,
            )
        )
    for pair in near_duplicates:
        if pair.left_split == "reward" and pair.right_split == "eval_heldout":
            batch_issues.append(
                _issue(
                    "leakage.cross_split_near_duplicate",
                    "conversations/reference_answer",
                    f"{pair.left_id} and {pair.right_id} score {pair.similarity}",
                    pair.left_id,
                )
            )

    return BatchValidationReport(
        results=results,
        batch_issues=tuple(batch_issues),
        near_duplicates=tuple(near_duplicates),
    )


def check_split_invariants(
    split_config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Check character/source_work disjointness and local-only eval placement."""

    issues: list[ValidationIssue] = []
    split_characters = {
        split: set(_coerce_str_list(split_config.get(split, ()))) for split in VALID_SPLITS
    }

    for split, characters in split_characters.items():
        if len(characters) != len(_coerce_str_list(split_config.get(split, ()))):
            issues.append(
                _issue(
                    "split.duplicate_character",
                    split,
                    "split contains duplicate character entries",
                )
            )

    for left, right in itertools.combinations(VALID_SPLITS, 2):
        overlap = split_characters[left] & split_characters[right]
        if overlap:
            issues.append(
                _issue(
                    "leakage.character_split_overlap",
                    f"{left}/{right}",
                    f"characters appear in both splits: {sorted(overlap)}",
                )
            )

    for character in HSR_LOCAL_ONLY_CHARACTERS:
        for split in ("sft", "reward"):
            if character in split_characters[split]:
                issues.append(
                    _issue(
                        "leakage.hsr_not_eval_only",
                        split,
                        f"{character} must be eval_heldout/local-only if present",
                    )
                )

    source_work_by_character = _source_work_mapping_from_split(split_config)
    if source_work_by_character:
        _check_source_work_disjointness(
            split_characters,
            source_work_by_character,
            issues,
        )

    if records:
        record_work_by_character = {
            str(record.get("character")): str(record.get("source_work"))
            for record in records
            if _is_non_empty_str(record.get("character")) and _is_non_empty_str(record.get("source_work"))
        }
        if record_work_by_character:
            merged_mapping = {**source_work_by_character, **record_work_by_character}
            _check_source_work_disjointness(split_characters, merged_mapping, issues)

        works_by_seen_split: dict[str, set[str]] = {}
        chars_by_seen_split: dict[str, set[str]] = {}
        for record in records:
            split = record.get("split")
            if split not in VALID_SPLITS:
                continue
            character = _optional_str(record.get("character"))
            source_work = _optional_str(record.get("source_work"))
            if character:
                chars_by_seen_split.setdefault(character, set()).add(str(split))
            if source_work:
                works_by_seen_split.setdefault(source_work, set()).add(str(split))
            if character in HSR_LOCAL_ONLY_CHARACTERS and split != "eval_heldout":
                issues.append(
                    _issue(
                        "leakage.hsr_not_eval_only",
                        "split",
                        f"{character} record must be eval_heldout/local-only",
                        _optional_str(record.get("id")),
                    )
                )

        for character, splits in sorted(chars_by_seen_split.items()):
            if len(splits) > 1:
                issues.append(
                    _issue(
                        "leakage.character_record_split_overlap",
                        "character",
                        f"{character} appears in multiple record splits: {sorted(splits)}",
                    )
                )
        for source_work, splits in sorted(works_by_seen_split.items()):
            if len(splits) > 1:
                issues.append(
                    _issue(
                        "leakage.source_work_record_split_overlap",
                        "source_work",
                        f"{source_work} appears in multiple record splits: {sorted(splits)}",
                    )
                )

    return issues


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_scalar_fields(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
) -> None:
    for field_name in ("id", "character", "source_work", "profile", "gold_focus_attr", "reference_answer", "source", "split"):
        if field_name in record and not _is_non_empty_str(record.get(field_name)):
            issues.append(_issue("schema.non_empty_string", field_name, "field must be a non-empty string", record_id))

    if "character_cluster" in record and not isinstance(record.get("character_cluster"), int):
        issues.append(_issue("schema.character_cluster", "character_cluster", "field must be an int", record_id))

    rejected_answer = record.get("rejected_answer")
    if "rejected_answer" in record and rejected_answer is not None and not isinstance(rejected_answer, str):
        issues.append(_issue("schema.rejected_answer", "rejected_answer", "field must be null or a string", record_id))

    synth_meta = record.get("synth_meta")
    if "synth_meta" in record and synth_meta is not None and not isinstance(synth_meta, Mapping):
        issues.append(_issue("schema.synth_meta", "synth_meta", "field must be null or an object", record_id))

    source = record.get("source")
    if isinstance(source, str) and source not in VALID_SOURCES:
        issues.append(
            _issue(
                "provenance.source_unknown",
                "source",
                f"source must be one of {sorted(VALID_SOURCES)}",
                record_id,
            )
        )

    split = record.get("split")
    if isinstance(split, str) and split not in VALID_SPLITS:
        issues.append(
            _issue(
                "split.unknown",
                "split",
                f"split must be one of {list(VALID_SPLITS)}",
                record_id,
            )
        )


def _validate_conversations(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
) -> None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        issues.append(_issue("schema.conversations", "conversations", "field must be a non-empty list", record_id))
        return

    for index, turn in enumerate(conversations):
        field_name = f"conversations[{index}]"
        if not isinstance(turn, Mapping):
            issues.append(_issue("schema.conversation_turn", field_name, "turn must be an object", record_id))
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role not in VALID_ROLES:
            issues.append(_issue("schema.conversation_role", f"{field_name}.role", "invalid role", record_id))
        if not _is_non_empty_str(content):
            issues.append(_issue("schema.conversation_content", f"{field_name}.content", "content must be non-empty", record_id))


def _validate_focus(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
) -> None:
    gold_focus = record.get("gold_focus")
    if not isinstance(gold_focus, list) or not gold_focus:
        issues.append(_issue("label.gold_focus", "gold_focus", "field must be a non-empty list", record_id))
        return

    seen: set[str] = set()
    for index, label in enumerate(gold_focus):
        if not isinstance(label, str):
            issues.append(_issue("label.type", f"gold_focus[{index}]", "label must be a string", record_id))
            continue
        if label not in FOCUS_LABELS:
            issues.append(
                _issue(
                    "label.illegal",
                    f"gold_focus[{index}]",
                    f"illegal focus label: {label}",
                    record_id,
                )
            )
        if label in seen:
            issues.append(_issue("label.duplicate", f"gold_focus[{index}]", "duplicate focus label", record_id))
        seen.add(label)


def _validate_reference_and_dpo(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
    *,
    check_cjk: bool,
) -> None:
    reference = record.get("reference_answer")
    gold_attr = record.get("gold_focus_attr")
    rejected = record.get("rejected_answer")

    if isinstance(reference, str):
        if not reference.strip():
            issues.append(_issue("reference.empty", "reference_answer", "reference must be non-empty", record_id))
        if len(reference) > 1000:
            issues.append(_issue("reference.too_long", "reference_answer", "reference is unexpectedly long", record_id))
        if isinstance(gold_attr, str) and normalize_text(reference) == normalize_text(gold_attr):
            issues.append(
                _issue(
                    "reference.equals_gold_attr",
                    "reference_answer",
                    "reference must not equal gold_focus_attr",
                    record_id,
                )
            )
        if normalize_text(reference) in {normalize_text(label) for label in FOCUS_LABELS}:
            issues.append(
                _issue(
                    "reference.equals_focus_label",
                    "reference_answer",
                    "reference must not equal a focus label",
                    record_id,
                )
            )

    requires_cjk = _record_requires_cjk(record)
    if check_cjk and requires_cjk and isinstance(reference, str) and not has_cjk(reference):
        issues.append(_issue("reference.not_cjk", "reference_answer", "CJK record requires a CJK reference", record_id))

    if isinstance(rejected, str):
        if not rejected.strip():
            issues.append(_issue("dpo.rejected_empty", "rejected_answer", "rejected answer must be non-empty if present", record_id))
        if isinstance(reference, str) and normalize_text(reference) == normalize_text(rejected):
            issues.append(
                _issue(
                    "dpo.chosen_equals_rejected",
                    "rejected_answer",
                    "DPO chosen/reference must differ from rejected",
                    record_id,
                )
            )
        if check_cjk and requires_cjk and not has_cjk(rejected):
            issues.append(_issue("dpo.rejected_not_cjk", "rejected_answer", "CJK record requires a CJK rejected answer", record_id))


def _validate_provenance(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
) -> None:
    source = record.get("source")
    synth_meta = record.get("synth_meta")
    rejected = record.get("rejected_answer")
    source_requires_meta = source in SYNTHETIC_SOURCES

    if source_requires_meta and not isinstance(synth_meta, Mapping):
        issues.append(
            _issue(
                "provenance.synth_meta_missing",
                "synth_meta",
                "synthetic sources require synth_meta",
                record_id,
            )
        )

    if isinstance(rejected, str) and rejected.strip() and not isinstance(synth_meta, Mapping):
        issues.append(
            _issue(
                "provenance.rejected_strategy_missing",
                "synth_meta.rejected_strategy",
                "rejected_answer requires synth_meta.rejected_strategy",
                record_id,
            )
        )

    if isinstance(synth_meta, Mapping):
        for key in ("generator", "prompt_id"):
            if not _is_non_empty_str(synth_meta.get(key)):
                issues.append(
                    _issue(
                        "provenance.synth_meta_field_missing",
                        f"synth_meta.{key}",
                        "synth_meta field is required",
                        record_id,
                    )
                )
        if "lore_sources" in synth_meta and not isinstance(synth_meta.get("lore_sources"), list):
            issues.append(
                _issue(
                    "provenance.lore_sources",
                    "synth_meta.lore_sources",
                    "lore_sources must be a list when present",
                    record_id,
                )
            )
        strategy = synth_meta.get("rejected_strategy")
        if isinstance(rejected, str) and rejected.strip() and not _is_non_empty_str(strategy):
            issues.append(
                _issue(
                    "provenance.rejected_strategy_missing",
                    "synth_meta.rejected_strategy",
                    "rejected_answer requires rejected_strategy",
                    record_id,
                )
            )
        if (rejected is None or rejected == "") and _is_non_empty_str(strategy):
            issues.append(
                _issue(
                    "provenance.rejected_strategy_without_rejected",
                    "synth_meta.rejected_strategy",
                    "rejected_strategy is present without rejected_answer",
                    record_id,
                )
            )


def _validate_split_membership(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
    *,
    split_config: Mapping[str, Any] | None,
) -> None:
    split = record.get("split")
    character = record.get("character")
    if character in HSR_LOCAL_ONLY_CHARACTERS and split != "eval_heldout":
        issues.append(
            _issue(
                "leakage.hsr_not_eval_only",
                "split",
                f"{character} must be eval_heldout/local-only",
                record_id,
            )
        )

    if not split_config or not isinstance(character, str) or split not in VALID_SPLITS:
        return

    character_to_split: dict[str, str] = {}
    duplicated: set[str] = set()
    for split_name in VALID_SPLITS:
        for split_character in _coerce_str_list(split_config.get(split_name, ())):
            if split_character in character_to_split and character_to_split[split_character] != split_name:
                duplicated.add(split_character)
            character_to_split[split_character] = split_name

    if character in duplicated:
        issues.append(
            _issue(
                "leakage.character_split_overlap",
                "character",
                f"{character} appears in multiple split_config splits",
                record_id,
            )
        )

    expected_split = character_to_split.get(character)
    if expected_split is None:
        issues.append(
            _issue(
                "split.character_missing",
                "character",
                f"{character} is absent from split_config",
                record_id,
            )
        )
    elif expected_split != split:
        issues.append(
            _issue(
                "split.mismatch",
                "split",
                f"record split={split!r} but split_config has {expected_split!r}",
                record_id,
            )
        )


def _validate_round_trip(
    record: Mapping[str, Any],
    issues: list[ValidationIssue],
    record_id: str | None,
) -> None:
    gold_focus = record.get("gold_focus")
    gold_attr = record.get("gold_focus_attr")
    reference = record.get("reference_answer")
    if not (
        isinstance(gold_focus, list)
        and gold_focus
        and all(isinstance(label, str) and label in FOCUS_LABELS for label in gold_focus)
        and isinstance(gold_attr, str)
        and isinstance(reference, str)
    ):
        return

    parsed = parse_completion(make_perfect_completion(record))
    if not parsed.well_formed:
        issues.append(
            _issue(
                "parseability.round_trip_format",
                "<completion>",
                "perfect completion did not parse as well formed",
                record_id,
            )
        )
        return
    if focus_overlap(parsed.focus, gold_focus) != 1.0 or set(parsed.focus) != set(gold_focus):
        issues.append(
            _issue(
                "parseability.round_trip_focus",
                "gold_focus",
                "perfect completion focus labels did not round-trip",
                record_id,
            )
        )
    if parsed.focus_attr != gold_attr.strip():
        issues.append(
            _issue(
                "parseability.round_trip_attr",
                "gold_focus_attr",
                "perfect completion focus_attr did not round-trip",
                record_id,
            )
        )
    if parsed.answer != reference.strip():
        issues.append(
            _issue(
                "parseability.round_trip_reference",
                "reference_answer",
                "perfect completion reference answer did not round-trip",
                record_id,
            )
        )


def _check_source_work_disjointness(
    split_characters: Mapping[str, set[str]],
    source_work_by_character: Mapping[str, str],
    issues: list[ValidationIssue],
) -> None:
    works_by_split: dict[str, set[str]] = {}
    for split, characters in split_characters.items():
        works_by_split[split] = {
            source_work_by_character[character]
            for character in characters
            if character in source_work_by_character
        }

    for left, right in itertools.combinations(VALID_SPLITS, 2):
        overlap = works_by_split.get(left, set()) & works_by_split.get(right, set())
        if overlap:
            issues.append(
                _issue(
                    "leakage.source_work_split_overlap",
                    f"{left}/{right}",
                    f"source_work appears in both splits: {sorted(overlap)}",
                )
            )


def _source_work_mapping_from_split(split_config: Mapping[str, Any]) -> dict[str, str]:
    value = split_config.get("_source_work_by_character", {})
    if not isinstance(value, Mapping):
        return {}
    return {
        str(character): str(source_work)
        for character, source_work in value.items()
        if _is_non_empty_str(character) and _is_non_empty_str(source_work)
    }


def _record_requires_cjk(record: Mapping[str, Any]) -> bool:
    text_fields = [
        record.get("character"),
        record.get("source_work"),
        record.get("profile"),
        record.get("gold_focus_attr"),
    ]
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        text_fields.extend(
            turn.get("content")
            for turn in conversations
            if isinstance(turn, Mapping)
        )
    return any(has_cjk(value) for value in text_fields)


def _split_focus_labels(raw: str) -> list[str]:
    return [part.strip() for part in _FOCUS_SPLIT_RE.split(raw.strip()) if part.strip()]


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _record_id(record: Mapping[str, Any], *, fallback: str = "<unknown>") -> str:
    value = record.get("id")
    return value if isinstance(value, str) and value.strip() else fallback


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _issue(code: str, field_name: str, message: str, record_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, field=field_name, message=message, record_id=record_id)
