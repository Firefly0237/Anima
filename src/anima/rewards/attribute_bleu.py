"""CJK-safe chrF reward over ``<focus_attr>`` text."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from anima.rewards.parsing import ensure_completion_list, is_output_contract_valid, parse_completion

CHRF_METRIC_NAME = "native_chrf_beta2_char_order6_whitespace_stripped"
CHRF_CHAR_ORDER = 6
CHRF_BETA = 2.0


def chrf_score(
    prediction: Any,
    reference: Any,
    *,
    char_order: int = CHRF_CHAR_ORDER,
    beta: float = CHRF_BETA,
) -> float:
    """Small dependency-free chrF implementation pinned for CJK reward tests."""

    pred = _normalize_metric_text(prediction)
    ref = _normalize_metric_text(reference)
    if not pred or not ref:
        return 0.0
    if pred == ref:
        return 1.0

    precisions: list[float] = []
    recalls: list[float] = []
    max_order = max(1, min(char_order, max(len(pred), len(ref))))
    for order in range(1, max_order + 1):
        pred_counts = _char_ngrams(pred, order)
        ref_counts = _char_ngrams(ref, order)
        if not pred_counts and not ref_counts:
            continue
        if not pred_counts or not ref_counts:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        overlap = sum((pred_counts & ref_counts).values())
        precisions.append(overlap / sum(pred_counts.values()))
        recalls.append(overlap / sum(ref_counts.values()))

    if not precisions or not recalls:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    beta_sq = beta * beta
    return float((1 + beta_sq) * precision * recall / (beta_sq * precision + recall))


def attribute_score(completion: Any, gold_focus_attr: Any) -> float:
    parsed = parse_completion(completion)
    if not is_output_contract_valid(parsed) or not parsed.focus_attr:
        return 0.0
    return chrf_score(parsed.focus_attr, gold_focus_attr)


def attribute_reward(
    completions: Any,
    gold_focus_attr: Any = None,
    **_kwargs: Any,
) -> list[float]:
    """TRL-compatible reward function using ``gold_focus_attr``."""

    batch = ensure_completion_list(completions)
    return [
        attribute_score(completion, _gold_text_at(gold_focus_attr, index, len(batch)))
        for index, completion in enumerate(batch)
    ]


def _normalize_metric_text(text: Any) -> str:
    return re.sub(r"\s+", "", "" if text is None else str(text).strip())


def _char_ngrams(text: str, order: int) -> Counter[str]:
    if order <= 0 or len(text) < order:
        return Counter()
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def _gold_text_at(values: Any, index: int, batch_size: int) -> Any:
    if values is None:
        return ""
    if batch_size == 1 and isinstance(values, (list, tuple)) and len(values) == 1:
        return values[0]
    if batch_size > 1 and isinstance(values, (list, tuple)) and len(values) == batch_size:
        return values[index]
    return values
