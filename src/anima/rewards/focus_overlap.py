"""Focus-label reward: graded overlap plus strict EM side metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anima.rewards.parsing import (
    ensure_completion_list,
    is_output_contract_valid,
    normalize_focus_labels,
    parse_completion,
)

FOCUS_OVERLAP_METRIC = "f1"


@dataclass(frozen=True)
class FocusOverlapResult:
    reward: float
    strict_em: float
    predicted: tuple[str, ...]
    gold: tuple[str, ...]
    illegal_predicted: tuple[str, ...]
    metric: str = FOCUS_OVERLAP_METRIC


def score_focus_overlap(
    completion: Any,
    gold_focus: Any,
    *,
    metric: str = FOCUS_OVERLAP_METRIC,
) -> FocusOverlapResult:
    parsed = parse_completion(completion)
    predicted = tuple(parsed.focus_labels)
    gold = normalize_focus_labels(gold_focus)
    pred_set = set(predicted)
    gold_set = set(gold)
    illegal_count = len(parsed.illegal_focus_labels)

    valid = is_output_contract_valid(parsed)
    if not valid or not pred_set or not gold_set:
        reward = 0.0
    else:
        true_positive = len(pred_set & gold_set)
        if metric == "jaccard":
            reward = true_positive / (len(pred_set | gold_set) + illegal_count)
        elif metric == "f1":
            reward = (2 * true_positive) / (len(pred_set) + illegal_count + len(gold_set))
        else:
            raise ValueError(f"Unsupported focus overlap metric: {metric}")

    strict_em = float(valid and bool(gold_set) and pred_set == gold_set and not parsed.illegal_focus_labels)
    return FocusOverlapResult(
        reward=float(reward),
        strict_em=strict_em,
        predicted=predicted,
        gold=gold,
        illegal_predicted=tuple(parsed.illegal_focus_labels),
        metric=metric,
    )


def focus_overlap_score(completion: Any, gold_focus: Any) -> float:
    return score_focus_overlap(completion, gold_focus).reward


def focus_strict_em(completion: Any, gold_focus: Any) -> float:
    return score_focus_overlap(completion, gold_focus).strict_em


def focus_overlap_reward(completions: Any, gold_focus: Any = None, **_kwargs: Any) -> list[float]:
    """TRL-compatible reward function using the ``gold_focus`` dataset column."""

    batch = ensure_completion_list(completions)
    return [
        score_focus_overlap(completion, _gold_focus_at(gold_focus, index, len(batch))).reward
        for index, completion in enumerate(batch)
    ]


def focus_strict_em_metric(completions: Any, gold_focus: Any = None, **_kwargs: Any) -> list[float]:
    """Side metric for logging only; do not include in optimized reward weights."""

    batch = ensure_completion_list(completions)
    return [
        score_focus_overlap(completion, _gold_focus_at(gold_focus, index, len(batch))).strict_em
        for index, completion in enumerate(batch)
    ]


def _gold_focus_at(gold_focus: Any, index: int, batch_size: int) -> Any:
    if gold_focus is None:
        return ()
    if (
        batch_size == 1
        and isinstance(gold_focus, (list, tuple))
        and len(gold_focus) == 1
        and isinstance(gold_focus[0], (list, tuple, set))
    ):
        return gold_focus[0]
    if batch_size == 1:
        return gold_focus
    if isinstance(gold_focus, (list, tuple)) and len(gold_focus) == batch_size:
        if all(isinstance(item, str) for item in gold_focus):
            return gold_focus
        return gold_focus[index]
    return gold_focus
