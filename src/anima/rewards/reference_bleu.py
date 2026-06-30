"""CJK-safe chrF reward over the ``\\boxed{}`` answer."""

from __future__ import annotations

from typing import Any

from anima.rewards.attribute_bleu import chrf_score
from anima.rewards.parsing import ensure_completion_list, is_output_contract_valid, parse_completion

REFERENCE_METRIC_NAME = "native_chrf_beta2_char_order6_whitespace_stripped"


def reference_score(completion: Any, reference_answer: Any) -> float:
    parsed = parse_completion(completion)
    if not is_output_contract_valid(parsed) or not parsed.boxed_answer:
        return 0.0
    return chrf_score(parsed.boxed_answer, reference_answer)


def reference_reward(
    completions: Any,
    reference_answer: Any = None,
    **_kwargs: Any,
) -> list[float]:
    """TRL-compatible reward function using ``reference_answer``."""

    batch = ensure_completion_list(completions)
    return [
        reference_score(completion, _gold_text_at(reference_answer, index, len(batch)))
        for index, completion in enumerate(batch)
    ]


def _gold_text_at(values: Any, index: int, batch_size: int) -> Any:
    if values is None:
        return ""
    if batch_size == 1 and isinstance(values, (list, tuple)) and len(values) == 1:
        return values[0]
    if batch_size > 1 and isinstance(values, (list, tuple)) and len(values) == batch_size:
        return values[index]
    return values
