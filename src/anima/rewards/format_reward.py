"""Format reward for the required think/focus/focus_attr/boxed structure."""

from __future__ import annotations

from typing import Any

from anima.rewards.parsing import ensure_completion_list, is_output_contract_valid


def format_score(completion: Any) -> float:
    return float(is_output_contract_valid(completion))


def format_reward(completions: Any, **_kwargs: Any) -> list[float]:
    """TRL-compatible 0/1 format reward."""

    return [format_score(completion) for completion in ensure_completion_list(completions)]
