"""Weighted reward wiring for TRL GRPOTrainer."""

from __future__ import annotations

from typing import Any, Callable

from anima.rewards.attribute_bleu import CHRF_METRIC_NAME, attribute_reward
from anima.rewards.focus_overlap import FOCUS_OVERLAP_METRIC, focus_overlap_reward
from anima.rewards.format_reward import format_reward
from anima.rewards.parsing import ensure_completion_list
from anima.rewards.reference_bleu import REFERENCE_METRIC_NAME, reference_reward

REWARD_WEIGHTS: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
reward_weights: list[float] = list(REWARD_WEIGHTS)
reward_funcs: list[Callable[..., list[float]]] = [
    focus_overlap_reward,
    attribute_reward,
    reference_reward,
    format_reward,
]

PINNED_REWARD_METRICS = {
    "focus": FOCUS_OVERLAP_METRIC,
    "attribute": CHRF_METRIC_NAME,
    "reference": REFERENCE_METRIC_NAME,
    "format": "exact_structure_0_1",
}

if abs(sum(REWARD_WEIGHTS) - 1.0) > 1e-12:  # pragma: no cover - import-time guard.
    raise RuntimeError(f"Reward weights must sum to 1.0, got {sum(REWARD_WEIGHTS)}")


def get_reward_funcs() -> list[Callable[..., list[float]]]:
    """Return a fresh TRL-compatible reward function list."""

    return list(reward_funcs)


def get_reward_weights() -> list[float]:
    """Return a fresh TRL ``reward_weights`` list."""

    return list(reward_weights)


def component_rewards(completions: Any, **kwargs: Any) -> dict[str, list[float]]:
    return {
        "focus": focus_overlap_reward(completions, **kwargs),
        "attribute": attribute_reward(completions, **kwargs),
        "reference": reference_reward(completions, **kwargs),
        "format": format_reward(completions, **kwargs),
    }


def combined_reward(completions: Any, **kwargs: Any) -> list[float]:
    """Single reward function variant for callers that do not use TRL weights."""

    batch_size = len(ensure_completion_list(completions))
    components = component_rewards(completions, **kwargs)
    return [
        sum(
            weight * components[name][index]
            for weight, name in zip(REWARD_WEIGHTS, ("focus", "attribute", "reference", "format"))
        )
        for index in range(batch_size)
    ]
