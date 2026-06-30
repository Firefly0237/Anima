"""Optional per-character reward normalization scaffold.

This is deliberately *not* used by the W2 v1 reward. If a later W4 ablation
enables character-conditioned normalization, normalize either rewards or
advantages, not both, and log the EMA decay/warmup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


@dataclass
class EmaStats:
    count: int = 0
    mean: float = 0.0
    second_moment: float = 0.0

    def update(self, value: float, decay: float) -> None:
        if self.count == 0:
            self.mean = value
            self.second_moment = value * value
        else:
            self.mean = decay * self.mean + (1.0 - decay) * value
            self.second_moment = decay * self.second_moment + (1.0 - decay) * value * value
        self.count += 1

    @property
    def variance(self) -> float:
        return max(0.0, self.second_moment - self.mean * self.mean)


@dataclass
class CharacterRewardNormalizer:
    """EMA reward standardizer keyed by character or cluster id."""

    decay: float = 0.99
    warmup: int = 8
    epsilon: float = 1e-6
    stats: dict[str, EmaStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def normalize(self, key: str, value: float, *, update: bool = True) -> float:
        stats = self.stats.setdefault(str(key), EmaStats())
        if stats.count < self.warmup:
            normalized = float(value)
        else:
            denom = math.sqrt(stats.variance + self.epsilon)
            normalized = (float(value) - stats.mean) / denom
        if update:
            stats.update(float(value), self.decay)
        return normalized

    def normalize_batch(
        self,
        keys: Iterable[str],
        values: Iterable[float],
        *,
        update: bool = True,
    ) -> list[float]:
        return [
            self.normalize(key, value, update=update)
            for key, value in zip(keys, values, strict=True)
        ]
