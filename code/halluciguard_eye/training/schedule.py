"""Learning-rate schedules for the reported three stages."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class ScheduleConfig:
    total_steps: int
    warmup_steps: int = 0
    minimum_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total steps must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup steps must be in [0, total steps)")
        if not 0 <= self.minimum_ratio <= 1:
            raise ValueError("minimum learning-rate ratio must be in [0, 1]")


def cosine_multiplier(step: int, config: ScheduleConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.minimum_ratio + (1.0 - config.minimum_ratio) * cosine


def cosine_schedule(optimizer: Optimizer, config: ScheduleConfig) -> LambdaLR:
    return LambdaLR(optimizer, lambda step: cosine_multiplier(step, config))


def constant_schedule(optimizer: Optimizer) -> LambdaLR:
    return LambdaLR(optimizer, lambda _: 1.0)
