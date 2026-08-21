"""Patient-clustered confidence intervals. Ref: Sec. III-G."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")
G = TypeVar("G", bound=Hashable)


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    replicates: int

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1:
            raise ValueError("confidence level must be in (0, 1)")
        if self.replicates <= 0:
            raise ValueError("replicate count must be positive")
        if self.lower > self.upper:
            raise ValueError("confidence interval is reversed")


def cluster_indices(groups: Sequence[G]) -> Mapping[G, tuple[int, ...]]:
    mapping: dict[G, list[int]] = {}
    for index, group in enumerate(groups):
        mapping.setdefault(group, []).append(index)
    return {group: tuple(indices) for group, indices in mapping.items()}


def cluster_bootstrap_interval(
    values: Sequence[T],
    groups: Sequence[G],
    statistic: Callable[[Sequence[T]], float],
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    if len(values) != len(groups) or not values:
        raise ValueError("values and groups must be nonempty and equal")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap controls are invalid")
    members = cluster_indices(groups)
    group_names = tuple(members)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.choice(len(group_names), size=len(group_names), replace=True)
        sampled_values = [
            values[index]
            for group_index in sampled_groups
            for index in members[group_names[int(group_index)]]
        ]
        samples[replicate] = statistic(sampled_values)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return ConfidenceInterval(statistic(values), float(lower), float(upper), confidence, replicates)


def paired_cluster_bootstrap_difference(
    left: Sequence[T],
    right: Sequence[T],
    groups: Sequence[G],
    statistic: Callable[[Sequence[T]], float],
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    if len(left) != len(right) or len(left) != len(groups) or not left:
        raise ValueError("paired values and groups must be nonempty and equal")
    members = cluster_indices(groups)
    group_names = tuple(members)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.choice(len(group_names), size=len(group_names), replace=True)
        indices = [
            index
            for group_index in sampled_groups
            for index in members[group_names[int(group_index)]]
        ]
        differences[replicate] = statistic([left[index] for index in indices]) - statistic(
            [right[index] for index in indices]
        )
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, (alpha, 1.0 - alpha))
    estimate = statistic(left) - statistic(right)
    return ConfidenceInterval(estimate, float(lower), float(upper), confidence, replicates)


def randomly_select_one_per_group(groups: Sequence[G], seed: int) -> tuple[int, ...]:
    members = cluster_indices(groups)
    rng = np.random.default_rng(seed)
    return tuple(sorted(int(rng.choice(indices)) for indices in members.values()))


def select_worse_per_group(groups: Sequence[G], severity: Sequence[float]) -> tuple[int, ...]:
    if len(groups) != len(severity):
        raise ValueError("groups and severity must have equal length")
    members = cluster_indices(groups)
    return tuple(
        sorted(
            max(indices, key=lambda index: (severity[index], -index))
            for indices in members.values()
        )
    )
