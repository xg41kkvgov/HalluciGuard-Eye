"""Stratified and patient-grouped splits. Ref: Sec. III-F and III-G."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class SplitAssignment:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (set(self.train), set(self.validation), set(self.test))
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("split indices overlap")
        if any(index < 0 for group in groups for index in group):
            raise ValueError("split indices must be nonnegative")

    @property
    def size(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def labels(self, labels: Sequence[int]) -> dict[str, tuple[int, ...]]:
        return {
            "train": tuple(labels[index] for index in self.train),
            "validation": tuple(labels[index] for index in self.validation),
            "test": tuple(labels[index] for index in self.test),
        }


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0 < train_fraction < 1:
        raise ValueError("train fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")


def stratified_split(
    labels: Sequence[int],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SplitAssignment:
    _validate_fractions(train_fraction, validation_fraction)
    if not labels:
        raise ValueError("labels cannot be empty")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in sorted(set(labels)):
        indices = np.asarray([index for index, value in enumerate(labels) if value == label])
        rng.shuffle(indices)
        count = len(indices)
        train_end = int(round(count * train_fraction))
        validation_end = train_end + int(round(count * validation_fraction))
        train.extend(indices[:train_end].tolist())
        validation.extend(indices[train_end:validation_end].tolist())
        test.extend(indices[validation_end:].tolist())
    return SplitAssignment(
        tuple(sorted(train)),
        tuple(sorted(validation)),
        tuple(sorted(test)),
    )


def grouped_stratified_split(
    labels: Sequence[int],
    groups: Sequence[T],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SplitAssignment:
    _validate_fractions(train_fraction, validation_fraction)
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have equal length")
    if not labels:
        raise ValueError("labels cannot be empty")
    members: dict[T, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    group_labels: dict[T, int] = {}
    for group, indices in members.items():
        values, counts = np.unique([labels[index] for index in indices], return_counts=True)
        group_labels[group] = int(values[int(np.argmax(counts))])
    rng = np.random.default_rng(seed)
    assignments: dict[str, list[T]] = {"train": [], "validation": [], "test": []}
    for label in sorted(set(group_labels.values())):
        current = [group for group, value in group_labels.items() if value == label]
        rng.shuffle(current)
        count = len(current)
        train_end = int(round(count * train_fraction))
        validation_end = train_end + int(round(count * validation_fraction))
        assignments["train"].extend(current[:train_end])
        assignments["validation"].extend(current[train_end:validation_end])
        assignments["test"].extend(current[validation_end:])
    return SplitAssignment(
        tuple(sorted(index for group in assignments["train"] for index in members[group])),
        tuple(sorted(index for group in assignments["validation"] for index in members[group])),
        tuple(sorted(index for group in assignments["test"] for index in members[group])),
    )


def kfold_group_assignments(
    groups: Sequence[T], folds: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    if folds < 2:
        raise ValueError("fold count must be at least two")
    unique = list(dict.fromkeys(groups))
    if folds > len(unique):
        raise ValueError("fold count cannot exceed group count")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    group_fold = {group: index % folds for index, group in enumerate(unique)}
    return tuple(
        tuple(index for index, group in enumerate(groups) if group_fold[group] == fold)
        for fold in range(folds)
    )


def assert_group_isolation(assignment: SplitAssignment, groups: Sequence[T]) -> None:
    split_groups = [
        {groups[index] for index in assignment.train},
        {groups[index] for index in assignment.validation},
        {groups[index] for index in assignment.test},
    ]
    if (
        split_groups[0] & split_groups[1]
        or split_groups[0] & split_groups[2]
        or split_groups[1] & split_groups[2]
    ):
        raise AssertionError("a group occurs in multiple splits")


def split_distribution(
    assignment: SplitAssignment, labels: Sequence[int]
) -> Mapping[str, Mapping[int, float]]:
    output: dict[str, dict[int, float]] = {}
    for name, indices in (
        ("train", assignment.train),
        ("validation", assignment.validation),
        ("test", assignment.test),
    ):
        if not indices:
            output[name] = {}
            continue
        values, counts = np.unique([labels[index] for index in indices], return_counts=True)
        output[name] = {
            int(value): float(count / len(indices))
            for value, count in zip(values, counts, strict=False)
        }
    return output
