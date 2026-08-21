"""Hallucination rate and factual accuracy. Ref: Sec. III-I and Table 3."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class HallucinationKind(str, Enum):
    ANATOMICAL = "anatomical"
    PATHOLOGICAL = "pathological"
    RECOMMENDATION = "recommendation"
    FAITHFULNESS = "faithfulness"
    FACTUALITY = "factuality"


@dataclass(frozen=True)
class HallucinationEvent:
    response_id: str
    kind: HallucinationKind
    serious: bool
    supported: bool
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.response_id.strip():
            raise ValueError("response identifier cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("event confidence must be in [0, 1]")


def hallucination_rate(has_hallucination: Sequence[bool]) -> float:
    if not has_hallucination:
        raise ValueError("hallucination labels cannot be empty")
    return float(np.mean(np.asarray(has_hallucination, dtype=np.float64)))


def response_hallucination_labels(
    response_ids: Sequence[str],
    events: Iterable[HallucinationEvent],
    serious_only: bool = False,
) -> tuple[bool, ...]:
    event_ids = {
        event.response_id
        for event in events
        if not event.supported and (event.serious or not serious_only)
    }
    return tuple(identifier in event_ids for identifier in response_ids)


def hallucination_breakdown(events: Iterable[HallucinationEvent]) -> Mapping[str, int]:
    counts = {kind.value: 0 for kind in HallucinationKind}
    for event in events:
        if not event.supported:
            counts[event.kind.value] += 1
    return counts


def factual_accuracy(
    predicted_entities: Sequence[set[int]], reference_entities: Sequence[set[int]]
) -> float:
    if len(predicted_entities) != len(reference_entities) or not predicted_entities:
        raise ValueError("predicted and reference entity collections must be nonempty and equal")
    correct = 0
    total = 0
    for predicted, reference in zip(predicted_entities, reference_entities, strict=False):
        correct += len(predicted & reference)
        total += len(predicted)
    return correct / total if total else 1.0


def entity_precision_recall_f1(
    predicted: Sequence[set[int]], reference: Sequence[set[int]]
) -> dict[str, float]:
    if len(predicted) != len(reference) or not predicted:
        raise ValueError("entity collections must be nonempty and equal")
    true_positive = sum(
        len(left & right) for left, right in zip(predicted, reference, strict=False)
    )
    false_positive = sum(
        len(left - right) for left, right in zip(predicted, reference, strict=False)
    )
    false_negative = sum(
        len(right - left) for left, right in zip(predicted, reference, strict=False)
    )
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def relative_hallucination_reduction(baseline_rate: float, model_rate: float) -> float:
    if not 0 <= baseline_rate <= 1 or not 0 <= model_rate <= 1:
        raise ValueError("hallucination rates must be in [0, 1]")
    if baseline_rate == 0:
        return 0.0 if model_rate == 0 else float("-inf")
    return (baseline_rate - model_rate) / baseline_rate


def serious_event_rate(response_ids: Sequence[str], events: Iterable[HallucinationEvent]) -> float:
    labels = response_hallucination_labels(response_ids, events, serious_only=True)
    return hallucination_rate(labels)


def event_confusion_matrix(
    predicted: Sequence[bool],
    reference: Sequence[bool],
) -> tuple[int, int, int, int]:
    if len(predicted) != len(reference) or not predicted:
        raise ValueError("event labels must be nonempty and equal")
    true_positive = sum(left and right for left, right in zip(predicted, reference, strict=False))
    false_positive = sum(
        left and not right for left, right in zip(predicted, reference, strict=False)
    )
    true_negative = sum(
        not left and not right for left, right in zip(predicted, reference, strict=False)
    )
    false_negative = sum(
        not left and right for left, right in zip(predicted, reference, strict=False)
    )
    return true_positive, false_positive, true_negative, false_negative
