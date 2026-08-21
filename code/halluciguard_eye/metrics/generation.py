"""Generation quality metrics reported in Sec. III-I."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np


def ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if order <= 0:
        raise ValueError("ngram order must be positive")
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def modified_precision(
    candidate: Sequence[str], references: Sequence[Sequence[str]], order: int
) -> tuple[int, int]:
    candidate_counts = ngrams(candidate, order)
    maximum_reference: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        reference_counts = ngrams(reference, order)
        for gram, count in reference_counts.items():
            maximum_reference[gram] = max(maximum_reference[gram], count)
    clipped = sum(min(count, maximum_reference[gram]) for gram, count in candidate_counts.items())
    total = sum(candidate_counts.values())
    return clipped, total


def closest_reference_length(candidate_length: int, references: Sequence[Sequence[str]]) -> int:
    lengths = [len(reference) for reference in references]
    if not lengths:
        raise ValueError("BLEU requires references")
    return min(lengths, key=lambda length: (abs(length - candidate_length), length))


def bleu4(
    candidate: Sequence[str], references: Sequence[Sequence[str]], smoothing: float = 1.0
) -> float:
    if not candidate or not references:
        return 0.0
    precisions: list[float] = []
    for order in range(1, 5):
        clipped, total = modified_precision(candidate, references, order)
        precisions.append((clipped + smoothing) / (total + smoothing))
    reference_length = closest_reference_length(len(candidate), references)
    brevity = (
        1.0
        if len(candidate) > reference_length
        else np.exp(1.0 - reference_length / len(candidate))
    )
    return float(brevity * np.exp(np.mean(np.log(precisions))))


def longest_common_subsequence(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: Sequence[str], reference: Sequence[str], beta: float = 1.0) -> float:
    if beta <= 0:
        raise ValueError("ROUGE beta must be positive")
    if not candidate or not reference:
        return 0.0
    length = longest_common_subsequence(candidate, reference)
    precision = length / len(candidate)
    recall = length / len(reference)
    denominator = recall + beta**2 * precision
    return (1 + beta**2) * precision * recall / denominator if denominator else 0.0


def bert_score_from_embeddings(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[1] != reference.shape[1]:
        raise ValueError("embedding matrices must have equal widths")
    candidate = candidate / np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-12)
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    similarities = candidate @ reference.T
    precision = float(similarities.max(axis=1).mean()) if len(candidate) else 0.0
    recall = float(similarities.max(axis=0).mean()) if len(reference) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def corpus_average(values: Iterable[float]) -> float:
    scores = tuple(values)
    if not scores:
        raise ValueError("metric values cannot be empty")
    return float(np.mean(scores))
