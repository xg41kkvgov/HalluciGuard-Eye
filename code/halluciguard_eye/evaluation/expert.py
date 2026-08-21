"""Expert Likert evaluation and Fleiss kappa. Ref: Sec. IV-E and Table 7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertRating:
    response_id: str
    reviewer_id: str
    accuracy: int
    completeness: int
    clinical_relevance: int
    trustworthiness: int

    def __post_init__(self) -> None:
        if not self.response_id.strip() or not self.reviewer_id.strip():
            raise ValueError("rating identifiers cannot be empty")
        values = (self.accuracy, self.completeness, self.clinical_relevance, self.trustworthiness)
        if any(not 1 <= value <= 5 for value in values):
            raise ValueError("Likert ratings must be in [1, 5]")


def mean_ratings(ratings: Sequence[ExpertRating]) -> Mapping[str, float]:
    if not ratings:
        raise ValueError("expert ratings cannot be empty")
    return {
        "accuracy": float(np.mean([rating.accuracy for rating in ratings])),
        "completeness": float(np.mean([rating.completeness for rating in ratings])),
        "clinical_relevance": float(np.mean([rating.clinical_relevance for rating in ratings])),
        "trustworthiness": float(np.mean([rating.trustworthiness for rating in ratings])),
    }


def rating_matrix(ratings: Sequence[ExpertRating], attribute: str) -> np.ndarray:
    if attribute not in {"accuracy", "completeness", "clinical_relevance", "trustworthiness"}:
        raise ValueError("unknown expert rating attribute")
    responses = sorted({rating.response_id for rating in ratings})
    reviewers = sorted({rating.reviewer_id for rating in ratings})
    lookup = {
        (rating.response_id, rating.reviewer_id): getattr(rating, attribute) for rating in ratings
    }
    if any((response, reviewer) not in lookup for response in responses for reviewer in reviewers):
        raise ValueError("expert rating matrix is incomplete")
    matrix = np.zeros((len(responses), 5), dtype=np.int64)
    for row, response in enumerate(responses):
        for reviewer in reviewers:
            matrix[row, lookup[(response, reviewer)] - 1] += 1
    return matrix


def fleiss_kappa(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("Fleiss kappa matrix must contain items and categories")
    reviewer_counts = matrix.sum(axis=1)
    if np.any(reviewer_counts != reviewer_counts[0]) or reviewer_counts[0] < 2:
        raise ValueError("every item must have the same number of at least two reviewers")
    reviewers = float(reviewer_counts[0])
    agreement = ((matrix**2).sum(axis=1) - reviewers) / (reviewers * (reviewers - 1))
    category_proportions = matrix.sum(axis=0) / matrix.sum()
    expected = float((category_proportions**2).sum())
    observed = float(agreement.mean())
    return (observed - expected) / (1.0 - expected) if expected < 1 else 1.0


def expert_summary(ratings: Sequence[ExpertRating]) -> Mapping[str, float]:
    summary = dict(mean_ratings(ratings))
    kappas = [
        fleiss_kappa(rating_matrix(ratings, attribute))
        for attribute in ("accuracy", "completeness", "clinical_relevance", "trustworthiness")
    ]
    summary["fleiss_kappa"] = float(np.mean(kappas))
    return summary
