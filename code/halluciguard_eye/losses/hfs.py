"""Hierarchical factuality objectives. Ref: Eq. (19)-(24)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class HFSWeights:
    anatomical: float = 0.25
    pathological: float = 0.40
    recommendation: float = 0.35

    def __post_init__(self) -> None:
        values = (self.anatomical, self.pathological, self.recommendation)
        if any(value < 0 for value in values):
            raise ValueError("HFS weights must be nonnegative")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("HFS weights must sum to one")

    def tensor(self, reference: Tensor) -> Tensor:
        return reference.new_tensor((self.anatomical, self.pathological, self.recommendation))


DEFAULT_HFS_WEIGHTS = HFSWeights()


def hierarchical_factuality_score(
    components: Tensor, weights: HFSWeights = DEFAULT_HFS_WEIGHTS
) -> Tensor:
    if components.shape[-1] != 3:
        raise ValueError("HFS components must end with three values")
    if torch.any(components < 0) or torch.any(components > 1):
        raise ValueError("HFS components must be in [0, 1]")
    return (components * weights.tensor(components)).sum(dim=-1)


def hierarchical_factuality_loss(
    predicted_components: Tensor,
    target_components: Tensor,
    weights: HFSWeights = DEFAULT_HFS_WEIGHTS,
    direct_score_weight: float = 1.0,
) -> Tensor:
    if predicted_components.shape != target_components.shape or predicted_components.shape[-1] != 3:
        raise ValueError("predicted and target HFS components must match")
    if direct_score_weight < 0:
        raise ValueError("direct score weight must be nonnegative")
    component_weights = weights.tensor(predicted_components)
    component_loss = functional.binary_cross_entropy(
        predicted_components,
        target_components,
        reduction="none",
    )
    component_loss = (component_loss * component_weights).sum(dim=-1).mean()
    predicted_score = hierarchical_factuality_score(predicted_components, weights)
    target_score = hierarchical_factuality_score(target_components, weights)
    score_loss = functional.smooth_l1_loss(predicted_score, target_score)
    return component_loss + direct_score_weight * score_loss


def pathological_similarity_loss(
    predicted_embeddings: Tensor,
    reference_embeddings: Tensor,
    severity_weights: Tensor,
) -> Tensor:
    if predicted_embeddings.shape != reference_embeddings.shape or predicted_embeddings.ndim != 2:
        raise ValueError("pathology embedding matrices must match")
    if severity_weights.shape != predicted_embeddings.shape[:1]:
        raise ValueError("severity weights must match pathology count")
    if torch.any(severity_weights <= 0):
        raise ValueError("severity weights must be positive")
    similarities = functional.cosine_similarity(predicted_embeddings, reference_embeddings, dim=-1)
    return ((1.0 - similarities) * severity_weights).sum() / severity_weights.sum()


def anatomical_verification_loss(
    predicted: Tensor, verified: Tensor, mask: Tensor | None = None
) -> Tensor:
    if predicted.shape != verified.shape:
        raise ValueError("anatomical predictions and labels must match")
    losses = functional.binary_cross_entropy(
        predicted, verified.to(predicted.dtype), reduction="none"
    )
    if mask is None:
        return losses.mean()
    if mask.shape != losses.shape:
        raise ValueError("anatomical mask shape mismatch")
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def recommendation_consistency_loss(probabilities: Tensor, supported: Tensor) -> Tensor:
    if probabilities.shape != supported.shape:
        raise ValueError("recommendation probabilities and labels must match")
    return functional.binary_cross_entropy(probabilities, supported.to(probabilities.dtype))


def one_minus_hfs(components: Tensor, weights: HFSWeights = DEFAULT_HFS_WEIGHTS) -> Tensor:
    return 1.0 - hierarchical_factuality_score(components, weights).mean()
