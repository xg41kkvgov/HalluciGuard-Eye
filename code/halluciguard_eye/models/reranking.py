"""Candidate response reranking. Ref: Sec. III-C.3 and Eq. (13)-(14)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CandidateScore:
    index: int
    log_probability: float
    alignment: float
    grounding: float
    composite: float


def composite_scores(
    log_probabilities: Tensor,
    alignment_scores: Tensor,
    grounding_scores: Tensor,
    alignment_weight: float = 0.3,
    grounding_weight: float = 0.4,
) -> Tensor:
    if (
        log_probabilities.shape != alignment_scores.shape
        or log_probabilities.shape != grounding_scores.shape
    ):
        raise ValueError("candidate score tensors must have equal shapes")
    if log_probabilities.ndim != 1:
        raise ValueError("candidate score tensors must be vectors")
    return (
        log_probabilities
        + alignment_weight * alignment_scores
        + grounding_weight * grounding_scores
    )


def rerank_candidates(
    log_probabilities: Tensor,
    alignment_scores: Tensor,
    grounding_scores: Tensor,
    alignment_weight: float = 0.3,
    grounding_weight: float = 0.4,
) -> tuple[int, tuple[CandidateScore, ...]]:
    scores = composite_scores(
        log_probabilities,
        alignment_scores,
        grounding_scores,
        alignment_weight,
        grounding_weight,
    )
    records = tuple(
        CandidateScore(
            index=index,
            log_probability=float(log_probabilities[index].item()),
            alignment=float(alignment_scores[index].item()),
            grounding=float(grounding_scores[index].item()),
            composite=float(scores[index].item()),
        )
        for index in range(len(scores))
    )
    return int(torch.argmax(scores).item()), records


def sequence_log_probabilities(
    logits: Tensor, token_ids: Tensor, ignore_index: int = -100
) -> Tensor:
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2]:
        raise ValueError("logits and token identifiers have incompatible shapes")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    safe_ids = token_ids.clamp_min(0)
    selected = log_probabilities.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
    mask = token_ids != ignore_index
    return (selected * mask).sum(dim=-1)


def select_candidates(values: Sequence[str], index: int) -> str:
    if index < 0 or index >= len(values):
        raise IndexError("candidate index is out of range")
    return values[index]
