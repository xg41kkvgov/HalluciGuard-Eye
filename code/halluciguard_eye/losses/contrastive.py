"""Image-text contrastive alignment. Ref: Eq. (7)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional


def cosine_similarity_matrix(left: Tensor, right: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("similarity inputs must be matrices with equal widths")
    return functional.normalize(left, dim=-1) @ functional.normalize(right, dim=-1).T


def image_text_contrastive_loss(
    visual: Tensor,
    text: Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> Tensor:
    if visual.shape != text.shape:
        raise ValueError("paired visual and text representations must match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = cosine_similarity_matrix(visual, text) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    visual_loss = functional.cross_entropy(logits, labels)
    if not symmetric:
        return visual_loss
    text_loss = functional.cross_entropy(logits.T, labels)
    return 0.5 * (visual_loss + text_loss)


def hard_negative_contrastive_loss(
    anchor: Tensor,
    positive: Tensor,
    negatives: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    if anchor.ndim != 2 or positive.shape != anchor.shape:
        raise ValueError("anchors and positives must be equal matrices")
    if (
        negatives.ndim != 3
        or negatives.shape[0] != anchor.shape[0]
        or negatives.shape[2] != anchor.shape[1]
    ):
        raise ValueError("negative shape must be batch x negatives x width")
    anchor = functional.normalize(anchor, dim=-1)
    positive = functional.normalize(positive, dim=-1)
    negatives = functional.normalize(negatives, dim=-1)
    positive_logits = (anchor * positive).sum(dim=-1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", anchor, negatives)
    logits = torch.cat((positive_logits, negative_logits), dim=-1) / temperature
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return functional.cross_entropy(logits, labels)


def alignment_recall_at_k(similarities: Tensor, k: int) -> Tensor:
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("paired similarity matrix must be square")
    if not 1 <= k <= similarities.shape[1]:
        raise ValueError("k is out of range")
    indices = torch.topk(similarities, k, dim=1).indices
    targets = torch.arange(similarities.shape[0], device=similarities.device)[:, None]
    return (indices == targets).any(dim=1).float().mean()
