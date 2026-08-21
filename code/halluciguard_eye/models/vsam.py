"""Visual-Semantic Alignment Module. Ref: Sec. III-C.1 and Eq. (3)-(8)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from halluciguard_eye.models.layers import masked_mean


@dataclass(frozen=True)
class AlignmentOutput:
    visual_to_text: Tensor
    text_to_visual: Tensor
    visual_context: Tensor
    text_context: Tensor
    score: Tensor
    visual_pooled: Tensor
    text_pooled: Tensor


class DirectedCrossAttention(nn.Module):
    def __init__(
        self, query_width: int, key_width: int, attention_width: int, output_width: int
    ) -> None:
        super().__init__()
        if min(query_width, key_width, attention_width, output_width) <= 0:
            raise ValueError("cross-attention widths must be positive")
        self.attention_width = attention_width
        self.query = nn.Linear(query_width, attention_width, bias=False)
        self.key = nn.Linear(key_width, attention_width, bias=False)
        self.value = nn.Linear(key_width, output_width, bias=False)

    def forward(
        self, query: Tensor, key_value: Tensor, key_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("cross-attention inputs must be three-dimensional")
        scores = self.query(query) @ self.key(key_value).transpose(-1, -2)
        scores = scores / math.sqrt(self.attention_width)
        if key_mask is not None:
            if key_mask.shape != key_value.shape[:2]:
                raise ValueError("key mask shape mismatch")
            scores = scores.masked_fill(~key_mask[:, None].bool(), torch.finfo(scores.dtype).min)
        weights = functional.softmax(scores.float(), dim=-1).to(scores.dtype)
        return weights @ self.value(key_value), weights


class VisualSemanticAlignmentModule(nn.Module):
    def __init__(
        self,
        visual_width: int,
        text_width: int,
        attention_width: int = 64,
        shared_width: int = 512,
    ) -> None:
        super().__init__()
        self.visual_width = visual_width
        self.text_width = text_width
        self.shared_width = shared_width
        self.visual_to_text = DirectedCrossAttention(
            visual_width, text_width, attention_width, shared_width
        )
        self.text_to_visual = DirectedCrossAttention(
            text_width, visual_width, attention_width, shared_width
        )
        self.visual_pool_projection = nn.Linear(visual_width, shared_width)
        self.text_pool_projection = nn.Linear(text_width, shared_width)
        self.score_network = nn.Sequential(
            nn.Linear(shared_width * 3, shared_width),
            nn.GELU(),
            nn.Linear(shared_width, 1),
        )

    def forward(
        self,
        visual: Tensor,
        text: Tensor,
        visual_mask: Tensor | None = None,
        text_mask: Tensor | None = None,
    ) -> AlignmentOutput:
        if visual.ndim != 3 or visual.shape[-1] != self.visual_width:
            raise ValueError("visual token shape mismatch")
        if text.ndim != 3 or text.shape[-1] != self.text_width:
            raise ValueError("text token shape mismatch")
        visual_context, visual_to_text = self.visual_to_text(visual, text, text_mask)
        text_context, text_to_visual = self.text_to_visual(text, visual, visual_mask)
        visual_summary = masked_mean(visual_context, visual_mask)
        text_summary = masked_mean(text_context, text_mask)
        agreement = visual_summary * text_summary
        score_features = torch.cat((visual_summary, text_summary, agreement), dim=-1)
        score = torch.sigmoid(self.score_network(score_features)).squeeze(-1)
        visual_pooled = functional.normalize(
            self.visual_pool_projection(masked_mean(visual, visual_mask)), dim=-1
        )
        text_pooled = functional.normalize(
            self.text_pool_projection(masked_mean(text, text_mask)), dim=-1
        )
        return AlignmentOutput(
            visual_to_text,
            text_to_visual,
            visual_context,
            text_context,
            score,
            visual_pooled,
            text_pooled,
        )

    def contrastive_logits(
        self, visual_pooled: Tensor, text_pooled: Tensor, temperature: float = 0.07
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("contrastive temperature must be positive")
        if visual_pooled.shape != text_pooled.shape or visual_pooled.ndim != 2:
            raise ValueError("pooled visual and text tensors must match")
        return visual_pooled @ text_pooled.T / temperature

    def contrastive_loss(
        self, visual_pooled: Tensor, text_pooled: Tensor, temperature: float = 0.07
    ) -> Tensor:
        logits = self.contrastive_logits(visual_pooled, text_pooled, temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        visual_loss = functional.cross_entropy(logits, labels)
        text_loss = functional.cross_entropy(logits.T, labels)
        return 0.5 * (visual_loss + text_loss)


def alignment_heatmap(
    output: AlignmentOutput, visual_tokens_excluding_class: bool = True
) -> Tensor:
    weights = output.text_to_visual.mean(dim=1)
    if visual_tokens_excluding_class and weights.shape[-1] > 1:
        weights = weights[..., 1:]
    return weights.mean(dim=1)


def paired_cosine_similarity(visual: Tensor, text: Tensor) -> Tensor:
    if visual.shape != text.shape:
        raise ValueError("paired tensors must have equal shapes")
    return functional.cosine_similarity(visual, text, dim=-1)
