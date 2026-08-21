"""Transformer and fusion primitives. Ref: Fig. 2 and Eq. (15)-(18)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def validate_sequence(tensor: Tensor, width: int, name: str) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape batch x tokens x width")
    if tensor.shape[-1] != width:
        raise ValueError(f"{name} has width {tensor.shape[-1]}, expected {width}")


def masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    if values.ndim < 2:
        raise ValueError("masked mean requires a sequence axis")
    if mask is None:
        return values.mean(dim=1)
    if mask.shape != values.shape[:2]:
        raise ValueError("mask shape must match the first two value dimensions")
    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (values * weights.unsqueeze(-1)).sum(dim=1) / denominator


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if width <= 0 or epsilon <= 0:
            raise ValueError("RMSNorm arguments must be positive")
        self.width = width
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.width:
            raise ValueError("RMSNorm input width mismatch")
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance.to(values.dtype) + self.epsilon)
        return normalized * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden_width or int(math.ceil(8 * width / 3 / 256) * 256)
        if width <= 0 or hidden <= 0:
            raise ValueError("SwiGLU widths must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = functional.silu(self.gate(values)) * self.value(values)
        return self.output(self.dropout(hidden))


class MultiHeadAttention(nn.Module):
    def __init__(
        self, width: int, heads: int, key_width: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("attention width must be divisible by head count")
        projected = key_width or width
        if projected % heads:
            raise ValueError("projected width must be divisible by head count")
        self.width = width
        self.heads = heads
        self.projected = projected
        self.head_width = projected // heads
        self.query = nn.Linear(width, projected, bias=False)
        self.key = nn.Linear(width, projected, bias=False)
        self.value = nn.Linear(width, projected, bias=False)
        self.output = nn.Linear(projected, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, values: Tensor) -> Tensor:
        batch, tokens, _ = values.shape
        return values.view(batch, tokens, self.heads, self.head_width).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        validate_sequence(query, self.width, "query")
        validate_sequence(key_value, self.width, "key_value")
        q = self._heads(self.query(query))
        k = self._heads(self.key(key_value))
        v = self._heads(self.value(key_value))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_width)
        if causal:
            query_tokens = query.shape[1]
            key_tokens = key_value.shape[1]
            causal_mask = torch.ones(
                query_tokens, key_tokens, dtype=torch.bool, device=query.device
            ).triu(1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            if attention_mask.ndim != 4:
                raise ValueError("attention mask must have two or four dimensions")
            scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)
        weights = functional.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = self.dropout(weights)
        context = weights @ v
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(query.shape[0], query.shape[1], self.projected)
        )
        return self.output(context), weights


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        norm = RMSNorm if rms_norm else nn.LayerNorm
        self.norm_attention = norm(width)
        self.attention = MultiHeadAttention(width, heads, dropout=dropout)
        self.norm_mlp = norm(width)
        hidden = max(width, int(width * mlp_ratio))
        self.mlp = (
            SwiGLU(width, hidden, dropout)
            if rms_norm
            else nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, width),
            )
        )

    def forward(
        self, values: Tensor, mask: Tensor | None = None, causal: bool = False
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm_attention(values)
        attended, weights = self.attention(normalized, normalized, mask, causal)
        values = values + attended
        values = values + self.mlp(self.norm_mlp(values))
        return values, weights


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = MultiHeadAttention(width, heads, dropout=dropout)
        self.output_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(
        self, query: Tensor, context: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        attended, weights = self.attention(self.query_norm(query), self.context_norm(context), mask)
        output = query + attended
        return output + self.mlp(self.output_norm(output)), weights


@dataclass(frozen=True)
class FusionOutput:
    hidden: Tensor
    gate: Tensor
    fused_context: Tensor


class GatedEvidenceFusion(nn.Module):
    def __init__(self, hidden_width: int, attention_width: int, evidence_width: int) -> None:
        super().__init__()
        if hidden_width <= 0 or attention_width <= 0 or evidence_width <= 0:
            raise ValueError("fusion widths must be positive")
        self.hidden_width = hidden_width
        self.attention_projection = nn.Linear(attention_width, hidden_width)
        self.evidence_projection = nn.Linear(evidence_width, hidden_width)
        self.gate = nn.Linear(hidden_width * 3, hidden_width)
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_width * 2, hidden_width * 2),
            nn.GELU(),
            nn.Linear(hidden_width * 2, hidden_width),
        )

    def forward(self, hidden: Tensor, attention: Tensor, evidence: Tensor) -> FusionOutput:
        validate_sequence(hidden, self.hidden_width, "hidden")
        if attention.ndim != 2 or evidence.ndim != 2:
            raise ValueError("attention and evidence contexts must be batch matrices")
        attention_hidden = self.attention_projection(attention)
        evidence_hidden = self.evidence_projection(evidence)
        tokens = hidden.shape[1]
        attention_tokens = attention_hidden[:, None].expand(-1, tokens, -1)
        evidence_tokens = evidence_hidden[:, None].expand(-1, tokens, -1)
        gate = torch.sigmoid(
            self.gate(torch.cat((hidden, attention_tokens, evidence_tokens), dim=-1))
        )
        context = self.context_mlp(torch.cat((attention_tokens, evidence_tokens), dim=-1))
        fused = gate * hidden + (1.0 - gate) * context
        return FusionOutput(fused, gate, context)


class EvidenceEncoder(nn.Module):
    def __init__(
        self, vocabulary_size: int, width: int, layers: int, heads: int, max_tokens: int
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0 or max_tokens <= 0:
            raise ValueError("evidence encoder sizes must be positive")
        self.width = width
        self.max_tokens = max_tokens
        self.tokens = nn.Embedding(vocabulary_size, width)
        self.positions = nn.Parameter(torch.empty(1, max_tokens, width))
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads, mlp_ratio=4.0) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.positions, std=0.02)

    def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_tokens:
            raise ValueError("evidence token shape is invalid")
        hidden = self.tokens(token_ids) + self.positions[:, : token_ids.shape[1]]
        for block in self.blocks:
            hidden, _ = block(hidden, mask)
        return masked_mean(self.norm(hidden), mask)


def causal_mask(length: int, device: torch.device | str) -> Tensor:
    if length <= 0:
        raise ValueError("causal mask length must be positive")
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()
