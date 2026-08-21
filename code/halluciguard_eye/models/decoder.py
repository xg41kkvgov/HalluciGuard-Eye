"""Vicuna-compatible decoder interface. Ref: Sec. III-J and Fig. 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from halluciguard_eye.models.layers import RMSNorm, TransformerBlock


@dataclass(frozen=True)
class DecoderOutput:
    hidden: Tensor
    logits: Tensor
    attention_maps: tuple[Tensor, ...]


class ClinicalLanguageDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        width: int,
        layers: int,
        heads: int,
        max_tokens: int,
        visual_width: int,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0 or width <= 0 or layers <= 0 or max_tokens <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.vocabulary_size = vocabulary_size
        self.width = width
        self.max_tokens = max_tokens
        self.token_embedding = nn.Embedding(vocabulary_size, width)
        self.position_embedding = nn.Parameter(torch.empty(1, max_tokens, width))
        self.visual_projection = nn.Linear(visual_width, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads, mlp_ratio=2.7, rms_norm=True) for _ in range(layers)
        )
        self.norm = RMSNorm(width)
        self.output = nn.Linear(width, vocabulary_size, bias=False)
        if tie_embeddings:
            self.output.weight = self.token_embedding.weight
        nn.init.normal_(self.position_embedding, std=0.02)

    def embed(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_tokens:
            raise ValueError("token identifiers have an invalid shape")
        return self.token_embedding(token_ids) + self.position_embedding[:, : token_ids.shape[1]]

    def forward(
        self,
        token_ids: Tensor,
        visual_context: Tensor | None = None,
        return_attention: bool = False,
    ) -> DecoderOutput:
        hidden = self.embed(token_ids)
        if visual_context is not None:
            if visual_context.ndim != 2 or visual_context.shape[0] != hidden.shape[0]:
                raise ValueError("visual context must have shape batch x width")
            hidden = hidden + self.visual_projection(visual_context)[:, None]
        attention_maps: list[Tensor] = []
        for block in self.blocks:
            hidden, attention = block(hidden, causal=True)
            if return_attention:
                attention_maps.append(attention)
        hidden = self.norm(hidden)
        return DecoderOutput(hidden, self.output(hidden), tuple(attention_maps))

    @torch.no_grad()
    def generate(
        self,
        prefix: Tensor,
        visual_context: Tensor | None,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("maximum new token count cannot be negative")
        if temperature <= 0:
            raise ValueError("generation temperature must be positive")
        tokens = prefix
        for _ in range(max_new_tokens):
            if tokens.shape[1] >= self.max_tokens:
                break
            logits = self(tokens, visual_context).logits[:, -1] / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                probabilities = torch.softmax(values, dim=-1)
                selected = torch.multinomial(probabilities, 1, generator=generator)
                next_token = indices.gather(1, selected)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    @torch.no_grad()
    def beam_candidates(
        self,
        prefix: Tensor,
        visual_context: Tensor | None,
        candidate_count: int,
        max_new_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        if prefix.shape[0] != 1:
            raise ValueError("beam candidate generation supports one sample")
        if candidate_count <= 0:
            raise ValueError("candidate count must be positive")
        beams = [(prefix, 0.0)]
        for _ in range(max_new_tokens):
            expanded: list[tuple[Tensor, float]] = []
            for tokens, score in beams:
                logits = self(tokens, visual_context).logits[:, -1]
                log_probabilities = torch.log_softmax(logits, dim=-1)
                values, indices = torch.topk(log_probabilities, candidate_count)
                for value, index in zip(values[0], indices[0], strict=False):
                    appended = torch.cat((tokens, index.view(1, 1)), dim=1)
                    expanded.append((appended, score + float(value.item())))
            beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:candidate_count]
        tokens = torch.cat([item[0] for item in beams], dim=0)
        scores = torch.tensor([item[1] for item in beams], device=prefix.device)
        return tokens, scores
