"""Clinical Factuality Grounding Network. Ref: Sec. III-C.2 and Eq. (9)-(18)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from halluciguard_eye.models.layers import EvidenceEncoder, FusionOutput, GatedEvidenceFusion


@dataclass(frozen=True)
class GroundingOutput:
    query: Tensor
    grounding_score: Tensor
    evidence: Tensor
    fusion: FusionOutput


class ClinicalFactualityGroundingNetwork(nn.Module):
    def __init__(
        self,
        entity_count: int,
        knowledge_width: int,
        hidden_width: int,
        attention_width: int,
        evidence_vocabulary: int,
        evidence_layers: int,
        evidence_heads: int,
        max_evidence_tokens: int,
    ) -> None:
        super().__init__()
        if entity_count <= 0:
            raise ValueError("entity count must be positive")
        self.entity_count = entity_count
        self.knowledge_width = knowledge_width
        self.entity_embeddings = nn.Embedding(entity_count, knowledge_width)
        self.claim_projection = nn.Linear(hidden_width, knowledge_width)
        self.evidence_encoder = EvidenceEncoder(
            evidence_vocabulary,
            hidden_width,
            evidence_layers,
            evidence_heads,
            max_evidence_tokens,
        )
        self.fusion = GatedEvidenceFusion(hidden_width, attention_width, hidden_width)
        nn.init.normal_(self.entity_embeddings.weight, std=0.02)

    def entity_query(self, entity_ids: Tensor, entity_mask: Tensor | None = None) -> Tensor:
        if entity_ids.ndim != 2:
            raise ValueError("entity identifiers must have shape batch x entities")
        if torch.any(entity_ids < 0) or torch.any(entity_ids >= self.entity_count):
            raise ValueError("entity identifier is out of range")
        embeddings = functional.normalize(self.entity_embeddings(entity_ids), dim=-1)
        if entity_mask is None:
            return functional.normalize(embeddings.mean(dim=1), dim=-1)
        if entity_mask.shape != entity_ids.shape:
            raise ValueError("entity mask shape mismatch")
        weights = entity_mask.to(embeddings.dtype)
        query = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        query = query / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return functional.normalize(query, dim=-1)

    def evidence_similarity(self, claim_hidden: Tensor, evidence_entity_ids: Tensor) -> Tensor:
        if claim_hidden.ndim != 3:
            raise ValueError("claim hidden states must be three-dimensional")
        if evidence_entity_ids.ndim != 2 or evidence_entity_ids.shape[0] != claim_hidden.shape[0]:
            raise ValueError("evidence entity shape mismatch")
        claims = functional.normalize(self.claim_projection(claim_hidden), dim=-1)
        evidence = functional.normalize(self.entity_embeddings(evidence_entity_ids), dim=-1)
        similarities = claims @ evidence.transpose(-1, -2)
        return similarities.max(dim=-1).values.mean(dim=-1)

    def forward(
        self,
        hidden: Tensor,
        alignment_context: Tensor,
        response_entity_ids: Tensor,
        evidence_entity_ids: Tensor,
        evidence_token_ids: Tensor,
        response_entity_mask: Tensor | None = None,
        evidence_token_mask: Tensor | None = None,
    ) -> GroundingOutput:
        query = self.entity_query(response_entity_ids, response_entity_mask)
        grounding_score = self.evidence_similarity(hidden, evidence_entity_ids)
        evidence = self.evidence_encoder(evidence_token_ids, evidence_token_mask)
        fusion = self.fusion(hidden, alignment_context, evidence)
        return GroundingOutput(query, grounding_score, evidence, fusion)

    def sync_knowledge_embeddings(self, source: Tensor) -> None:
        if source.shape != self.entity_embeddings.weight.shape:
            raise ValueError("knowledge embedding shape mismatch")
        with torch.no_grad():
            self.entity_embeddings.weight.copy_(source)


def pad_entity_sequences(
    sequences: Sequence[Sequence[int]], padding: int = 0
) -> tuple[Tensor, Tensor]:
    if not sequences:
        raise ValueError("entity sequences cannot be empty")
    length = max(len(sequence) for sequence in sequences)
    if length == 0:
        length = 1
    identifiers = torch.full((len(sequences), length), padding, dtype=torch.long)
    mask = torch.zeros((len(sequences), length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        if sequence:
            identifiers[row, : len(sequence)] = torch.tensor(sequence)
            mask[row, : len(sequence)] = True
    return identifiers, mask
