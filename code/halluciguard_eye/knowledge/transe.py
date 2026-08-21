"""TransE embeddings and margin loss. Ref: Sec. III-B.3 and Eq. (2)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from halluciguard_eye.knowledge.graph import Triplet


@dataclass(frozen=True)
class CorruptedBatch:
    positive: Tensor
    negative: Tensor

    def __post_init__(self) -> None:
        if self.positive.ndim != 2 or self.positive.shape[-1] != 3:
            raise ValueError("positive triplets must have shape N x 3")
        if self.negative.shape != self.positive.shape:
            raise ValueError("positive and negative triplet shapes must match")


class TransE(nn.Module):
    def __init__(
        self, entity_count: int, relation_count: int, dimension: int = 256, margin: float = 1.0
    ) -> None:
        super().__init__()
        if entity_count <= 0 or relation_count <= 0 or dimension <= 0:
            raise ValueError("TransE dimensions must be positive")
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.dimension = dimension
        self.margin = margin
        self.entity_embeddings = nn.Embedding(entity_count, dimension)
        self.relation_embeddings = nn.Embedding(relation_count, dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6.0 / self.dimension**0.5
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        self.normalize_entities()

    @torch.no_grad()
    def normalize_entities(self) -> None:
        self.entity_embeddings.weight.copy_(
            functional.normalize(self.entity_embeddings.weight, dim=-1)
        )

    def distance(self, triplets: Tensor) -> Tensor:
        if triplets.ndim != 2 or triplets.shape[-1] != 3:
            raise ValueError("triplets must have shape N x 3")
        head = self.entity_embeddings(triplets[:, 0])
        relation = self.relation_embeddings(triplets[:, 1])
        tail = self.entity_embeddings(triplets[:, 2])
        return torch.linalg.vector_norm(head + relation - tail, ord=2, dim=-1)

    def forward(self, positive: Tensor, negative: Tensor) -> Tensor:
        if positive.shape != negative.shape:
            raise ValueError("positive and negative batches must match")
        positive_distance = self.distance(positive)
        negative_distance = self.distance(negative)
        return functional.relu(self.margin + positive_distance - negative_distance).mean()

    def score_all_tails(self, heads: Tensor, relations: Tensor) -> Tensor:
        query = self.entity_embeddings(heads) + self.relation_embeddings(relations)
        tails = self.entity_embeddings.weight
        return -torch.cdist(query, tails, p=2)

    def score_all_heads(self, relations: Tensor, tails: Tensor) -> Tensor:
        query = self.entity_embeddings(tails) - self.relation_embeddings(relations)
        heads = self.entity_embeddings.weight
        return -torch.cdist(query, heads, p=2)

    def entity_vectors(self, identifiers: Tensor) -> Tensor:
        return functional.normalize(self.entity_embeddings(identifiers), dim=-1)

    def relation_vectors(self, identifiers: Tensor) -> Tensor:
        return self.relation_embeddings(identifiers)


def triplets_to_tensor(triplets: Iterable[Triplet], device: torch.device | str = "cpu") -> Tensor:
    values = [[triplet.head, triplet.relation, triplet.tail] for triplet in triplets]
    if not values:
        return torch.empty((0, 3), dtype=torch.long, device=device)
    return torch.tensor(values, dtype=torch.long, device=device)


def corrupt_triplets(
    positive: Tensor,
    entity_count: int,
    generator: torch.Generator,
) -> Tensor:
    if positive.ndim != 2 or positive.shape[-1] != 3:
        raise ValueError("positive triplets must have shape N x 3")
    if entity_count <= 1:
        raise ValueError("entity count must exceed one")
    negative = positive.clone()
    replace_head = torch.rand(positive.shape[0], generator=generator, device=positive.device) < 0.5
    replacements = torch.randint(
        0,
        entity_count - 1,
        (positive.shape[0],),
        generator=generator,
        device=positive.device,
    )
    original = torch.where(replace_head, positive[:, 0], positive[:, 2])
    replacements += (replacements >= original).to(replacements.dtype)
    negative[replace_head, 0] = replacements[replace_head]
    negative[~replace_head, 2] = replacements[~replace_head]
    return negative


def negative_sampling_batch(
    positive: Tensor,
    entity_count: int,
    ratio: int,
    generator: torch.Generator,
) -> CorruptedBatch:
    if ratio <= 0:
        raise ValueError("negative sampling ratio must be positive")
    expanded = positive.repeat_interleave(ratio, dim=0)
    negative = corrupt_triplets(expanded, entity_count, generator)
    return CorruptedBatch(expanded, negative)


@torch.no_grad()
def link_prediction_ranks(model: TransE, triplets: Tensor, batch_size: int = 256) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    ranks: list[Tensor] = []
    for start in range(0, len(triplets), batch_size):
        batch = triplets[start : start + batch_size]
        scores = model.score_all_tails(batch[:, 0], batch[:, 1])
        target_scores = scores.gather(1, batch[:, 2, None])
        ranks.append((scores > target_scores).sum(dim=1) + 1)
    return torch.cat(ranks) if ranks else torch.empty(0, dtype=torch.long, device=triplets.device)


@torch.no_grad()
def link_prediction_metrics(model: TransE, triplets: Tensor) -> dict[str, float]:
    ranks = link_prediction_ranks(model, triplets).float()
    if not len(ranks):
        raise ValueError("link prediction requires triplets")
    return {
        "mean_reciprocal_rank": float((1.0 / ranks).mean().item()),
        "hits_at_10": float((ranks <= 10).float().mean().item()),
        "mean_rank": float(ranks.mean().item()),
    }
