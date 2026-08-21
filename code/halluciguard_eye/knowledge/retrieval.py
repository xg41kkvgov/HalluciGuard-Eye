"""Query-driven OCKG retrieval. Ref: Sec. III-C.2 and Eq. (9)-(12)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from halluciguard_eye.knowledge.graph import ClinicalKnowledgeGraph, Triplet


@dataclass(frozen=True)
class ScoredTriplet:
    triplet: Triplet
    score: float
    hop: int

    def __post_init__(self) -> None:
        if not -1 <= self.score <= 1:
            raise ValueError("retrieval score must be in [-1, 1]")
        if self.hop < 0:
            raise ValueError("hop must be nonnegative")


@dataclass(frozen=True)
class RetrievedSubgraph:
    query: Tensor
    items: tuple[ScoredTriplet, ...]
    linearized: tuple[str, ...]

    def triplets(self) -> tuple[Triplet, ...]:
        return tuple(item.triplet for item in self.items)

    def entity_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {value for item in self.items for value in (item.triplet.head, item.triplet.tail)}
            )
        )

    def top(self, count: int) -> RetrievedSubgraph:
        if count < 0:
            raise ValueError("count cannot be negative")
        selected = self.items[:count]
        return RetrievedSubgraph(self.query, selected, self.linearized[:count])


class KnowledgeRetriever:
    def __init__(
        self,
        graph: ClinicalKnowledgeGraph,
        entity_embeddings: Tensor,
        relation_embeddings: Tensor,
        retrieval_threshold: float = 0.75,
        expansion_threshold: float = 0.6,
        hops: int = 2,
    ) -> None:
        if entity_embeddings.ndim != 2 or relation_embeddings.ndim != 2:
            raise ValueError("embedding matrices must be two-dimensional")
        if len(entity_embeddings) <= max(graph.entities, default=-1):
            raise ValueError("entity embedding matrix does not cover the graph")
        if len(relation_embeddings) <= max(graph.relations, default=-1):
            raise ValueError("relation embedding matrix does not cover the graph")
        if entity_embeddings.shape[1] != relation_embeddings.shape[1]:
            raise ValueError("entity and relation embedding widths must match")
        if not 0 <= retrieval_threshold <= 1 or not 0 <= expansion_threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")
        if hops < 0:
            raise ValueError("hop count cannot be negative")
        self.graph = graph
        self.entity_embeddings = functional.normalize(entity_embeddings, dim=-1)
        self.relation_embeddings = functional.normalize(relation_embeddings, dim=-1)
        self.retrieval_threshold = retrieval_threshold
        self.expansion_threshold = expansion_threshold
        self.hops = hops

    def entity_query(self, entity_ids: Sequence[int]) -> Tensor:
        if not entity_ids:
            raise ValueError("entity query requires at least one entity")
        identifiers = torch.tensor(
            entity_ids, dtype=torch.long, device=self.entity_embeddings.device
        )
        return functional.normalize(self.entity_embeddings[identifiers].mean(dim=0), dim=-1)

    def response_query(
        self, text: str, fallback_encoder: Callable[[str], Tensor] | None = None
    ) -> Tensor:
        entities = self.graph.find_entities(text)
        if entities:
            return self.entity_query([entity.identifier for entity in entities])
        if fallback_encoder is None:
            raise ValueError(
                "response contains no known entities and no fallback encoder was supplied"
            )
        query = fallback_encoder(text)
        if query.shape != (self.entity_embeddings.shape[1],):
            raise ValueError("fallback query has the wrong shape")
        return functional.normalize(query, dim=-1)

    def initial_scores(self, query: Tensor) -> dict[Triplet, float]:
        if query.shape != (self.entity_embeddings.shape[1],):
            raise ValueError("query has the wrong shape")
        scores: dict[Triplet, float] = {}
        for triplet in self.graph.triplets:
            head = float(torch.dot(self.entity_embeddings[triplet.head], query).item())
            tail = float(torch.dot(self.entity_embeddings[triplet.tail], query).item())
            score = max(head, tail)
            if score > self.retrieval_threshold:
                scores[triplet] = score
        return scores

    def relation_score(self, triplet: Triplet, query: Tensor) -> float:
        head = self.entity_embeddings[triplet.head]
        tail = self.entity_embeddings[triplet.tail]
        relation = self.relation_embeddings[triplet.relation]
        structural = functional.cosine_similarity((head + relation)[None], tail[None]).item()
        relevance = max(torch.dot(head, query).item(), torch.dot(tail, query).item())
        relation_weight = self.graph.relation(triplet.relation).weight
        score = 0.5 * structural + 0.5 * relevance
        return float(max(-1.0, min(1.0, score * min(1.0, relation_weight))))

    def expand(self, query: Tensor, initial: Mapping[Triplet, float]) -> tuple[ScoredTriplet, ...]:
        selected: dict[Triplet, ScoredTriplet] = {
            triplet: ScoredTriplet(triplet, score, 0) for triplet, score in initial.items()
        }
        frontier = {
            identifier for triplet in initial for identifier in (triplet.head, triplet.tail)
        }
        for hop in range(1, self.hops + 1):
            candidates = {
                triplet
                for identifier in frontier
                for triplet in self.graph.incident(identifier)
                if triplet not in selected
            }
            accepted: list[ScoredTriplet] = []
            for triplet in candidates:
                score = self.relation_score(triplet, query)
                if score > self.expansion_threshold:
                    accepted.append(ScoredTriplet(triplet, score, hop))
            if not accepted:
                break
            selected.update((item.triplet, item) for item in accepted)
            frontier = {
                value for item in accepted for value in (item.triplet.head, item.triplet.tail)
            }
        return tuple(
            sorted(selected.values(), key=lambda item: (-item.score, item.hop, item.triplet))
        )

    def retrieve(self, query: Tensor, limit: int | None = None) -> RetrievedSubgraph:
        initial = self.initial_scores(query)
        items = self.expand(query, initial)
        if limit is not None:
            if limit < 0:
                raise ValueError("retrieval limit cannot be negative")
            items = items[:limit]
        linearized = self.graph.linearize(item.triplet for item in items)
        return RetrievedSubgraph(query, items, linearized)

    def claim_grounding_score(
        self, claim_embeddings: Tensor, subgraph: RetrievedSubgraph
    ) -> Tensor:
        if (
            claim_embeddings.ndim != 2
            or claim_embeddings.shape[1] != self.entity_embeddings.shape[1]
        ):
            raise ValueError("claim embeddings have the wrong shape")
        entity_ids = subgraph.entity_ids()
        if not entity_ids:
            return claim_embeddings.new_zeros(())
        identifiers = torch.tensor(
            entity_ids, dtype=torch.long, device=self.entity_embeddings.device
        )
        evidence = self.entity_embeddings[identifiers]
        claims = functional.normalize(claim_embeddings, dim=-1)
        similarities = claims @ evidence.T
        return similarities.max(dim=1).values.mean()


def mean_entity_query(entity_embeddings: Tensor, entity_ids: Iterable[int]) -> Tensor:
    identifiers = tuple(entity_ids)
    if not identifiers:
        raise ValueError("entity identifiers cannot be empty")
    index = torch.tensor(identifiers, dtype=torch.long, device=entity_embeddings.device)
    return functional.normalize(entity_embeddings[index].mean(dim=0), dim=-1)
