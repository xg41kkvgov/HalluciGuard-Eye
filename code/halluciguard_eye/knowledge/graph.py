"""OCKG typed graph. Ref: Sec. III-B, Eq. (2), and Table 1."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EntityType(str, Enum):
    ANATOMY = "Anatomy"
    PATHOLOGY = "Pathology"
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    PROCEDURE = "Procedure"
    FINDING = "Finding"


class RelationFamily(str, Enum):
    HIERARCHICAL = "Hierarchical"
    CAUSAL = "Causal"
    SPATIAL = "Spatial"
    CLINICAL = "Clinical"


@dataclass(frozen=True)
class Entity:
    identifier: int
    name: str
    entity_type: EntityType
    aliases: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if self.identifier < 0:
            raise ValueError("entity identifier must be nonnegative")
        if not self.name.strip():
            raise ValueError("entity name cannot be empty")
        normalized = [alias.strip().casefold() for alias in self.aliases]
        if any(not alias for alias in normalized):
            raise ValueError("entity aliases cannot be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("entity aliases must be unique")

    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class Relation:
    identifier: int
    name: str
    family: RelationFamily
    inverse_name: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.identifier < 0:
            raise ValueError("relation identifier must be nonnegative")
        if not self.name.strip():
            raise ValueError("relation name cannot be empty")
        if self.weight <= 0:
            raise ValueError("relation weight must be positive")


@dataclass(frozen=True, order=True)
class Triplet:
    head: int
    relation: int
    tail: int
    source: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.head < 0 or self.relation < 0 or self.tail < 0:
            raise ValueError("triplet identifiers must be nonnegative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("triplet confidence must be in [0, 1]")


class ClinicalKnowledgeGraph:
    def __init__(
        self,
        entities: Iterable[Entity],
        relations: Iterable[Relation],
        triplets: Iterable[Triplet],
    ) -> None:
        entity_items = tuple(entities)
        relation_items = tuple(relations)
        triplet_items = tuple(triplets)
        self.entities = {entity.identifier: entity for entity in entity_items}
        self.relations = {relation.identifier: relation for relation in relation_items}
        self.triplets = tuple(sorted(set(triplet_items)))
        self._validate(entity_items, relation_items)
        self._outgoing: dict[int, list[Triplet]] = {identifier: [] for identifier in self.entities}
        self._incoming: dict[int, list[Triplet]] = {identifier: [] for identifier in self.entities}
        self._pair_relations: dict[tuple[int, int], list[int]] = {}
        for triplet in self.triplets:
            self._outgoing[triplet.head].append(triplet)
            self._incoming[triplet.tail].append(triplet)
            self._pair_relations.setdefault((triplet.head, triplet.tail), []).append(
                triplet.relation
            )

    def _validate(self, entities: Sequence[Entity], relations: Sequence[Relation]) -> None:
        if len(self.entities) != len(entities):
            raise ValueError("entity identifiers must be unique")
        if len(self.relations) != len(relations):
            raise ValueError("relation identifiers must be unique")
        normalized_names = [entity.name.casefold() for entity in entities]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("entity names must be unique")
        relation_names = [relation.name.casefold() for relation in relations]
        if len(relation_names) != len(set(relation_names)):
            raise ValueError("relation names must be unique")
        for triplet in self.triplets:
            if triplet.head not in self.entities or triplet.tail not in self.entities:
                raise ValueError("triplet references an unknown entity")
            if triplet.relation not in self.relations:
                raise ValueError("triplet references an unknown relation")

    def __len__(self) -> int:
        return len(self.triplets)

    def entity(self, identifier: int) -> Entity:
        return self.entities[identifier]

    def relation(self, identifier: int) -> Relation:
        return self.relations[identifier]

    def outgoing(self, identifier: int) -> tuple[Triplet, ...]:
        return tuple(self._outgoing.get(identifier, ()))

    def incoming(self, identifier: int) -> tuple[Triplet, ...]:
        return tuple(self._incoming.get(identifier, ()))

    def incident(self, identifier: int) -> tuple[Triplet, ...]:
        return tuple(dict.fromkeys((*self.outgoing(identifier), *self.incoming(identifier))))

    def neighbors(self, identifier: int) -> tuple[int, ...]:
        values = {triplet.tail for triplet in self.outgoing(identifier)}
        values.update(triplet.head for triplet in self.incoming(identifier))
        return tuple(sorted(values))

    def relation_ids(self, head: int, tail: int) -> tuple[int, ...]:
        return tuple(self._pair_relations.get((head, tail), ()))

    def find_entities(self, text: str) -> tuple[Entity, ...]:
        normalized = text.casefold()
        matches: list[Entity] = []
        for entity in self.entities.values():
            if any(name.casefold() in normalized for name in entity.names()):
                matches.append(entity)
        return tuple(sorted(matches, key=lambda entity: entity.identifier))

    def shortest_path(
        self, source: int, target: int, max_length: int
    ) -> tuple[Triplet, ...] | None:
        if source not in self.entities or target not in self.entities:
            raise KeyError("source and target must be graph entities")
        if max_length < 0:
            raise ValueError("maximum path length cannot be negative")
        if source == target:
            return ()
        queue: deque[tuple[int, tuple[Triplet, ...]]] = deque([(source, ())])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_length:
                continue
            for triplet in self.outgoing(current):
                candidate = (*path, triplet)
                if triplet.tail == target:
                    return candidate
                if triplet.tail not in visited:
                    visited.add(triplet.tail)
                    queue.append((triplet.tail, candidate))
        return None

    def reachable(self, source: int, max_length: int) -> tuple[int, ...]:
        visited = {source}
        frontier = {source}
        for _ in range(max_length):
            next_frontier = {
                neighbor for current in frontier for neighbor in self.neighbors(current)
            } - visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        visited.remove(source)
        return tuple(sorted(visited))

    def subgraph(self, triplets: Iterable[Triplet]) -> ClinicalKnowledgeGraph:
        selected = tuple(dict.fromkeys(triplets))
        entity_ids = {item for triplet in selected for item in (triplet.head, triplet.tail)}
        relation_ids = {triplet.relation for triplet in selected}
        return ClinicalKnowledgeGraph(
            (self.entities[identifier] for identifier in sorted(entity_ids)),
            (self.relations[identifier] for identifier in sorted(relation_ids)),
            selected,
        )

    def statistics(self) -> dict[str, object]:
        entity_types = {kind.value: 0 for kind in EntityType}
        relation_families = {family.value: 0 for family in RelationFamily}
        for entity in self.entities.values():
            entity_types[entity.entity_type.value] += 1
        for triplet in self.triplets:
            relation_families[self.relations[triplet.relation].family.value] += 1
        degree = sum(len(self.incident(identifier)) for identifier in self.entities)
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "triplets": len(self.triplets),
            "average_degree": degree / len(self.entities) if self.entities else 0.0,
            "entity_types": entity_types,
            "relation_families": relation_families,
        }

    def linearize(self, triplets: Iterable[Triplet]) -> tuple[str, ...]:
        return tuple(
            " ".join(
                (
                    self.entity(item.head).name,
                    self.relation(item.relation).name,
                    self.entity(item.tail).name,
                )
            )
            for item in triplets
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "entities": [
                {
                    "identifier": entity.identifier,
                    "name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "aliases": list(entity.aliases),
                    "source": entity.source,
                }
                for entity in self.entities.values()
            ],
            "relations": [
                {
                    "identifier": relation.identifier,
                    "name": relation.name,
                    "family": relation.family.value,
                    "inverse_name": relation.inverse_name,
                    "weight": relation.weight,
                }
                for relation in self.relations.values()
            ],
            "triplets": [
                {
                    "head": triplet.head,
                    "relation": triplet.relation,
                    "tail": triplet.tail,
                    "source": triplet.source,
                    "confidence": triplet.confidence,
                }
                for triplet in self.triplets
            ],
        }
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str | Path) -> ClinicalKnowledgeGraph:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, Mapping):
            raise ValueError("knowledge graph document must be a mapping")
        entities = (
            Entity(
                identifier=int(item["identifier"]),
                name=str(item["name"]),
                entity_type=EntityType(str(item["entity_type"])),
                aliases=tuple(str(value) for value in item.get("aliases", ())),
                source=str(item.get("source", "")),
            )
            for item in raw["entities"]
        )
        relations = (
            Relation(
                identifier=int(item["identifier"]),
                name=str(item["name"]),
                family=RelationFamily(str(item["family"])),
                inverse_name=str(item.get("inverse_name", "")),
                weight=float(item.get("weight", 1.0)),
            )
            for item in raw["relations"]
        )
        triplets = (
            Triplet(
                head=int(item["head"]),
                relation=int(item["relation"]),
                tail=int(item["tail"]),
                source=str(item.get("source", "")),
                confidence=float(item.get("confidence", 1.0)),
            )
            for item in raw["triplets"]
        )
        return cls(entities, relations, triplets)


def validate_paper_statistics(graph: ClinicalKnowledgeGraph) -> dict[str, bool]:
    statistics = graph.statistics()
    return {
        "entity_count": statistics["entities"] == 47_832,
        "relation_type_count": statistics["relations"] == 23,
        "triplet_count": statistics["triplets"] == 156_294,
    }
