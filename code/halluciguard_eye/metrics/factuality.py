"""HFS measurement. Ref: Sec. III-D and Eq. (19)-(22)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from halluciguard_eye.data.annotations import AnatomicalReference, PathologyReference
from halluciguard_eye.knowledge.graph import ClinicalKnowledgeGraph


@dataclass(frozen=True)
class FactualityComponents:
    anatomical_accuracy: float
    pathological_characterization: float
    clinical_recommendation_validity: float

    def __post_init__(self) -> None:
        values = (
            self.anatomical_accuracy,
            self.pathological_characterization,
            self.clinical_recommendation_validity,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("factuality components must be in [0, 1]")

    def hfs(self, weights: tuple[float, float, float] = (0.25, 0.40, 0.35)) -> float:
        return hierarchical_factuality_score(
            self.anatomical_accuracy,
            self.pathological_characterization,
            self.clinical_recommendation_validity,
            weights,
        )


def hierarchical_factuality_score(
    anatomical_accuracy: float,
    pathological_characterization: float,
    clinical_recommendation_validity: float,
    weights: tuple[float, float, float] = (0.25, 0.40, 0.35),
) -> float:
    values = (anatomical_accuracy, pathological_characterization, clinical_recommendation_validity)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("HFS components must be in [0, 1]")
    if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("HFS weights must be nonnegative and sum to one")
    return float(sum(weight * value for weight, value in zip(weights, values, strict=False)))


def anatomical_accuracy(verification: Sequence[bool]) -> float:
    if not verification:
        return 1.0
    return float(np.mean(np.asarray(verification, dtype=np.float64)))


def verify_anatomical_references(
    mentioned: Sequence[AnatomicalReference],
    detected: Sequence[AnatomicalReference],
    iou_threshold: float = 0.5,
) -> tuple[bool, ...]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in [0, 1]")
    results: list[bool] = []
    for reference in mentioned:
        matches = [
            candidate
            for candidate in detected
            if candidate.name.casefold() == reference.name.casefold() and candidate.visible
        ]
        results.append(
            any(reference.box.iou(candidate.box) >= iou_threshold for candidate in matches)
        )
    return tuple(results)


def pathological_characterization(
    response_pathologies: Sequence[PathologyReference],
    reference_pathologies: Sequence[PathologyReference],
    similarity: Callable[[PathologyReference, PathologyReference], float],
) -> float:
    if not response_pathologies:
        return 1.0 if not reference_pathologies else 0.0
    weighted = 0.0
    total_weight = 0.0
    for response in response_pathologies:
        score = max(
            (similarity(response, reference) for reference in reference_pathologies), default=0.0
        )
        score = min(1.0, max(0.0, score))
        weighted += response.severity.weight * score
        total_weight += response.severity.weight
    return weighted / total_weight


def lexical_pathology_similarity(left: PathologyReference, right: PathologyReference) -> float:
    name_score = float(left.name.casefold() == right.name.casefold())
    left_locations = {value.casefold() for value in left.locations}
    right_locations = {value.casefold() for value in right.locations}
    location_union = left_locations | right_locations
    location_score = (
        len(left_locations & right_locations) / len(location_union) if location_union else 1.0
    )
    left_attributes = {value.casefold() for value in left.attributes}
    right_attributes = {value.casefold() for value in right.attributes}
    attribute_union = left_attributes | right_attributes
    attribute_score = (
        len(left_attributes & right_attributes) / len(attribute_union) if attribute_union else 1.0
    )
    severity_score = float(left.severity is right.severity)
    return 0.5 * name_score + 0.2 * location_score + 0.15 * attribute_score + 0.15 * severity_score


def clinical_recommendation_validity(
    recommendation_entities: Sequence[int],
    diagnosis_entities: Sequence[int],
    graph: ClinicalKnowledgeGraph,
    maximum_path_length: int = 3,
) -> float:
    if not recommendation_entities:
        return 1.0
    if not diagnosis_entities:
        return 0.0
    supported = 0
    for recommendation in recommendation_entities:
        valid = any(
            graph.shortest_path(diagnosis, recommendation, maximum_path_length) is not None
            for diagnosis in diagnosis_entities
        )
        supported += int(valid)
    return supported / len(recommendation_entities)


def batch_hfs(components: Iterable[FactualityComponents]) -> dict[str, float]:
    values = tuple(components)
    if not values:
        raise ValueError("factuality component collection cannot be empty")
    anatomy = np.asarray([item.anatomical_accuracy for item in values])
    pathology = np.asarray([item.pathological_characterization for item in values])
    recommendation = np.asarray([item.clinical_recommendation_validity for item in values])
    hfs_values = np.asarray([item.hfs() for item in values])
    return {
        "anatomical_accuracy": float(anatomy.mean()),
        "pathological_characterization": float(pathology.mean()),
        "clinical_recommendation_validity": float(recommendation.mean()),
        "hierarchical_factuality_score": float(hfs_values.mean()),
    }


def graph_distance_similarity(
    left_entity: int,
    right_entity: int,
    graph: ClinicalKnowledgeGraph,
    maximum_distance: int = 4,
) -> float:
    if left_entity == right_entity:
        return 1.0
    path = graph.shortest_path(left_entity, right_entity, maximum_distance)
    if path is None:
        path = graph.shortest_path(right_entity, left_entity, maximum_distance)
    if path is None:
        return 0.0
    return 1.0 / (1.0 + len(path))


def multi_source_distance(
    graph: ClinicalKnowledgeGraph,
    sources: Sequence[int],
    maximum_distance: int,
) -> Mapping[int, int]:
    distances: dict[int, int] = {source: 0 for source in sources}
    queue: deque[int] = deque(sources)
    while queue:
        current = queue.popleft()
        if distances[current] >= maximum_distance:
            continue
        for neighbor in graph.neighbors(current):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances
