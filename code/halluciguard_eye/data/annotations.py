"""Clinical annotation records used by HFS. Ref: Sec. III-D and III-G."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


class ClaimLevel(str, Enum):
    ANATOMICAL = "anatomical"
    PATHOLOGICAL = "pathological"
    RECOMMENDATION = "recommendation"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    VISION_THREATENING = "vision_threatening"

    @property
    def weight(self) -> float:
        return {
            Severity.MILD: 1.0,
            Severity.MODERATE: 2.0,
            Severity.VISION_THREATENING: 3.0,
        }[self]


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("bounding box coordinates must be normalized")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding box must have positive area")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def intersection(self, other: BoundingBox) -> float:
        width = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        height = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        return width * height

    def union(self, other: BoundingBox) -> float:
        return self.area + other.area - self.intersection(other)

    def iou(self, other: BoundingBox) -> float:
        union = self.union(other)
        return self.intersection(other) / union if union else 0.0


@dataclass(frozen=True)
class AnatomicalReference:
    name: str
    box: BoundingBox
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("anatomical name cannot be empty")


@dataclass(frozen=True)
class PathologyReference:
    name: str
    severity: Severity
    locations: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("pathology name cannot be empty")
        if len(set(self.locations)) != len(self.locations):
            raise ValueError("pathology locations must be unique")
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("pathology attributes must be unique")


@dataclass(frozen=True)
class RecommendationReference:
    action: str
    diagnosis: str
    urgency: str

    def __post_init__(self) -> None:
        if not self.action.strip() or not self.diagnosis.strip() or not self.urgency.strip():
            raise ValueError("recommendation fields cannot be empty")


@dataclass(frozen=True)
class ClinicalClaim:
    text: str
    level: ClaimLevel
    entity_ids: tuple[int, ...]
    confidence: float = 1.0
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("claim text cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("claim confidence must be in [0, 1]")
        if any(entity_id < 0 for entity_id in self.entity_ids):
            raise ValueError("entity identifiers must be nonnegative")


@dataclass(frozen=True)
class ClinicalAnnotation:
    image_id: str
    diagnosis: str
    grade: int
    anatomical: tuple[AnatomicalReference, ...] = ()
    pathologies: tuple[PathologyReference, ...] = ()
    recommendations: tuple[RecommendationReference, ...] = ()
    gradable: bool = True
    reviewer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.image_id.strip() or not self.diagnosis.strip():
            raise ValueError("annotation identifiers cannot be empty")
        if self.grade < 0:
            raise ValueError("grade must be nonnegative")
        anatomy_names = [item.name.casefold() for item in self.anatomical]
        if len(anatomy_names) != len(set(anatomy_names)):
            raise ValueError("anatomical references must be unique")


@dataclass(frozen=True)
class ResponseAnnotation:
    response_id: str
    image_id: str
    text: str
    claims: tuple[ClinicalClaim, ...]
    candidate_log_probability: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.response_id.strip() or not self.image_id.strip():
            raise ValueError("response identifiers cannot be empty")
        if not self.text.strip():
            raise ValueError("response text cannot be empty")

    def claims_at(self, level: ClaimLevel) -> tuple[ClinicalClaim, ...]:
        return tuple(claim for claim in self.claims if claim.level is level)

    def mentioned_entities(self) -> tuple[int, ...]:
        return tuple(sorted({entity for claim in self.claims for entity in claim.entity_ids}))


def validate_annotation_pair(annotation: ClinicalAnnotation, response: ResponseAnnotation) -> None:
    if annotation.image_id != response.image_id:
        raise ValueError("clinical annotation and response image identifiers differ")


def group_claims_by_level(
    claims: Iterable[ClinicalClaim],
) -> dict[ClaimLevel, tuple[ClinicalClaim, ...]]:
    grouped: dict[ClaimLevel, list[ClinicalClaim]] = {level: [] for level in ClaimLevel}
    for claim in claims:
        grouped[claim.level].append(claim)
    return {level: tuple(values) for level, values in grouped.items()}


def reviewer_agreement_labels(
    labels_by_reviewer: Mapping[str, Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if not labels_by_reviewer:
        raise ValueError("reviewer labels cannot be empty")
    lengths = {len(labels) for labels in labels_by_reviewer.values()}
    if len(lengths) != 1:
        raise ValueError("all reviewers must label the same number of items")
    reviewer_order = sorted(labels_by_reviewer)
    item_count = next(iter(lengths))
    return tuple(
        tuple(labels_by_reviewer[reviewer][item] for reviewer in reviewer_order)
        for item in range(item_count)
    )
