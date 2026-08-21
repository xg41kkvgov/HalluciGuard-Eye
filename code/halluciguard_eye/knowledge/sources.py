"""OCKG source provenance and validation. Ref: Sec. III-B.1 and III-B.2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum


class Authority(str, Enum):
    CLINICAL_GUIDELINE = "clinical_guideline"
    CLASSIFICATION = "classification"
    GRADING_SCALE = "grading_scale"
    PEER_REVIEWED_LITERATURE = "peer_reviewed_literature"


@dataclass(frozen=True)
class KnowledgeSource:
    identifier: str
    title: str
    authority: Authority
    publisher: str
    version: str
    url: str
    accessed: str

    def __post_init__(self) -> None:
        values = (
            self.identifier,
            self.title,
            self.publisher,
            self.version,
            self.url,
            self.accessed,
        )
        if any(not value.strip() for value in values):
            raise ValueError("knowledge source fields cannot be empty")
        if not self.url.startswith("https://"):
            raise ValueError("knowledge source URL must use HTTPS")


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    locator: str
    quotation_digest: str
    reviewer_count: int
    accepted: bool

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.locator.strip():
            raise ValueError("source evidence identifiers cannot be empty")
        if len(self.quotation_digest) != 64:
            raise ValueError("quotation digest must be SHA-256 hexadecimal")
        if self.reviewer_count < 0:
            raise ValueError("reviewer count cannot be negative")


class SourceRegistry:
    def __init__(self, sources: Iterable[KnowledgeSource]) -> None:
        items = tuple(sources)
        self._sources = {source.identifier: source for source in items}
        if len(self._sources) != len(items):
            raise ValueError("knowledge source identifiers must be unique")

    def __len__(self) -> int:
        return len(self._sources)

    def get(self, identifier: str) -> KnowledgeSource:
        return self._sources[identifier]

    def by_authority(self, authority: Authority) -> tuple[KnowledgeSource, ...]:
        return tuple(source for source in self._sources.values() if source.authority is authority)

    def validate_evidence(self, evidence: Sequence[SourceEvidence]) -> tuple[str, ...]:
        errors: list[str] = []
        for item in evidence:
            if item.source_id not in self._sources:
                errors.append(f"unknown source {item.source_id}")
            if item.accepted and item.reviewer_count < 1:
                errors.append(f"accepted evidence {item.locator} has no reviewer")
        return tuple(errors)

    def coverage(self, evidence: Sequence[SourceEvidence]) -> Mapping[str, int]:
        counts = {identifier: 0 for identifier in self._sources}
        for item in evidence:
            if item.accepted and item.source_id in counts:
                counts[item.source_id] += 1
        return counts


def authoritative_source_categories() -> tuple[tuple[str, Authority], ...]:
    return (
        ("American Academy of Ophthalmology", Authority.CLINICAL_GUIDELINE),
        ("European Society of Retina Specialists", Authority.CLINICAL_GUIDELINE),
        ("International Council of Ophthalmology", Authority.CLINICAL_GUIDELINE),
        ("ICD-11", Authority.CLASSIFICATION),
        ("International Clinical Diabetic Retinopathy Severity Scale", Authority.GRADING_SCALE),
        ("Age-Related Eye Disease Study classification", Authority.GRADING_SCALE),
        ("PubMed ophthalmology literature", Authority.PEER_REVIEWED_LITERATURE),
    )
