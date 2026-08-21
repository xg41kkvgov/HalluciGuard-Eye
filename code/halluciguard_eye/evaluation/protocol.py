"""Frozen public and retrospective evaluation protocol. Ref: Sec. III-F to III-I."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    modality: str
    train_count: int
    validation_count: int
    test_count: int
    classes: int
    external_only: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or self.modality not in {"fundus", "oct"}:
            raise ValueError("dataset identity is invalid")
        if min(self.train_count, self.validation_count, self.test_count) < 0 or self.classes <= 1:
            raise ValueError("dataset counts are invalid")
        if self.external_only and (self.train_count or self.validation_count):
            raise ValueError("external datasets cannot have train or validation counts")


@dataclass(frozen=True)
class EvaluationProtocol:
    datasets: tuple[DatasetProtocol, ...]
    seed: int
    runs: int
    confidence: float
    beam_candidates: int
    thresholds_frozen: bool
    patient_clustered: bool

    @classmethod
    def paper_protocol(cls) -> EvaluationProtocol:
        return cls(
            datasets=(
                DatasetProtocol("EyePACS", "fundus", 28_101, 3_513, 3_512, 5),
                DatasetProtocol("Messidor-2", "fundus", 0, 0, 1_748, 5, True),
                DatasetProtocol("OCTDL", "oct", 1_445, 309, 310, 6),
                DatasetProtocol("Kermany OCT", "oct", 75_136, 8_348, 1_000, 4),
            ),
            seed=42,
            runs=3,
            confidence=0.95,
            beam_candidates=5,
            thresholds_frozen=True,
            patient_clustered=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "datasets": [dataset.__dict__ for dataset in self.datasets],
            "seed": self.seed,
            "runs": self.runs,
            "confidence": self.confidence,
            "beam_candidates": self.beam_candidates,
            "thresholds_frozen": self.thresholds_frozen,
            "patient_clustered": self.patient_clustered,
        }

    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        payload = self.as_dict() | {"digest": self.digest()}
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)


def validate_reported_result_schema(results: Mapping[str, Any]) -> tuple[str, ...]:
    required = {
        "hallucination_rate",
        "hierarchical_factuality_score",
        "factual_accuracy",
        "sensitivity",
        "specificity",
        "auc",
        "quadratic_weighted_kappa",
        "f1",
    }
    missing = required - set(results)
    errors: list[str] = []
    if missing:
        errors.append(f"missing metrics: {sorted(missing)}")
    for name in required & set(results):
        value = results[name]
        if not isinstance(value, int | float):
            errors.append(f"{name} is not numeric")
    return tuple(errors)


def assert_no_patient_overlap(split_patient_ids: Mapping[str, Sequence[str]]) -> None:
    sets = {name: set(values) for name, values in split_patient_ids.items()}
    names = sorted(sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise AssertionError(f"patient overlap between {left_name} and {right_name}")


def serious_clinical_hallucination(
    unsupported: bool,
    missed_vision_threatening_disease: bool,
    fabricated_vision_threatening_disease: bool,
    incorrect_referral: bool,
) -> bool:
    clinical_consequence = (
        missed_vision_threatening_disease
        or fabricated_vision_threatening_disease
        or incorrect_referral
    )
    return unsupported and clinical_consequence
