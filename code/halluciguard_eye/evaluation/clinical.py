"""Single-center retrospective analysis. Ref: Sec. III-G."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from halluciguard_eye.evaluation.bootstrap import ConfidenceInterval, cluster_bootstrap_interval
from halluciguard_eye.metrics.diagnostic import sensitivity_specificity
from halluciguard_eye.metrics.hallucination import hallucination_rate


@dataclass(frozen=True)
class ClinicalEyeRecord:
    eye_id: str
    patient_id: str
    calendar_period: str
    camera_type: str
    age: int
    sex: str
    gradable: bool
    reference_grade: int
    predicted_grade: int
    serious_hallucination: bool

    def __post_init__(self) -> None:
        if not self.eye_id.strip() or not self.patient_id.strip():
            raise ValueError("clinical record identifiers cannot be empty")
        if self.age < 18:
            raise ValueError("retrospective cohort includes adults only")
        if self.reference_grade not in range(5) or self.predicted_grade not in range(5):
            raise ValueError("DR grades must be in [0, 4]")


def referable(grade: int) -> bool:
    if grade not in range(5):
        raise ValueError("DR grade must be in [0, 4]")
    return grade >= 2


def clinical_primary_outcomes(records: Sequence[ClinicalEyeRecord]) -> Mapping[str, float]:
    if not records:
        raise ValueError("clinical records cannot be empty")
    labels = [referable(record.reference_grade) for record in records]
    predictions = [referable(record.predicted_grade) for record in records]
    sensitivity, specificity = sensitivity_specificity(labels, predictions)
    serious_rate = hallucination_rate([record.serious_hallucination for record in records])
    return {
        "referable_dr_sensitivity": sensitivity,
        "referable_dr_specificity": specificity,
        "serious_clinical_hallucination_rate": serious_rate,
    }


def patient_clustered_outcome_intervals(
    records: Sequence[ClinicalEyeRecord],
    replicates: int = 2000,
    seed: int = 42,
) -> Mapping[str, ConfidenceInterval]:
    groups = [record.patient_id for record in records]
    sensitivity_values = [
        (referable(record.reference_grade), referable(record.predicted_grade)) for record in records
    ]
    serious_values = [record.serious_hallucination for record in records]

    def sensitivity_statistic(values: Sequence[tuple[bool, bool]]) -> float:
        positives = [prediction for label, prediction in values if label]
        return float(np.mean(positives)) if positives else 1.0

    return {
        "referable_dr_sensitivity": cluster_bootstrap_interval(
            sensitivity_values, groups, sensitivity_statistic, replicates=replicates, seed=seed
        ),
        "serious_clinical_hallucination_rate": cluster_bootstrap_interval(
            serious_values,
            groups,
            lambda values: float(np.mean(values)),
            replicates=replicates,
            seed=seed,
        ),
    }


def stratified_outcomes(
    records: Iterable[ClinicalEyeRecord], field: str
) -> Mapping[str, Mapping[str, float]]:
    if field not in {"calendar_period", "camera_type", "sex", "gradable"}:
        raise ValueError("unsupported clinical stratification field")
    groups: dict[str, list[ClinicalEyeRecord]] = {}
    for record in records:
        groups.setdefault(str(getattr(record, field)), []).append(record)
    return {name: clinical_primary_outcomes(values) for name, values in groups.items()}


def cohort_requirements(records: Sequence[ClinicalEyeRecord]) -> Mapping[str, bool]:
    patients = {record.patient_id for record in records}
    years = {record.calendar_period for record in records}
    return {
        "at_least_800_eyes": len(records) >= 800,
        "more_than_400_patients": len(patients) > 400,
        "four_calendar_periods": len(years) == 4,
        "all_adults": all(record.age >= 18 for record in records),
    }
