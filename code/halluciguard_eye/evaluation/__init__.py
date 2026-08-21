"""Public benchmark and patient-clustered clinical evaluation."""

from halluciguard_eye.evaluation.bootstrap import cluster_bootstrap_interval
from halluciguard_eye.evaluation.clinical import ClinicalEyeRecord
from halluciguard_eye.evaluation.expert import ExpertRating
from halluciguard_eye.evaluation.gee import fit_logistic_gee
from halluciguard_eye.evaluation.protocol import EvaluationProtocol

__all__ = [
    "ClinicalEyeRecord",
    "EvaluationProtocol",
    "ExpertRating",
    "cluster_bootstrap_interval",
    "fit_logistic_gee",
]
