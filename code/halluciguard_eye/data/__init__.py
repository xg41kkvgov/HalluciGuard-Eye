"""Ophthalmic image, text, graph, and clinical cohort data interfaces."""

from halluciguard_eye.data.annotations import ClinicalAnnotation, ClinicalClaim, ResponseAnnotation
from halluciguard_eye.data.datasets import OphthalmicDataset, OphthalmicSample
from halluciguard_eye.data.manifest import DatasetManifest, ManifestRecord
from halluciguard_eye.data.splits import SplitAssignment, stratified_split
from halluciguard_eye.data.tokenization import ClinicalTokenizer, Vocabulary

__all__ = [
    "ClinicalAnnotation",
    "ClinicalClaim",
    "ClinicalTokenizer",
    "DatasetManifest",
    "ManifestRecord",
    "OphthalmicDataset",
    "OphthalmicSample",
    "ResponseAnnotation",
    "SplitAssignment",
    "Vocabulary",
    "stratified_split",
]
