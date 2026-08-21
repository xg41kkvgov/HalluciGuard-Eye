from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from halluciguard_eye.data.annotations import (
    AnatomicalReference,
    BoundingBox,
    PathologyReference,
    Severity,
)
from halluciguard_eye.data.datasets import (
    OphthalmicDataset,
    crop_fundus_field,
    letterbox_resize,
    normalize_imagenet,
)
from halluciguard_eye.data.manifest import DatasetManifest, ManifestRecord, file_sha256
from halluciguard_eye.data.splits import (
    assert_group_isolation,
    grouped_stratified_split,
    stratified_split,
)
from halluciguard_eye.data.tokenization import ClinicalTokenizer, build_vocabulary, segment_claims
from halluciguard_eye.knowledge.graph import (
    ClinicalKnowledgeGraph,
    Entity,
    EntityType,
    Relation,
    RelationFamily,
    Triplet,
)
from halluciguard_eye.metrics.factuality import (
    anatomical_accuracy,
    clinical_recommendation_validity,
    hierarchical_factuality_score,
    lexical_pathology_similarity,
    pathological_characterization,
)
from PIL import Image


def small_graph() -> ClinicalKnowledgeGraph:
    entities = (
        Entity(0, "diabetic retinopathy", EntityType.DISEASE),
        Entity(1, "microaneurysm", EntityType.PATHOLOGY),
        Entity(2, "macula", EntityType.ANATOMY),
        Entity(3, "anti-VEGF", EntityType.PROCEDURE),
        Entity(4, "diabetic macular edema", EntityType.DISEASE),
    )
    relations = (
        Relation(0, "causes", RelationFamily.CAUSAL),
        Relation(1, "located-in", RelationFamily.SPATIAL),
        Relation(2, "treated-by", RelationFamily.CLINICAL),
    )
    triplets = (
        Triplet(0, 0, 1),
        Triplet(1, 1, 2),
        Triplet(0, 2, 3),
        Triplet(4, 2, 3),
    )
    return ClinicalKnowledgeGraph(entities, relations, triplets)


def test_bounding_box_geometry() -> None:
    left = BoundingBox(0.0, 0.0, 0.5, 0.5)
    right = BoundingBox(0.25, 0.25, 0.75, 0.75)
    assert left.area == pytest.approx(0.25)
    assert left.intersection(right) == pytest.approx(0.0625)
    assert left.union(right) == pytest.approx(0.4375)
    assert left.iou(right) == pytest.approx(1 / 7)


def test_bounding_box_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        BoundingBox(-0.1, 0.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        BoundingBox(0.5, 0.0, 0.5, 0.5)


def test_manifest_round_trip(tmp_path: Path) -> None:
    records = (
        ManifestRecord("a", "train/a.png", 0, "p1", "fundus", "train"),
        ManifestRecord("b", "test/b.png", 1, "p2", "fundus", "test"),
    )
    manifest = DatasetManifest("sample", records)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path)
    loaded = DatasetManifest.from_csv("sample", path)
    assert tuple(loaded) == records
    assert loaded.digest() == manifest.digest()
    assert loaded.class_counts() == {0: 1, 1: 1}
    assert loaded.split_counts() == {"train": 1, "test": 1}


def test_file_sha256_is_content_derived(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"ophthalmic evidence")
    expected = hashlib.sha256(b"ophthalmic evidence").hexdigest()
    assert file_sha256(path) == expected


def test_dataset_reads_real_image(tmp_path: Path) -> None:
    image_path = tmp_path / "retina.png"
    array = np.zeros((18, 24, 3), dtype=np.uint8)
    array[3:15, 6:18] = (170, 40, 20)
    Image.fromarray(array).save(image_path)
    manifest = DatasetManifest(
        "sample",
        (ManifestRecord("retina", "retina.png", 2, "p1", "fundus", "train"),),
    )
    dataset = OphthalmicDataset(tmp_path, manifest, image_size=28)
    sample = dataset[0]
    assert sample.image.shape == (28, 28, 3)
    assert sample.image.dtype == np.float32
    assert sample.label == 2
    assert np.isfinite(sample.image).all()


def test_crop_fundus_field_removes_dark_border() -> None:
    array = np.zeros((20, 30, 3), dtype=np.uint8)
    array[5:15, 10:20] = 100
    cropped = crop_fundus_field(Image.fromarray(array))
    assert cropped.size == (10, 10)


def test_letterbox_resize_preserves_square_canvas() -> None:
    image = Image.new("RGB", (20, 10), (255, 255, 255))
    resized = letterbox_resize(image, 32)
    assert resized.size == (32, 32)
    output = np.asarray(resized)
    assert np.all(output[:8] == 0)
    assert np.all(output[8:24] == 255)


def test_imagenet_normalization_matches_manual_formula() -> None:
    image = np.ones((2, 2, 3), dtype=np.float32) * 0.5
    normalized = normalize_imagenet(image)
    expected = (np.array([0.5, 0.5, 0.5]) - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    assert np.allclose(normalized[0, 0], expected)


def test_stratified_split_has_full_disjoint_coverage() -> None:
    labels = [0] * 20 + [1] * 20 + [2] * 20
    split = stratified_split(labels, 0.7, 0.15, seed=42)
    assert split.size == len(labels)
    assert set(split.train) | set(split.validation) | set(split.test) == set(range(len(labels)))
    for indices in (split.train, split.validation, split.test):
        counts = np.bincount([labels[index] for index in indices], minlength=3)
        assert counts.max() == counts.min()


def test_grouped_split_isolates_patients() -> None:
    groups = [f"p{index // 2}" for index in range(60)]
    labels = [index // 10 for index in range(60)]
    split = grouped_stratified_split(labels, groups, 0.6, 0.2, seed=42)
    assert_group_isolation(split, groups)
    assert split.size == len(labels)


def test_clinical_tokenizer_round_trip() -> None:
    documents = ["Microaneurysm is located in the macula.", "Recommend anti-VEGF treatment."]
    vocabulary = build_vocabulary(documents, maximum_size=64)
    tokenizer = ClinicalTokenizer(vocabulary)
    encoded = tokenizer.encode(documents[0], maximum_length=12)
    assert len(encoded.identifiers) == 12
    assert sum(encoded.attention_mask) == 9
    decoded = tokenizer.decode(encoded.identifiers)
    assert "microaneurysm" in decoded
    assert "macula" in decoded


def test_claim_segmentation() -> None:
    text = "The macula is visible. Microaneurysms are present; recommend review."
    claims = segment_claims(text)
    assert claims == (
        "The macula is visible.",
        "Microaneurysms are present",
        "recommend review.",
    )


def test_graph_index_and_shortest_path() -> None:
    graph = small_graph()
    assert len(graph) == 4
    assert graph.neighbors(0) == (1, 3)
    path = graph.shortest_path(0, 2, 3)
    assert path is not None
    assert [(item.head, item.tail) for item in path] == [(0, 1), (1, 2)]
    assert graph.shortest_path(2, 3, 3) is None


def test_graph_find_entities_uses_aliases() -> None:
    graph = ClinicalKnowledgeGraph(
        (Entity(0, "diabetic retinopathy", EntityType.DISEASE, aliases=("DR",)),),
        (),
        (),
    )
    assert graph.find_entities("The patient has DR.")[0].identifier == 0


def test_graph_linearization() -> None:
    graph = small_graph()
    assert graph.linearize((Triplet(0, 2, 3),)) == ("diabetic retinopathy treated-by anti-VEGF",)


def test_anatomical_accuracy_is_direct_mean() -> None:
    labels = [True, False, True, True]
    independent = sum(int(value) for value in labels) / len(labels)
    assert anatomical_accuracy(labels) == independent


def test_hfs_is_weighted_sum() -> None:
    expected = 0.25 * 0.8 + 0.40 * 0.6 + 0.35 * 1.0
    assert hierarchical_factuality_score(0.8, 0.6, 1.0) == pytest.approx(expected)


def test_pathological_score_respects_severity_weights() -> None:
    response = (
        PathologyReference("microaneurysm", Severity.MILD, ("macula",)),
        PathologyReference("neovascularization", Severity.VISION_THREATENING, ("disc",)),
    )
    reference = (PathologyReference("microaneurysm", Severity.MILD, ("macula",)),)
    score = pathological_characterization(response, reference, lexical_pathology_similarity)
    unrelated_similarity = 0.15
    expected = (1.0 * 1.0 + 3.0 * unrelated_similarity) / 4.0
    assert score == pytest.approx(expected)


def test_recommendation_validity_uses_three_hop_graph_path() -> None:
    graph = small_graph()
    assert clinical_recommendation_validity((3,), (0,), graph, 3) == 1.0
    assert clinical_recommendation_validity((2,), (4,), graph, 3) == 0.0


def test_anatomical_reference_validation() -> None:
    reference = AnatomicalReference("macula", BoundingBox(0.4, 0.4, 0.6, 0.6))
    assert reference.visible
