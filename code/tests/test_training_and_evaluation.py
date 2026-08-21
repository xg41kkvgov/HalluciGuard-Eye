from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from halluciguard_eye.evaluation.bootstrap import (
    cluster_bootstrap_interval,
    randomly_select_one_per_group,
    select_worse_per_group,
)
from halluciguard_eye.evaluation.clinical import ClinicalEyeRecord, clinical_primary_outcomes
from halluciguard_eye.evaluation.expert import (
    ExpertRating,
    fleiss_kappa,
    mean_ratings,
    rating_matrix,
)
from halluciguard_eye.evaluation.gee import (
    exchangeable_correlation,
    fit_logistic_gee,
    intercept_and_treatment,
    logistic,
    predict_probability,
)
from halluciguard_eye.evaluation.protocol import EvaluationProtocol, serious_clinical_hallucination
from halluciguard_eye.losses.stages import StageLoss
from halluciguard_eye.metrics.diagnostic import (
    binary_auc,
    classification_metrics,
    quadratic_weighted_kappa,
    sensitivity_specificity,
)
from halluciguard_eye.metrics.generation import bert_score_from_embeddings, bleu4, rouge_l
from halluciguard_eye.metrics.hallucination import (
    HallucinationEvent,
    HallucinationKind,
    entity_precision_recall_f1,
    hallucination_rate,
    relative_hallucination_reduction,
)
from halluciguard_eye.training.checkpoint import load_checkpoint, save_checkpoint
from halluciguard_eye.training.engine import Trainer, TrainingBatch, TrainingStage
from halluciguard_eye.training.schedule import ScheduleConfig, cosine_multiplier
from halluciguard_eye.training.seed import set_seed
from torch import nn


class TinyClinicalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layer(images)


def tiny_batch() -> TrainingBatch:
    images = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 1])
    return TrainingBatch(
        images=images,
        token_ids=torch.zeros(4, 2, dtype=torch.long),
        labels=labels,
        response_entity_ids=torch.zeros(4, 1, dtype=torch.long),
        evidence_entity_ids=torch.zeros(4, 1, dtype=torch.long),
        evidence_token_ids=torch.zeros(4, 2, dtype=torch.long),
        hfs_targets=torch.zeros(4, 3),
    )


def tiny_loss(model: nn.Module, batch: TrainingBatch, stage: TrainingStage) -> StageLoss:
    logits = model(batch.images)
    cross_entropy = nn.functional.cross_entropy(logits, batch.labels)
    stage_penalty = logits.new_tensor(float(stage - 1) * 0.0)
    return StageLoss(cross_entropy + stage_penalty, {"generation": cross_entropy})


def test_forward_loss_backward_and_parameter_update() -> None:
    set_seed(42)
    model = TinyClinicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    trainer = Trainer(model, optimizer, tiny_loss, "cpu", gradient_clipping=1.0)
    batch = tiny_batch()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = trainer.train_step((batch,), TrainingStage.FACTUALITY_INSTRUCTION)
    after = dict(model.named_parameters())
    assert result.loss > 0
    assert result.gradient_norm > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_single_batch_overfit_regression() -> None:
    set_seed(7)
    model = TinyClinicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.08, weight_decay=0.0)
    trainer = Trainer(model, optimizer, tiny_loss, "cpu", gradient_clipping=5.0)
    batch = tiny_batch()
    losses = [
        trainer.train_step((batch,), TrainingStage.FACTUALITY_INSTRUCTION).loss for _ in range(40)
    ]
    assert losses[-1] < losses[0] * 0.2
    predictions = model(batch.images).argmax(dim=-1)
    assert torch.equal(predictions, batch.labels)


def test_atomic_checkpoint_save_and_load(tmp_path: Path) -> None:
    set_seed(19)
    model = TinyClinicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, step=3, epoch=2, stage=1, seed=19, metadata={"source": "test"}
    )
    assert path.is_file()
    assert not tuple(tmp_path.glob("*.tmp"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    metadata = load_checkpoint(path, model, optimizer)
    assert metadata == {
        "step": 3,
        "epoch": 2,
        "stage": 1,
        "seed": 19,
        "metadata": {"source": "test"},
    }
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])


def test_checkpoint_restores_random_state(tmp_path: Path) -> None:
    set_seed(23)
    model = TinyClinicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, step=0, epoch=0, stage=1, seed=23)
    expected = torch.rand(5)
    torch.rand(100)
    load_checkpoint(path, model, optimizer, restore_rng=True)
    actual = torch.rand(5)
    assert torch.equal(actual, expected)


def test_cosine_schedule_boundaries() -> None:
    config = ScheduleConfig(total_steps=100, warmup_steps=10, minimum_ratio=0.1)
    assert cosine_multiplier(0, config) == pytest.approx(0.1)
    assert cosine_multiplier(9, config) == pytest.approx(1.0)
    assert cosine_multiplier(100, config) == pytest.approx(0.1)


def test_hallucination_metrics() -> None:
    assert hallucination_rate([True, False, False, True]) == 0.5
    reduction = relative_hallucination_reduction(0.384, 0.222)
    assert reduction == pytest.approx((0.384 - 0.222) / 0.384)
    metrics = entity_precision_recall_f1([{1, 2}, {3}], [{1, 4}, {3, 5}])
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 4)


def test_diagnostic_metrics_against_manual_counts() -> None:
    labels = [False, False, True, True, True]
    predictions = [False, True, True, False, True]
    sensitivity, specificity = sensitivity_specificity(labels, predictions)
    assert sensitivity == pytest.approx(2 / 3)
    assert specificity == pytest.approx(1 / 2)
    assert binary_auc(labels, [0.1, 0.4, 0.8, 0.3, 0.9]) == pytest.approx(5 / 6)


def test_quadratic_kappa_perfect_and_imperfect() -> None:
    labels = [0, 1, 2, 3, 4]
    assert quadratic_weighted_kappa(labels, labels, 5) == 1.0
    reversed_predictions = list(reversed(labels))
    assert quadratic_weighted_kappa(labels, reversed_predictions, 5) < 0
    metrics = classification_metrics(labels, labels, 5, positive_threshold=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_generation_metrics() -> None:
    candidate = "microaneurysm located in temporal macula".split()
    reference = "microaneurysm is located in the temporal macula".split()
    assert 0 < bleu4(candidate, (reference,)) <= 1
    assert rouge_l(candidate, reference) == pytest.approx(5 / 7 * 2 / (1 + 5 / 7))
    embeddings = np.eye(3)
    score = bert_score_from_embeddings(embeddings, embeddings)
    assert score == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_cluster_bootstrap_keeps_fellow_eyes_together() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    groups = ["p1", "p1", "p2", "p2", "p3", "p3"]
    interval = cluster_bootstrap_interval(values, groups, np.mean, replicates=200, seed=42)
    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower <= interval.estimate <= interval.upper
    selected = randomly_select_one_per_group(groups, seed=42)
    assert len(selected) == 3
    assert len({groups[index] for index in selected}) == 3
    worse = select_worse_per_group(groups, values)
    assert [values[index] for index in worse] == [1.0, 1.0, 0.0]


def test_expert_fleiss_kappa() -> None:
    ratings = [
        ExpertRating("r1", "a", 5, 4, 4, 4),
        ExpertRating("r1", "b", 5, 4, 4, 4),
        ExpertRating("r2", "a", 2, 3, 3, 2),
        ExpertRating("r2", "b", 2, 3, 3, 2),
    ]
    means = mean_ratings(ratings)
    assert means["accuracy"] == 3.5
    assert fleiss_kappa(rating_matrix(ratings, "accuracy")) == 1.0


def test_clinical_primary_outcomes() -> None:
    records = (
        ClinicalEyeRecord("e1", "p1", "y1", "c1", 50, "F", True, 3, 3, False),
        ClinicalEyeRecord("e2", "p1", "y1", "c1", 50, "F", True, 2, 1, True),
        ClinicalEyeRecord("e3", "p2", "y2", "c2", 60, "M", True, 0, 0, False),
        ClinicalEyeRecord("e4", "p3", "y2", "c2", 70, "F", True, 1, 2, False),
    )
    outcomes = clinical_primary_outcomes(records)
    assert outcomes["referable_dr_sensitivity"] == 0.5
    assert outcomes["referable_dr_specificity"] == 0.5
    assert outcomes["serious_clinical_hallucination_rate"] == 0.25


def test_serious_hallucination_definition() -> None:
    assert serious_clinical_hallucination(True, True, False, False)
    assert serious_clinical_hallucination(True, False, False, True)
    assert not serious_clinical_hallucination(False, True, False, False)
    assert not serious_clinical_hallucination(True, False, False, False)


def test_paper_evaluation_protocol_is_frozen() -> None:
    protocol = EvaluationProtocol.paper_protocol()
    assert protocol.seed == 42
    assert protocol.runs == 3
    assert protocol.beam_candidates == 5
    assert protocol.patient_clustered
    assert protocol.datasets[0].test_count == 3512
    assert protocol.datasets[1].external_only
    assert len(protocol.digest()) == 64


def test_hallucination_event_validation() -> None:
    event = HallucinationEvent("r1", HallucinationKind.RECOMMENDATION, True, False, 0.9)
    assert event.serious
    with pytest.raises(ValueError):
        HallucinationEvent("r1", HallucinationKind.FACTUALITY, False, False, 1.1)


def test_logistic_is_stable_for_extreme_values() -> None:
    values = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    probabilities = logistic(values)
    assert np.isfinite(probabilities).all()
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[2] == pytest.approx(0.5)
    assert probabilities[-1] == pytest.approx(1.0)
    assert np.all(np.diff(probabilities) > 0)


def test_exchangeable_correlation_matrix() -> None:
    matrix = exchangeable_correlation(3, 0.25)
    expected = np.array(
        [
            [1.0, 0.25, 0.25],
            [0.25, 1.0, 0.25],
            [0.25, 0.25, 1.0],
        ]
    )
    assert np.array_equal(matrix, expected)
    assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_logistic_gee_detects_treatment_direction() -> None:
    treatment = [False, True] * 20
    labels = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
        ],
        dtype=np.float64,
    )
    groups = [f"p{index // 2}" for index in range(40)]
    design = intercept_and_treatment(treatment)
    result = fit_logistic_gee(design, labels, groups)
    assert result.converged
    assert result.coefficients[1] > 0
    probabilities = predict_probability(result, design)
    assert (
        probabilities[np.asarray(treatment)].mean() > probabilities[~np.asarray(treatment)].mean()
    )
    assert result.confidence_intervals().shape == (2, 2)
