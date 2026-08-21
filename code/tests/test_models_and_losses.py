from __future__ import annotations

import math

import pytest
import torch
from halluciguard_eye.config import ModelConfig
from halluciguard_eye.knowledge.graph import (
    ClinicalKnowledgeGraph,
    Entity,
    EntityType,
    Relation,
    RelationFamily,
    Triplet,
)
from halluciguard_eye.knowledge.retrieval import KnowledgeRetriever
from halluciguard_eye.knowledge.transe import TransE, corrupt_triplets, negative_sampling_batch
from halluciguard_eye.losses.contrastive import image_text_contrastive_loss
from halluciguard_eye.losses.hfs import (
    HFSWeights,
    hierarchical_factuality_loss,
    hierarchical_factuality_score,
)
from halluciguard_eye.losses.stages import instruction_loss, pretraining_loss, rlcf_loss
from halluciguard_eye.models.full_model import HalluciGuardEye
from halluciguard_eye.models.layers import GatedEvidenceFusion, RMSNorm, masked_mean
from halluciguard_eye.models.reranking import composite_scores, rerank_candidates
from halluciguard_eye.models.vision import VisionTransformer
from halluciguard_eye.models.vsam import VisualSemanticAlignmentModule


def model_config() -> ModelConfig:
    return ModelConfig(
        image_size=28,
        patch_size=7,
        vision_width=32,
        language_width=48,
        knowledge_width=16,
        alignment_width=8,
        evidence_width=48,
        vision_layers=2,
        decoder_layers=2,
        attention_heads=4,
        max_text_tokens=12,
    )


def test_rms_norm_has_unit_root_mean_square() -> None:
    norm = RMSNorm(4, epsilon=1e-12)
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    output = norm(values)
    rms = output.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6)


def test_masked_mean_ignores_padding() -> None:
    values = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert masked_mean(values, mask).item() == pytest.approx(2.0)


def test_vision_transformer_shape() -> None:
    model = VisionTransformer(28, 7, 32, 2, 4)
    images = torch.randn(3, 3, 28, 28)
    output = model(images, return_attention=True)
    assert output.tokens.shape == (3, 17, 32)
    assert output.pooled.shape == (3, 32)
    assert len(output.attention_maps) == 2
    assert output.attention_maps[0].shape == (3, 4, 17, 17)


def test_vsam_bidirectional_shapes_and_range() -> None:
    model = VisualSemanticAlignmentModule(32, 48, 8, 24)
    visual = torch.randn(2, 17, 32)
    text = torch.randn(2, 12, 48)
    output = model(visual, text)
    assert output.visual_to_text.shape == (2, 17, 12)
    assert output.text_to_visual.shape == (2, 12, 17)
    assert output.visual_context.shape == (2, 17, 24)
    assert output.text_context.shape == (2, 12, 24)
    assert torch.all((output.score >= 0) & (output.score <= 1))
    assert torch.allclose(output.visual_to_text.sum(dim=-1), torch.ones(2, 17), atol=1e-6)
    assert torch.allclose(output.text_to_visual.sum(dim=-1), torch.ones(2, 12), atol=1e-6)


def test_contrastive_loss_prefers_matched_pairs() -> None:
    matched = torch.eye(4)
    shuffled = matched.roll(1, dims=0)
    matched_loss = image_text_contrastive_loss(matched, matched)
    shuffled_loss = image_text_contrastive_loss(matched, shuffled)
    assert matched_loss < shuffled_loss


def test_transe_distance_matches_manual_l2() -> None:
    model = TransE(3, 1, dimension=2, margin=1.0)
    with torch.no_grad():
        model.entity_embeddings.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        model.relation_embeddings.weight.copy_(torch.tensor([[1.0, 0.0]]))
    triplets = torch.tensor([[0, 0, 1], [0, 0, 2]])
    distances = model.distance(triplets)
    assert distances[0].item() == pytest.approx(0.0)
    assert distances[1].item() == pytest.approx(math.sqrt(2))


def test_transe_corruption_changes_one_endpoint() -> None:
    positive = torch.tensor([[0, 0, 1], [2, 0, 3], [4, 0, 5]])
    generator = torch.Generator().manual_seed(42)
    negative = corrupt_triplets(positive, 8, generator)
    changed = (positive != negative).sum(dim=1)
    assert torch.all(changed == 1)
    assert torch.all(positive[:, 1] == negative[:, 1])


def test_negative_sampling_ratio() -> None:
    positive = torch.tensor([[0, 0, 1], [2, 0, 3]])
    batch = negative_sampling_batch(positive, 8, 4, torch.Generator().manual_seed(1))
    assert batch.positive.shape == (8, 3)
    assert batch.negative.shape == (8, 3)


def test_retriever_follows_equation_threshold_and_two_hops() -> None:
    graph = ClinicalKnowledgeGraph(
        (
            Entity(0, "disease", EntityType.DISEASE),
            Entity(1, "lesion", EntityType.PATHOLOGY),
            Entity(2, "macula", EntityType.ANATOMY),
            Entity(3, "procedure", EntityType.PROCEDURE),
        ),
        (
            Relation(0, "causes", RelationFamily.CAUSAL),
            Relation(1, "located-in", RelationFamily.SPATIAL),
            Relation(2, "treated-by", RelationFamily.CLINICAL),
        ),
        (Triplet(0, 0, 1), Triplet(1, 1, 2), Triplet(0, 2, 3)),
    )
    entity_embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=torch.float32
    )
    relation_embeddings = torch.tensor([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.3]])
    retriever = KnowledgeRetriever(graph, entity_embeddings, relation_embeddings, 0.7, 0.0, 2)
    query = torch.tensor([1.0, 0.0])
    result = retriever.retrieve(query)
    assert len(result.items) == 3
    assert result.entity_ids() == (0, 1, 2, 3)


def test_gated_fusion_matches_equation_extremes() -> None:
    fusion = GatedEvidenceFusion(4, 3, 5)
    hidden = torch.randn(2, 6, 4)
    attention = torch.randn(2, 3)
    evidence = torch.randn(2, 5)
    output = fusion(hidden, attention, evidence)
    assert output.hidden.shape == hidden.shape
    assert output.gate.shape == hidden.shape
    reconstructed = output.gate * hidden + (1 - output.gate) * output.fused_context
    assert torch.allclose(output.hidden, reconstructed)


def test_candidate_reranking_matches_manual_equation() -> None:
    log_probability = torch.tensor([-2.0, -1.9, -2.1])
    alignment = torch.tensor([0.5, 0.1, 0.9])
    grounding = torch.tensor([0.2, 0.1, 0.8])
    expected = log_probability + 0.3 * alignment + 0.4 * grounding
    scores = composite_scores(log_probability, alignment, grounding)
    index, records = rerank_candidates(log_probability, alignment, grounding)
    assert torch.allclose(scores, expected)
    assert index == int(torch.argmax(expected))
    assert records[index].composite == pytest.approx(float(expected[index]))


def test_hfs_tensor_score_matches_independent_sum() -> None:
    components = torch.tensor([[0.8, 0.6, 1.0], [0.4, 0.5, 0.2]])
    score = hierarchical_factuality_score(components)
    expected = torch.tensor(
        [
            0.25 * 0.8 + 0.40 * 0.6 + 0.35 * 1.0,
            0.25 * 0.4 + 0.40 * 0.5 + 0.35 * 0.2,
        ]
    )
    assert torch.allclose(score, expected)


def test_hfs_loss_has_gradient() -> None:
    predictions = torch.tensor([[0.4, 0.5, 0.6]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    loss = hierarchical_factuality_loss(predictions, targets, HFSWeights())
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert torch.any(predictions.grad != 0)


def test_stage_loss_equations() -> None:
    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value)

    pretrain = pretraining_loss(scalar(2.0), scalar(3.0), scalar(4.0))
    assert pretrain.total.item() == pytest.approx(2.0 + 0.5 * 3.0 + 0.3 * 4.0)
    instruct = instruction_loss(scalar(2.0), scalar(3.0), scalar(4.0), scalar(5.0))
    assert instruct.total.item() == pytest.approx(2.0 + 0.4 * 3.0 + 0.3 * 4.0 + 0.3 * 5.0)
    rlcf = rlcf_loss(torch.tensor([-1.0]), torch.tensor([2.0]), scalar(0.5))
    assert rlcf.total.item() == pytest.approx(2.0 + 0.1 * 0.5)


def test_full_forward_shapes() -> None:
    model = HalluciGuardEye(model_config(), vocabulary_size=64, entity_count=24)
    output = model(
        images=torch.randn(2, 3, 28, 28),
        token_ids=torch.randint(0, 64, (2, 12)),
        response_entity_ids=torch.randint(0, 24, (2, 3)),
        evidence_entity_ids=torch.randint(0, 24, (2, 5)),
        evidence_token_ids=torch.randint(0, 64, (2, 12)),
    )
    assert output.logits.shape == (2, 12, 64)
    assert output.alignment.score.shape == (2,)
    assert output.grounding.grounding_score.shape == (2,)
    components = model.factuality_components(output)
    assert components.shape == (2, 3)
    assert torch.all((components >= 0) & (components <= 1))
