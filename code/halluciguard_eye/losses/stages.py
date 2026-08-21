"""Three-stage HalluciGuard-Eye training losses. Ref: Eq. (23)-(25)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class StageLoss:
    total: Tensor
    components: dict[str, Tensor]

    def detached(self) -> dict[str, float]:
        return {"total": float(self.total.detach().item())} | {
            name: float(value.detach().item()) for name, value in self.components.items()
        }


def masked_language_model_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("language logits and labels have incompatible shapes")
    return functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=ignore_index
    )


def image_text_matching_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim not in {1, 2}:
        raise ValueError("image-text matching logits must be a vector or matrix")
    if logits.ndim == 1:
        if labels.shape != logits.shape:
            raise ValueError("binary image-text labels must match logits")
        return functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
    if labels.shape != logits.shape[:1]:
        raise ValueError("multiclass image-text labels must match batch")
    return functional.cross_entropy(logits, labels.long())


def pretraining_loss(
    mlm: Tensor,
    image_text_matching: Tensor,
    knowledge_graph_embedding: Tensor,
    itm_weight: float = 0.5,
    kge_weight: float = 0.3,
) -> StageLoss:
    if itm_weight < 0 or kge_weight < 0:
        raise ValueError("pretraining loss weights must be nonnegative")
    total = mlm + itm_weight * image_text_matching + kge_weight * knowledge_graph_embedding
    return StageLoss(
        total,
        {
            "masked_language_model": mlm,
            "image_text_matching": image_text_matching,
            "knowledge_graph_embedding": knowledge_graph_embedding,
        },
    )


def instruction_loss(
    generation: Tensor,
    alignment: Tensor,
    grounding: Tensor,
    hfs: Tensor,
    alignment_weight: float = 0.4,
    grounding_weight: float = 0.3,
    hfs_weight: float = 0.3,
) -> StageLoss:
    weights = (alignment_weight, grounding_weight, hfs_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("instruction loss weights must be nonnegative")
    total = (
        generation + alignment_weight * alignment + grounding_weight * grounding + hfs_weight * hfs
    )
    return StageLoss(
        total,
        {
            "generation": generation,
            "alignment": alignment,
            "grounding": grounding,
            "hierarchical_factuality": hfs,
        },
    )


def token_kl_divergence(
    policy_logits: Tensor, reference_logits: Tensor, mask: Tensor | None = None
) -> Tensor:
    if policy_logits.shape != reference_logits.shape or policy_logits.ndim != 3:
        raise ValueError("policy and reference logits must be equal three-dimensional tensors")
    policy_log_probabilities = functional.log_softmax(policy_logits, dim=-1)
    reference_log_probabilities = functional.log_softmax(reference_logits, dim=-1)
    policy_probabilities = policy_log_probabilities.exp()
    divergence = (
        policy_probabilities * (policy_log_probabilities - reference_log_probabilities)
    ).sum(dim=-1)
    if mask is None:
        return divergence.mean()
    if mask.shape != divergence.shape:
        raise ValueError("KL mask shape mismatch")
    weights = mask.to(divergence.dtype)
    return (divergence * weights).sum() / weights.sum().clamp_min(1.0)


def rlcf_loss(
    log_probabilities: Tensor,
    clinical_rewards: Tensor,
    kl_divergence: Tensor,
    kl_weight: float = 0.1,
    baseline: Tensor | None = None,
) -> StageLoss:
    if log_probabilities.shape != clinical_rewards.shape:
        raise ValueError("log probabilities and clinical rewards must match")
    if kl_weight < 0:
        raise ValueError("KL weight must be nonnegative")
    advantages = clinical_rewards if baseline is None else clinical_rewards - baseline
    policy = -(log_probabilities * advantages.detach()).mean()
    total = policy + kl_weight * kl_divergence
    return StageLoss(total, {"policy": policy, "kl_divergence": kl_divergence})


def grounding_ranking_loss(
    correct_scores: Tensor, incorrect_scores: Tensor, margin: float = 0.1
) -> Tensor:
    if correct_scores.shape != incorrect_scores.shape:
        raise ValueError("correct and incorrect grounding scores must match")
    if margin < 0:
        raise ValueError("grounding margin must be nonnegative")
    return functional.relu(margin - correct_scores + incorrect_scores).mean()


def clinical_reward(
    hfs: Tensor,
    serious_hallucination: Tensor,
    diagnostic_correctness: Tensor,
    hallucination_penalty: float = 1.0,
    diagnostic_weight: float = 0.5,
) -> Tensor:
    if hfs.shape != serious_hallucination.shape or hfs.shape != diagnostic_correctness.shape:
        raise ValueError("clinical reward components must match")
    return (
        hfs
        + diagnostic_weight * diagnostic_correctness
        - hallucination_penalty * serious_hallucination
    )
