"""Dual-stream HalluciGuard-Eye network. Ref: Fig. 1, Fig. 2, and Eq. (3)-(18)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from halluciguard_eye.config import ModelConfig
from halluciguard_eye.models.decoder import ClinicalLanguageDecoder, DecoderOutput
from halluciguard_eye.models.grounding import ClinicalFactualityGroundingNetwork, GroundingOutput
from halluciguard_eye.models.vision import VisionOutput, VisionTransformer
from halluciguard_eye.models.vsam import AlignmentOutput, VisualSemanticAlignmentModule


@dataclass(frozen=True)
class ModelOutput:
    vision: VisionOutput
    decoder: DecoderOutput
    alignment: AlignmentOutput
    grounding: GroundingOutput
    logits: Tensor


class HalluciGuardEye(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary_size: int,
        entity_count: int,
        evidence_vocabulary: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        evidence_vocabulary = evidence_vocabulary or vocabulary_size
        self.vision = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            width=config.vision_width,
            layers=config.vision_layers,
            heads=config.attention_heads,
        )
        self.decoder = ClinicalLanguageDecoder(
            vocabulary_size=vocabulary_size,
            width=config.language_width,
            layers=config.decoder_layers,
            heads=config.attention_heads,
            max_tokens=config.max_text_tokens,
            visual_width=config.vision_width,
        )
        self.alignment = VisualSemanticAlignmentModule(
            visual_width=config.vision_width,
            text_width=config.language_width,
            attention_width=config.alignment_width,
            shared_width=config.language_width,
        )
        self.grounding = ClinicalFactualityGroundingNetwork(
            entity_count=entity_count,
            knowledge_width=config.knowledge_width,
            hidden_width=config.language_width,
            attention_width=config.language_width,
            evidence_vocabulary=evidence_vocabulary,
            evidence_layers=max(1, min(4, config.decoder_layers)),
            evidence_heads=config.attention_heads,
            max_evidence_tokens=config.max_text_tokens,
        )
        self.factuality_head = nn.Sequential(
            nn.Linear(config.language_width, config.language_width),
            nn.GELU(),
            nn.Linear(config.language_width, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
        response_entity_ids: Tensor,
        evidence_entity_ids: Tensor,
        evidence_token_ids: Tensor,
        token_mask: Tensor | None = None,
        response_entity_mask: Tensor | None = None,
        evidence_token_mask: Tensor | None = None,
    ) -> ModelOutput:
        vision = self.vision(images)
        decoder = self.decoder(token_ids, vision.pooled)
        visual_mask = torch.ones(vision.tokens.shape[:2], dtype=torch.bool, device=images.device)
        alignment = self.alignment(
            vision.tokens,
            decoder.hidden,
            visual_mask,
            token_mask,
        )
        alignment_context = alignment.visual_context.mean(dim=1)
        grounding = self.grounding(
            decoder.hidden,
            alignment_context,
            response_entity_ids,
            evidence_entity_ids,
            evidence_token_ids,
            response_entity_mask,
            evidence_token_mask,
        )
        logits = self.decoder.output(self.decoder.norm(grounding.fusion.hidden))
        return ModelOutput(vision, decoder, alignment, grounding, logits)

    def factuality_components(self, output: ModelOutput, mask: Tensor | None = None) -> Tensor:
        hidden = output.grounding.fusion.hidden
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.to(hidden.dtype)
            pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.factuality_head(pooled)

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim < 2 or name.endswith("bias") or "norm" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def freeze_vision(self) -> None:
        for parameter in self.vision.parameters():
            parameter.requires_grad = False

    def unfreeze_vision(self) -> None:
        for parameter in self.vision.parameters():
            parameter.requires_grad = True
