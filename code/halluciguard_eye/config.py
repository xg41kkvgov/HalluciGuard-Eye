"""Typed configuration for the architecture and experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

T = TypeVar("T")


def _required(mapping: Mapping[str, Any], key: str, expected: type[T]) -> T:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise ValueError(f"{key} must have type {expected.__name__}")
    return cast(T, value)


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


@dataclass(frozen=True)
class ModelConfig:
    image_size: int
    patch_size: int
    vision_width: int
    language_width: int
    knowledge_width: int
    alignment_width: int
    evidence_width: int
    vision_layers: int
    decoder_layers: int
    attention_heads: int
    max_text_tokens: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ModelConfig:
        values = {name: int(_number(raw, name)) for name in cls.__dataclass_fields__}
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image and patch sizes must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image size must be divisible by patch size")
        if self.vision_width % self.attention_heads:
            raise ValueError("vision width must be divisible by attention heads")
        if self.language_width % self.attention_heads:
            raise ValueError("language width must be divisible by attention heads")
        for value in (
            self.vision_width,
            self.language_width,
            self.knowledge_width,
            self.alignment_width,
            self.evidence_width,
            self.vision_layers,
            self.decoder_layers,
            self.attention_heads,
            self.max_text_tokens,
        ):
            if value <= 0:
                raise ValueError("model dimensions and layer counts must be positive")


@dataclass(frozen=True)
class KnowledgeConfig:
    entity_count: int
    relation_count: int
    triplet_count: int
    embedding_dimension: int
    margin: float
    negative_sampling_ratio: int
    embedding_epochs: int
    embedding_learning_rate: float
    retrieval_threshold: float
    expansion_threshold: float
    reasoning_hops: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> KnowledgeConfig:
        config = cls(
            entity_count=int(_number(raw, "entity_count")),
            relation_count=int(_number(raw, "relation_count")),
            triplet_count=int(_number(raw, "triplet_count")),
            embedding_dimension=int(_number(raw, "embedding_dimension")),
            margin=_number(raw, "margin"),
            negative_sampling_ratio=int(_number(raw, "negative_sampling_ratio")),
            embedding_epochs=int(_number(raw, "embedding_epochs")),
            embedding_learning_rate=_number(raw, "embedding_learning_rate"),
            retrieval_threshold=_number(raw, "retrieval_threshold"),
            expansion_threshold=_number(raw, "expansion_threshold"),
            reasoning_hops=int(_number(raw, "reasoning_hops")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            self.entity_count,
            self.relation_count,
            self.triplet_count,
            self.embedding_dimension,
            self.negative_sampling_ratio,
            self.embedding_epochs,
            self.reasoning_hops,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("knowledge graph counts must be positive")
        if self.margin <= 0 or self.embedding_learning_rate <= 0:
            raise ValueError("embedding optimization values must be positive")
        if not 0 <= self.retrieval_threshold <= 1:
            raise ValueError("retrieval threshold must be in [0, 1]")
        if not 0 <= self.expansion_threshold <= 1:
            raise ValueError("expansion threshold must be in [0, 1]")


@dataclass(frozen=True)
class GenerationConfig:
    candidates: int
    alignment_weight: float
    grounding_weight: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GenerationConfig:
        config = cls(
            candidates=int(_number(raw, "candidates")),
            alignment_weight=_number(raw, "alignment_weight"),
            grounding_weight=_number(raw, "grounding_weight"),
        )
        if config.candidates <= 0:
            raise ValueError("candidate count must be positive")
        return config


@dataclass(frozen=True)
class LossConfig:
    contrastive_temperature: float
    pretrain_itm_weight: float
    pretrain_kge_weight: float
    instruct_alignment_weight: float
    instruct_grounding_weight: float
    instruct_hfs_weight: float
    rlcf_kl_weight: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> LossConfig:
        values = {name: _number(raw, name) for name in cls.__dataclass_fields__}
        config = cls(**values)
        if config.contrastive_temperature <= 0:
            raise ValueError("contrastive temperature must be positive")
        if any(value < 0 for value in values.values()):
            raise ValueError("loss weights must be nonnegative")
        return config


@dataclass(frozen=True)
class HFSConfig:
    anatomical_weight: float
    pathological_weight: float
    recommendation_weight: float
    recommendation_max_path: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> HFSConfig:
        config = cls(
            anatomical_weight=_number(raw, "anatomical_weight"),
            pathological_weight=_number(raw, "pathological_weight"),
            recommendation_weight=_number(raw, "recommendation_weight"),
            recommendation_max_path=int(_number(raw, "recommendation_max_path")),
        )
        total = config.anatomical_weight + config.pathological_weight + config.recommendation_weight
        if abs(total - 1.0) > 1e-8:
            raise ValueError("HFS weights must sum to one")
        if config.recommendation_max_path <= 0:
            raise ValueError("recommendation path length must be positive")
        return config


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    stage_epochs: tuple[int, int, int]
    world_size: int
    mixed_precision: bool
    scheduler: str
    warmup_steps: int
    weight_decay: float
    gradient_clipping: float
    precision: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TrainingConfig:
        stages = raw.get("stage_epochs")
        if not isinstance(stages, list) or len(stages) != 3:
            raise ValueError("stage_epochs must have three values")
        config = cls(
            learning_rate=_number(raw, "learning_rate"),
            batch_size=int(_number(raw, "batch_size")),
            stage_epochs=cast(tuple[int, int, int], tuple(int(value) for value in stages)),
            world_size=int(_number(raw, "world_size")),
            mixed_precision=_required(raw, "mixed_precision", bool),
            scheduler=_required(raw, "scheduler", str),
            warmup_steps=int(_number(raw, "warmup_steps"))
            if isinstance(raw.get("warmup_steps"), int | float)
            else 0,
            weight_decay=_number(raw, "weight_decay")
            if isinstance(raw.get("weight_decay"), int | float)
            else 0.01,
            gradient_clipping=_number(raw, "gradient_clipping")
            if isinstance(raw.get("gradient_clipping"), int | float)
            else 1.0,
            precision=_required(raw, "precision", str),
        )
        if config.learning_rate <= 0 or config.batch_size <= 0 or config.world_size <= 0:
            raise ValueError("training scale values must be positive")
        if any(value <= 0 for value in config.stage_epochs):
            raise ValueError("stage epochs must be positive")
        return config


@dataclass(frozen=True)
class ProjectConfig:
    project_title: str
    seed: int
    model: ModelConfig
    knowledge: KnowledgeConfig
    generation: GenerationConfig
    loss: LossConfig
    hfs: HFSConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ProjectConfig:
        return cls(
            project_title=_required(raw, "project_title", str),
            seed=int(_number(raw, "seed")),
            model=ModelConfig.from_mapping(_required(raw, "model", dict)),
            knowledge=KnowledgeConfig.from_mapping(_required(raw, "knowledge", dict)),
            generation=GenerationConfig.from_mapping(_required(raw, "generation", dict)),
            loss=LossConfig.from_mapping(_required(raw, "loss", dict)),
            hfs=HFSConfig.from_mapping(_required(raw, "hfs", dict)),
            training=TrainingConfig.from_mapping(_required(raw, "training", dict)),
        )


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ProjectConfig.from_mapping(raw)
