"""Three-stage training engine. Ref: Sec. III-E and Eq. (23)-(25)."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from halluciguard_eye.losses.stages import StageLoss


class TrainingStage(IntEnum):
    KNOWLEDGE_PRETRAINING = 1
    FACTUALITY_INSTRUCTION = 2
    CLINICAL_FEEDBACK = 3


@dataclass(frozen=True)
class TrainingBatch:
    images: Tensor
    token_ids: Tensor
    labels: Tensor
    response_entity_ids: Tensor
    evidence_entity_ids: Tensor
    evidence_token_ids: Tensor
    hfs_targets: Tensor
    token_mask: Tensor | None = None

    def to(self, device: torch.device | str) -> TrainingBatch:
        return TrainingBatch(
            images=self.images.to(device),
            token_ids=self.token_ids.to(device),
            labels=self.labels.to(device),
            response_entity_ids=self.response_entity_ids.to(device),
            evidence_entity_ids=self.evidence_entity_ids.to(device),
            evidence_token_ids=self.evidence_token_ids.to(device),
            hfs_targets=self.hfs_targets.to(device),
            token_mask=self.token_mask.to(device) if self.token_mask is not None else None,
        )


@dataclass(frozen=True)
class StepResult:
    loss: float
    components: Mapping[str, float]
    gradient_norm: float
    learning_rate: float


LossFunction = Callable[[nn.Module, TrainingBatch, TrainingStage], StageLoss]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: LossFunction,
        device: torch.device | str,
        gradient_clipping: float = 1.0,
        gradient_accumulation: int = 1,
        mixed_precision: bool = False,
    ) -> None:
        if gradient_clipping <= 0 or gradient_accumulation <= 0:
            raise ValueError("gradient controls must be positive")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation = gradient_accumulation
        self.mixed_precision = mixed_precision
        self.step = 0

    def _autocast(self) -> contextlib.AbstractContextManager[None]:
        if not self.mixed_precision:
            return contextlib.nullcontext()
        device_type = self.device.type
        dtype = torch.bfloat16 if device_type in {"cpu", "cuda"} else torch.float16
        return torch.autocast(device_type=device_type, dtype=dtype)

    def train_step(self, batches: Iterable[TrainingBatch], stage: TrainingStage) -> StepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_value = 0.0
        component_values: dict[str, float] = {}
        consumed = 0
        for consumed, batch in enumerate(batches, start=1):
            if consumed > self.gradient_accumulation:
                raise ValueError("more batches supplied than gradient accumulation permits")
            batch = batch.to(self.device)
            with self._autocast():
                result = self.loss_function(self.model, batch, stage)
                loss = result.total / self.gradient_accumulation
            loss.backward()
            total_value += float(result.total.detach().item())
            for name, value in result.components.items():
                component_values[name] = component_values.get(name, 0.0) + float(
                    value.detach().item()
                )
        if consumed == 0:
            raise ValueError("training step requires at least one batch")
        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("gradient norm is not finite")
        self.optimizer.step()
        self.step += 1
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        divisor = float(consumed)
        return StepResult(
            total_value / divisor,
            {name: value / divisor for name, value in component_values.items()},
            float(gradient_norm.item()),
            learning_rate,
        )

    @torch.no_grad()
    def evaluate(self, batches: Iterable[TrainingBatch], stage: TrainingStage) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for current_count, batch in enumerate(batches, start=1):
            count = current_count
            result = self.loss_function(self.model, batch.to(self.device), stage)
            totals["loss"] = totals.get("loss", 0.0) + float(result.total.item())
            for name, value in result.components.items():
                totals[name] = totals.get(name, 0.0) + float(value.item())
        if count == 0:
            raise ValueError("evaluation requires at least one batch")
        return {name: value / count for name, value in totals.items()}


def batches_of(values: Iterable[TrainingBatch], count: int) -> Iterator[tuple[TrainingBatch, ...]]:
    if count <= 0:
        raise ValueError("batch group count must be positive")
    group: list[TrainingBatch] = []
    for value in values:
        group.append(value)
        if len(group) == count:
            yield tuple(group)
            group = []
    if group:
        yield tuple(group)
