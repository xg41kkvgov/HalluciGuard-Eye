"""Atomic checkpoints containing complete random state."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from halluciguard_eye.training.seed import RandomState, capture_random_state, restore_random_state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    epoch: int,
    stage: int,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "stage": stage,
        "seed": seed,
        "random_state": capture_random_state(seed),
        "metadata": dict(metadata or {}),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint is missing model state")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError("checkpoint is missing optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
    random_state = checkpoint.get("random_state")
    if restore_rng:
        if not isinstance(random_state, RandomState):
            raise ValueError("checkpoint is missing typed random state")
        restore_random_state(random_state)
    return {
        "step": int(checkpoint["step"]),
        "epoch": int(checkpoint["epoch"]),
        "stage": int(checkpoint["stage"]),
        "seed": int(checkpoint["seed"]),
        "metadata": dict(checkpoint.get("metadata", {})),
    }


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        "step": int(checkpoint["step"]),
        "epoch": int(checkpoint["epoch"]),
        "stage": int(checkpoint["stage"]),
        "seed": int(checkpoint["seed"]),
        "metadata": dict(checkpoint.get("metadata", {})),
    }
