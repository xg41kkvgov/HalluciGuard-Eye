"""Random state control and checkpoint restoration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RandomState:
    seed: int
    python: object
    numpy: dict[str, Any]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]


def set_seed(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic


def capture_random_state(seed: int) -> RandomState:
    cuda_states = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return RandomState(
        seed=seed,
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=cuda_states,
    )


def restore_random_state(state: RandomState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state.torch_cuda))
