"""Training, optimization, checkpoints, and distributed utilities."""

from halluciguard_eye.training.checkpoint import load_checkpoint, save_checkpoint
from halluciguard_eye.training.engine import Trainer, TrainingBatch
from halluciguard_eye.training.schedule import ScheduleConfig, cosine_schedule
from halluciguard_eye.training.seed import set_seed

__all__ = [
    "ScheduleConfig",
    "Trainer",
    "TrainingBatch",
    "cosine_schedule",
    "load_checkpoint",
    "save_checkpoint",
    "set_seed",
]
