from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def safe_makedirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seeds(seed_value: int, max_epochs: int = 0, distributed_rank: int = 0) -> None:
    seed = int(seed_value) + int(max_epochs) * 17 + int(distributed_rank) * 97
    random.seed(seed)
    np.random.seed(seed)


def check_and_fix_inf_nan(input_tensor, *args, hard_max: float = 1e8):
    _ = args
    tensor = jnp.asarray(input_tensor)
    tensor = jnp.nan_to_num(tensor, nan=0.0, posinf=hard_max, neginf=-hard_max)
    tensor = jnp.clip(tensor, -hard_max, hard_max)
    return tensor


@dataclass
class AverageMeter:
    name: str
    fmt: str = ":.4f"
    value: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / self.count


@dataclass
class DurationMeter:
    name: str
    device: Any = None
    fmt: str = ":.4f"
    _start: float = 0.0

    def __post_init__(self):
        self._start = time.time()

    def update(self) -> float:
        return time.time() - self._start


def is_sequence_of_primitives(data: Any) -> bool:
    if isinstance(data, (list, tuple)):
        return all(isinstance(x, (str, int, float, bool)) for x in data)
    return False


def copy_data_to_device(data, device, *args, **kwargs):
    _ = device, args, kwargs
    if isinstance(data, np.ndarray):
        return jnp.asarray(data)
    return data


def move_to_device(data, device):
    return copy_data_to_device(data, device)


def get_resume_checkpoint(save_dir: str) -> str | None:
    path = Path(save_dir)
    if not path.exists():
        return None
    candidates = sorted(path.glob("*.ckpt"))
    return str(candidates[-1]) if candidates else None


def model_summary(model, max_depth: int = 2, *args, **kwargs) -> str:
    _ = max_depth, args, kwargs
    return f"{type(model).__name__}"


__all__ = [
    "safe_makedirs",
    "set_seeds",
    "check_and_fix_inf_nan",
    "AverageMeter",
    "DurationMeter",
    "is_sequence_of_primitives",
    "copy_data_to_device",
    "move_to_device",
    "get_resume_checkpoint",
    "model_summary",
]
