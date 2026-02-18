from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np


def seed_worker(worker_id: int, base_seed: int, epoch: int = 0) -> int:
    worker_seed = int(base_seed) + int(epoch) * 997 + int(worker_id) * 37
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    return worker_seed


def get_worker_info() -> dict:
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def get_worker_init_fn(
    seed: int,
    num_workers: int,
    epoch: int,
    worker_init_fn: Optional[Callable] = None,
) -> Callable[[int], None]:
    _ = num_workers

    def _init(worker_id: int):
        seed_worker(worker_id=worker_id, base_seed=seed, epoch=epoch)
        if worker_init_fn is not None:
            worker_init_fn(worker_id)

    return _init
