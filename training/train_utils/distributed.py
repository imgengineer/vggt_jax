from __future__ import annotations

import os


def is_dist_avail_and_initialized() -> bool:
    return False


def get_rank() -> int:
    return 0


def get_world_size() -> int:
    return 1


def get_machine_local_and_dist_rank() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist_rank = int(os.environ.get("RANK", "0"))
    return local_rank, dist_rank


def barrier() -> None:
    return None


__all__ = [
    "is_dist_avail_and_initialized",
    "get_rank",
    "get_world_size",
    "get_machine_local_and_dist_rank",
    "barrier",
]
