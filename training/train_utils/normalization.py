from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from .general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor, name: str = "tensor") -> None:
    _ = name
    tensor = jnp.asarray(input_tensor)
    if jnp.any(~jnp.isfinite(tensor)):
        raise ValueError(f"{name} contains invalid values")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: jnp.ndarray,
    cam_points: Optional[jnp.ndarray] = None,
    world_points: Optional[jnp.ndarray] = None,
    depths: Optional[jnp.ndarray] = None,
    point_masks: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    extrinsics = check_and_fix_inf_nan(extrinsics)
    if cam_points is not None:
        cam_points = check_and_fix_inf_nan(cam_points)
    if world_points is not None:
        world_points = check_and_fix_inf_nan(world_points)
    if depths is not None:
        depths = check_and_fix_inf_nan(depths)
    _ = point_masks
    return extrinsics, cam_points, world_points, depths


__all__ = ["check_valid_tensor", "normalize_camera_extrinsics_and_points_batch"]
