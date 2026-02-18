from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .distortion import apply_distortion


def img_from_cam_np(
    intrinsics: np.ndarray,
    points_cam: np.ndarray,
    extra_params: np.ndarray | None = None,
    default: float = 0.0,
) -> np.ndarray:
    z = points_cam[:, 2:3, :]
    points_cam_norm = points_cam / z
    uv = points_cam_norm[:, :2, :]

    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = np.stack([np.asarray(uu), np.asarray(vv)], axis=1)

    ones = np.ones_like(uv[:, :1, :])
    points_cam_h = np.concatenate([uv, ones], axis=1)
    points2d_h = np.einsum("bij,bjk->bik", intrinsics, points_cam_h)
    points2d = np.nan_to_num(points2d_h[:, :2, :], nan=default)
    return points2d.transpose(0, 2, 1)


def project_3D_points_np(
    points3D: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray | None = None,
    extra_params: np.ndarray | None = None,
    *,
    default: float = 0.0,
    only_points_cam: bool = False,
):
    n = points3D.shape[0]
    batch = extrinsics.shape[0]

    points3d_h = np.concatenate([points3D, np.ones((n, 1), dtype=points3D.dtype)], axis=1)
    points3d_h = np.broadcast_to(points3d_h, (batch, n, 4))
    points_cam = np.einsum("bij,bnj->bni", extrinsics, points3d_h).transpose(0, 2, 1)

    if only_points_cam:
        return None, points_cam
    if intrinsics is None:
        raise ValueError("`intrinsics` must be provided unless only_points_cam=True")

    points2d = img_from_cam_np(intrinsics, points_cam, extra_params=extra_params, default=default)
    return points2d, points_cam


def img_from_cam(
    intrinsics: jnp.ndarray,
    points_cam: jnp.ndarray,
    extra_params: jnp.ndarray | None = None,
    default: float = 0.0,
) -> jnp.ndarray:
    points_cam = points_cam / points_cam[:, 2:3, :]
    uv = points_cam[:, :2, :]

    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = jnp.stack([jnp.asarray(uu), jnp.asarray(vv)], axis=1)

    points_cam_h = jnp.concatenate([uv, jnp.ones_like(uv[:, :1, :])], axis=1)
    points2d_h = jnp.einsum("bij,bjk->bik", intrinsics, points_cam_h)
    points2d = jnp.nan_to_num(points2d_h[:, :2, :], nan=default)
    return points2d.transpose(0, 2, 1)


def project_3D_points(
    points3D: jnp.ndarray,
    extrinsics: jnp.ndarray,
    intrinsics: jnp.ndarray | None = None,
    extra_params: jnp.ndarray | None = None,
    default: float = 0.0,
    only_points_cam: bool = False,
):
    n = points3D.shape[0]
    batch = extrinsics.shape[0]

    points3d_h = jnp.concatenate([points3D, jnp.ones_like(points3D[..., 0:1])], axis=1)
    points3d_h = jnp.broadcast_to(points3d_h, (batch, n, 4))
    points_cam = jnp.einsum("bij,bnj->bin", extrinsics, points3d_h)

    if only_points_cam:
        return None, points_cam
    if intrinsics is None:
        raise ValueError("`intrinsics` must be provided unless only_points_cam=True")

    points2d = img_from_cam(intrinsics, points_cam, extra_params, default)
    return points2d, points_cam


__all__ = ["img_from_cam_np", "project_3D_points_np", "img_from_cam", "project_3D_points"]
