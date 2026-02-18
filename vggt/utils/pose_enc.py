from __future__ import annotations

import numpy as np

from vggt.utils.rotation import quat_to_mat


def pose_encoding_to_extri_intri(
    pose_encoding: np.ndarray,
    image_size_hw: tuple[int, int] | None = None,
    *,
    pose_encoding_type: str = "absT_quaR_FoV",
    build_intrinsics: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    intrinsics = None

    pose_encoding = np.asarray(pose_encoding, dtype=np.float32)
    if pose_encoding.ndim == 2:
        pose_encoding = pose_encoding[None, ...]

    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError(f"Unsupported pose_encoding_type: {pose_encoding_type}")

    if image_size_hw is None and build_intrinsics:
        raise ValueError("image_size_hw is required when build_intrinsics=True")

    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = np.concatenate([R, T[..., None]], axis=-1).astype(np.float32)

    if build_intrinsics:
        H, W = image_size_hw  # type: ignore[misc]
        fy = (H / 2.0) / np.tan(fov_h / 2.0)
        fx = (W / 2.0) / np.tan(fov_w / 2.0)
        intrinsics = np.zeros(pose_encoding.shape[:2] + (3, 3), dtype=np.float32)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2.0
        intrinsics[..., 1, 2] = H / 2.0
        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics

