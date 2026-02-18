from __future__ import annotations

import numpy as np


def quat_to_mat(quaternions: np.ndarray) -> np.ndarray:
    """Quaternion (XYZW, scalar-last) to rotation matrix."""
    quaternions = np.asarray(quaternions, dtype=np.float32)
    i, j, k, r = np.split(quaternions, 4, axis=-1)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1, keepdims=True)

    o = np.concatenate(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    out_shape = quaternions.shape[:-1] + (3, 3)
    return o.reshape(out_shape).astype(np.float32)

