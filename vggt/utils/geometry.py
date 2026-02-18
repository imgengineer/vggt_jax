from __future__ import annotations

import numpy as np


def closed_form_inverse_se3(se3: np.ndarray, R: np.ndarray | None = None, T: np.ndarray | None = None) -> np.ndarray:
    se3 = np.asarray(se3)
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    R_transposed = np.transpose(R, (0, 2, 1))
    top_right = -np.matmul(R_transposed, T)
    inverted_matrix = np.tile(np.eye(4, dtype=se3.dtype), (len(R_transposed), 1, 1))
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth_map = np.asarray(depth_map, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    H, W = depth_map.shape

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_map is None:
        raise ValueError("depth_map is None")

    depth_map = np.asarray(depth_map, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)

    point_mask = depth_map > eps
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    return world_coords_points.astype(np.float32), cam_coords_points.astype(np.float32), point_mask


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    depth_map = np.asarray(depth_map)
    extrinsics_cam = np.asarray(extrinsics_cam)
    intrinsics_cam = np.asarray(intrinsics_cam)

    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx],
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points_list.append(cur_world_points)
    return np.stack(world_points_list, axis=0).astype(np.float32)

