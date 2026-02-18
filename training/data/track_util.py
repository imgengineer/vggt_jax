from __future__ import annotations

import numpy as np

from vggt.dependency.projection import project_3D_points_np


def _sample_positive_world_points(
    world_points0: np.ndarray,
    point_mask0: np.ndarray,
    target_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_ys, valid_xs = np.where(point_mask0 > 0)
    if valid_ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    take = min(target_num, valid_ys.size)
    picked = np.random.choice(valid_ys.size, size=take, replace=False)
    ys = valid_ys[picked]
    xs = valid_xs[picked]
    world = world_points0[ys, xs].astype(np.float32)
    uv = np.stack([xs, ys], axis=-1).astype(np.float32)
    return world, uv


def _project_tracks(
    sampled_world_points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depths: np.ndarray,
    rel_thres: float,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = extrinsics.shape[0]
    num_tracks = sampled_world_points.shape[0]
    if num_tracks == 0:
        return (
            np.zeros((seq_len, 0, 2), dtype=np.float32),
            np.zeros((seq_len, 0), dtype=bool),
        )

    uv, points_cam = project_3D_points_np(sampled_world_points, extrinsics, intrinsics)
    uv = uv.astype(np.float32)
    z = points_cam[:, 2, :].astype(np.float32)

    h, w = depths.shape[-2:]
    x = uv[..., 0]
    y = uv[..., 1]
    inside = (x >= 0) & (x <= (w - 1)) & (y >= 0) & (y <= (h - 1))

    x_int = np.clip(np.round(x).astype(np.int32), 0, w - 1)
    y_int = np.clip(np.round(y).astype(np.int32), 0, h - 1)
    sampled_depth = depths[np.arange(seq_len)[:, None], y_int, x_int]

    valid_depth = sampled_depth > 0
    close_depth = np.abs(sampled_depth - z) < np.maximum(sampled_depth, z) * rel_thres
    vis = inside & valid_depth & close_depth

    return uv, vis


def _build_negative_tracks(
    seq_len: int,
    num_tracks: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_tracks <= 0:
        return np.zeros((seq_len, 0, 2), dtype=np.float32), np.zeros((seq_len, 0), dtype=bool)

    u0 = np.random.randint(0, width, size=(1, num_tracks))
    v0 = np.random.randint(0, height, size=(1, num_tracks))
    uv0 = np.stack([u0, v0], axis=-1).astype(np.float32)

    jitter_u = np.random.uniform(-0.05 * width, 0.05 * width, size=(seq_len, num_tracks))
    jitter_v = np.random.uniform(-0.05 * height, 0.05 * height, size=(seq_len, num_tracks))
    tracks = np.zeros((seq_len, num_tracks, 2), dtype=np.float32)
    tracks[..., 0] = uv0[..., 0] + jitter_u
    tracks[..., 1] = uv0[..., 1] + jitter_v

    tracks[..., 0] = np.clip(tracks[..., 0], 0.0, float(width - 1))
    tracks[..., 1] = np.clip(tracks[..., 1], 0.0, float(height - 1))
    vis = np.zeros((seq_len, num_tracks), dtype=bool)
    return tracks, vis


def build_tracks_by_depth(
    extrinsics,
    intrinsics,
    world_points,
    depths,
    point_masks,
    images,
    pos_rel_thres: float = 0.05,
    neg_epipolar_thres: float = 16.0,
    target_track_num: int = 1024,
    seq_name: str | None = None,
):
    _ = neg_epipolar_thres, seq_name
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    world_points = np.asarray(world_points, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)
    point_masks = np.asarray(point_masks).astype(bool)
    images = np.asarray(images)

    seq_len, height, width = images.shape[0], images.shape[1], images.shape[2]
    target_track_num = int(target_track_num)

    max_positive = int(target_track_num * 1.5)
    sampled_world, _ = _sample_positive_world_points(
        world_points[0], point_masks[0], target_num=max_positive
    )
    positive_tracks, positive_vis = _project_tracks(
        sampled_world,
        extrinsics,
        intrinsics,
        depths,
        rel_thres=float(pos_rel_thres),
    )

    if positive_tracks.shape[1] > target_track_num:
        valid_count = positive_vis.sum(axis=0)
        keep = np.argsort(-valid_count)[:target_track_num]
        positive_tracks = positive_tracks[:, keep]
        positive_vis = positive_vis[:, keep]

    num_positive = positive_tracks.shape[1]
    num_negative = max(0, target_track_num - num_positive)
    negative_tracks, negative_vis = _build_negative_tracks(
        seq_len=seq_len,
        num_tracks=num_negative,
        height=height,
        width=width,
    )

    tracks = np.concatenate([positive_tracks, negative_tracks], axis=1).astype(np.float32)
    vis = np.concatenate([positive_vis, negative_vis], axis=1).astype(bool)
    pos_mask = np.zeros((tracks.shape[1],), dtype=bool)
    pos_mask[:num_positive] = True

    if tracks.shape[1] < target_track_num:
        pad = target_track_num - tracks.shape[1]
        tracks = np.concatenate([tracks, np.zeros((seq_len, pad, 2), dtype=np.float32)], axis=1)
        vis = np.concatenate([vis, np.zeros((seq_len, pad), dtype=bool)], axis=1)
        pos_mask = np.concatenate([pos_mask, np.zeros((pad,), dtype=bool)], axis=0)

    if tracks.shape[1] > target_track_num:
        tracks = tracks[:, :target_track_num]
        vis = vis[:, :target_track_num]
        pos_mask = pos_mask[:target_track_num]

    return tracks, vis, pos_mask
