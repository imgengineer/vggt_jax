from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .vggsfm_utils import (
    build_vggsfm_tracker,
    calculate_index_mappings,
    extract_keypoints,
    generate_rank_by_dino,
    initialize_feature_extractors,
    predict_tracks_in_chunks,
    switch_tensor_order,
)


def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
    keypoint_extractor: str = "good",
    max_points_num: int = 163840,
    fine_tracking: bool = True,
    complete_non_vis: bool = True,
):
    _ = masks, complete_non_vis
    images = jnp.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"Expected images shape [S,C,H,W] or [S,H,W,C], got {images.shape}")
    if images.shape[1] == 3:
        images_b = images[None, ...]
    elif images.shape[-1] == 3:
        images_b = images.transpose(0, 3, 1, 2)[None, ...]
    else:
        raise ValueError(f"Cannot infer channel axis from {images.shape}")

    tracker = build_vggsfm_tracker()
    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num)
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    keypoint_extractors = initialize_feature_extractors(max_query_pts, extractor_method=keypoint_extractor)
    fmaps_for_tracker = tracker.process_images_to_fmaps(images_b)

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    for query_index in query_frame_indexes:
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images_b[0],
            conf,
            points_3d,
            fmaps_for_tracker[0],
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
        )
        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        if pred_conf is not None:
            pred_confs.append(pred_conf)
        if pred_point_3d is not None:
            pred_points_3d.append(pred_point_3d)
        if pred_color is not None:
            pred_colors.append(pred_color)

    tracks = np.concatenate(pred_tracks, axis=1) if pred_tracks else np.zeros((images.shape[0], 0, 2), dtype=np.float32)
    vis_scores = np.concatenate(pred_vis_scores, axis=1) if pred_vis_scores else np.zeros((images.shape[0], 0), dtype=np.float32)
    confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    points3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    colors = np.concatenate(pred_colors, axis=0) if pred_colors else None
    return tracks, vis_scores, confs, points3d, colors


def _forward_on_query(
    query_index: int,
    images: jnp.ndarray,
    conf,
    points_3d,
    fmaps_for_tracker: jnp.ndarray,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
):
    _ = max_points_num
    frame_num = images.shape[0]
    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)

    query_points_int = np.asarray(jnp.round(query_points[0]).astype(jnp.int32))
    query_points_int[:, 0] = np.clip(query_points_int[:, 0], 0, images.shape[-1] - 1)
    query_points_int[:, 1] = np.clip(query_points_int[:, 1], 0, images.shape[-2] - 1)
    colors = np.asarray(images[query_index][:, query_points_int[:, 1], query_points_int[:, 0]].transpose(1, 0))
    colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    pred_conf = None
    pred_point_3d = None
    if conf is not None and points_3d is not None:
        conf_np = np.asarray(conf)
        points_np = np.asarray(points_3d)
        pred_conf = conf_np[query_index][query_points_int[:, 1], query_points_int[:, 0]]
        pred_point_3d = points_np[query_index][query_points_int[:, 1], query_points_int[:, 0]]

    reorder = calculate_index_mappings(query_index, frame_num)
    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder, dim=0)
    images_feed = images_feed[None, ...]
    fmaps_feed = fmaps_feed[None, ...]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, [query_points], fmaps_feed, fine_tracking=fine_tracking
    )
    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder, dim=1)
    pred_track = np.asarray(pred_track[0])
    pred_vis = np.asarray(pred_vis[0])
    return pred_track, pred_vis, pred_conf, pred_point_3d, colors


__all__ = ["predict_tracks"]
