from __future__ import annotations

from typing import Any
import jax.numpy as jnp
import numpy as np

from .vggsfm_tracker import TrackerPredictor


def build_vggsfm_tracker(model_path: str | None = None):
    _ = model_path
    return TrackerPredictor().eval()


def _to_shwc(images: jnp.ndarray | np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"Expected shape [S,3,H,W] or [S,H,W,3], got {arr.shape}")
    if arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Cannot infer channel axis from shape {arr.shape}")


def farthest_point_sampling(distance_matrix: np.ndarray, num_samples: int, most_common_frame_index: int = 0) -> list[int]:
    dist = np.clip(np.asarray(distance_matrix), a_min=0.0, a_max=None)
    num_frames = dist.shape[0]
    selected = [int(most_common_frame_index)]
    check_distances = dist[selected[0]].copy()
    check_distances[selected[0]] = 0.0

    while len(selected) < min(int(num_samples), num_frames):
        farthest = int(np.argmax(check_distances))
        selected.append(farthest)
        check_distances = dist[farthest].copy()
        check_distances[selected] = 0.0
    return selected


def generate_rank_by_dino(
    images,
    query_frame_num: int,
    image_size: int = 336,
    model_name: str = "dinov2_vitb14_reg",
    device: str = "cuda",
    spatial_similarity: bool = False,
):
    _ = image_size, model_name, device, spatial_similarity
    frames = _to_shwc(images)
    feat = frames.reshape(frames.shape[0], -1).astype(np.float32)
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)

    sim = feat @ feat.T
    np.fill_diagonal(sim, -1e9)
    similarity_sum = sim.sum(axis=1)
    most_common = int(np.argmax(similarity_sum))
    distance = 100.0 - sim
    return farthest_point_sampling(distance, query_frame_num, most_common)


def calculate_index_mappings(query_index: int, S: int, device: Any | None = None):
    _ = device
    new_order = np.arange(S, dtype=np.int32)
    new_order[0], new_order[query_index] = query_index, 0
    return new_order


def switch_tensor_order(tensors, order, dim: int = 1):
    order = np.asarray(order, dtype=np.int32)
    outputs = []
    for tensor in tensors:
        if tensor is None:
            outputs.append(None)
            continue
        arr = jnp.asarray(tensor)
        outputs.append(jnp.take(arr, jnp.asarray(order), axis=dim))
    return outputs


def initialize_feature_extractors(
    max_query_num: int,
    det_thres: float = 0.005,
    extractor_method: str = "aliked",
    device: str = "cuda",
):
    _ = det_thres, device
    methods = [m.strip().lower() for m in extractor_method.split("+") if m.strip()]
    if not methods:
        methods = ["good"]
    return {name: {"max_query_num": int(max_query_num)} for name in methods}


def _image_to_gray_u8(query_image) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV is required for keypoint extraction. Install `opencv-python`.") from exc

    img = np.asarray(query_image)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape [3,H,W] or [H,W,3], got {img.shape}")
    img = np.clip(img, 0.0, 1.0)
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray


def extract_keypoints(query_image, extractors, round_keypoints: bool = True):
    gray = _image_to_gray_u8(query_image)
    keypoints: list[np.ndarray] = []

    for cfg in extractors.values():
        max_num = int(cfg.get("max_query_num", 2048))
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_num,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3,
        )
        if points is None:
            continue
        points = points.reshape(-1, 2)
        if round_keypoints:
            points = np.round(points)
        keypoints.append(points.astype(np.float32))

    if not keypoints:
        h, w = gray.shape
        ys = np.linspace(0, h - 1, num=16, dtype=np.float32)
        xs = np.linspace(0, w - 1, num=16, dtype=np.float32)
        mesh = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        keypoints = [mesh]

    points_all = np.concatenate(keypoints, axis=0)
    return jnp.asarray(points_all[None, ...], dtype=jnp.float32)


def predict_tracks_in_chunks(
    track_predictor,
    images_feed,
    query_points_list,
    fmaps_feed,
    fine_tracking: bool,
    num_splits: int | None = None,
    fine_chunk: int = 40960,
):
    _ = num_splits, fine_chunk
    if not isinstance(query_points_list, (list, tuple)):
        query_points_list = [query_points_list]

    track_list = []
    vis_list = []
    score_list = []

    for split_points in query_points_list:
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed, split_points, fmaps=fmaps_feed, fine_tracking=fine_tracking
        )
        track_list.append(jnp.asarray(fine_pred_track))
        vis_list.append(jnp.asarray(pred_vis))
        score_list.append(jnp.asarray(pred_score))

    fine_pred_track = jnp.concatenate(track_list, axis=2)
    pred_vis = jnp.concatenate(vis_list, axis=2)
    pred_score = jnp.concatenate(score_list, axis=2) if score_list else None
    return fine_pred_track, pred_vis, pred_score


__all__ = [
    "build_vggsfm_tracker",
    "generate_rank_by_dino",
    "farthest_point_sampling",
    "calculate_index_mappings",
    "switch_tensor_order",
    "initialize_feature_extractors",
    "extract_keypoints",
    "predict_tracks_in_chunks",
]
