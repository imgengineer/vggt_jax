from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import trimesh


def _to_nhwc_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must be 4D, got {images.shape}")
    if images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] == 3:
        return images
    raise ValueError(f"images must be NCHW or NHWC with 3 channels, got {images.shape}")


def _as_4x4(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.ndim != 3:
        raise ValueError(f"extrinsic must be [S,3,4] or [S,4,4], got {extrinsic.shape}")
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsic must be [S,3,4] or [S,4,4], got {extrinsic.shape}")
    out = np.tile(np.eye(4, dtype=np.float32), (extrinsic.shape[0], 1, 1))
    out[:, :3, :4] = extrinsic
    return out


def _select_frame_indices(filter_by_frames: str, num_frames: int) -> np.ndarray:
    if filter_by_frames in ("all", "All"):
        return np.arange(num_frames, dtype=np.int32)

    try:
        selected = int(str(filter_by_frames).split(":")[0])
    except (TypeError, ValueError, IndexError):
        return np.arange(num_frames, dtype=np.int32)

    if selected < 0 or selected >= num_frames:
        return np.arange(num_frames, dtype=np.int32)
    return np.asarray([selected], dtype=np.int32)


def _make_camera_marker(scale: float, color_rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    cone = trimesh.creation.cone(radius=max(scale * 0.03, 1e-4), height=max(scale * 0.08, 1e-4), sections=4)
    cone.apply_translation([0.0, 0.0, -scale * 0.04])
    cone.visual.face_colors[:, :3] = np.asarray(color_rgb, dtype=np.uint8)
    return cone


def _color_from_index(index: int, total: int) -> tuple[int, int, int]:
    if total <= 1:
        hue = 0.0
    else:
        hue = float(index) / float(total - 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def predictions_to_glb(
    predictions: dict[str, Any],
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: str | None = None,
    prediction_mode: str = "Predicted Pointmap",
) -> trimesh.Scene:
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if "images" not in predictions:
        raise ValueError("predictions must contain `images`")

    images_nhwc = _to_nhwc_images(np.asarray(predictions["images"]))

    if "Pointmap" in prediction_mode:
        world_points = np.asarray(predictions.get("world_points", predictions.get("world_points_from_depth")))
        conf_map = np.asarray(
            predictions.get("world_points_conf", predictions.get("depth_conf", np.ones(world_points.shape[:-1])))
        )
    else:
        world_points = np.asarray(predictions["world_points_from_depth"])
        conf_map = np.asarray(predictions.get("depth_conf", np.ones(world_points.shape[:-1])))

    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError(f"world_points must be [S,H,W,3], got {world_points.shape}")
    if conf_map.shape != world_points.shape[:-1]:
        raise ValueError(f"confidence map shape mismatch: {conf_map.shape} vs {world_points.shape[:-1]}")

    frame_indices = _select_frame_indices(filter_by_frames, world_points.shape[0])
    world_points = world_points[frame_indices]
    conf_map = conf_map[frame_indices]
    images_nhwc = images_nhwc[frame_indices]

    vertices = world_points.reshape(-1, 3)
    colors = np.clip(images_nhwc.reshape(-1, 3) * 255.0, 0.0, 255.0).astype(np.uint8)
    confidence = conf_map.reshape(-1)

    if mask_sky and target_dir is not None:
        sky_mask_dir = Path(target_dir) / "sky_masks"
        if sky_mask_dir.exists():
            sky_mask_list = sorted(sky_mask_dir.glob("*"))
            if len(sky_mask_list) >= len(frame_indices):
                masks = []
                for mask_path in sky_mask_list[: len(frame_indices)]:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    if mask.shape[:2] != conf_map.shape[1:3]:
                        mask = cv2.resize(mask, (conf_map.shape[2], conf_map.shape[1]), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask > 0)
                if len(masks) == len(frame_indices):
                    sky_conf_mask = np.stack(masks, axis=0).reshape(-1)
                    confidence = confidence * sky_conf_mask.astype(confidence.dtype)

    conf_thres = float(conf_thres)
    if confidence.size == 0:
        threshold = 0.0
    elif conf_thres <= 0.0:
        threshold = 0.0
    else:
        threshold = float(np.percentile(confidence, conf_thres))

    mask = (confidence >= threshold) & (confidence > 1e-6)

    if mask_black_bg:
        mask &= colors.sum(axis=1) > 16
    if mask_white_bg:
        mask &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    vertices = vertices[mask]
    colors = colors[mask]

    if vertices.shape[0] == 0:
        vertices = np.zeros((1, 3), dtype=np.float32)
        colors = np.full((1, 3), 255, dtype=np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    if show_cam and "extrinsic" in predictions:
        extrinsic = np.asarray(predictions["extrinsic"])
        extrinsic = _as_4x4(extrinsic)[frame_indices]

        if vertices.shape[0] >= 2:
            lo = np.percentile(vertices, 5, axis=0)
            hi = np.percentile(vertices, 95, axis=0)
            scene_scale = float(np.linalg.norm(hi - lo))
        else:
            scene_scale = 1.0
        scene_scale = max(scene_scale, 1e-3)

        for idx, world_to_cam in enumerate(extrinsic):
            cam_to_world = np.linalg.inv(world_to_cam)
            color = _color_from_index(idx, len(extrinsic))
            marker = _make_camera_marker(scene_scale, color)
            marker.apply_transform(cam_to_world)
            scene.add_geometry(marker)

    return scene


def download_file_from_url(url: str, output_path: str, *, timeout: int = 120) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    return str(output)


def segment_sky(image_path: str, onnx_session: Any, mask_path: str | None = None) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(rgb, (2, 0, 1))[None, ...]

    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: x})[0]
    mask = np.asarray(output)

    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        mask = mask[0]
    mask = cv2.resize(mask.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)

    if mask_path is not None:
        path = Path(mask_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), (mask * 255.0).astype(np.uint8))

    return mask
