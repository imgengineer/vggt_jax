from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def color_from_xy(x: float, y: float, W: int, H: int, cmap_name: str = "hsv") -> tuple[float, float, float]:
    import matplotlib.cm

    x_norm = x / max(W - 1, 1)
    y_norm = y / max(H - 1, 1)
    c = (x_norm + y_norm) / 2.0
    r, g, b, _ = matplotlib.cm.get_cmap(cmap_name)(c)
    return float(r), float(g), float(b)


def get_track_colors_by_position(
    tracks_b: np.ndarray,
    vis_mask_b: np.ndarray | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    cmap_name: str = "hsv",
) -> np.ndarray:
    tracks_b = np.asarray(tracks_b)
    if tracks_b.ndim != 3 or tracks_b.shape[-1] != 2:
        raise ValueError(f"Expected tracks shape [S,N,2], got {tracks_b.shape}")
    seq_len, num_tracks, _ = tracks_b.shape

    if vis_mask_b is None:
        vis_mask_b = np.ones((seq_len, num_tracks), dtype=bool)
    else:
        vis_mask_b = np.asarray(vis_mask_b).astype(bool)

    if image_width is None or image_height is None:
        raise ValueError("image_width and image_height are required")

    colors = np.zeros((num_tracks, 3), dtype=np.uint8)
    for track_idx in range(num_tracks):
        visible = np.where(vis_mask_b[:, track_idx])[0]
        if visible.size == 0:
            continue
        first = int(visible[0])
        x, y = tracks_b[first, track_idx].tolist()
        r, g, b = color_from_xy(x, y, W=image_width, H=image_height, cmap_name=cmap_name)
        colors[track_idx] = np.asarray([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
    return colors


def _to_numpy_image_sequence(images, image_format: str) -> np.ndarray:
    images_np = np.asarray(images)
    if images_np.ndim == 5:
        if images_np.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 5D images, got {images_np.shape}")
        images_np = images_np[0]

    if images_np.ndim != 4:
        raise ValueError(f"Expected image tensor shape [S,3,H,W] or [S,H,W,3], got {images_np.shape}")

    if image_format.upper() == "CHW":
        if images_np.shape[1] != 3:
            raise ValueError(f"Expected CHW images with C=3, got {images_np.shape}")
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    elif image_format.upper() == "HWC":
        if images_np.shape[-1] != 3:
            raise ValueError(f"Expected HWC images with C=3, got {images_np.shape}")
    else:
        raise ValueError(f"Unknown image_format={image_format}")
    return images_np


def visualize_tracks_on_images(
    images,
    tracks,
    track_vis_mask=None,
    out_dir: str = "track_visuals_concat_by_xy",
    image_format: str = "CHW",
    normalize_mode: str | None = "[0,1]",
    cmap_name: str = "hsv",
    frames_per_row: int = 4,
    save_grid: bool = True,
):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV is required for visualize_tracks_on_images. Install `opencv-python`.") from exc

    tracks_np = np.asarray(tracks)
    if tracks_np.ndim == 4:
        if tracks_np.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for tracks, got {tracks_np.shape}")
        tracks_np = tracks_np[0]
        if track_vis_mask is not None:
            track_vis_mask = np.asarray(track_vis_mask)[0]

    if tracks_np.ndim != 3 or tracks_np.shape[-1] != 2:
        raise ValueError(f"Expected tracks shape [S,N,2], got {tracks_np.shape}")

    images_np = _to_numpy_image_sequence(images, image_format=image_format)
    seq_len, height, width, _ = images_np.shape
    if tracks_np.shape[0] != seq_len:
        raise ValueError(f"tracks frame count {tracks_np.shape[0]} != image frame count {seq_len}")

    if track_vis_mask is not None:
        vis_mask_np = np.asarray(track_vis_mask).astype(bool)
        if vis_mask_np.ndim == 3:
            if vis_mask_np.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for track_vis_mask, got {vis_mask_np.shape}")
            vis_mask_np = vis_mask_np[0]
    else:
        vis_mask_np = None

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    colors_rgb = get_track_colors_by_position(
        tracks_np,
        vis_mask_b=vis_mask_np,
        image_width=width,
        image_height=height,
        cmap_name=cmap_name,
    )

    frame_images: list[np.ndarray] = []
    for frame_idx in range(seq_len):
        img = images_np[frame_idx].astype(np.float32)
        if normalize_mode == "[0,1]":
            img = np.clip(img, 0.0, 1.0) * 255.0
        elif normalize_mode == "[-1,1]":
            img = np.clip((img + 1.0) * 0.5, 0.0, 1.0) * 255.0
        img_uint8 = img.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        if vis_mask_np is None:
            valid_indices = range(tracks_np.shape[1])
        else:
            valid_indices = np.where(vis_mask_np[frame_idx])[0]

        for track_idx in valid_indices:
            x, y = tracks_np[frame_idx, track_idx]
            r, g, b = colors_rgb[track_idx]
            cv2.circle(img_bgr, (int(round(x)), int(round(y))), radius=3, color=(int(b), int(g), int(r)), thickness=-1)

        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame_images.append(frame_rgb)
        cv2.imwrite(str(out_path / f"frame_{frame_idx:04d}.png"), img_bgr)

    if save_grid:
        num_rows = (seq_len + frames_per_row - 1) // frames_per_row
        rows = []
        for row in range(num_rows):
            start = row * frames_per_row
            end = min(start + frames_per_row, seq_len)
            row_img = np.concatenate(frame_images[start:end], axis=1)
            if end - start < frames_per_row:
                pad_width = (frames_per_row - (end - start)) * width
                pad = np.zeros((height, pad_width, 3), dtype=np.uint8)
                row_img = np.concatenate([row_img, pad], axis=1)
            rows.append(row_img)
        grid_rgb = np.concatenate(rows, axis=0)
        cv2.imwrite(str(out_path / "tracks_grid.png"), cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))

    print(f"[INFO] Saved {seq_len} frames to {out_path}")


__all__ = ["color_from_xy", "get_track_colors_by_position", "visualize_tracks_on_images"]
