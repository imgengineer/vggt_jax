from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image


def resolve_image_paths(image_dir: str) -> list[str]:
    root = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(str(p) for p in root.iterdir() if p.suffix.lower() in exts)


def _open_rgb_image(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def load_and_preprocess_images_square_np(
    image_paths: list[str],
    target_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Center-pad images to square then resize.

    Returns:
      - images: float32 array [N, target_size, target_size, 3] in [0, 1]
      - original_coords: float32 array [N, 6] storing
          [x1, y1, x2, y2, width, height] in the resized square canvas.
    """
    if len(image_paths) == 0:
        raise ValueError("At least 1 image is required")

    images: list[np.ndarray] = []
    original_coords: list[np.ndarray] = []

    for image_path in image_paths:
        img = _open_rgb_image(image_path)
        width, height = img.size

        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        scale = float(target_size) / float(max_dim)
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale
        original_coords.append(np.asarray([x1, y1, x2, y2, width, height], dtype=np.float32))

        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        arr = np.asarray(square_img, dtype=np.float32) / 255.0
        images.append(arr)

    stacked = np.stack(images, axis=0).astype(np.float32)
    coords = np.stack(original_coords, axis=0).astype(np.float32)
    return stacked, coords


def load_and_preprocess_images_square(
    image_paths: list[str],
    target_size: int = 1024,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    images, coords = load_and_preprocess_images_square_np(image_paths, target_size=target_size)
    return jnp.asarray(images), jnp.asarray(coords)
