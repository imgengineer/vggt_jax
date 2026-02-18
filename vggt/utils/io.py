from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image


def load_and_preprocess_images_np(image_paths: list[str], target_size: int = 518) -> np.ndarray:
    if len(image_paths) == 0:
        raise ValueError("At least one image path is required")

    tensors = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")

        width, height = image.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(arr)

    max_h = max(t.shape[0] for t in tensors)
    max_w = max(t.shape[1] for t in tensors)
    padded = []
    for tensor in tensors:
        h, w, channels = tensor.shape
        out = np.ones((max_h, max_w, channels), dtype=np.float32)
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        out[y0 : y0 + h, x0 : x0 + w, :] = tensor
        padded.append(out)

    return np.stack(padded, axis=0)


def load_and_preprocess_images(image_paths: list[str], target_size: int = 518) -> jnp.ndarray:
    stacked = load_and_preprocess_images_np(image_paths, target_size=target_size)
    return jnp.asarray(stacked)


def resolve_image_paths(image_dir: str) -> list[str]:
    root = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(str(p) for p in root.iterdir() if p.suffix.lower() in exts)
    return paths
