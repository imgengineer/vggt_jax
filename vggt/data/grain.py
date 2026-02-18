from __future__ import annotations

from typing import Any

import grain

from vggt.utils.io import load_and_preprocess_images_np, resolve_image_paths


def create_image_folder_iter_dataset(
    image_dir: str,
    *,
    batch_size: int = 1,
    target_size: int = 518,
    shuffle: bool = False,
    seed: int | None = 0,
    num_epochs: int | None = 1,
    drop_remainder: bool = False,
    num_threads: int = 16,
    prefetch_buffer_size: int = 16,
) -> grain.IterDataset[dict[str, Any]]:
    """Create a simple Grain pipeline that loads images from a folder.

    Yields dict batches:
      - images: np.ndarray[batch, H, W, 3] float32 in [0, 1]
      - paths:  list[str]
    """

    image_paths = resolve_image_paths(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    ds: grain.MapDataset[int] = grain.MapDataset.range(0, len(image_paths))
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.repeat(num_epochs=num_epochs)

    def batch_fn(indices: list[int]) -> dict[str, Any]:
        paths = [image_paths[int(i)] for i in indices]
        images = load_and_preprocess_images_np(paths, target_size=target_size)
        return {"images": images, "paths": paths}

    ds = ds.batch(batch_size, drop_remainder=drop_remainder, batch_fn=batch_fn)
    read_options = grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch_buffer_size)
    return ds.to_iter_dataset(read_options=read_options)
