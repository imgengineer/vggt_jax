from __future__ import annotations

import bisect
import random
from abc import ABC

import jax.numpy as jnp
import numpy as np

try:
    from hydra.utils import instantiate
except ImportError:  # pragma: no cover
    def instantiate(*args, **kwargs):
        _ = args, kwargs
        raise ImportError("Missing `hydra-core`. Install with `bash scripts/setup_uv.sh train`.")

from .augmentation import get_image_augmentation
from .track_util import build_tracks_by_depth


def _to_array_or_none(items, dtype=None):
    if not items:
        return None
    if items[0] is None:
        return None
    arr = np.stack(items)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


class ComposedDataset(ABC):
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        _ = kwargs
        base_dataset_list = [instantiate(dataset_cfg, common_conf=common_config) for dataset_cfg in dataset_configs]
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        self.cojitter = bool(common_config.augs.cojitter)
        self.cojitter_ratio = float(common_config.augs.cojitter_ratio)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        self.fixed_num_images = int(common_config.fix_img_num)
        self.fixed_aspect_ratio = float(common_config.fix_aspect_ratio)
        self.load_track = bool(common_config.load_track)
        self.track_num = int(common_config.track_num)
        self.training = bool(common_config.training)
        self.common_config = common_config
        self.total_samples = len(self.base_dataset)

    def __len__(self):
        return self.total_samples

    def _apply_image_aug(self, images: np.ndarray) -> np.ndarray:
        if not self.training or self.image_aug is None:
            return images

        if self.cojitter and random.random() > self.cojitter_ratio:
            params = self.image_aug.sample_params()
            return np.stack([self.image_aug(img, params=params) for img in images], axis=0)

        return np.stack([self.image_aug(img) for img in images], axis=0)

    def __getitem__(self, idx_tuple):
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        batch = self.base_dataset[idx_tuple]

        images = np.stack(batch["images"]).astype(np.float32)
        images = images / 255.0 if images.max() > 1.0 else images
        images = self._apply_image_aug(images)

        depths = _to_array_or_none(batch["depths"], dtype=np.float32)
        extrinsics = np.stack(batch["extrinsics"]).astype(np.float32)
        intrinsics = np.stack(batch["intrinsics"]).astype(np.float32)
        cam_points = _to_array_or_none(batch["cam_points"], dtype=np.float32)
        world_points = _to_array_or_none(batch["world_points"], dtype=np.float32)
        point_masks = _to_array_or_none(batch["point_masks"])
        if point_masks is not None:
            point_masks = point_masks.astype(bool)
        ids = np.asarray(batch["ids"], dtype=np.int32)

        sample = {
            "seq_name": batch["seq_name"],
            "ids": jnp.asarray(ids),
            "images": jnp.asarray(images),
            "depths": jnp.asarray(depths) if depths is not None else None,
            "extrinsics": jnp.asarray(extrinsics),
            "intrinsics": jnp.asarray(intrinsics),
            "cam_points": jnp.asarray(cam_points) if cam_points is not None else None,
            "world_points": jnp.asarray(world_points) if world_points is not None else None,
            "point_masks": jnp.asarray(point_masks) if point_masks is not None else None,
        }

        if self.load_track:
            if batch.get("tracks") is not None:
                tracks = np.stack(batch["tracks"]).astype(np.float32)
                track_vis_mask = np.stack(batch["track_masks"]).astype(bool)
                valid_indices = np.where(track_vis_mask[0])[0]

                if valid_indices.size == 0:
                    tracks = np.zeros((images.shape[0], self.track_num, 2), dtype=np.float32)
                    track_vis_mask = np.zeros((images.shape[0], self.track_num), dtype=bool)
                    track_positive_mask = np.zeros((self.track_num,), dtype=bool)
                else:
                    if valid_indices.size >= self.track_num:
                        sampled = np.random.permutation(valid_indices)[: self.track_num]
                    else:
                        sampled = np.random.choice(valid_indices, size=self.track_num, replace=True)
                    tracks = tracks[:, sampled, :]
                    track_vis_mask = track_vis_mask[:, sampled]
                    track_positive_mask = np.ones((self.track_num,), dtype=bool)
            else:
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
                    extrinsics,
                    intrinsics,
                    world_points,
                    depths,
                    point_masks,
                    images,
                    target_track_num=self.track_num,
                    seq_name=batch["seq_name"],
                )

            sample["tracks"] = jnp.asarray(tracks)
            sample["track_vis_mask"] = jnp.asarray(track_vis_mask)
            sample["track_positive_mask"] = jnp.asarray(track_positive_mask)

        return sample


class TupleConcatDataset:
    def __init__(self, datasets, common_config):
        self.datasets = list(datasets)
        self.inside_random = bool(common_config.inside_random)
        self.cumulative_sizes = []
        running = 0
        for ds in self.datasets:
            running += len(ds)
            self.cumulative_sizes.append(running)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        idx_tuple = idx if isinstance(idx, tuple) else None
        if idx_tuple is not None:
            idx = int(idx_tuple[0])
        idx = int(idx)

        if self.inside_random:
            idx = random.randint(0, len(self) - 1)

        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise ValueError("Index out of range")

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]

        if idx_tuple is None:
            idx_tuple = (sample_idx, 1, 1.0)
        elif len(idx_tuple) == 3:
            idx_tuple = (sample_idx, idx_tuple[1], idx_tuple[2])
        else:
            raise ValueError("Tuple index must be (seq_idx, num_images, aspect_ratio)")

        return self.datasets[dataset_idx][idx_tuple]
