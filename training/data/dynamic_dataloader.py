from __future__ import annotations

import math
import random
from typing import Callable, Iterator, Optional

import jax.numpy as jnp
import numpy as np

try:
    from hydra.utils import instantiate
except ImportError:  # pragma: no cover
    def instantiate(*args, **kwargs):
        _ = args, kwargs
        raise ImportError("Missing `hydra-core`. Install with `bash scripts/setup_uv.sh train`.")

from .worker_fn import get_worker_init_fn


def _default_collate(batch: list[dict]):
    if not batch:
        return {}
    collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if first is None:
            collated[key] = None
        elif isinstance(first, str):
            collated[key] = values
        else:
            collated[key] = jnp.asarray(np.stack([np.asarray(v) for v in values], axis=0))
    return collated


class DynamicDistributedSampler:
    def __init__(self, dataset, seed: int = 42, shuffle: bool = True):
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample_indices(self) -> np.ndarray:
        n = len(self.dataset)
        indices = np.arange(n, dtype=np.int32)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)
        return indices

    def __len__(self):
        return len(self.dataset)


class DynamicBatchSampler:
    def __init__(
        self,
        sampler: DynamicDistributedSampler,
        aspect_ratio_range,
        image_num_range,
        epoch: int = 0,
        seed: int = 42,
        max_img_per_gpu: int = 48,
    ):
        self.sampler = sampler
        self.aspect_ratio_range = tuple(aspect_ratio_range)
        self.image_num_range = tuple(int(v) for v in image_num_range)
        self.max_img_per_gpu = int(max_img_per_gpu)
        self.seed = int(seed)
        self.epoch = int(epoch)

        self.possible_nums = np.arange(self.image_num_range[0], self.image_num_range[1] + 1, dtype=np.int32)
        self.weights = np.ones_like(self.possible_nums, dtype=np.float32)
        self.weights = self.weights / self.weights.sum()
        self._rng = random.Random(self.seed + self.epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.sampler.set_epoch(epoch)
        self._rng.seed(self.seed + epoch)

    def __iter__(self):
        indices = self.sampler.sample_indices()
        cursor = 0
        while cursor < len(indices):
            random_image_num = int(np.random.choice(self.possible_nums, p=self.weights))
            random_aspect_ratio = round(
                self._rng.uniform(float(self.aspect_ratio_range[0]), float(self.aspect_ratio_range[1])),
                2,
            )
            batch_size = max(1, int(math.floor(self.max_img_per_gpu / max(1, random_image_num))))
            current = indices[cursor : cursor + batch_size]
            cursor += batch_size
            yield [(int(idx), random_image_num, random_aspect_ratio) for idx in current]

    def __len__(self):
        n = len(self.sampler)
        avg_imgs = (self.image_num_range[0] + self.image_num_range[1]) / 2.0
        avg_bs = max(1, int(math.floor(self.max_img_per_gpu / max(1.0, avg_imgs))))
        return int(math.ceil(n / avg_bs))


class _SimpleDataLoader:
    def __init__(self, dataset, batch_sampler, collate_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn or _default_collate

    def __iter__(self) -> Iterator[dict]:
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[idx_tuple] for idx_tuple in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self):
        return len(self.batch_sampler)


class DynamicJaxDataset:
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
    ) -> None:
        _ = num_workers, pin_memory, drop_last, worker_init_fn, persistent_workers
        self.dataset_config = dataset
        self.common_config = common_config
        self.seed = int(seed)
        self.collate_fn = collate_fn
        self.max_img_per_gpu = int(max_img_per_gpu)

        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)
        self.aspect_ratio_range = common_config.augs.aspects
        self.image_num_range = common_config.img_nums

        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"Invalid aspect range: {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"Invalid image_num range: {self.image_num_range}")

        self.sampler = DynamicDistributedSampler(self.dataset, seed=self.seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=self.seed,
            max_img_per_gpu=self.max_img_per_gpu,
        )

    def get_loader(self, epoch: int):
        self.sampler.set_epoch(epoch)
        self.batch_sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        get_worker_init_fn(seed=self.seed, num_workers=0, epoch=epoch, worker_init_fn=None)
        return _SimpleDataLoader(self.dataset, self.batch_sampler, collate_fn=self.collate_fn)


# Backward-compatible name used by old config.
DynamicTorchDataset = DynamicJaxDataset
