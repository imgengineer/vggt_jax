from __future__ import annotations

import numpy as np
from PIL import Image, ImageFile

from .dataset_util import (
    crop_image_depth_and_intrinsic_by_pp,
    depth_to_world_coords_points,
    resize_image_depth_and_intrinsic,
    rotate_90_degrees,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset:
    def __init__(self, common_conf):
        self.img_size = int(common_conf.img_size)
        self.patch_size = int(common_conf.patch_size)
        self.aug_scale = common_conf.augs.scales
        self.rescale = bool(common_conf.rescale)
        self.rescale_aug = bool(common_conf.rescale_aug)
        self.landscape_check = bool(common_conf.landscape_check)

    def __len__(self):
        return int(self.len_train)

    def __getitem__(self, idx_tuple):
        seq_index, img_per_seq, aspect_ratio = idx_tuple
        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        raise NotImplementedError

    def get_target_shape(self, aspect_ratio: float) -> np.ndarray:
        short_size = int(self.img_size * float(aspect_ratio))
        if short_size % self.patch_size != 0:
            short_size = (short_size // self.patch_size) * self.patch_size
        return np.array([short_size, self.img_size], dtype=np.int32)

    def process_one_image(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        extri_opencv: np.ndarray,
        intri_opencv: np.ndarray,
        original_size: np.ndarray,
        target_image_shape: np.ndarray,
        track: np.ndarray | None = None,
        filepath: str | None = None,
        safe_bound: int = 4,
    ):
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        if getattr(self, "training", False) and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(self.aug_scale[0], self.aug_scale[1], 2)
            random_h_scale = min(float(random_h_scale), 1.0)
            random_w_scale = min(float(random_w_scale), 1.0)
            aug_size = (original_size * np.array([random_h_scale, random_w_scale])).astype(np.int32)
        else:
            aug_size = original_size

        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image,
            depth_map,
            intri_opencv,
            aug_size,
            track=track,
            filepath=filepath,
        )

        original_size = np.array(image.shape[:2], dtype=np.int32)
        target_shape = target_image_shape

        rotate_to_portrait = False
        if self.landscape_check and original_size[0] > 1.25 * original_size[1]:
            if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                target_shape = np.array([target_image_shape[1], target_image_shape[0]], dtype=np.int32)
                rotate_to_portrait = True

        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image,
                depth_map,
                intri_opencv,
                target_shape,
                original_size,
                track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug,
            )

        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image,
            depth_map,
            intri_opencv,
            target_shape,
            track=track,
            filepath=filepath,
            strict=True,
        )

        if rotate_to_portrait:
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depth_map, extri_opencv, intri_opencv
        )
        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def get_nearby_ids(self, ids, full_seq_num: int, expand_ratio=None, expand_range=None):
        if len(ids) == 0:
            raise ValueError("No IDs provided.")
        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0

        total_ids = len(ids)
        start_idx = int(ids[0])
        if expand_range is None:
            expand_range = int(total_ids * float(expand_ratio))

        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)
        valid_range = np.arange(low_bound, high_bound)

        sampled_ids = np.random.choice(valid_range, size=(total_ids - 1), replace=True)
        return np.insert(sampled_ids, 0, start_idx)
