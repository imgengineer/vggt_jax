from __future__ import annotations

import random
import importlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

cv2 = None


@dataclass
class _AugParams:
    apply_jitter: bool
    brightness: float
    contrast: float
    saturation: float
    gray: bool
    blur: bool


class ImageAugmentation:
    def __init__(self, color_jitter: Optional[Dict[str, float]] = None, gray_scale: bool = True, gau_blur: bool = False):
        defaults = {
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
            "hue": 0.1,
            "p": 0.9,
        }
        merged = {**defaults, **(color_jitter or {})}
        self.color_jitter = merged
        self.gray_scale = bool(gray_scale)
        self.gau_blur = bool(gau_blur)

    def sample_params(self) -> _AugParams:
        return _AugParams(
            apply_jitter=(random.random() < float(self.color_jitter["p"])),
            brightness=1.0 + random.uniform(-float(self.color_jitter["brightness"]), float(self.color_jitter["brightness"])),
            contrast=1.0 + random.uniform(-float(self.color_jitter["contrast"]), float(self.color_jitter["contrast"])),
            saturation=1.0 + random.uniform(-float(self.color_jitter["saturation"]), float(self.color_jitter["saturation"])),
            gray=self.gray_scale and (random.random() < 0.05),
            blur=self.gau_blur and (random.random() < 0.05),
        )

    @staticmethod
    def _apply_single(image: np.ndarray, params: _AugParams) -> np.ndarray:
        x = image.astype(np.float32, copy=True)
        if x.max() > 1.0:
            x = x / 255.0

        if params.apply_jitter:
            x = x * params.brightness
            mean = x.mean(axis=(0, 1), keepdims=True)
            x = (x - mean) * params.contrast + mean
            gray = x.mean(axis=-1, keepdims=True)
            x = gray + (x - gray) * params.saturation

        if params.gray:
            gray = x.mean(axis=-1, keepdims=True)
            x = np.repeat(gray, repeats=3, axis=-1)

        if params.blur:
            global cv2
            if cv2 is None:
                try:
                    cv2 = importlib.import_module("cv2")
                except Exception:  # pragma: no cover
                    cv2 = False
            if cv2:
                x = cv2.GaussianBlur(x, (5, 5), sigmaX=0.5, sigmaY=0.5)

        return np.clip(x, 0.0, 1.0).astype(np.float32)

    def __call__(self, image: np.ndarray, params: _AugParams | None = None) -> np.ndarray:
        if params is None:
            params = self.sample_params()
        return self._apply_single(image, params)


def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False,
) -> ImageAugmentation | None:
    return ImageAugmentation(color_jitter=color_jitter, gray_scale=gray_scale, gau_blur=gau_blur)
