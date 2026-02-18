from __future__ import annotations

import jax
import jax.numpy as jnp

from .track_modules.base_track_predictor import BaseTrackerPredictor
from .track_modules.blocks import BasicEncoder, ShallowEncoder
from .track_modules.track_refine import refine_track


def _to_bschw(images: jnp.ndarray) -> jnp.ndarray:
    images = jnp.asarray(images)
    if images.ndim != 5:
        raise ValueError(f"Expected images with shape [B,S,C,H,W] or [B,S,H,W,C], got {images.shape}")
    if images.shape[2] == 3:
        return images
    if images.shape[-1] == 3:
        return images.transpose(0, 1, 4, 2, 3)
    raise ValueError(f"Cannot infer channel axis from {images.shape}")


class TrackerPredictor:
    def __init__(self, **extra_args):
        _ = extra_args
        self.coarse_stride = 4
        self.coarse_down_ratio = 2
        self.coarse_fnet = BasicEncoder(stride=self.coarse_stride)
        self.coarse_predictor = BaseTrackerPredictor(stride=self.coarse_stride)
        self.fine_fnet = ShallowEncoder(stride=1, out_dim=32)
        self.fine_predictor = BaseTrackerPredictor(
            stride=1,
            depth=4,
            corr_levels=3,
            corr_radius=3,
            latent_dim=32,
            hidden_size=256,
            use_spaceatt=False,
        )

    def to(self, *args, **kwargs):
        _ = args, kwargs
        return self

    def eval(self):
        return self

    def train(self, mode: bool = True):
        _ = mode
        return self

    def process_images_to_fmaps(self, images: jnp.ndarray) -> jnp.ndarray:
        images = _to_bschw(images)
        batch_size, seq_len, channels, height, width = images.shape
        flat = images.reshape(batch_size * seq_len, channels, height, width)
        if self.coarse_down_ratio > 1:
            out_h = max(1, height // self.coarse_down_ratio)
            out_w = max(1, width // self.coarse_down_ratio)
            flat = jax.image.resize(
                flat,
                shape=(flat.shape[0], flat.shape[1], out_h, out_w),
                method="bilinear",
                antialias=False,
            )
        fmaps = self.coarse_fnet(flat)
        return fmaps.reshape(batch_size, seq_len, fmaps.shape[1], fmaps.shape[2], fmaps.shape[3])

    def __call__(
        self,
        images: jnp.ndarray,
        query_points: jnp.ndarray,
        fmaps: jnp.ndarray | None = None,
        coarse_iters: int = 6,
        inference: bool = True,
        fine_tracking: bool = True,
        fine_chunk: int = 40960,
    ):
        _ = inference, fine_chunk
        images = _to_bschw(images)
        query_points = jnp.asarray(query_points)
        if query_points.ndim == 2:
            query_points = query_points[None, ...]

        if fmaps is None:
            fmaps = self.process_images_to_fmaps(images)

        coarse_pred_list, pred_vis, pred_score = self.coarse_predictor(
            query_points=query_points,
            fmaps=fmaps,
            iters=coarse_iters,
            down_ratio=self.coarse_down_ratio,
        )
        coarse_pred_track = coarse_pred_list[-1]

        if fine_tracking:
            fine_pred_track, fine_score = refine_track(
                images, self.fine_fnet, self.fine_predictor, coarse_pred_track, compute_score=True
            )
            pred_score = fine_score
        else:
            fine_pred_track = coarse_pred_track
            pred_score = jnp.ones_like(pred_vis)

        return fine_pred_track, coarse_pred_track, pred_vis, pred_score


__all__ = ["TrackerPredictor"]
