from __future__ import annotations

import jax.numpy as jnp

from vggt.heads.track_ops import (
    _sample_nchw,
    get_2d_embedding,
    get_2d_sincos_pos_embed,
    sample_features4d,
)


def bilinear_sampler(
    input: jnp.ndarray,
    coords: jnp.ndarray,
    align_corners: bool = True,
    padding_mode: str = "border",
) -> jnp.ndarray:
    _ = align_corners
    return _sample_nchw(input, coords, padding_mode=padding_mode)


__all__ = ["get_2d_sincos_pos_embed", "get_2d_embedding", "bilinear_sampler", "sample_features4d"]
