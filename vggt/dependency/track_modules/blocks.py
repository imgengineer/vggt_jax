from __future__ import annotations

import jax
import jax.numpy as jnp

from vggt.heads.track_ops import CorrBlock


class EfficientUpdateFormer:
    def __init__(
        self,
        space_depth: int = 6,
        time_depth: int = 6,
        input_dim: int = 320,
        hidden_size: int = 384,
        num_heads: int = 8,
        output_dim: int = 130,
        mlp_ratio: float = 4.0,
        add_space_attn: bool = True,
        num_virtual_tracks: int = 64,
    ):
        _ = space_depth, time_depth, input_dim, hidden_size, num_heads, mlp_ratio, add_space_attn, num_virtual_tracks
        self.output_dim = output_dim

    def __call__(self, input_tensor: jnp.ndarray, mask: jnp.ndarray | None = None):
        _ = mask
        shape = input_tensor.shape[:-1] + (self.output_dim,)
        return jnp.zeros(shape, dtype=input_tensor.dtype), None


class BasicEncoder:
    def __init__(self, stride: int = 4, out_dim: int = 128):
        self.stride = int(max(stride, 1))
        self.out_dim = int(out_dim)

    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(images)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got {x.shape}")
        if self.stride > 1:
            out_h = max(1, x.shape[-2] // self.stride)
            out_w = max(1, x.shape[-1] // self.stride)
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1], out_h, out_w), method="bilinear", antialias=False)
        if x.shape[1] >= self.out_dim:
            return x[:, : self.out_dim]
        pad = self.out_dim - x.shape[1]
        return jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))


class ShallowEncoder(BasicEncoder):
    pass


def compute_corr_level(fmap1: jnp.ndarray, fmap2s: jnp.ndarray, channels: int) -> jnp.ndarray:
    corrs = jnp.einsum("bsnc,bsch->bsnh", fmap1, fmap2s, precision=jax.lax.Precision.HIGHEST)
    return corrs / jnp.sqrt(jnp.asarray(channels, dtype=fmap1.dtype))


__all__ = ["EfficientUpdateFormer", "CorrBlock", "BasicEncoder", "ShallowEncoder", "compute_corr_level"]
