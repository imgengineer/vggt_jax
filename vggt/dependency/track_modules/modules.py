from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


class ResidualBlock:
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1, kernel_size: int = 3):
        _ = in_planes, planes, norm_fn, stride, kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)


@dataclass
class Mlp:
    in_features: int
    hidden_features: int | None = None
    out_features: int | None = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = self.hidden_features or self.in_features
        out = self.out_features or self.in_features
        if x.shape[-1] != hidden:
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, max(hidden - x.shape[-1], 0))])
            x = x[..., :hidden]
        x = jax.nn.gelu(x, approximate=False)
        if hidden != out:
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, max(out - hidden, 0))])
            x = x[..., :out]
        return x


class AttnBlock:
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs):
        _ = num_heads, kwargs
        self.hidden_size = hidden_size
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        _ = mask
        return x + self.mlp(x)


class CrossAttnBlock:
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int = 1, mlp_ratio: float = 4.0, **kwargs):
        _ = context_dim, num_heads, kwargs
        self.hidden_size = hidden_size
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def __call__(self, x: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        _ = mask
        context_mean = jnp.mean(context, axis=-2, keepdims=True)
        context_mean = jnp.broadcast_to(context_mean, x.shape)
        return x + self.mlp(x + context_mean)


__all__ = ["ResidualBlock", "Mlp", "AttnBlock", "CrossAttnBlock"]
