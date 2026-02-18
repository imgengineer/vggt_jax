from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class LayerScale(nnx.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        self.gamma = nnx.Param(jnp.full((dim,), float(init_values), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self.gamma[...]

