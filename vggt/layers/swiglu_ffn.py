from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class SwiGLUFFN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nnx.Linear(in_features, hidden_features * 2, use_bias=bias, rngs=rngs)
        self.w3 = nnx.Linear(hidden_features, out_features, use_bias=bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        return self.w3(jax.nn.silu(x1) * x2)


class SwiGLUFFNFused(SwiGLUFFN):
    pass

