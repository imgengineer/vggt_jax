from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx


class Mlp(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[jnp.ndarray], jnp.ndarray] = nnx.gelu,
        drop: float = 0.0,
        bias: bool = True,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nnx.Linear(in_features, hidden_features, use_bias=bias, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, use_bias=bias, rngs=rngs)
        self.act_layer = act_layer
        self.drop1 = nnx.Dropout(float(drop))
        self.drop2 = nnx.Dropout(float(drop))
        self.deterministic = False

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray:
        deterministic = bool(self.deterministic)

        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop1(x, rngs=rngs, deterministic=deterministic)
        x = self.fc2(x)
        x = self.drop2(x, rngs=rngs, deterministic=deterministic)
        return x
