from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def drop_path(
    x: jnp.ndarray,
    drop_prob: float = 0.0,
    training: bool = False,
    *,
    rng: jax.Array | None = None,
) -> jnp.ndarray:
    if drop_prob <= 0.0 or not training:
        return x
    if rng is None:
        raise ValueError("drop_path requires `rng` when training=True and drop_prob>0")

    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape).astype(x.dtype)
    return x * mask / keep_prob


class DropPath(nnx.Module):
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = float(drop_prob)
        self.deterministic = False

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray:
        deterministic = bool(self.deterministic)
        if self.drop_prob <= 0.0 or deterministic:
            return x
        if rngs is None:
            raise ValueError("DropPath requires `rngs` in non-deterministic mode")
        return drop_path(x, drop_prob=self.drop_prob, training=True, rng=rngs.dropout())
