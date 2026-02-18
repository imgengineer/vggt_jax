from __future__ import annotations

import jax.numpy as jnp


def global_norm(tree) -> jnp.ndarray:
    leaves = []
    if isinstance(tree, dict):
        for value in tree.values():
            leaves.append(global_norm(value))
        return jnp.sqrt(jnp.sum(jnp.stack([v * v for v in leaves]))) if leaves else jnp.asarray(0.0)
    arr = jnp.asarray(tree)
    return jnp.sqrt(jnp.sum(arr * arr))


def clip_by_global_norm(grads, max_norm: float):
    norm = global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))

    def _scale(x):
        if isinstance(x, dict):
            return {k: _scale(v) for k, v in x.items()}
        return jnp.asarray(x) * scale

    return _scale(grads), norm


class GradientClipper:
    def __init__(self, configs=None, max_norm: float | None = None):
        self.configs = configs or []
        if max_norm is not None:
            self.max_norm = float(max_norm)
        elif self.configs:
            first = self.configs[0]
            self.max_norm = float(getattr(first, "max_norm", 1.0))
        else:
            self.max_norm = 1.0

    def __call__(self, grads):
        clipped, _ = clip_by_global_norm(grads, self.max_norm)
        return clipped


__all__ = ["global_norm", "clip_by_global_norm", "GradientClipper"]
