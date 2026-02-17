import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Rope2DConfig:
    frequency: float = 100.0


def build_2d_positions(batch_size: int, height: int, width: int, dtype=jnp.float32) -> jnp.ndarray:
    y = jnp.arange(height, dtype=dtype)
    x = jnp.arange(width, dtype=dtype)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    pos = jnp.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1)
    pos = jnp.broadcast_to(pos[None, :, :], (batch_size, pos.shape[0], 2))
    return pos


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _rope_1d(tokens: jnp.ndarray, positions: jnp.ndarray, base_frequency: float) -> jnp.ndarray:
    dim = tokens.shape[-1]
    assert dim % 2 == 0
    inv_freq = 1.0 / (base_frequency ** (jnp.arange(0, dim, 2, dtype=tokens.dtype) / dim))
    angles = positions[..., None] * inv_freq[None, None, :]
    angles = jnp.concatenate([angles, angles], axis=-1)
    cos = jnp.cos(angles)[:, None, :, :]
    sin = jnp.sin(angles)[:, None, :, :]
    return tokens * cos + _rotate_half(tokens) * sin


def apply_rope_2d(q_or_k: jnp.ndarray, positions: jnp.ndarray, cfg: Rope2DConfig) -> jnp.ndarray:
    assert q_or_k.ndim == 4
    head_dim = q_or_k.shape[-1]
    assert head_dim % 2 == 0
    vertical, horizontal = jnp.split(q_or_k, 2, axis=-1)
    y_pos = positions[..., 0]
    x_pos = positions[..., 1]
    vertical = _rope_1d(vertical, y_pos, cfg.frequency)
    horizontal = _rope_1d(horizontal, x_pos, cfg.frequency)
    return jnp.concatenate([vertical, horizontal], axis=-1)
