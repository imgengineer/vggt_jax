import jax
import jax.numpy as jnp


def make_sincos_pos_embed(embed_dim: int, pos: jnp.ndarray, omega_0: float = 100.0) -> jnp.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    pos = pos.reshape(-1)
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (omega_0**omega)

    out = jnp.einsum("m,d->md", pos.astype(jnp.float32), omega, precision=jax.lax.Precision.HIGHEST)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    return jnp.concatenate([emb_sin, emb_cos], axis=1)


def position_grid_to_embed(pos_grid: jnp.ndarray, embed_dim: int, omega_0: float = 100.0) -> jnp.ndarray:
    if pos_grid.ndim != 3 or pos_grid.shape[-1] != 2:
        raise ValueError(f"Expected pos_grid shape (H, W, 2), got {pos_grid.shape}")

    height, width, _ = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, 2)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = jnp.concatenate([emb_x, emb_y], axis=-1)
    return emb.reshape(height, width, embed_dim)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    if dtype is None:
        dtype = jnp.float32

    x_coords = jnp.linspace(left_x, right_x, num=width, dtype=dtype)
    y_coords = jnp.linspace(top_y, bottom_y, num=height, dtype=dtype)

    uu, vv = jnp.meshgrid(x_coords, y_coords, indexing="xy")
    return jnp.stack((uu, vv), axis=-1)
