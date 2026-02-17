import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from vggt_jax.layers.rope import Rope2DConfig, apply_rope_2d


@dataclasses.dataclass(frozen=True)
class AttentionConfig:
    dim: int
    num_heads: int
    qkv_bias: bool = True
    proj_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    qk_norm: bool = True
    rope_frequency: float = 100.0
    use_rope: bool = True


class Attention(nnx.Module):
    def __init__(self, cfg: AttentionConfig, *, rngs: nnx.Rngs):
        if cfg.dim % cfg.num_heads != 0:
            raise ValueError(f"dim={cfg.dim} must be divisible by num_heads={cfg.num_heads}")
        self.cfg = cfg
        self.head_dim = cfg.dim // cfg.num_heads
        self.qkv = nnx.Linear(cfg.dim, 3 * cfg.dim, use_bias=cfg.qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(cfg.dim, cfg.dim, use_bias=cfg.proj_bias, rngs=rngs)
        self.attn_drop = nnx.Dropout(cfg.attn_drop)
        self.proj_drop = nnx.Dropout(cfg.proj_drop)
        self.q_norm = nnx.LayerNorm(self.head_dim, epsilon=1e-5, rngs=rngs) if cfg.qk_norm else None
        self.k_norm = nnx.LayerNorm(self.head_dim, epsilon=1e-5, rngs=rngs) if cfg.qk_norm else None
        self.rope_cfg = Rope2DConfig(frequency=cfg.rope_frequency)

    def __call__(
        self, x: jnp.ndarray, pos: jnp.ndarray | None, *, rngs: nnx.Rngs | None, deterministic: bool
    ) -> jnp.ndarray:
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, tokens, 3, self.cfg.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.cfg.use_rope and pos is not None:
            q = apply_rope_2d(q, pos, self.rope_cfg)
            k = apply_rope_2d(k, pos, self.rope_cfg)

        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
        )
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out, rngs=rngs)
        return out
