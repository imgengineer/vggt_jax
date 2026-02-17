import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from vggt_jax.layers.attention import Attention, AttentionConfig


@dataclasses.dataclass(frozen=True)
class BlockConfig:
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    init_values: float = 0.01
    qk_norm: bool = True
    rope_frequency: float = 100.0
    use_rope: bool = True


class LayerScale(nnx.Module):
    def __init__(self, dim: int, init_values: float, *, rngs: nnx.Rngs):
        _ = rngs
        self.gamma = nnx.Param(jnp.ones((dim,), dtype=jnp.float32) * init_values)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self.gamma[...]


class Mlp(nnx.Module):
    def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool, drop: float, *, rngs: nnx.Rngs):
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, use_bias=ffn_bias, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, use_bias=ffn_bias, rngs=rngs)
        self.drop = nnx.Dropout(drop)

    def __call__(self, x: jnp.ndarray, *, rngs: nnx.Rngs | None) -> jnp.ndarray:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.drop(x, rngs=rngs)
        x = self.fc2(x)
        x = self.drop(x, rngs=rngs)
        return x


class Block(nnx.Module):
    def __init__(self, cfg: BlockConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.norm1 = nnx.LayerNorm(cfg.dim, epsilon=1e-5, rngs=rngs)
        self.attn = Attention(
            AttentionConfig(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                qkv_bias=cfg.qkv_bias,
                proj_bias=cfg.proj_bias,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.drop,
                qk_norm=cfg.qk_norm,
                rope_frequency=cfg.rope_frequency,
                use_rope=cfg.use_rope,
            ),
            rngs=rngs,
        )
        self.ls1 = LayerScale(cfg.dim, cfg.init_values, rngs=rngs)
        self.norm2 = nnx.LayerNorm(cfg.dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(cfg.dim, cfg.mlp_ratio, cfg.ffn_bias, cfg.drop, rngs=rngs)
        self.ls2 = LayerScale(cfg.dim, cfg.init_values, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, pos: jnp.ndarray | None, *, rngs: nnx.Rngs | None, deterministic: bool
    ) -> jnp.ndarray:
        x = x + self.ls1(self.attn(self.norm1(x), pos, rngs=rngs, deterministic=deterministic))
        x = x + self.ls2(self.mlp(self.norm2(x), rngs=rngs))
        return x
