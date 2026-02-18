import dataclasses

import jax.numpy as jnp
from flax import nnx

from vggt.layers.attention import Attention, AttentionConfig
from vggt.layers.layer_scale import LayerScale
from vggt.layers.mlp import Mlp


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


class Block(nnx.Module):
    def __init__(self, cfg: BlockConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.deterministic = False
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
        self.ls1 = LayerScale(cfg.dim, cfg.init_values)
        self.norm2 = nnx.LayerNorm(cfg.dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(
            in_features=cfg.dim,
            hidden_features=int(cfg.dim * cfg.mlp_ratio),
            out_features=cfg.dim,
            drop=cfg.drop,
            bias=cfg.ffn_bias,
            rngs=rngs,
        )
        self.ls2 = LayerScale(cfg.dim, cfg.init_values)

    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray | None,
        *,
        rngs: nnx.Rngs | None,
    ) -> jnp.ndarray:
        x = x + self.ls1(self.attn(self.norm1(x), pos, rngs=rngs))
        x = x + self.ls2(self.mlp(self.norm2(x), rngs=rngs))
        return x
