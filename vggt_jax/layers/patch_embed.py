import dataclasses

import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class PatchEmbedConfig:
    patch_size: int = 14
    in_chans: int = 3
    embed_dim: int = 1024


class PatchEmbed(nnx.Module):
    def __init__(self, cfg: PatchEmbedConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.proj = nnx.Conv(
            in_features=cfg.in_chans,
            out_features=cfg.embed_dim,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x_bchw: jnp.ndarray) -> jnp.ndarray:
        x = jnp.transpose(x_bchw, (0, 2, 3, 1))
        x = self.proj(x)
        batch_size, patch_h, patch_w, channels = x.shape
        x = x.reshape(batch_size, patch_h * patch_w, channels)
        return x
