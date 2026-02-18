from __future__ import annotations

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from vggt.layers.block import Block, BlockConfig
from vggt.layers.patch_embed import PatchEmbed, PatchEmbedConfig


class BlockChunk(nnx.Module):
    def __init__(self, blocks: Sequence[Block]):
        self.blocks = nnx.List(list(blocks))
        self.deterministic = False

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, pos=None, rngs=rngs)
        return x


class DinoVisionTransformer(nnx.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: float | None = None,
        embed_layer: type[PatchEmbed] = PatchEmbed,
        ffn_layer: str = "mlp",
        block_chunks: int = 1,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        qk_norm: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        _ = drop_path_rate, drop_path_uniform, init_values
        if ffn_layer not in {"mlp", "swiglu", "swiglufused", "identity"}:
            raise NotImplementedError(f"Unsupported ffn_layer={ffn_layer}")
        if rngs is None:
            rngs = nnx.Rngs(0)

        if isinstance(img_size, int):
            img_h, img_w = img_size, img_size
        else:
            img_h, img_w = int(img_size[0]), int(img_size[1])

        self.deterministic = False
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            PatchEmbedConfig(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            ),
            rngs=rngs,
        )

        num_patches = (img_h // patch_size) * (img_w // patch_size)
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 1e-6)
        self.pos_embed = nnx.Param(jax.random.normal(rngs.params(), (1, num_patches + 1, embed_dim)) * 0.02)
        self.register_tokens = (
            nnx.Param(jax.random.normal(rngs.params(), (1, num_register_tokens, embed_dim)) * 1e-6)
            if num_register_tokens > 0
            else None
        )

        block_cfg = BlockConfig(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            use_rope=False,
            init_values=0.0 if init_values is None else float(init_values),
        )
        blocks = [Block(block_cfg, rngs=rngs) for _ in range(depth)]
        if block_chunks > 0:
            chunk_size = max(1, depth // block_chunks)
            self.blocks = nnx.List(
                [BlockChunk(blocks[i : i + chunk_size]) for i in range(0, depth, chunk_size)]
            )
        else:
            self.blocks = nnx.List(blocks)

        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.mask_token = nnx.Param(jnp.zeros((1, embed_dim), dtype=jnp.float32))

    def interpolate_pos_encoding(self, x: jnp.ndarray, width: int, height: int) -> jnp.ndarray:
        npatch = x.shape[1] - 1
        total_patches = self.pos_embed.shape[1] - 1
        if npatch == total_patches and width == height:
            return self.pos_embed[...].astype(x.dtype)

        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        out_h = height // self.patch_size
        out_w = width // self.patch_size
        m = int(math.sqrt(total_patches))
        if total_patches != m * m:
            raise ValueError(f"Expected square patch grid, got {total_patches}")

        patch_pos_embed = patch_pos_embed.reshape(1, m, m, dim)
        patch_pos_embed = jax.image.resize(
            patch_pos_embed,
            shape=(1, out_h, out_w, dim),
            method="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.reshape(1, out_h * out_w, dim)
        return jnp.concatenate((class_pos_embed, patch_pos_embed), axis=1).astype(x.dtype)

    def prepare_tokens_with_masks(self, x: jnp.ndarray, masks: jnp.ndarray | None = None) -> jnp.ndarray:
        batch_size, _, height, width = x.shape
        x = self.patch_embed(x)

        if masks is not None:
            mask_token = jnp.broadcast_to(self.mask_token[...], x.shape)
            x = jnp.where(masks[..., None], mask_token.astype(x.dtype), x)

        cls_tokens = jnp.broadcast_to(self.cls_token[...], (batch_size, 1, self.embed_dim))
        x = jnp.concatenate((cls_tokens, x), axis=1)
        x = x + self.interpolate_pos_encoding(x, width=width, height=height)

        if self.register_tokens is not None:
            reg_tokens = jnp.broadcast_to(
                self.register_tokens[...],
                (batch_size, self.num_register_tokens, self.embed_dim),
            )
            x = jnp.concatenate((x[:, :1], reg_tokens, x[:, 1:]), axis=1)
        return x

    def forward_features(
        self,
        x: jnp.ndarray | list[jnp.ndarray],
        masks: jnp.ndarray | list[jnp.ndarray] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> dict[str, jnp.ndarray] | list[dict[str, jnp.ndarray]]:
        if isinstance(x, list):
            masks_list: list[jnp.ndarray | None]
            if masks is None:
                masks_list = [None] * len(x)
            elif isinstance(masks, list):
                masks_list = list(masks)
            else:
                raise ValueError("When `x` is a list, `masks` must be None or list")

            tokens_list = [self.prepare_tokens_with_masks(im, m) for im, m in zip(x, masks_list, strict=True)]
            for blk in self.blocks:
                tokens_list = [blk(t, rngs=rngs) for t in tokens_list]
            return [
                {
                    "x_norm_clstoken": self.norm(t)[:, 0],
                    "x_norm_regtokens": self.norm(t)[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": self.norm(t)[:, self.num_register_tokens + 1 :],
                    "x_prenorm": t,
                    "masks": m,
                }
                for t, m in zip(tokens_list, masks_list, strict=True)
            ]

        tokens = self.prepare_tokens_with_masks(x, masks if isinstance(masks, jnp.ndarray) else None)
        for blk in self.blocks:
            tokens = blk(tokens, rngs=rngs)
        tokens_norm = self.norm(tokens)
        return {
            "x_norm_clstoken": tokens_norm[:, 0],
            "x_norm_regtokens": tokens_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": tokens_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": tokens,
            "masks": masks,
        }

    def __call__(
        self,
        x: jnp.ndarray | list[jnp.ndarray],
        masks: jnp.ndarray | list[jnp.ndarray] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray | list[dict[str, jnp.ndarray]]:
        out = self.forward_features(x, masks=masks, rngs=rngs)
        if isinstance(out, list):
            return out
        return out["x_norm_clstoken"]


__all__ = ["BlockChunk", "DinoVisionTransformer"]
