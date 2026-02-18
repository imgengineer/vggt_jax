import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from vggt.holders import PatchEmbedBackboneHolder
from vggt.layers.block import Block, BlockConfig
from vggt.layers.patch_embed import PatchEmbed, PatchEmbedConfig
from vggt.layers.rope import build_2d_positions

_RESNET_MEAN = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
_RESNET_STD = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)


@dataclasses.dataclass(frozen=True)
class AggregatorConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    qk_norm: bool = True
    aa_order: tuple[str, str] = ("frame", "global")
    aa_block_size: int = 1
    rope_freq: float = 100.0
    init_values: float = 0.01
    patch_embed: str = "dinov2_vitl14_reg"
    patch_embed_interpolate_antialias: bool = True
    patch_embed_interpolate_offset: float = 0.0
    use_patch_embed_holder_runtime: bool = True
    enable_patch_embed_holder: bool = True


def _slice_expand_and_flatten(token_tensor: jnp.ndarray, batch_size: int, seq_len: int) -> jnp.ndarray:
    query = jnp.broadcast_to(token_tensor[:, 0:1], (batch_size, 1) + token_tensor.shape[2:])
    others = jnp.broadcast_to(token_tensor[:, 1:2], (batch_size, max(seq_len - 1, 0)) + token_tensor.shape[2:])
    combined = jnp.concatenate([query, others], axis=1)
    return combined.reshape(batch_size * seq_len, *combined.shape[2:])


def _torch_layer_norm(x: jnp.ndarray, holder, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * holder.weight[...] + holder.bias[...]


def _torch_linear(x: jnp.ndarray, holder) -> jnp.ndarray:
    y = jnp.einsum("...i,oi->...o", x, holder.weight[...], precision=lax.Precision.HIGHEST)
    if hasattr(holder, "bias"):
        y = y + holder.bias[...]
    return y


def _torch_conv2d_nchw(x: jnp.ndarray, holder, *, stride: int = 1, padding: int = 0) -> jnp.ndarray:
    y = lax.conv_general_dilated(
        lhs=x,
        rhs=holder.weight[...],
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=lax.Precision.HIGHEST,
    )
    if hasattr(holder, "bias"):
        y = y + holder.bias[...][None, :, None, None]
    return y


class Aggregator(nnx.Module):
    def __init__(self, cfg: AggregatorConfig, *, rngs: nnx.Rngs):
        if cfg.depth % cfg.aa_block_size != 0:
            raise ValueError(f"depth={cfg.depth} must be divisible by aa_block_size={cfg.aa_block_size}")

        self.cfg = cfg
        self.deterministic = False
        self.patch_embed = PatchEmbed(
            PatchEmbedConfig(
                patch_size=cfg.patch_size,
                in_chans=3,
                embed_dim=cfg.embed_dim,
            ),
            rngs=rngs,
        )

        if cfg.enable_patch_embed_holder:
            self.patch_embed_holder = PatchEmbedBackboneHolder(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_register_tokens=cfg.num_register_tokens,
            )
        else:
            self.patch_embed_holder = None

        block_cfg = BlockConfig(
            dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            proj_bias=cfg.proj_bias,
            ffn_bias=cfg.ffn_bias,
            drop=0.0,
            attn_drop=0.0,
            init_values=cfg.init_values,
            qk_norm=cfg.qk_norm,
            rope_frequency=cfg.rope_freq,
            use_rope=cfg.rope_freq > 0,
        )
        self.frame_blocks = nnx.List([Block(block_cfg, rngs=rngs) for _ in range(cfg.depth)])
        self.global_blocks = nnx.List([Block(block_cfg, rngs=rngs) for _ in range(cfg.depth)])

        self.camera_token = nnx.Param(jax.random.normal(rngs.params(), (1, 2, 1, cfg.embed_dim)) * 1e-6)
        self.register_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 2, cfg.num_register_tokens, cfg.embed_dim)) * 1e-6
        )

        self.patch_start_idx = 1 + cfg.num_register_tokens
        self.aa_block_num = cfg.depth // cfg.aa_block_size

    def _normalize(self, images: jnp.ndarray) -> jnp.ndarray:
        mean = _RESNET_MEAN[None, None, None, None, :]
        std = _RESNET_STD[None, None, None, None, :]
        return (images - mean) / std

    def _patch_embed_num_heads(self) -> int:
        patch_embed_name = self.cfg.patch_embed
        if "vits14" in patch_embed_name:
            return 6
        if "vitb14" in patch_embed_name:
            return 12
        if "vitl14" in patch_embed_name:
            return 16
        if "vitg2" in patch_embed_name:
            return 24
        return self.cfg.num_heads

    def _torch_vit_attention(self, x: jnp.ndarray, attn_holder, *, num_heads: int) -> jnp.ndarray:
        batch_size, tokens, channels = x.shape
        head_dim = channels // num_heads

        qkv = _torch_linear(x, attn_holder.qkv)
        qkv = qkv.reshape(batch_size, tokens, 3, num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=lax.Precision.HIGHEST) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v, precision=lax.Precision.HIGHEST)

        out = out.transpose(0, 2, 1, 3).reshape(batch_size, tokens, channels)
        out = _torch_linear(out, attn_holder.proj)
        return out

    def _torch_vit_mlp(self, x: jnp.ndarray, mlp_holder) -> jnp.ndarray:
        x = _torch_linear(x, mlp_holder.fc1)
        x = jax.nn.gelu(x, approximate=False)
        x = _torch_linear(x, mlp_holder.fc2)
        return x

    def _torch_vit_block(self, x: jnp.ndarray, block_holder, *, num_heads: int) -> jnp.ndarray:
        attn_in = _torch_layer_norm(x, block_holder.norm1, eps=1e-6)
        x = x + block_holder.ls1.gamma[...] * self._torch_vit_attention(attn_in, block_holder.attn, num_heads=num_heads)

        mlp_in = _torch_layer_norm(x, block_holder.norm2, eps=1e-6)
        x = x + block_holder.ls2.gamma[...] * self._torch_vit_mlp(mlp_in, block_holder.mlp)
        return x

    def _interpolate_pos_encoding(self, x: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        holder = self.patch_embed_holder
        pos_embed = holder.pos_embed[...]

        npatch = x.shape[1] - 1
        total_patches = pos_embed.shape[1] - 1

        if npatch == total_patches and height == width:
            return pos_embed.astype(x.dtype)

        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]

        dim = x.shape[-1]
        out_h = height // self.cfg.patch_size
        out_w = width // self.cfg.patch_size

        m = int(math.sqrt(total_patches))
        if total_patches != m * m:
            raise ValueError(f"Expected square patch grid, got {total_patches}")

        patch_pos_embed = patch_pos_embed.reshape(1, m, m, dim)
        patch_pos_embed = jax.image.resize(
            patch_pos_embed,
            shape=(1, out_h, out_w, dim),
            method="bicubic",
            antialias=self.cfg.patch_embed_interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.reshape(1, out_h * out_w, dim)

        return jnp.concatenate((class_pos_embed, patch_pos_embed), axis=1).astype(x.dtype)

    def _patch_embed_forward_from_holder(self, images_bhwc: jnp.ndarray) -> jnp.ndarray:
        holder = self.patch_embed_holder
        images_bchw = jnp.transpose(images_bhwc, (0, 3, 1, 2))

        tokens = _torch_conv2d_nchw(images_bchw, holder.patch_embed.proj, stride=self.cfg.patch_size, padding=0)
        batch_size, channels, patch_h, patch_w = tokens.shape
        tokens = tokens.reshape(batch_size, channels, patch_h * patch_w).transpose(0, 2, 1)

        cls_token = jnp.broadcast_to(holder.cls_token[...], (batch_size, 1, holder.cls_token.shape[-1]))
        tokens = jnp.concatenate((cls_token, tokens), axis=1)
        tokens = tokens + self._interpolate_pos_encoding(tokens, images_bhwc.shape[1], images_bhwc.shape[2])

        if self.cfg.num_register_tokens > 0:
            register_tokens = jnp.broadcast_to(
                holder.register_tokens[...],
                (batch_size, self.cfg.num_register_tokens, holder.register_tokens.shape[-1]),
            )
            tokens = jnp.concatenate((tokens[:, :1], register_tokens, tokens[:, 1:]), axis=1)

        patch_embed_heads = self._patch_embed_num_heads()
        for block_holder in holder.blocks:
            tokens = self._torch_vit_block(tokens, block_holder, num_heads=patch_embed_heads)

        tokens = _torch_layer_norm(tokens, holder.norm, eps=1e-6)
        return tokens[:, 1 + self.cfg.num_register_tokens :]

    def _patch_embed_forward(self, images_bhwc: jnp.ndarray) -> jnp.ndarray:
        use_holder_runtime = (
            self.patch_embed_holder is not None
            and self.cfg.use_patch_embed_holder_runtime
            and "conv" not in self.cfg.patch_embed
        )

        if use_holder_runtime:
            return self._patch_embed_forward_from_holder(images_bhwc)

        return self.patch_embed(images_bhwc)

    def __call__(
        self,
        images: jnp.ndarray,
        *,
        rngs: nnx.Rngs | None,
    ) -> tuple[list[jnp.ndarray], int]:
        batch_size, seq_len, height, width, channels = images.shape
        if channels != 3:
            raise ValueError(f"Expected images with C=3, got {channels}")

        images = self._normalize(images)
        flat_images = images.reshape(batch_size * seq_len, height, width, channels)
        patch_tokens = self._patch_embed_forward(flat_images)
        _, _, embed_dim = patch_tokens.shape

        camera_token = _slice_expand_and_flatten(self.camera_token[...], batch_size, seq_len)
        register_token = _slice_expand_and_flatten(self.register_token[...], batch_size, seq_len)
        tokens = jnp.concatenate([camera_token, register_token, patch_tokens], axis=1)

        patch_h = height // self.cfg.patch_size
        patch_w = width // self.cfg.patch_size
        pos = build_2d_positions(batch_size * seq_len, patch_h, patch_w, dtype=tokens.dtype)
        pos = pos + 1
        pos_special = jnp.zeros((batch_size * seq_len, self.patch_start_idx, 2), dtype=pos.dtype)
        pos = jnp.concatenate([pos_special, pos], axis=1)

        _, full_tokens, _ = tokens.shape

        frame_idx = 0
        global_idx = 0
        outputs = []
        for _ in range(self.aa_block_num):
            frame_intermediates = []
            global_intermediates = []
            for attn_type in self.cfg.aa_order:
                if attn_type == "frame":
                    if tokens.shape != (batch_size * seq_len, full_tokens, embed_dim):
                        tokens = tokens.reshape(batch_size, seq_len, full_tokens, embed_dim).reshape(
                            batch_size * seq_len, full_tokens, embed_dim
                        )
                    if pos.shape != (batch_size * seq_len, full_tokens, 2):
                        pos = pos.reshape(batch_size, seq_len, full_tokens, 2).reshape(batch_size * seq_len, full_tokens, 2)
                    for _ in range(self.cfg.aa_block_size):
                        tokens = self.frame_blocks[frame_idx](tokens, pos, rngs=rngs)
                        frame_idx += 1
                        frame_intermediates.append(tokens.reshape(batch_size, seq_len, full_tokens, embed_dim))
                elif attn_type == "global":
                    if tokens.shape != (batch_size, seq_len * full_tokens, embed_dim):
                        tokens = tokens.reshape(batch_size, seq_len, full_tokens, embed_dim).reshape(
                            batch_size, seq_len * full_tokens, embed_dim
                        )
                    if pos.shape != (batch_size, seq_len * full_tokens, 2):
                        pos = pos.reshape(batch_size, seq_len, full_tokens, 2).reshape(batch_size, seq_len * full_tokens, 2)
                    for _ in range(self.cfg.aa_block_size):
                        tokens = self.global_blocks[global_idx](tokens, pos, rngs=rngs)
                        global_idx += 1
                        global_intermediates.append(tokens.reshape(batch_size, seq_len, full_tokens, embed_dim))
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for frame_tensor, global_tensor in zip(frame_intermediates, global_intermediates, strict=True):
                outputs.append(jnp.concatenate([frame_tensor, global_tensor], axis=-1))

            tokens = tokens.reshape(batch_size, seq_len, full_tokens, embed_dim).reshape(batch_size * seq_len, full_tokens, embed_dim)
            pos = pos.reshape(batch_size, seq_len, full_tokens, 2).reshape(batch_size * seq_len, full_tokens, 2)

        return outputs, self.patch_start_idx

__all__ = [
    "AggregatorConfig",
    "Aggregator",
]
