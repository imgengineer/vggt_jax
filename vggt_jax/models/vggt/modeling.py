import dataclasses
import math
import os
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from flax import nnx
from huggingface_hub import snapshot_download

from vggt_jax.heads.head_act import activate_head, activate_pose
from vggt_jax.heads.track_ops import run_tracker_predictor
from vggt_jax.heads.utils import create_uv_grid, position_grid_to_embed
from vggt_jax.holders import DPTTorchHolder, DPTTorchHolderConfig, PatchEmbedBackboneHolder, TrackerHolder
from vggt_jax.layers.block import Block, BlockConfig
from vggt_jax.layers.patch_embed import PatchEmbed, PatchEmbedConfig
from vggt_jax.layers.rope import build_2d_positions

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
        mean = _RESNET_MEAN[None, None, :, None, None]
        std = _RESNET_STD[None, None, :, None, None]
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

    def _patch_embed_forward_from_holder(self, images_bchw: jnp.ndarray) -> jnp.ndarray:
        holder = self.patch_embed_holder

        tokens = _torch_conv2d_nchw(images_bchw, holder.patch_embed.proj, stride=self.cfg.patch_size, padding=0)
        batch_size, channels, patch_h, patch_w = tokens.shape
        tokens = tokens.reshape(batch_size, channels, patch_h * patch_w).transpose(0, 2, 1)

        cls_token = jnp.broadcast_to(holder.cls_token[...], (batch_size, 1, holder.cls_token.shape[-1]))
        tokens = jnp.concatenate((cls_token, tokens), axis=1)
        tokens = tokens + self._interpolate_pos_encoding(tokens, images_bchw.shape[-2], images_bchw.shape[-1])

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

    def _patch_embed_forward(self, images_bchw: jnp.ndarray) -> jnp.ndarray:
        use_holder_runtime = (
            self.patch_embed_holder is not None
            and self.cfg.use_patch_embed_holder_runtime
            and "conv" not in self.cfg.patch_embed
        )

        if use_holder_runtime:
            return self._patch_embed_forward_from_holder(images_bchw)

        return self.patch_embed(images_bchw)

    def __call__(
        self, images: jnp.ndarray, *, rngs: nnx.Rngs | None, deterministic: bool
    ) -> tuple[list[jnp.ndarray], int]:
        batch_size, seq_len, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected images with C=3, got {channels}")

        images = self._normalize(images)
        flat_images = images.reshape(batch_size * seq_len, channels, height, width)
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
                        tokens = self.frame_blocks[frame_idx](tokens, pos, rngs=rngs, deterministic=deterministic)
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
                        tokens = self.global_blocks[global_idx](tokens, pos, rngs=rngs, deterministic=deterministic)
                        global_idx += 1
                        global_intermediates.append(tokens.reshape(batch_size, seq_len, full_tokens, embed_dim))
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for frame_tensor, global_tensor in zip(frame_intermediates, global_intermediates, strict=True):
                outputs.append(jnp.concatenate([frame_tensor, global_tensor], axis=-1))

            tokens = tokens.reshape(batch_size, seq_len, full_tokens, embed_dim).reshape(batch_size * seq_len, full_tokens, embed_dim)
            pos = pos.reshape(batch_size, seq_len, full_tokens, 2).reshape(batch_size * seq_len, full_tokens, 2)

        return outputs, self.patch_start_idx


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale) + shift


@dataclasses.dataclass(frozen=True)
class CameraHeadConfig:
    dim_in: int = 2048
    trunk_depth: int = 4
    num_heads: int = 16
    mlp_ratio: float = 4.0
    init_values: float = 0.01
    pose_encoding_type: str = "absT_quaR_FoV"
    trans_act: str = "linear"
    quat_act: str = "linear"
    fl_act: str = "relu"


class CameraHead(nnx.Module):
    def __init__(self, cfg: CameraHeadConfig, *, rngs: nnx.Rngs):
        if cfg.pose_encoding_type != "absT_quaR_FoV":
            raise ValueError(f"Unsupported pose encoding type: {cfg.pose_encoding_type}")
        self.cfg = cfg
        self.target_dim = 9
        block_cfg = BlockConfig(
            dim=cfg.dim_in,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            init_values=cfg.init_values,
            use_rope=False,
            qk_norm=False,
        )
        self.trunk = nnx.List([Block(block_cfg, rngs=rngs) for _ in range(cfg.trunk_depth)])
        self.token_norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-5, rngs=rngs)
        self.trunk_norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-5, rngs=rngs)
        self.empty_pose_tokens = nnx.Param(jnp.zeros((1, 1, self.target_dim), dtype=jnp.float32))
        self.embed_pose = nnx.Linear(self.target_dim, cfg.dim_in, rngs=rngs)
        self.pose_ln_mod = nnx.Linear(cfg.dim_in, 3 * cfg.dim_in, use_bias=True, rngs=rngs)
        self.adaln_norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-6, use_bias=False, use_scale=False, rngs=rngs)
        self.pose_branch_1 = nnx.Linear(cfg.dim_in, cfg.dim_in // 2, rngs=rngs)
        self.pose_branch_2 = nnx.Linear(cfg.dim_in // 2, self.target_dim, rngs=rngs)

    def _trunk_forward(self, x: jnp.ndarray, *, rngs: nnx.Rngs | None, deterministic: bool) -> jnp.ndarray:
        for block in self.trunk:
            x = block(x, pos=None, rngs=rngs, deterministic=deterministic)
        return x

    def _pose_branch(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.pose_branch_1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.pose_branch_2(x)
        return x

    def __call__(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        num_iterations: int = 4,
        *,
        rngs: nnx.Rngs | None,
        deterministic: bool,
    ) -> list[jnp.ndarray]:
        tokens = aggregated_tokens_list[-1]
        pose_tokens = self.token_norm(tokens[:, :, 0])

        batch_size, seq_len, _ = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list: list[jnp.ndarray] = []
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                pose_input = jnp.broadcast_to(self.empty_pose_tokens[...], (batch_size, seq_len, self.target_dim))
            else:
                pose_input = jax.lax.stop_gradient(pred_pose_enc)

            mod_input = self.embed_pose(pose_input)
            mod_input = jax.nn.silu(mod_input)
            shift, scale, gate = jnp.split(self.pose_ln_mod(mod_input), 3, axis=-1)

            pose_tokens_mod = gate * modulate(self.adaln_norm(pose_tokens), shift, scale)
            pose_tokens_mod = pose_tokens_mod + pose_tokens
            pose_tokens_mod = self._trunk_forward(pose_tokens_mod, rngs=rngs, deterministic=deterministic)

            delta = self._pose_branch(self.trunk_norm(pose_tokens_mod))
            pred_pose_enc = delta if pred_pose_enc is None else pred_pose_enc + delta

            activated = activate_pose(
                pred_pose_enc,
                trans_act=self.cfg.trans_act,
                quat_act=self.cfg.quat_act,
                fl_act=self.cfg.fl_act,
            )
            pred_pose_enc_list.append(activated)

        return pred_pose_enc_list


@dataclasses.dataclass(frozen=True)
class DPTHeadConfig:
    dim_in: int = 2048
    patch_size: int = 14
    output_dim: int = 4
    activation: str = "inv_log"
    conf_activation: str = "expp1"
    intermediate_layer_idx: tuple[int, int, int, int] = (4, 11, 17, 23)
    feature_only: bool = False
    features: int = 256
    out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    pos_embed: bool = True
    down_ratio: int = 1
    frames_chunk_size: int | None = 8
    use_torch_runtime: bool = True
    enable_torch_holder: bool = True


class _SimpleRefine(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        return x


def _torch_conv2d(x: jnp.ndarray, holder, *, stride: int = 1, padding: int = 0) -> jnp.ndarray:
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


def _torch_conv_transpose2d(x: jnp.ndarray, holder, *, stride: int = 1) -> jnp.ndarray:
    y = lax.conv_transpose(
        lhs=x,
        rhs=holder.weight[...],
        strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        transpose_kernel=True,
        precision=lax.Precision.HIGHEST,
    )
    if hasattr(holder, "bias"):
        y = y + holder.bias[...][None, :, None, None]
    return y


def _resize_align_corners_nchw(x: jnp.ndarray, size: tuple[int, int]) -> jnp.ndarray:
    batch_size, channels, height, width = x.shape
    out_h, out_w = size

    if out_h == 1:
        ys = jnp.zeros((1,), dtype=x.dtype)
    else:
        ys = jnp.linspace(0.0, float(height - 1), out_h, dtype=x.dtype)

    if out_w == 1:
        xs = jnp.zeros((1,), dtype=x.dtype)
    else:
        xs = jnp.linspace(0.0, float(width - 1), out_w, dtype=x.dtype)

    y0 = jnp.floor(ys).astype(jnp.int32)
    x0 = jnp.floor(xs).astype(jnp.int32)
    y1 = jnp.clip(y0 + 1, 0, height - 1)
    x1 = jnp.clip(x0 + 1, 0, width - 1)

    wy = ys - y0.astype(x.dtype)
    wx = xs - x0.astype(x.dtype)

    x00 = x[:, :, y0[:, None], x0[None, :]]
    x10 = x[:, :, y1[:, None], x0[None, :]]
    x01 = x[:, :, y0[:, None], x1[None, :]]
    x11 = x[:, :, y1[:, None], x1[None, :]]

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = wy[:, None] * (1.0 - wx)[None, :]
    wc = (1.0 - wy)[:, None] * wx[None, :]
    wd = wy[:, None] * wx[None, :]

    return (
        x00 * wa[None, None, :, :]
        + x10 * wb[None, None, :, :]
        + x01 * wc[None, None, :, :]
        + x11 * wd[None, None, :, :]
    )


def _custom_interpolate_nchw(
    x: jnp.ndarray,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> jnp.ndarray:
    if mode != "bilinear" or not align_corners:
        raise ValueError("Only bilinear + align_corners=True is supported")

    if size is None:
        if scale_factor is None:
            raise ValueError("Either size or scale_factor must be provided")
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    int_max = 1610612736
    input_elements = int(size[0]) * int(size[1]) * int(x.shape[0]) * int(x.shape[1])

    if input_elements > int_max:
        num_chunks = (input_elements // int_max) + 1
        chunks = jnp.array_split(x, num_chunks, axis=0)
        resized_chunks = [_resize_align_corners_nchw(chunk, size) for chunk in chunks]
        return jnp.concatenate(resized_chunks, axis=0)

    return _resize_align_corners_nchw(x, size)


def _residual_conv_unit(x: jnp.ndarray, holder) -> jnp.ndarray:
    residual = jax.nn.relu(x)
    out = _torch_conv2d(residual, holder.conv1, stride=1, padding=1)
    out = jax.nn.relu(out)
    out = _torch_conv2d(out, holder.conv2, stride=1, padding=1)
    return out + residual


def _feature_fusion_block(
    x: jnp.ndarray,
    holder,
    *,
    residual: jnp.ndarray | None,
    size: tuple[int, int] | None,
) -> jnp.ndarray:
    output = x
    if getattr(holder, "has_residual", False):
        if residual is None:
            raise ValueError("Residual input is required for this fusion block")
        res = _residual_conv_unit(residual, holder.resConfUnit1)
        output = output + res

    output = _residual_conv_unit(output, holder.resConfUnit2)

    if size is None:
        target_size = (output.shape[-2] * 2, output.shape[-1] * 2)
    else:
        target_size = size

    output = _custom_interpolate_nchw(output, size=target_size, mode="bilinear", align_corners=True)
    output = _torch_conv2d(output, holder.out_conv, stride=1, padding=0)
    return output


class DPTHead(nnx.Module):
    def __init__(self, cfg: DPTHeadConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-5, rngs=rngs)

        self.proj = nnx.List([nnx.Conv(cfg.dim_in, cfg.dim_in // 2, kernel_size=(1, 1), rngs=rngs) for _ in range(4)])
        self.refines = nnx.List([_SimpleRefine(cfg.dim_in // 2, rngs=rngs) for _ in range(4)])

        self.fuse = nnx.Conv(cfg.dim_in // 2, cfg.dim_in // 2, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.out1 = nnx.Conv(cfg.dim_in // 2, cfg.dim_in // 4, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.out2 = nnx.Conv(cfg.dim_in // 4, cfg.output_dim, kernel_size=(1, 1), rngs=rngs)

        if cfg.enable_torch_holder:
            self.torch_holder = DPTTorchHolder(
                DPTTorchHolderConfig(
                    dim_in=cfg.dim_in,
                    output_dim=cfg.output_dim,
                    out_channels=cfg.out_channels,
                    features=cfg.features,
                    feature_only=cfg.feature_only,
                )
            )
        else:
            self.torch_holder = None

    def _resize_to(self, x: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        return jax.image.resize(x, shape=(x.shape[0], height, width, x.shape[-1]), method="bilinear")

    def _tokens_to_map(self, tokens: jnp.ndarray, patch_h: int, patch_w: int) -> jnp.ndarray:
        batch_size, seq_len, patch_count, channels = tokens.shape
        x = tokens.reshape(batch_size * seq_len, patch_count, channels)
        x = self.norm(x)
        x = x.reshape(batch_size * seq_len, patch_h, patch_w, channels)
        return x

    def _apply_pos_embed_nchw(self, x: jnp.ndarray, width: int, height: int, ratio: float = 0.1) -> jnp.ndarray:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = jnp.transpose(pos_embed, (2, 0, 1))[None]
        pos_embed = jnp.broadcast_to(pos_embed, (x.shape[0],) + pos_embed.shape[1:])
        return x + pos_embed

    def _apply_resize_layer(self, x: jnp.ndarray, layer_idx: int) -> jnp.ndarray:
        resize_layer = self.torch_holder.resize_layers[layer_idx]
        if layer_idx == 0:
            return _torch_conv_transpose2d(x, resize_layer, stride=4)
        if layer_idx == 1:
            return _torch_conv_transpose2d(x, resize_layer, stride=2)
        if layer_idx == 2:
            return x
        return _torch_conv2d(x, resize_layer, stride=2, padding=1)

    def _scratch_forward_torch(self, features: list[jnp.ndarray]) -> jnp.ndarray:
        layer_1, layer_2, layer_3, layer_4 = features
        scratch = self.torch_holder.scratch

        layer_1_rn = _torch_conv2d(layer_1, scratch.layer1_rn, stride=1, padding=1)
        layer_2_rn = _torch_conv2d(layer_2, scratch.layer2_rn, stride=1, padding=1)
        layer_3_rn = _torch_conv2d(layer_3, scratch.layer3_rn, stride=1, padding=1)
        layer_4_rn = _torch_conv2d(layer_4, scratch.layer4_rn, stride=1, padding=1)

        out = _feature_fusion_block(
            layer_4_rn,
            scratch.refinenet4,
            residual=None,
            size=(layer_3_rn.shape[-2], layer_3_rn.shape[-1]),
        )
        out = _feature_fusion_block(
            out,
            scratch.refinenet3,
            residual=layer_3_rn,
            size=(layer_2_rn.shape[-2], layer_2_rn.shape[-1]),
        )
        out = _feature_fusion_block(
            out,
            scratch.refinenet2,
            residual=layer_2_rn,
            size=(layer_1_rn.shape[-2], layer_1_rn.shape[-1]),
        )
        out = _feature_fusion_block(
            out,
            scratch.refinenet1,
            residual=layer_1_rn,
            size=None,
        )

        out = _torch_conv2d(out, scratch.output_conv1, stride=1, padding=1)
        return out

    def _forward_torch_impl(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx]

        batch_size, seq_len, _, height, width = images.shape
        patch_h = height // self.cfg.patch_size
        patch_w = width // self.cfg.patch_size

        features = []
        max_index = len(aggregated_tokens_list) - 1

        for dpt_idx, source_layer_idx in enumerate(self.cfg.intermediate_layer_idx):
            layer_idx = min(source_layer_idx, max_index)
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(batch_size * seq_len, -1, x.shape[-1])
            x = _torch_layer_norm(x, self.torch_holder.norm, eps=1e-5)
            x = jnp.transpose(x, (0, 2, 1)).reshape(batch_size * seq_len, x.shape[-1], patch_h, patch_w)

            x = _torch_conv2d(x, self.torch_holder.projects[dpt_idx], stride=1, padding=0)
            if self.cfg.pos_embed:
                x = self._apply_pos_embed_nchw(x, width, height)
            x = self._apply_resize_layer(x, dpt_idx)

            features.append(x)

        out = self._scratch_forward_torch(features)
        target_size = (
            int(patch_h * self.cfg.patch_size / self.cfg.down_ratio),
            int(patch_w * self.cfg.patch_size / self.cfg.down_ratio),
        )
        out = _custom_interpolate_nchw(out, size=target_size, mode="bilinear", align_corners=True)

        if self.cfg.pos_embed:
            out = self._apply_pos_embed_nchw(out, width, height)

        if self.cfg.feature_only:
            return out.reshape(batch_size, seq_len, *out.shape[1:])

        out = _torch_conv2d(out, self.torch_holder.scratch.output_conv2[0], stride=1, padding=1)
        out = jax.nn.relu(out)
        out = _torch_conv2d(out, self.torch_holder.scratch.output_conv2[2], stride=1, padding=0)

        preds, conf = activate_head(out, activation=self.cfg.activation, conf_activation=self.cfg.conf_activation)
        preds = preds.reshape(batch_size, seq_len, *preds.shape[1:])
        conf = conf.reshape(batch_size, seq_len, *conf.shape[1:])
        return preds, conf

    def _forward_simple(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
        batch_size, seq_len, _, height, width = images.shape
        patch_h = height // self.cfg.patch_size
        patch_w = width // self.cfg.patch_size

        features = []
        layer_indices = list(self.cfg.intermediate_layer_idx)
        max_index = len(aggregated_tokens_list) - 1
        for head_idx, layer_idx in enumerate(layer_indices):
            layer_idx = min(layer_idx, max_index)
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            x = self._tokens_to_map(x, patch_h, patch_w)
            x = self.proj[head_idx](x)

            if head_idx == 0:
                target_h, target_w = patch_h * 4, patch_w * 4
            elif head_idx == 1:
                target_h, target_w = patch_h * 2, patch_w * 2
            elif head_idx == 2:
                target_h, target_w = patch_h, patch_w
            else:
                target_h, target_w = max(patch_h // 2, 1), max(patch_w // 2, 1)

            x = self._resize_to(x, target_h, target_w)
            x = self.refines[head_idx](x)
            features.append(x)

        fused = features[0]
        for x in features[1:]:
            x = self._resize_to(x, fused.shape[1], fused.shape[2])
            fused = fused + x

        fused = self.fuse(fused)
        fused = jax.nn.relu(fused)
        fused = self._resize_to(fused, height, width)

        if self.cfg.feature_only:
            fused = jnp.transpose(fused, (0, 3, 1, 2))
            return fused.reshape(batch_size, seq_len, *fused.shape[1:])

        out = self.out1(fused)
        out = jax.nn.relu(out)
        out = self.out2(out)
        out = jnp.transpose(out, (0, 3, 1, 2))

        preds, conf = activate_head(out, activation=self.cfg.activation, conf_activation=self.cfg.conf_activation)
        preds = preds.reshape(batch_size, seq_len, *preds.shape[1:])
        conf = conf.reshape(batch_size, seq_len, *conf.shape[1:])
        return preds, conf

    def __call__(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: int,
        frames_chunk_size: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
        if self.torch_holder is None or not self.cfg.use_torch_runtime:
            return self._forward_simple(aggregated_tokens_list, images, patch_start_idx)

        _, seq_len, _, _, _ = images.shape
        if frames_chunk_size is None:
            frames_chunk_size = self.cfg.frames_chunk_size

        if frames_chunk_size is None or frames_chunk_size >= seq_len:
            return self._forward_torch_impl(aggregated_tokens_list, images, patch_start_idx)

        if frames_chunk_size <= 0:
            raise ValueError(f"frames_chunk_size must be positive, got {frames_chunk_size}")

        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, seq_len, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, seq_len)
            chunk = self._forward_torch_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx=frames_start_idx,
                frames_end_idx=frames_end_idx,
            )

            if self.cfg.feature_only:
                all_preds.append(chunk)
            else:
                chunk_preds, chunk_conf = chunk
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        if self.cfg.feature_only:
            return jnp.concatenate(all_preds, axis=1)

        return jnp.concatenate(all_preds, axis=1), jnp.concatenate(all_conf, axis=1)


@dataclasses.dataclass(frozen=True)
class TrackHeadConfig:
    dim_in: int = 2048
    patch_size: int = 14
    features: int = 128
    iters: int = 4
    stride: int = 2
    corr_levels: int = 7
    corr_radius: int = 4
    hidden_size: int = 384
    tracker_depth: int = 6
    tracker_num_heads: int = 8
    max_scale: int = 518
    enable_tracker_holder: bool = True
    enable_torch_holder: bool = True


class TrackHead(nnx.Module):
    def __init__(self, cfg: TrackHeadConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.feature_extractor = DPTHead(
            DPTHeadConfig(
                dim_in=cfg.dim_in,
                patch_size=cfg.patch_size,
                output_dim=max(cfg.features, 4),
                feature_only=True,
                features=cfg.features,
                down_ratio=2,
                pos_embed=False,
                enable_torch_holder=cfg.enable_torch_holder,
            ),
            rngs=rngs,
        )
        if cfg.enable_tracker_holder:
            self.tracker = TrackerHolder(
                latent_dim=cfg.features,
                hidden_size=cfg.hidden_size,
                corr_levels=cfg.corr_levels,
                corr_radius=cfg.corr_radius,
                depth=cfg.tracker_depth,
                use_spaceatt=True,
            )
        else:
            self.tracker = None

    def _normalize_feature_dim(self, feature_maps: jnp.ndarray) -> jnp.ndarray:
        channels = feature_maps.shape[2]
        if channels == self.cfg.features:
            return feature_maps
        if channels > self.cfg.features:
            return feature_maps[:, :, : self.cfg.features]

        pad_width = ((0, 0), (0, 0), (0, self.cfg.features - channels), (0, 0), (0, 0))
        return jnp.pad(feature_maps, pad_width)

    def _fallback_predict(
        self,
        query_points: jnp.ndarray,
        batch_size: int,
        seq_len: int,
        track_iters: int,
        dtype,
    ) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        track_list = []
        tiled = jnp.broadcast_to(query_points[:, None, :, :], (batch_size, seq_len, query_points.shape[1], 2))
        for _ in range(track_iters):
            track_list.append(tiled)
        vis = jnp.ones((batch_size, seq_len, query_points.shape[1]), dtype=dtype)
        conf = jnp.ones((batch_size, seq_len, query_points.shape[1]), dtype=dtype)
        return track_list, vis, conf

    def __call__(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: int,
        query_points: jnp.ndarray | None,
        iters: int | None = None,
    ) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        batch_size, seq_len, _, _, _ = images.shape
        if query_points is None:
            raise ValueError("query_points is required for track prediction")
        if query_points.ndim == 2:
            query_points = query_points[None, ...]

        track_iters = self.cfg.iters if iters is None else iters

        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        if feature_maps.ndim != 5:
            raise ValueError(f"Expected 5D feature map tensor, got {feature_maps.shape}")

        if feature_maps.shape[-1] == self.cfg.features:
            feature_maps = jnp.transpose(feature_maps, (0, 1, 4, 2, 3))

        feature_maps = self._normalize_feature_dim(feature_maps)

        if self.tracker is None:
            return self._fallback_predict(query_points, batch_size, seq_len, track_iters, images.dtype)

        coord_preds, vis, conf = run_tracker_predictor(
            self.tracker,
            query_points=query_points,
            fmaps=feature_maps,
            iters=track_iters,
            stride=self.cfg.stride,
            corr_levels=self.cfg.corr_levels,
            corr_radius=self.cfg.corr_radius,
            max_scale=self.cfg.max_scale,
            num_heads=self.cfg.tracker_num_heads,
            apply_sigmoid=True,
        )

        if conf is None:
            conf = jnp.ones_like(vis)

        return coord_preds, vis, conf


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    aa_order: tuple[str, str] = ("frame", "global")
    aa_block_size: int = 1
    rope_freq: float = 100.0
    init_values: float = 0.01
    patch_embed: str = "dinov2_vitl14_reg"
    enable_camera: bool = True
    enable_point: bool = True
    enable_depth: bool = True
    enable_track: bool = True
    enable_conversion_holders: bool = True

    @classmethod
    def vggt_base(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def vggt_tiny(cls) -> "ModelConfig":
        return cls(
            img_size=224,
            patch_size=14,
            embed_dim=256,
            depth=8,
            num_heads=8,
            num_register_tokens=2,
            patch_embed="conv",
            enable_conversion_holders=False,
        )


class VGGT(nnx.Module):
    def __init__(
        self,
        cfg: ModelConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_depth: bool = True,
        enable_track: bool = True,
        enable_conversion_holders: bool = True,
    ):
        if cfg is not None:
            non_default_args = [
                ("img_size", img_size, 518),
                ("patch_size", patch_size, 14),
                ("embed_dim", embed_dim, 1024),
                ("enable_camera", enable_camera, True),
                ("enable_point", enable_point, True),
                ("enable_depth", enable_depth, True),
                ("enable_track", enable_track, True),
                ("enable_conversion_holders", enable_conversion_holders, True),
            ]
            unexpected = [name for name, value, default in non_default_args if value != default]
            if unexpected:
                raise ValueError(f"Pass either cfg or scalar args, got both: {unexpected}")
        else:
            cfg = ModelConfig(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                enable_camera=enable_camera,
                enable_point=enable_point,
                enable_depth=enable_depth,
                enable_track=enable_track,
                enable_conversion_holders=enable_conversion_holders,
            )

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.cfg = cfg
        # Match PyTorch defaults: freshly constructed modules are in training mode.
        # NNX's `train()` / `eval()` toggles `deterministic`; we reuse that here.
        self.deterministic = False
        self.aggregator = Aggregator(
            AggregatorConfig(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                num_register_tokens=cfg.num_register_tokens,
                aa_order=cfg.aa_order,
                aa_block_size=cfg.aa_block_size,
                rope_freq=cfg.rope_freq,
                init_values=cfg.init_values,
                patch_embed=cfg.patch_embed,
                enable_patch_embed_holder=cfg.enable_conversion_holders,
            ),
            rngs=rngs,
        )

        head_dim_in = 2 * cfg.embed_dim
        self.camera_head = (
            CameraHead(CameraHeadConfig(dim_in=head_dim_in, num_heads=cfg.num_heads), rngs=rngs)
            if cfg.enable_camera
            else None
        )
        self.point_head = (
            DPTHead(
                DPTHeadConfig(
                    dim_in=head_dim_in,
                    patch_size=cfg.patch_size,
                    output_dim=4,
                    activation="inv_log",
                    conf_activation="expp1",
                    features=256,
                    enable_torch_holder=cfg.enable_conversion_holders,
                ),
                rngs=rngs,
            )
            if cfg.enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                DPTHeadConfig(
                    dim_in=head_dim_in,
                    patch_size=cfg.patch_size,
                    output_dim=2,
                    activation="exp",
                    conf_activation="expp1",
                    features=256,
                    enable_torch_holder=cfg.enable_conversion_holders,
                ),
                rngs=rngs,
            )
            if cfg.enable_depth
            else None
        )
        self.track_head = (
            TrackHead(
                TrackHeadConfig(
                    dim_in=head_dim_in,
                    patch_size=cfg.patch_size,
                    enable_tracker_holder=cfg.enable_conversion_holders,
                    enable_torch_holder=cfg.enable_conversion_holders,
                ),
                rngs=rngs,
            )
            if cfg.enable_track
            else None
        )

    @classmethod
    def from_torch_checkpoint(
        cls,
        checkpoint_path: str,
        config: ModelConfig | None = None,
        *,
        strict: bool = False,
        include_track: bool = True,
        rngs: nnx.Rngs | None = None,
        return_report: bool = False,
    ):
        model, report = create_vggt_from_torch_checkpoint(
            checkpoint_path,
            config=config,
            strict=strict,
            include_track=include_track,
            rngs=rngs,
        )
        return (model, report) if return_report else model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "facebook/VGGT-1B",
        *,
        filename: str = "model.pt",
        cache_dir: str = "./weights",
        proxy_port: int | None = None,
        config: ModelConfig | None = None,
        strict: bool = False,
        include_track: bool = True,
        rngs: nnx.Rngs | None = None,
        return_report: bool = False,
    ):
        configure_proxy_env(proxy_port)
        local_dir = Path(download_pretrained_weights(repo_id=repo_id, cache_dir=cache_dir))
        checkpoint_path = local_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")
        model, report = create_vggt_from_torch_checkpoint(
            str(checkpoint_path),
            config=config,
            strict=strict,
            include_track=include_track,
            rngs=rngs,
        )
        return (model, report) if return_report else model

    def eval(self, **attributes):
        super().eval(**attributes)
        return self

    def train(self, mode: bool = True, **attributes):
        if mode:
            super().train(**attributes)
        else:
            super().eval(**attributes)
        return self

    def forward(self, images: jnp.ndarray, query_points: jnp.ndarray | None = None):
        return self(images, query_points)

    def __call__(
        self,
        images: jnp.ndarray,
        query_points: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
    ) -> dict[str, jnp.ndarray | list[jnp.ndarray]]:
        deterministic_flag = bool(self.deterministic) if deterministic is None else bool(deterministic)

        if images.ndim == 4:
            images = images[None, ...]
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points[None, ...]

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images, rngs=rngs, deterministic=deterministic_flag
        )

        predictions: dict[str, jnp.ndarray | list[jnp.ndarray]] = {}
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(
                aggregated_tokens_list, rngs=rngs, deterministic=deterministic_flag
            )
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(aggregated_tokens_list, images, patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            world_points, world_points_conf = self.point_head(aggregated_tokens_list, images, patch_start_idx)
            predictions["world_points"] = world_points
            predictions["world_points_conf"] = world_points_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        # Match PyTorch: store images only during inference/eval mode.
        if deterministic_flag:
            predictions["images"] = images

        return predictions


@jax.jit
def forward(graphdef: nnx.GraphDef[nnx.Module], state: nnx.State, images: jax.Array, rngs: nnx.Rngs) -> dict:
    model = nnx.merge(graphdef, state)
    return model(images, rngs=rngs, deterministic=False)


Transform = str
NONE: Transform = "none"
LINEAR_WEIGHT: Transform = "linear_weight"
CONV_OIHW_TO_HWIO: Transform = "conv_oihw_to_hwio"


@dataclasses.dataclass(frozen=True)
class ConversionReport:
    total_source_keys: int
    mapped_keys: int
    loaded_keys: int
    skipped_unmapped: int
    skipped_missing_target: int
    skipped_shape_mismatch: int
    missing_target_leaves: int


def configure_proxy_env(proxy_port: int | None = None) -> None:
    if proxy_port is None:
        return
    http_proxy = f"http://127.0.0.1:{proxy_port}"
    all_proxy = f"http://127.0.0.1:{proxy_port}"
    os.environ["HTTP_PROXY"] = http_proxy
    os.environ["HTTPS_PROXY"] = http_proxy
    os.environ["ALL_PROXY"] = all_proxy


def download_pretrained_weights(
    repo_id: str = "facebook/VGGT-1B",
    cache_dir: str = "./weights",
    allow_patterns: tuple[str, ...] = ("*.pt", "*.bin", "*.safetensors", "*.json"),
) -> str:
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns),
    )
    return local_dir


def create_vggt_model(config: ModelConfig | None = None, *, rngs: nnx.Rngs | None = None) -> VGGT:
    if rngs is None:
        rngs = nnx.Rngs(0)
    if config is None:
        return VGGT(rngs=rngs)
    return VGGT(config, rngs=rngs)


def create_vggt_from_pretrained(
    repo_id: str = "facebook/VGGT-1B", cache_dir: str = "./weights"
) -> tuple[VGGT, Path]:
    config = ModelConfig.vggt_base()
    model = create_vggt_model(config)
    local_dir = Path(download_pretrained_weights(repo_id=repo_id, cache_dir=cache_dir))
    return model, local_dir


def _flatten_tree(tree: dict[str, Any], prefix: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    out: dict[tuple[Any, ...], Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(_flatten_tree(value, prefix + (key,)))
        return out
    out[prefix] = tree
    return out


def _unflatten_tree(flat: dict[tuple[Any, ...], Any]) -> dict[str, Any]:
    root: dict[Any, Any] = {}
    for path, value in flat.items():
        cursor = root
        for part in path[:-1]:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]
        cursor[path[-1]] = value
    return root


def _parse_path(path: str) -> tuple[Any, ...]:
    parts = []
    for token in path.split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token)
    return tuple(parts)


def _format_path(path: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in path)


def _map_block_param(rest: str) -> tuple[str, Transform] | None:
    mapping: dict[str, tuple[str, Transform]] = {
        "norm1.weight": ("norm1.scale", NONE),
        "norm1.bias": ("norm1.bias", NONE),
        "norm2.weight": ("norm2.scale", NONE),
        "norm2.bias": ("norm2.bias", NONE),
        "attn.q_norm.weight": ("attn.q_norm.scale", NONE),
        "attn.q_norm.bias": ("attn.q_norm.bias", NONE),
        "attn.k_norm.weight": ("attn.k_norm.scale", NONE),
        "attn.k_norm.bias": ("attn.k_norm.bias", NONE),
        "attn.qkv.weight": ("attn.qkv.kernel", LINEAR_WEIGHT),
        "attn.qkv.bias": ("attn.qkv.bias", NONE),
        "attn.proj.weight": ("attn.proj.kernel", LINEAR_WEIGHT),
        "attn.proj.bias": ("attn.proj.bias", NONE),
        "ls1.gamma": ("ls1.gamma", NONE),
        "ls2.gamma": ("ls2.gamma", NONE),
        "mlp.fc1.weight": ("mlp.fc1.kernel", LINEAR_WEIGHT),
        "mlp.fc1.bias": ("mlp.fc1.bias", NONE),
        "mlp.fc2.weight": ("mlp.fc2.kernel", LINEAR_WEIGHT),
        "mlp.fc2.bias": ("mlp.fc2.bias", NONE),
    }
    return mapping.get(rest)


def map_torch_key_to_nnx(source_key: str) -> tuple[str, Transform] | tuple[None, None]:
    source_key = source_key.removeprefix("module.")
    source_key = source_key.removeprefix("model.")

    direct_map: dict[str, tuple[str, Transform]] = {
        "aggregator.camera_token": ("aggregator.camera_token", NONE),
        "aggregator.register_token": ("aggregator.register_token", NONE),
        "camera_head.token_norm.weight": ("camera_head.token_norm.scale", NONE),
        "camera_head.token_norm.bias": ("camera_head.token_norm.bias", NONE),
        "camera_head.trunk_norm.weight": ("camera_head.trunk_norm.scale", NONE),
        "camera_head.trunk_norm.bias": ("camera_head.trunk_norm.bias", NONE),
        "camera_head.empty_pose_tokens": ("camera_head.empty_pose_tokens", NONE),
        "camera_head.embed_pose.weight": ("camera_head.embed_pose.kernel", LINEAR_WEIGHT),
        "camera_head.embed_pose.bias": ("camera_head.embed_pose.bias", NONE),
        "camera_head.poseLN_modulation.1.weight": ("camera_head.pose_ln_mod.kernel", LINEAR_WEIGHT),
        "camera_head.poseLN_modulation.1.bias": ("camera_head.pose_ln_mod.bias", NONE),
        "camera_head.pose_branch.fc1.weight": ("camera_head.pose_branch_1.kernel", LINEAR_WEIGHT),
        "camera_head.pose_branch.fc1.bias": ("camera_head.pose_branch_1.bias", NONE),
        "camera_head.pose_branch.fc2.weight": ("camera_head.pose_branch_2.kernel", LINEAR_WEIGHT),
        "camera_head.pose_branch.fc2.bias": ("camera_head.pose_branch_2.bias", NONE),
    }
    if source_key in direct_map:
        return direct_map[source_key]

    agg_match = re.match(r"^aggregator\.(frame_blocks|global_blocks)\.(\d+)\.(.+)$", source_key)
    if agg_match:
        block_group, block_idx, rest = agg_match.groups()
        mapped = _map_block_param(rest)
        if mapped is None:
            return None, None
        target_rest, transform = mapped
        return f"aggregator.{block_group}.{block_idx}.{target_rest}", transform

    if source_key.startswith("aggregator.patch_embed."):
        suffix = source_key.removeprefix("aggregator.patch_embed.")
        return f"aggregator.patch_embed_holder.{suffix}", NONE

    camera_match = re.match(r"^camera_head\.trunk\.(\d+)\.(.+)$", source_key)
    if camera_match:
        block_idx, rest = camera_match.groups()
        mapped = _map_block_param(rest)
        if mapped is None:
            return None, None
        target_rest, transform = mapped
        return f"camera_head.trunk.{block_idx}.{target_rest}", transform

    dpt_match = re.match(r"^(depth_head|point_head|track_head\.feature_extractor)\.(.+)$", source_key)
    if dpt_match:
        prefix, rest = dpt_match.groups()
        return f"{prefix}.torch_holder.{rest}", NONE

    if source_key.startswith("track_head.tracker."):
        return source_key, NONE

    return None, None


def _apply_transform(array: np.ndarray, transform: Transform) -> np.ndarray:
    if transform == NONE:
        return array
    if transform == LINEAR_WEIGHT:
        return array.T
    if transform == CONV_OIHW_TO_HWIO:
        return np.transpose(array, (2, 3, 1, 0))
    raise ValueError(f"Unsupported transform: {transform}")


def _load_torch_state_dict(checkpoint_path: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for .pt conversion. Install with: uv add --optional convert torch"
        ) from exc

    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                break
        else:
            if all(hasattr(value, "shape") for value in obj.values()):
                state_dict = obj
            else:
                raise ValueError(f"Unsupported checkpoint structure at {checkpoint_path}")
    else:
        raise ValueError(f"Unsupported checkpoint structure at {checkpoint_path}")

    normalized = {}
    for key, value in state_dict.items():
        stripped = key.removeprefix("module.")
        normalized[stripped] = value
    return normalized


def _to_numpy_tensor(torch_tensor: Any) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for conversion") from exc

    tensor = torch_tensor.detach().cpu()
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.numpy()


def create_vggt_from_torch_checkpoint(
    checkpoint_path: str,
    config: ModelConfig | None = None,
    *,
    strict: bool = False,
    include_track: bool = True,
    rngs: nnx.Rngs | None = None,
) -> tuple[VGGT, ConversionReport]:
    if config is None:
        config = ModelConfig.vggt_base()

    model = create_vggt_model(config, rngs=rngs)
    graph_def, abs_state = nnx.split(model)
    target_state = nnx.to_pure_dict(abs_state)
    flat_target = _flatten_tree(target_state)
    source_state = _load_torch_state_dict(checkpoint_path)

    mapped_keys = 0
    loaded_keys = 0
    skipped_unmapped = 0
    skipped_missing_target = 0
    skipped_shape_mismatch = 0

    for source_key, source_tensor in source_state.items():
        if not include_track and source_key.startswith("track_head.tracker."):
            skipped_unmapped += 1
            continue

        target_key, transform = map_torch_key_to_nnx(source_key)
        if target_key is None:
            skipped_unmapped += 1
            continue

        mapped_keys += 1
        target_path = _parse_path(target_key)
        if target_path not in flat_target:
            skipped_missing_target += 1
            continue

        array = _to_numpy_tensor(source_tensor)
        array = _apply_transform(array, transform)
        target_leaf = flat_target[target_path]
        if array.shape != target_leaf.shape:
            skipped_shape_mismatch += 1
            continue

        flat_target[target_path] = jnp.asarray(array, dtype=target_leaf.dtype)
        loaded_keys += 1

    updated_state = _unflatten_tree(flat_target)
    model = nnx.merge(graph_def, updated_state)

    missing_target_leaves = len(flat_target) - loaded_keys
    report = ConversionReport(
        total_source_keys=len(source_state),
        mapped_keys=mapped_keys,
        loaded_keys=loaded_keys,
        skipped_unmapped=skipped_unmapped,
        skipped_missing_target=skipped_missing_target,
        skipped_shape_mismatch=skipped_shape_mismatch,
        missing_target_leaves=missing_target_leaves,
    )

    if strict and (
        report.skipped_missing_target > 0 or report.skipped_shape_mismatch > 0 or report.skipped_unmapped > 0
    ):
        raise RuntimeError(
            "Strict conversion failed: "
            f"unmapped={report.skipped_unmapped}, "
            f"missing_target={report.skipped_missing_target}, "
            f"shape_mismatch={report.skipped_shape_mismatch}"
        )

    return model, report


def save_nnx_weights_npz(model: VGGT, output_path: str) -> None:
    _, state = nnx.split(model)
    flat = _flatten_tree(nnx.to_pure_dict(state))
    arrays = {_format_path(path): np.asarray(value) for path, value in flat.items()}
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)


def load_nnx_weights_npz(model: VGGT, npz_path: str, *, strict: bool = True) -> VGGT:
    graph_def, state = nnx.split(model)
    flat_target = _flatten_tree(nnx.to_pure_dict(state))

    data = np.load(npz_path)
    loaded = 0
    unexpected_keys: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key in data.files:
        path = _parse_path(key)
        if path not in flat_target:
            unexpected_keys.append(key)
            continue

        array = np.asarray(data[key])
        target_leaf = flat_target[path]
        if tuple(array.shape) != tuple(target_leaf.shape):
            shape_mismatches.append((key, tuple(array.shape), tuple(target_leaf.shape)))
            continue

        flat_target[path] = jnp.asarray(array, dtype=target_leaf.dtype)
        loaded += 1

    missing_keys = [key for key in (_format_path(path) for path in flat_target) if key not in data]

    if strict and (unexpected_keys or shape_mismatches or missing_keys):
        lines = ["Strict NPZ load failed:"]
        if unexpected_keys:
            lines.append(f"- unexpected_keys={len(unexpected_keys)}")
        if missing_keys:
            lines.append(f"- missing_keys={len(missing_keys)}")
        if shape_mismatches:
            example_key, src_shape, dst_shape = shape_mismatches[0]
            lines.append(f"- shape_mismatches={len(shape_mismatches)} (e.g. {example_key}: {src_shape} != {dst_shape})")
        raise RuntimeError("\n".join(lines))

    updated_state = _unflatten_tree(flat_target)
    _ = loaded
    return nnx.merge(graph_def, updated_state)


def create_vggt_from_nnx_npz(
    npz_path: str,
    config: ModelConfig | None = None,
    *,
    strict: bool = True,
    rngs: nnx.Rngs | None = None,
) -> VGGT:
    model = create_vggt_model(config, rngs=rngs)
    return load_nnx_weights_npz(model, npz_path, strict=strict)


__all__ = [
    "ModelConfig",
    "VGGT",
    "forward",
    "Transform",
    "NONE",
    "LINEAR_WEIGHT",
    "CONV_OIHW_TO_HWIO",
    "ConversionReport",
    "configure_proxy_env",
    "download_pretrained_weights",
    "create_vggt_model",
    "create_vggt_from_pretrained",
    "map_torch_key_to_nnx",
    "create_vggt_from_torch_checkpoint",
    "save_nnx_weights_npz",
    "load_nnx_weights_npz",
    "create_vggt_from_nnx_npz",
]
