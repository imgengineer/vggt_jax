import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from vggt.heads.head_act import activate_head
from vggt.heads.utils import create_uv_grid, position_grid_to_embed
from vggt.holders import DPTTorchHolder, DPTTorchHolderConfig


def _torch_layer_norm(x: jnp.ndarray, holder, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * holder.weight[...] + holder.bias[...]


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
        return jax.image.resize(
            x,
            shape=(x.shape[0], height, width, x.shape[-1]),
            method="bilinear",
            antialias=False,
        )

    def _infer_image_hw(self, images: jnp.ndarray) -> tuple[int, int, int, int]:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, T, H, W, C] or [B, T, C, H, W], got {images.shape}")
        batch_size, seq_len = images.shape[0], images.shape[1]
        if images.shape[-1] == 3:
            return batch_size, seq_len, images.shape[2], images.shape[3]
        if images.shape[2] == 3:
            return batch_size, seq_len, images.shape[3], images.shape[4]
        raise ValueError(f"Cannot infer image layout from shape {images.shape}")

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

        batch_size, seq_len, height, width = self._infer_image_hw(images)
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
        batch_size, seq_len, height, width = self._infer_image_hw(images)
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


__all__ = [
    "DPTHeadConfig",
    "DPTHead",
    "_custom_interpolate_nchw",
    "_residual_conv_unit",
    "_feature_fusion_block",
]
