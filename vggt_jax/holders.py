import dataclasses

import jax.numpy as jnp
from flax import nnx


class EmptyModule(nnx.Module):
    def __call__(self, x):
        return x


class TorchLinearHolder(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True):
        self.weight = nnx.Param(jnp.zeros((out_features, in_features), dtype=jnp.float32))
        if bias:
            self.bias = nnx.Param(jnp.zeros((out_features,), dtype=jnp.float32))


class TorchLayerNormHolder(nnx.Module):
    def __init__(self, dim: int):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))


class TorchGroupNormHolder(nnx.Module):
    def __init__(self, dim: int):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))


class TorchConv2dHolder(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, *, bias: bool = True, groups: int = 1):
        self.weight = nnx.Param(jnp.zeros((out_ch, in_ch // groups, kernel_size, kernel_size), dtype=jnp.float32))
        if bias:
            self.bias = nnx.Param(jnp.zeros((out_ch,), dtype=jnp.float32))


class TorchConvTranspose2dHolder(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, *, bias: bool = True, groups: int = 1):
        self.weight = nnx.Param(jnp.zeros((in_ch, out_ch // groups, kernel_size, kernel_size), dtype=jnp.float32))
        if bias:
            self.bias = nnx.Param(jnp.zeros((out_ch,), dtype=jnp.float32))


class TorchMlpHolder(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int | None = None):
        if out_features is None:
            out_features = in_features
        self.fc1 = TorchLinearHolder(in_features, hidden_features)
        self.fc2 = TorchLinearHolder(hidden_features, out_features)


class TorchQKVAttentionHolder(nnx.Module):
    def __init__(self, dim: int):
        self.qkv = TorchLinearHolder(dim, 3 * dim)
        self.proj = TorchLinearHolder(dim, dim)


class TorchLayerScaleHolder(nnx.Module):
    def __init__(self, dim: int):
        self.gamma = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))


class TorchVitBlockHolder(nnx.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        hidden = int(dim * mlp_ratio)
        self.norm1 = TorchLayerNormHolder(dim)
        self.attn = TorchQKVAttentionHolder(dim)
        self.ls1 = TorchLayerScaleHolder(dim)
        self.norm2 = TorchLayerNormHolder(dim)
        self.mlp = TorchMlpHolder(dim, hidden, dim)
        self.ls2 = TorchLayerScaleHolder(dim)


class PatchEmbedProjHolder(nnx.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        self.proj = TorchConv2dHolder(3, embed_dim, patch_size)


class PatchEmbedBackboneHolder(nnx.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, depth: int, num_register_tokens: int):
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim), dtype=jnp.float32))
        self.pos_embed = nnx.Param(jnp.zeros((1, num_patches + 1, embed_dim), dtype=jnp.float32))
        self.register_tokens = nnx.Param(jnp.zeros((1, num_register_tokens, embed_dim), dtype=jnp.float32))
        self.mask_token = nnx.Param(jnp.zeros((1, embed_dim), dtype=jnp.float32))
        self.patch_embed = PatchEmbedProjHolder(patch_size, embed_dim)
        self.blocks = nnx.List([TorchVitBlockHolder(embed_dim, mlp_ratio=4.0) for _ in range(depth)])
        self.norm = TorchLayerNormHolder(embed_dim)


class ResidualConvUnitHolder(nnx.Module):
    def __init__(self, features: int):
        self.conv1 = TorchConv2dHolder(features, features, kernel_size=3, bias=True)
        self.conv2 = TorchConv2dHolder(features, features, kernel_size=3, bias=True)


class FeatureFusionBlockHolder(nnx.Module):
    def __init__(self, features: int, has_residual: bool = True):
        self.out_conv = TorchConv2dHolder(features, features, kernel_size=1, bias=True)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnitHolder(features)
        self.resConfUnit2 = ResidualConvUnitHolder(features)


class ScratchHolder(nnx.Module):
    def __init__(self, in_shape: tuple[int, int, int, int], out_shape: int, output_dim: int, feature_only: bool):
        self.layer1_rn = TorchConv2dHolder(in_shape[0], out_shape, kernel_size=3, bias=False)
        self.layer2_rn = TorchConv2dHolder(in_shape[1], out_shape, kernel_size=3, bias=False)
        self.layer3_rn = TorchConv2dHolder(in_shape[2], out_shape, kernel_size=3, bias=False)
        self.layer4_rn = TorchConv2dHolder(in_shape[3], out_shape, kernel_size=3, bias=False)

        self.refinenet1 = FeatureFusionBlockHolder(out_shape, has_residual=True)
        self.refinenet2 = FeatureFusionBlockHolder(out_shape, has_residual=True)
        self.refinenet3 = FeatureFusionBlockHolder(out_shape, has_residual=True)
        self.refinenet4 = FeatureFusionBlockHolder(out_shape, has_residual=False)

        if feature_only:
            self.output_conv1 = TorchConv2dHolder(out_shape, out_shape, kernel_size=3, bias=True)
        else:
            self.output_conv1 = TorchConv2dHolder(out_shape, out_shape // 2, kernel_size=3, bias=True)
            self.output_conv2 = nnx.List(
                [
                    TorchConv2dHolder(out_shape // 2, 32, kernel_size=3, bias=True),
                    EmptyModule(),
                    TorchConv2dHolder(32, output_dim, kernel_size=1, bias=True),
                ]
            )


@dataclasses.dataclass(frozen=True)
class DPTTorchHolderConfig:
    dim_in: int
    output_dim: int
    out_channels: tuple[int, int, int, int]
    features: int
    feature_only: bool


class DPTTorchHolder(nnx.Module):
    def __init__(self, cfg: DPTTorchHolderConfig):
        self.norm = TorchLayerNormHolder(cfg.dim_in)
        self.projects = nnx.List(
            [
                TorchConv2dHolder(cfg.dim_in, cfg.out_channels[0], kernel_size=1, bias=True),
                TorchConv2dHolder(cfg.dim_in, cfg.out_channels[1], kernel_size=1, bias=True),
                TorchConv2dHolder(cfg.dim_in, cfg.out_channels[2], kernel_size=1, bias=True),
                TorchConv2dHolder(cfg.dim_in, cfg.out_channels[3], kernel_size=1, bias=True),
            ]
        )
        self.resize_layers = nnx.List(
            [
                TorchConvTranspose2dHolder(cfg.out_channels[0], cfg.out_channels[0], kernel_size=4, bias=True),
                TorchConvTranspose2dHolder(cfg.out_channels[1], cfg.out_channels[1], kernel_size=2, bias=True),
                EmptyModule(),
                TorchConv2dHolder(cfg.out_channels[3], cfg.out_channels[3], kernel_size=3, bias=True),
            ]
        )
        self.scratch = ScratchHolder(
            in_shape=cfg.out_channels,
            out_shape=cfg.features,
            output_dim=cfg.output_dim,
            feature_only=cfg.feature_only,
        )


class TorchMultiHeadAttentionHolder(nnx.Module):
    def __init__(self, hidden_size: int):
        self.in_proj_weight = nnx.Param(jnp.zeros((3 * hidden_size, hidden_size), dtype=jnp.float32))
        self.in_proj_bias = nnx.Param(jnp.zeros((3 * hidden_size,), dtype=jnp.float32))
        self.out_proj = TorchLinearHolder(hidden_size, hidden_size)


class AttnBlockHolder(nnx.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        hidden_mlp = int(hidden_size * mlp_ratio)
        self.norm1 = TorchLayerNormHolder(hidden_size)
        self.norm2 = TorchLayerNormHolder(hidden_size)
        self.attn = TorchMultiHeadAttentionHolder(hidden_size)
        self.mlp = TorchMlpHolder(hidden_size, hidden_mlp, hidden_size)


class CrossAttnBlockHolder(nnx.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        hidden_mlp = int(hidden_size * mlp_ratio)
        self.norm1 = TorchLayerNormHolder(hidden_size)
        self.norm_context = TorchLayerNormHolder(hidden_size)
        self.norm2 = TorchLayerNormHolder(hidden_size)
        self.cross_attn = TorchMultiHeadAttentionHolder(hidden_size)
        self.mlp = TorchMlpHolder(hidden_size, hidden_mlp, hidden_size)


class EfficientUpdateFormerHolder(nnx.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        time_depth: int,
        space_depth: int,
        num_virtual_tracks: int,
    ):
        self.input_norm = TorchLayerNormHolder(input_dim)
        self.input_transform = TorchLinearHolder(input_dim, hidden_size)
        self.output_norm = TorchLayerNormHolder(hidden_size)
        self.flow_head = TorchLinearHolder(hidden_size, output_dim)
        self.virual_tracks = nnx.Param(jnp.zeros((1, num_virtual_tracks, 1, hidden_size), dtype=jnp.float32))

        self.time_blocks = nnx.List([AttnBlockHolder(hidden_size) for _ in range(time_depth)])
        self.space_virtual_blocks = nnx.List([AttnBlockHolder(hidden_size) for _ in range(space_depth)])
        self.space_point2virtual_blocks = nnx.List([CrossAttnBlockHolder(hidden_size) for _ in range(space_depth)])
        self.space_virtual2point_blocks = nnx.List([CrossAttnBlockHolder(hidden_size) for _ in range(space_depth)])


class TrackerHolder(nnx.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 128,
        hidden_size: int = 384,
        corr_levels: int = 7,
        corr_radius: int = 4,
        depth: int = 6,
        use_spaceatt: bool = True,
        num_virtual_tracks: int = 64,
    ):
        corr_dim = corr_levels * ((corr_radius * 2 + 1) ** 2)
        self.corr_mlp = TorchMlpHolder(corr_dim, hidden_size, latent_dim)

        transformer_dim = latent_dim + latent_dim + latent_dim + 4
        self.query_ref_token = nnx.Param(jnp.zeros((1, 2, transformer_dim), dtype=jnp.float32))

        space_depth = depth if use_spaceatt else 0
        self.updateformer = EfficientUpdateFormerHolder(
            input_dim=transformer_dim,
            hidden_size=hidden_size,
            output_dim=latent_dim + 2,
            time_depth=depth,
            space_depth=space_depth,
            num_virtual_tracks=num_virtual_tracks,
        )

        self.fmap_norm = TorchLayerNormHolder(latent_dim)
        self.ffeat_norm = TorchGroupNormHolder(latent_dim)
        self.ffeat_updater = nnx.List([TorchLinearHolder(latent_dim, latent_dim), EmptyModule()])
        self.vis_predictor = nnx.List([TorchLinearHolder(latent_dim, 1)])
        self.conf_predictor = nnx.List([TorchLinearHolder(latent_dim, 1)])
