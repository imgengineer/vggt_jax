import dataclasses

import jax.numpy as jnp
from flax import nnx

from vggt.heads.dpt_head import DPTHead, DPTHeadConfig
from vggt.heads.track_ops import run_tracker_predictor
from vggt.holders import TrackerHolder


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


__all__ = [
    "TrackHeadConfig",
    "TrackHead",
]

