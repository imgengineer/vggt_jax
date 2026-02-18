from __future__ import annotations

import jax.numpy as jnp


class BaseTrackerPredictor:
    def __init__(
        self,
        stride: int = 1,
        corr_levels: int = 5,
        corr_radius: int = 4,
        latent_dim: int = 128,
        hidden_size: int = 384,
        use_spaceatt: bool = True,
        depth: int = 6,
        max_scale: int = 518,
        predict_conf: bool = True,
        **kwargs,
    ):
        _ = corr_levels, corr_radius, latent_dim, hidden_size, use_spaceatt, depth, max_scale, kwargs
        self.stride = stride
        self.predict_conf = predict_conf

    def __call__(
        self,
        query_points: jnp.ndarray,
        fmaps: jnp.ndarray | None = None,
        iters: int = 6,
        return_feat: bool = False,
        down_ratio: int = 1,
        apply_sigmoid: bool = True,
    ):
        _ = apply_sigmoid

        query_points = jnp.asarray(query_points)
        if query_points.ndim == 2:
            query_points = query_points[None, ...]

        batch_size, num_tracks, _ = query_points.shape
        seq_len = 1 if fmaps is None else int(fmaps.shape[1])

        coords = jnp.broadcast_to(query_points[:, None, :, :], (batch_size, seq_len, num_tracks, 2))
        coords = coords * float(max(self.stride * down_ratio, 1))
        coord_preds = [coords for _ in range(max(int(iters), 1))]

        vis = jnp.ones((batch_size, seq_len, num_tracks), dtype=query_points.dtype)
        conf = jnp.ones_like(vis) if self.predict_conf else None

        if return_feat:
            latent_dim = 128 if fmaps is None else int(fmaps.shape[2])
            track_feats = jnp.zeros((batch_size, seq_len, num_tracks, latent_dim), dtype=query_points.dtype)
            query_track_feat = jnp.zeros((batch_size, num_tracks, latent_dim), dtype=query_points.dtype)
            return coord_preds, vis, track_feats, query_track_feat, conf
        return coord_preds, vis, conf


__all__ = ["BaseTrackerPredictor"]
