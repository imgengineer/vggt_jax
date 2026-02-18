import jax
import jax.numpy as jnp
from flax import nnx

from vggt.heads.track_head import TrackHead, TrackHeadConfig


def test_track_head_tracker_forward_shapes_and_ref_frame_lock():
    cfg = TrackHeadConfig(
        dim_in=64,
        patch_size=8,
        features=16,
        iters=2,
        stride=2,
        corr_levels=2,
        corr_radius=2,
        hidden_size=32,
        tracker_depth=2,
        tracker_num_heads=4,
        enable_tracker_holder=True,
        enable_torch_holder=False,
    )
    model = TrackHead(cfg, rngs=nnx.Rngs(0))

    images = jax.random.uniform(jax.random.key(0), (1, 2, 32, 32, 3), dtype=jnp.float32)
    patch_count = (32 // cfg.patch_size) * (32 // cfg.patch_size)
    tokens = jax.random.uniform(jax.random.key(1), (1, 2, 1 + patch_count, cfg.dim_in), dtype=jnp.float32)

    query_points = jnp.array([[[8.0, 8.0], [12.0, 10.0], [20.0, 16.0]]], dtype=jnp.float32)

    track_list, vis, conf = model([tokens], images, patch_start_idx=1, query_points=query_points)

    assert len(track_list) == cfg.iters
    assert track_list[-1].shape == (1, 2, 3, 2)
    assert vis.shape == (1, 2, 3)
    assert conf.shape == (1, 2, 3)

    assert jnp.all(jnp.isfinite(track_list[-1]))
    assert jnp.all(jnp.isfinite(vis))
    assert jnp.all(jnp.isfinite(conf))

    assert jnp.allclose(track_list[-1][:, 0], query_points, atol=1e-4)
