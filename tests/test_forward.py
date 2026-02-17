import jax
import jax.numpy as jnp
from flax import nnx

from vggt_jax.models import ModelConfig, VGGT


def test_vggt_tiny_forward_shapes():
    cfg = ModelConfig.vggt_tiny()
    model = VGGT(cfg, rngs=nnx.Rngs(0))

    images = jax.random.uniform(jax.random.key(42), (1, 3, 3, 56, 56), dtype=jnp.float32)
    query_points = jnp.array([[[8.0, 8.0], [10.0, 12.0]]], dtype=jnp.float32)

    preds = model(images, query_points=query_points, deterministic=True)

    assert preds["pose_enc"].shape == (1, 3, 9)
    assert preds["depth"].shape == (1, 3, 56, 56, 1)
    assert preds["depth_conf"].shape == (1, 3, 56, 56)
    assert preds["world_points"].shape == (1, 3, 56, 56, 3)
    assert preds["world_points_conf"].shape == (1, 3, 56, 56)
    assert preds["track"].shape == (1, 3, 2, 2)
    assert preds["vis"].shape == (1, 3, 2)
    assert preds["conf"].shape == (1, 3, 2)
