import jax.numpy as jnp

from vggt_jax.heads.track_ops import _sample_nchw


def test_sample_nchw_1x1_matches_grid_sample_semantics():
    # For align_corners=True grid sampling on a 1x1 feature map,
    # coordinates should always sample the single pixel value.
    inp = jnp.array([[[[2.5]]]], dtype=jnp.float32)
    coords = jnp.array(
        [
            [
                [[-10.0, -10.0], [0.0, 0.0], [10.0, 10.0]],
                [[3.0, -7.0], [0.25, 0.75], [100.0, -100.0]],
            ]
        ],
        dtype=jnp.float32,
    )

    out = _sample_nchw(inp, coords, padding_mode="zeros")

    assert out.shape == (1, 1, 2, 3)
    assert jnp.allclose(out, 2.5, atol=1e-7)
