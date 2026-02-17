import jax.numpy as jnp

from vggt_jax.heads.dpt_head import _residual_conv_unit
from vggt_jax.holders import ResidualConvUnitHolder


def test_residual_conv_unit_uses_relu_residual_path():
    holder = ResidualConvUnitHolder(features=1)

    x = jnp.array([[[[-2.0, -1.0], [0.5, 3.0]]]], dtype=jnp.float32)
    y = _residual_conv_unit(x, holder)

    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(y, expected)
