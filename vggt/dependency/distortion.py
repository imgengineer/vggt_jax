from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def _is_numpy(x: Any) -> bool:
    return isinstance(x, np.ndarray)


def _as_jax(x: Any) -> jnp.ndarray:
    return jnp.asarray(x)


def _maybe_numpy(x: jnp.ndarray, *, like: Any) -> jnp.ndarray | np.ndarray:
    if _is_numpy(like):
        return np.asarray(x)
    return x


def apply_distortion(extra_params, u, v):
    extra_params_j = _as_jax(extra_params)
    u_j = _as_jax(u)
    v_j = _as_jax(v)

    num_params = extra_params_j.shape[1]

    if num_params == 1:
        k = extra_params_j[:, 0]
        u2 = u_j * u_j
        v2 = v_j * v_j
        r2 = u2 + v2
        radial = k[:, None] * r2
        du = u_j * radial
        dv = v_j * radial
    elif num_params == 2:
        k1, k2 = extra_params_j[:, 0], extra_params_j[:, 1]
        u2 = u_j * u_j
        v2 = v_j * v_j
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u_j * radial
        dv = v_j * radial
    elif num_params == 4:
        k1, k2, p1, p2 = (
            extra_params_j[:, 0],
            extra_params_j[:, 1],
            extra_params_j[:, 2],
            extra_params_j[:, 3],
        )
        u2 = u_j * u_j
        v2 = v_j * v_j
        uv = u_j * v_j
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u_j * radial + 2 * p1[:, None] * uv + p2[:, None] * (r2 + 2 * u2)
        dv = v_j * radial + 2 * p2[:, None] * uv + p1[:, None] * (r2 + 2 * v2)
    else:
        raise ValueError(f"Unsupported number of distortion parameters: {num_params}")

    out_u = u_j + du
    out_v = v_j + dv
    return _maybe_numpy(out_u, like=u), _maybe_numpy(out_v, like=v)


def single_undistortion(params, tracks_normalized):
    tracks_j = _as_jax(tracks_normalized)
    u = tracks_j[..., 0]
    v = tracks_j[..., 1]
    u_undist, v_undist = apply_distortion(params, u, v)
    stacked = jnp.stack([jnp.asarray(u_undist), jnp.asarray(v_undist)], axis=-1)
    return _maybe_numpy(stacked, like=tracks_normalized)


def iterative_undistortion(
    params,
    tracks_normalized,
    max_iterations: int = 100,
    max_step_norm: float = 1e-10,
    rel_step_size: float = 1e-6,
):
    tracks_j = _as_jax(tracks_normalized)
    params_j = _as_jax(params)

    u = tracks_j[..., 0]
    v = tracks_j[..., 1]
    original_u = u
    original_v = v

    eps = jnp.finfo(u.dtype).eps
    for _ in range(max_iterations):
        u_undist, v_undist = apply_distortion(params_j, u, v)
        u_undist = jnp.asarray(u_undist)
        v_undist = jnp.asarray(v_undist)

        dx = original_u - u_undist
        dy = original_v - v_undist

        step_u = jnp.maximum(jnp.abs(u) * rel_step_size, eps)
        step_v = jnp.maximum(jnp.abs(v) * rel_step_size, eps)

        j00 = (
            jnp.asarray(apply_distortion(params_j, u + step_u, v)[0])
            - jnp.asarray(apply_distortion(params_j, u - step_u, v)[0])
        ) / (2 * step_u)
        j01 = (
            jnp.asarray(apply_distortion(params_j, u, v + step_v)[0])
            - jnp.asarray(apply_distortion(params_j, u, v - step_v)[0])
        ) / (2 * step_v)
        j10 = (
            jnp.asarray(apply_distortion(params_j, u + step_u, v)[1])
            - jnp.asarray(apply_distortion(params_j, u - step_u, v)[1])
        ) / (2 * step_u)
        j11 = (
            jnp.asarray(apply_distortion(params_j, u, v + step_v)[1])
            - jnp.asarray(apply_distortion(params_j, u, v - step_v)[1])
        ) / (2 * step_v)

        jac = jnp.stack(
            [
                jnp.stack([j00 + 1.0, j01], axis=-1),
                jnp.stack([j10, j11 + 1.0], axis=-1),
            ],
            axis=-2,
        )
        rhs = jnp.stack([dx, dy], axis=-1)
        delta = jnp.linalg.solve(jac, rhs)

        u = u + delta[..., 0]
        v = v + delta[..., 1]

        if jnp.max(jnp.sum(delta * delta, axis=-1)) < max_step_norm:
            break

    out = jnp.stack([u, v], axis=-1)
    return _maybe_numpy(out, like=tracks_normalized)


__all__ = ["apply_distortion", "single_undistortion", "iterative_undistortion"]
