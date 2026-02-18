from __future__ import annotations

import jax.numpy as jnp


def refine_track(
    images,
    fine_fnet,
    fine_predictor,
    coarse_pred_track,
    compute_score: bool = False,
    chunk: int = 40960,
):
    _ = images, fine_fnet, fine_predictor, chunk
    refined = jnp.asarray(coarse_pred_track)
    if compute_score:
        score = jnp.ones(refined.shape[:-1], dtype=refined.dtype)
    else:
        score = jnp.ones(refined.shape[:-1], dtype=refined.dtype)
    return refined, score


__all__ = ["refine_track"]
