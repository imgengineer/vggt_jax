from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optax
from flax import nnx


@dataclass
class OptimBundle:
    optimizer: optax.GradientTransformation
    state: Any


def construct_optimizers(model, optim_conf):
    optimizer_conf = getattr(optim_conf, "optimizer", None)
    if optimizer_conf is None:
        lr = float(getattr(optim_conf, "lr", 1e-4))
        weight_decay = float(getattr(optim_conf, "weight_decay", 0.0))
        tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    else:
        lr = float(getattr(optimizer_conf, "lr", getattr(optim_conf, "lr", 1e-4)))
        weight_decay = float(getattr(optimizer_conf, "weight_decay", getattr(optim_conf, "weight_decay", 0.0)))
        tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)

    params = nnx.state(model, nnx.Param)
    state = tx.init(params)
    return OptimBundle(optimizer=tx, state=state)


__all__ = ["OptimBundle", "construct_optimizers"]
