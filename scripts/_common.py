from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from flax import nnx

from vggt.models import ModelConfig, VGGT
from vggt.models.vggt import create_vggt_from_nnx_npz, create_vggt_from_orbax_checkpoint

DEFAULT_REPO_ID = "facebook/VGGT-1B"
DEFAULT_FILENAME = "model.safetensors"
DEFAULT_CACHE_DIR = "./weights"
DEFAULT_ORBAX_PATH = "weights/orbax/facebook__VGGT-1B_model_full"
DEFAULT_NPZ_PATH = "weights/vggt_weights_full.npz"


@dataclass(frozen=True)
class ModelLoadSpec:
    tiny: bool
    weights_orbax: str | None
    weights_npz: str | None
    from_pretrained: bool
    prefer_hf_weights: bool


def add_model_loading_args(parser: argparse.ArgumentParser, *, include_tiny: bool) -> None:
    if include_tiny:
        parser.add_argument("--tiny", action="store_true", help="Use VGGT-Tiny config (224 resolution).")
    parser.add_argument("--weights-orbax", type=str, default=None, help="Path to an Orbax checkpoint directory.")
    parser.add_argument("--weights-npz", type=str, default=None, help="Path to converted NNX `.npz` weights.")
    parser.add_argument(
        "--from-pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load HF public weights via `VGGT.from_pretrained`.",
    )
    parser.add_argument(
        "--prefer-orbax",
        action="store_true",
        help="Prefer local Orbax/NPZ if available (otherwise HF-first).",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Hugging Face weight filename.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Hugging Face local cache dir.")


def resolve_model_config(*, tiny: bool) -> ModelConfig:
    return ModelConfig.vggt_tiny() if tiny else ModelConfig.vggt_base()


def resolve_target_size(target_size: int | None, *, tiny: bool) -> int:
    if target_size is not None:
        return int(target_size)
    return 224 if tiny else 518


def resolve_model_load_spec(
    *,
    tiny: bool,
    weights_orbax: str | None,
    weights_npz: str | None,
    from_pretrained: bool | None,
    prefer_orbax: bool,
) -> ModelLoadSpec:
    resolved_orbax = weights_orbax
    resolved_npz = weights_npz

    if prefer_orbax and not tiny:
        if resolved_orbax is None and Path(DEFAULT_ORBAX_PATH).exists():
            resolved_orbax = DEFAULT_ORBAX_PATH
        if resolved_npz is None and resolved_orbax is None and Path(DEFAULT_NPZ_PATH).exists():
            resolved_npz = DEFAULT_NPZ_PATH

    auto_from_pretrained = (
        (not tiny)
        and (not prefer_orbax)
        and (resolved_orbax is None)
        and (resolved_npz is None)
    )
    resolved_from_pretrained = auto_from_pretrained if from_pretrained is None else bool(from_pretrained)

    return ModelLoadSpec(
        tiny=tiny,
        weights_orbax=resolved_orbax,
        weights_npz=resolved_npz,
        from_pretrained=resolved_from_pretrained,
        prefer_hf_weights=not prefer_orbax,
    )


def load_vggt_model(
    *,
    config: ModelConfig,
    load_spec: ModelLoadSpec,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    cache_dir: str = DEFAULT_CACHE_DIR,
    rng_seed: int = 0,
    strict: bool = True,
) -> VGGT:
    rngs = nnx.Rngs(rng_seed)

    if load_spec.from_pretrained:
        model = VGGT.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            config=config,
            rngs=rngs,
            prefer_hf_weights=load_spec.prefer_hf_weights,
        )
    elif load_spec.weights_orbax is not None:
        model = create_vggt_from_orbax_checkpoint(load_spec.weights_orbax, config, rngs=rngs, strict=strict)
    elif load_spec.weights_npz is not None:
        model = create_vggt_from_nnx_npz(load_spec.weights_npz, config, rngs=rngs, strict=strict)
    else:
        model = VGGT(config, rngs=rngs)

    model.eval()
    return model
