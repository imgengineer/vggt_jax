import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from vggt.heads.camera_head import CameraHead, CameraHeadConfig
from vggt.heads.dpt_head import DPTHead, DPTHeadConfig
from vggt.heads.track_head import TrackHead, TrackHeadConfig
from vggt.models.aggregator import Aggregator, AggregatorConfig


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    aa_order: tuple[str, str] = ("frame", "global")
    aa_block_size: int = 1
    rope_freq: float = 100.0
    init_values: float = 0.01
    patch_embed: str = "dinov2_vitl14_reg"
    enable_camera: bool = True
    enable_point: bool = True
    enable_depth: bool = True
    enable_track: bool = True
    enable_conversion_holders: bool = True

    @classmethod
    def vggt_base(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def vggt_tiny(cls) -> "ModelConfig":
        return cls(
            img_size=224,
            patch_size=14,
            embed_dim=256,
            depth=8,
            num_heads=8,
            num_register_tokens=2,
            patch_embed="conv",
            enable_conversion_holders=False,
        )


class VGGT(nnx.Module):
    def __init__(
        self,
        cfg: ModelConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_depth: bool = True,
        enable_track: bool = True,
        enable_conversion_holders: bool = True,
    ):
        if cfg is not None:
            non_default_args = [
                ("img_size", img_size, 518),
                ("patch_size", patch_size, 14),
                ("embed_dim", embed_dim, 1024),
                ("enable_camera", enable_camera, True),
                ("enable_point", enable_point, True),
                ("enable_depth", enable_depth, True),
                ("enable_track", enable_track, True),
                ("enable_conversion_holders", enable_conversion_holders, True),
            ]
            unexpected = [name for name, value, default in non_default_args if value != default]
            if unexpected:
                raise ValueError(f"Pass either cfg or scalar args, got both: {unexpected}")
        else:
            cfg = ModelConfig(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                enable_camera=enable_camera,
                enable_point=enable_point,
                enable_depth=enable_depth,
                enable_track=enable_track,
                enable_conversion_holders=enable_conversion_holders,
            )

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.cfg = cfg
        # Match PyTorch defaults: freshly constructed modules are in training mode.
        # NNX's `train()` / `eval()` toggles `deterministic`; we reuse that here.
        self.deterministic = False
        self.aggregator = Aggregator(
            AggregatorConfig(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                num_register_tokens=cfg.num_register_tokens,
                aa_order=cfg.aa_order,
                aa_block_size=cfg.aa_block_size,
                rope_freq=cfg.rope_freq,
                init_values=cfg.init_values,
                patch_embed=cfg.patch_embed,
                enable_patch_embed_holder=cfg.enable_conversion_holders,
            ),
            rngs=rngs,
        )

        head_dim_in = 2 * cfg.embed_dim
        self.camera_head = (
            CameraHead(CameraHeadConfig(dim_in=head_dim_in, num_heads=cfg.num_heads), rngs=rngs)
            if cfg.enable_camera
            else None
        )
        self.point_head = (
            DPTHead(
                DPTHeadConfig(
                    dim_in=head_dim_in,
                    patch_size=cfg.patch_size,
                    output_dim=4,
                    activation="inv_log",
                    conf_activation="expp1",
                    features=256,
                    enable_torch_holder=cfg.enable_conversion_holders,
                ),
                rngs=rngs,
            )
            if cfg.enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                DPTHeadConfig(
                    dim_in=head_dim_in,
                    patch_size=cfg.patch_size,
                    output_dim=2,
                    activation="exp",
                    conf_activation="expp1",
                    features=256,
                    enable_torch_holder=cfg.enable_conversion_holders,
                ),
                rngs=rngs,
            )
            if cfg.enable_depth
            else None
        )
        self.track_head = (
            TrackHead(
                TrackHeadConfig(
                    dim_in=head_dim_in,
                    patch_size=cfg.patch_size,
                    enable_tracker_holder=cfg.enable_conversion_holders,
                    enable_torch_holder=cfg.enable_conversion_holders,
                ),
                rngs=rngs,
            )
            if cfg.enable_track
            else None
        )

    @classmethod
    def from_torch_checkpoint(
        cls,
        checkpoint_path: str,
        config: ModelConfig | None = None,
        *,
        strict: bool = False,
        include_track: bool = True,
        rngs: nnx.Rngs | None = None,
        return_report: bool = False,
    ):
        model, report = create_vggt_from_torch_checkpoint(
            checkpoint_path,
            config=config,
            strict=strict,
            include_track=include_track,
            rngs=rngs,
        )
        return (model, report) if return_report else model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "facebook/VGGT-1B",
        *,
        filename: str = "model.safetensors",
        cache_dir: str = "./weights",
        config: ModelConfig | None = None,
        strict: bool = False,
        include_track: bool = True,
        prefer_hf_weights: bool = True,
        save_orbax_cache: bool = False,
        rngs: nnx.Rngs | None = None,
        return_report: bool = False,
    ):
        orbax_dir = default_orbax_checkpoint_dir(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            include_track=include_track,
        )
        report_path = orbax_dir / "conversion_report.json"
        checkpoint_path: Path | None = None
        hf_error: Exception | None = None

        def _load_from_hf_checkpoint() -> tuple["VGGT", ConversionReport]:
            nonlocal checkpoint_path
            local_dir = Path(download_pretrained_weights(repo_id=repo_id, cache_dir=cache_dir))
            checkpoint_path = local_dir / filename
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")
            return create_vggt_from_torch_checkpoint(
                str(checkpoint_path),
                config=config,
                strict=strict,
                include_track=include_track,
                rngs=rngs,
            )

        if prefer_hf_weights:
            try:
                model, report = _load_from_hf_checkpoint()
                if save_orbax_cache:
                    save_nnx_weights_orbax(model, str(orbax_dir), force=True)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
                return (model, report) if return_report else model
            except Exception as exc:  # pragma: no cover
                hf_error = exc

        if (orbax_dir / "_CHECKPOINT_METADATA").exists():
            model = create_vggt_from_orbax_checkpoint(str(orbax_dir), config=config, strict=strict, rngs=rngs)
            if return_report and report_path.exists():
                report = ConversionReport(**json.loads(report_path.read_text(encoding="utf-8")))
                return model, report
            return model

        try:
            model, report = _load_from_hf_checkpoint()
        except Exception as exc:  # pragma: no cover
            if hf_error is not None:
                raise RuntimeError(
                    "Failed to load HF checkpoint and no Orbax fallback is available. "
                    f"checkpoint={checkpoint_path if checkpoint_path is not None else filename}, orbax={orbax_dir}"
                ) from hf_error
            raise RuntimeError(
                "Failed to load pretrained weights. "
                f"checkpoint={checkpoint_path if checkpoint_path is not None else filename}, orbax={orbax_dir}"
            ) from exc

        if save_orbax_cache:
            save_nnx_weights_orbax(model, str(orbax_dir), force=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
        return (model, report) if return_report else model

    def eval(self, **attributes):
        super().eval(**attributes)
        return self

    def train(self, mode: bool = True, **attributes):
        if mode:
            super().train(**attributes)
        else:
            super().eval(**attributes)
        return self

    def forward(self, images: jnp.ndarray, query_points: jnp.ndarray | None = None):
        return self(images, query_points)

    def __call__(
        self,
        images: jnp.ndarray,
        query_points: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> dict[str, jnp.ndarray | list[jnp.ndarray]]:
        deterministic_flag = bool(self.deterministic)

        if images.ndim == 4:
            if images.shape[-1] == 3:
                images = images[None, ...]
            elif images.shape[1] == 3:
                images = jnp.transpose(images, (0, 2, 3, 1))[None, ...]
            else:
                raise ValueError(f"Expected image shape [T,H,W,C] or [T,C,H,W], got {images.shape}")
        elif images.ndim == 5:
            if images.shape[-1] == 3:
                pass
            elif images.shape[2] == 3:
                images = jnp.transpose(images, (0, 1, 3, 4, 2))
            else:
                raise ValueError(f"Expected image shape [B,T,H,W,C] or [B,T,C,H,W], got {images.shape}")
        else:
            raise ValueError(f"Expected 4D/5D image tensor, got {images.shape}")

        if query_points is not None and query_points.ndim == 2:
            query_points = query_points[None, ...]

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, rngs=rngs)

        predictions: dict[str, jnp.ndarray | list[jnp.ndarray]] = {}
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list, rngs=rngs)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(aggregated_tokens_list, images, patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            world_points, world_points_conf = self.point_head(aggregated_tokens_list, images, patch_start_idx)
            predictions["world_points"] = world_points
            predictions["world_points_conf"] = world_points_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        # Match PyTorch: store images only during inference/eval mode.
        if deterministic_flag:
            predictions["images"] = images

        return predictions


@jax.jit
def forward(graphdef: nnx.GraphDef[nnx.Module], state: nnx.State, images: jax.Array, rngs: nnx.Rngs) -> dict:
    model = nnx.merge(graphdef, state)
    return model(images, rngs=rngs)


Transform = str
NONE: Transform = "none"
LINEAR_WEIGHT: Transform = "linear_weight"
CONV_OIHW_TO_HWIO: Transform = "conv_oihw_to_hwio"


@dataclasses.dataclass(frozen=True)
class ConversionReport:
    total_source_keys: int
    mapped_keys: int
    loaded_keys: int
    skipped_unmapped: int
    skipped_missing_target: int
    skipped_shape_mismatch: int
    missing_target_leaves: int


def download_pretrained_weights(
    repo_id: str = "facebook/VGGT-1B",
    cache_dir: str = "./weights",
    allow_patterns: tuple[str, ...] = ("*.safetensors", "*.npz", "*.json", "*.pt", "*.bin"),
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency `huggingface-hub`. Install it to download pretrained weights."
        ) from exc

    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns),
    )
    return local_dir


def create_vggt_model(config: ModelConfig | None = None, *, rngs: nnx.Rngs | None = None) -> VGGT:
    if rngs is None:
        rngs = nnx.Rngs(0)
    if config is None:
        return VGGT(rngs=rngs)
    return VGGT(config, rngs=rngs)


def create_vggt_from_pretrained(repo_id: str = "facebook/VGGT-1B", cache_dir: str = "./weights") -> tuple[VGGT, Path]:
    config = ModelConfig.vggt_base()
    model = VGGT.from_pretrained(
        repo_id=repo_id,
        cache_dir=cache_dir,
        config=config,
        prefer_hf_weights=True,
        save_orbax_cache=False,
    )
    local_dir = Path(download_pretrained_weights(repo_id=repo_id, cache_dir=cache_dir))
    return model, local_dir


def _flatten_tree(tree: dict[str, Any], prefix: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    out: dict[tuple[Any, ...], Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(_flatten_tree(value, prefix + (key,)))
        return out
    out[prefix] = tree
    return out


def _unflatten_tree(flat: dict[tuple[Any, ...], Any]) -> dict[str, Any]:
    root: dict[Any, Any] = {}
    for path, value in flat.items():
        cursor = root
        for part in path[:-1]:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]
        cursor[path[-1]] = value
    return root


def _parse_path(path: str) -> tuple[Any, ...]:
    parts = []
    for token in path.split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token)
    return tuple(parts)


def _format_path(path: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in path)


def _map_block_param(rest: str) -> tuple[str, Transform] | None:
    mapping: dict[str, tuple[str, Transform]] = {
        "norm1.weight": ("norm1.scale", NONE),
        "norm1.bias": ("norm1.bias", NONE),
        "norm2.weight": ("norm2.scale", NONE),
        "norm2.bias": ("norm2.bias", NONE),
        "attn.q_norm.weight": ("attn.q_norm.scale", NONE),
        "attn.q_norm.bias": ("attn.q_norm.bias", NONE),
        "attn.k_norm.weight": ("attn.k_norm.scale", NONE),
        "attn.k_norm.bias": ("attn.k_norm.bias", NONE),
        "attn.qkv.weight": ("attn.qkv.kernel", LINEAR_WEIGHT),
        "attn.qkv.bias": ("attn.qkv.bias", NONE),
        "attn.proj.weight": ("attn.proj.kernel", LINEAR_WEIGHT),
        "attn.proj.bias": ("attn.proj.bias", NONE),
        "ls1.gamma": ("ls1.gamma", NONE),
        "ls2.gamma": ("ls2.gamma", NONE),
        "mlp.fc1.weight": ("mlp.fc1.kernel", LINEAR_WEIGHT),
        "mlp.fc1.bias": ("mlp.fc1.bias", NONE),
        "mlp.fc2.weight": ("mlp.fc2.kernel", LINEAR_WEIGHT),
        "mlp.fc2.bias": ("mlp.fc2.bias", NONE),
    }
    return mapping.get(rest)


def map_torch_key_to_nnx(source_key: str) -> tuple[str, Transform] | tuple[None, None]:
    source_key = source_key.removeprefix("module.")
    source_key = source_key.removeprefix("model.")

    direct_map: dict[str, tuple[str, Transform]] = {
        "aggregator.camera_token": ("aggregator.camera_token", NONE),
        "aggregator.register_token": ("aggregator.register_token", NONE),
        "camera_head.token_norm.weight": ("camera_head.token_norm.scale", NONE),
        "camera_head.token_norm.bias": ("camera_head.token_norm.bias", NONE),
        "camera_head.trunk_norm.weight": ("camera_head.trunk_norm.scale", NONE),
        "camera_head.trunk_norm.bias": ("camera_head.trunk_norm.bias", NONE),
        "camera_head.empty_pose_tokens": ("camera_head.empty_pose_tokens", NONE),
        "camera_head.embed_pose.weight": ("camera_head.embed_pose.kernel", LINEAR_WEIGHT),
        "camera_head.embed_pose.bias": ("camera_head.embed_pose.bias", NONE),
        "camera_head.poseLN_modulation.1.weight": ("camera_head.pose_ln_mod.kernel", LINEAR_WEIGHT),
        "camera_head.poseLN_modulation.1.bias": ("camera_head.pose_ln_mod.bias", NONE),
        "camera_head.pose_branch.fc1.weight": ("camera_head.pose_branch_1.kernel", LINEAR_WEIGHT),
        "camera_head.pose_branch.fc1.bias": ("camera_head.pose_branch_1.bias", NONE),
        "camera_head.pose_branch.fc2.weight": ("camera_head.pose_branch_2.kernel", LINEAR_WEIGHT),
        "camera_head.pose_branch.fc2.bias": ("camera_head.pose_branch_2.bias", NONE),
    }
    if source_key in direct_map:
        return direct_map[source_key]

    agg_match = re.match(r"^aggregator\.(frame_blocks|global_blocks)\.(\d+)\.(.+)$", source_key)
    if agg_match:
        block_group, block_idx, rest = agg_match.groups()
        mapped = _map_block_param(rest)
        if mapped is None:
            return None, None
        target_rest, transform = mapped
        return f"aggregator.{block_group}.{block_idx}.{target_rest}", transform

    if source_key.startswith("aggregator.patch_embed."):
        suffix = source_key.removeprefix("aggregator.patch_embed.")
        return f"aggregator.patch_embed_holder.{suffix}", NONE

    camera_match = re.match(r"^camera_head\.trunk\.(\d+)\.(.+)$", source_key)
    if camera_match:
        block_idx, rest = camera_match.groups()
        mapped = _map_block_param(rest)
        if mapped is None:
            return None, None
        target_rest, transform = mapped
        return f"camera_head.trunk.{block_idx}.{target_rest}", transform

    dpt_match = re.match(r"^(depth_head|point_head|track_head\.feature_extractor)\.(.+)$", source_key)
    if dpt_match:
        prefix, rest = dpt_match.groups()
        return f"{prefix}.torch_holder.{rest}", NONE

    if source_key.startswith("track_head.tracker."):
        return source_key, NONE

    return None, None


def _apply_transform(array: np.ndarray, transform: Transform) -> np.ndarray:
    if transform == NONE:
        return array
    if transform == LINEAR_WEIGHT:
        return array.T
    if transform == CONV_OIHW_TO_HWIO:
        return np.transpose(array, (2, 3, 1, 0))
    raise ValueError(f"Unsupported transform: {transform}")


def _load_torch_state_dict(checkpoint_path: str) -> dict[str, Any]:
    suffix = Path(checkpoint_path).suffix.lower()

    if suffix == ".npz":
        data = np.load(checkpoint_path)
        return {str(key).removeprefix("module."): np.asarray(value) for key, value in data.items()}

    if suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Missing dependency `safetensors` for loading .safetensors checkpoints.") from exc

        state_dict: dict[str, np.ndarray] = {}
        with safe_open(checkpoint_path, framework="numpy") as f:
            for key in f.keys():
                state_dict[str(key).removeprefix("module.")] = np.asarray(f.get_tensor(key))
        return state_dict

    raise ValueError(
        f"Unsupported checkpoint format: {checkpoint_path}. "
        "Pure JAX path supports `.npz` and `.safetensors` directly."
    )


def _to_numpy_tensor(torch_tensor: Any) -> np.ndarray:
    return np.asarray(torch_tensor)


def create_vggt_from_torch_checkpoint(
    checkpoint_path: str,
    config: ModelConfig | None = None,
    *,
    strict: bool = False,
    include_track: bool = True,
    rngs: nnx.Rngs | None = None,
) -> tuple[VGGT, ConversionReport]:
    if config is None:
        config = ModelConfig.vggt_base()

    model = create_vggt_model(config, rngs=rngs)
    graph_def, abs_state = nnx.split(model)
    target_state = nnx.to_pure_dict(abs_state)
    flat_target = _flatten_tree(target_state)
    source_state = _load_torch_state_dict(checkpoint_path)

    mapped_keys = 0
    loaded_keys = 0
    skipped_unmapped = 0
    skipped_missing_target = 0
    skipped_shape_mismatch = 0

    for source_key, source_tensor in source_state.items():
        if not include_track and source_key.startswith("track_head.tracker."):
            skipped_unmapped += 1
            continue

        target_key, transform = map_torch_key_to_nnx(source_key)
        if target_key is None:
            skipped_unmapped += 1
            continue

        mapped_keys += 1
        target_path = _parse_path(target_key)
        if target_path not in flat_target:
            skipped_missing_target += 1
            continue

        array = _to_numpy_tensor(source_tensor)
        array = _apply_transform(array, transform)
        target_leaf = flat_target[target_path]
        if array.shape != target_leaf.shape:
            skipped_shape_mismatch += 1
            continue

        flat_target[target_path] = jnp.asarray(array, dtype=target_leaf.dtype)
        loaded_keys += 1

    updated_state = _unflatten_tree(flat_target)
    model = nnx.merge(graph_def, updated_state)

    missing_target_leaves = len(flat_target) - loaded_keys
    report = ConversionReport(
        total_source_keys=len(source_state),
        mapped_keys=mapped_keys,
        loaded_keys=loaded_keys,
        skipped_unmapped=skipped_unmapped,
        skipped_missing_target=skipped_missing_target,
        skipped_shape_mismatch=skipped_shape_mismatch,
        missing_target_leaves=missing_target_leaves,
    )

    if strict and (
        report.skipped_missing_target > 0 or report.skipped_shape_mismatch > 0 or report.skipped_unmapped > 0
    ):
        raise RuntimeError(
            "Strict conversion failed: "
            f"unmapped={report.skipped_unmapped}, "
            f"missing_target={report.skipped_missing_target}, "
            f"shape_mismatch={report.skipped_shape_mismatch}"
        )

    return model, report


def create_vggt_from_checkpoint(
    checkpoint_path: str,
    config: ModelConfig | None = None,
    *,
    strict: bool = False,
    include_track: bool = True,
    rngs: nnx.Rngs | None = None,
) -> tuple[VGGT, ConversionReport]:
    return create_vggt_from_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        strict=strict,
        include_track=include_track,
        rngs=rngs,
    )


def default_orbax_checkpoint_dir(
    *,
    repo_id: str,
    filename: str,
    cache_dir: str,
    include_track: bool,
) -> Path:
    safe_repo_id = repo_id.replace("/", "__")
    stem = Path(filename).stem
    suffix = "full" if include_track else "no_track"
    return Path(cache_dir) / "orbax" / f"{safe_repo_id}_{stem}_{suffix}"


def save_nnx_weights_orbax(model: VGGT, output_dir: str, *, force: bool = False) -> None:
    try:
        import orbax.checkpoint as oc
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing dependency: orbax-checkpoint") from exc

    _, state = nnx.split(model)
    item = nnx.to_pure_dict(state)
    item = jax.tree_util.tree_map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, item)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = oc.PyTreeCheckpointer()
    checkpointer.save(output_path, item, force=force)


def load_nnx_weights_orbax(model: VGGT, checkpoint_dir: str, *, strict: bool = True) -> VGGT:
    try:
        import orbax.checkpoint as oc
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing dependency: orbax-checkpoint") from exc

    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()

    graph_def, state = nnx.split(model)
    flat_target = _flatten_tree(nnx.to_pure_dict(state))

    checkpointer = oc.PyTreeCheckpointer()
    restored_tree = checkpointer.restore(checkpoint_path)
    raw_flat_restored = _flatten_tree(restored_tree)
    flat_restored: dict[tuple[Any, ...], Any] = {}
    for path, value in raw_flat_restored.items():
        normalized = tuple(int(part) if isinstance(part, str) and part.isdigit() else part for part in path)
        flat_restored[normalized] = value

    loaded = 0
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for path, target_leaf in flat_target.items():
        if path not in flat_restored:
            missing_keys.append(_format_path(path))
            continue

        array = np.asarray(flat_restored[path])
        if tuple(array.shape) != tuple(target_leaf.shape):
            shape_mismatches.append((_format_path(path), tuple(array.shape), tuple(target_leaf.shape)))
            continue

        flat_target[path] = jnp.asarray(array, dtype=target_leaf.dtype)
        loaded += 1

    for path in flat_restored:
        if path not in flat_target:
            unexpected_keys.append(_format_path(path))

    if strict and (unexpected_keys or missing_keys or shape_mismatches):
        lines = ["Strict Orbax load failed:"]
        if unexpected_keys:
            lines.append(f"- unexpected_keys={len(unexpected_keys)}")
        if missing_keys:
            lines.append(f"- missing_keys={len(missing_keys)}")
        if shape_mismatches:
            key, src_shape, dst_shape = shape_mismatches[0]
            lines.append(f"- shape_mismatches={len(shape_mismatches)} (e.g. {key}: {src_shape} != {dst_shape})")
        raise RuntimeError("\n".join(lines))

    if not strict and loaded == 0:
        raise RuntimeError(f"No Orbax leaves loaded from {checkpoint_path}")

    updated_state = _unflatten_tree(flat_target)
    return nnx.merge(graph_def, updated_state)


def create_vggt_from_orbax_checkpoint(
    checkpoint_dir: str,
    config: ModelConfig | None = None,
    *,
    strict: bool = True,
    rngs: nnx.Rngs | None = None,
) -> VGGT:
    model = create_vggt_model(config, rngs=rngs)
    return load_nnx_weights_orbax(model, checkpoint_dir, strict=strict)


def save_nnx_weights_npz(model: VGGT, output_path: str) -> None:
    _, state = nnx.split(model)
    flat = _flatten_tree(nnx.to_pure_dict(state))
    arrays = {_format_path(path): np.asarray(value) for path, value in flat.items()}
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)


def load_nnx_weights_npz(model: VGGT, npz_path: str, *, strict: bool = True) -> VGGT:
    graph_def, state = nnx.split(model)
    flat_target = _flatten_tree(nnx.to_pure_dict(state))

    data = np.load(npz_path)
    loaded = 0
    unexpected_keys: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key in data.files:
        path = _parse_path(key)
        if path not in flat_target:
            unexpected_keys.append(key)
            continue

        array = np.asarray(data[key])
        target_leaf = flat_target[path]
        if tuple(array.shape) != tuple(target_leaf.shape):
            shape_mismatches.append((key, tuple(array.shape), tuple(target_leaf.shape)))
            continue

        flat_target[path] = jnp.asarray(array, dtype=target_leaf.dtype)
        loaded += 1

    missing_keys = [key for key in (_format_path(path) for path in flat_target) if key not in data]

    if strict and (unexpected_keys or shape_mismatches or missing_keys):
        lines = ["Strict NPZ load failed:"]
        if unexpected_keys:
            lines.append(f"- unexpected_keys={len(unexpected_keys)}")
        if missing_keys:
            lines.append(f"- missing_keys={len(missing_keys)}")
        if shape_mismatches:
            example_key, src_shape, dst_shape = shape_mismatches[0]
            lines.append(f"- shape_mismatches={len(shape_mismatches)} (e.g. {example_key}: {src_shape} != {dst_shape})")
        raise RuntimeError("\n".join(lines))

    updated_state = _unflatten_tree(flat_target)
    _ = loaded
    return nnx.merge(graph_def, updated_state)


def create_vggt_from_nnx_npz(
    npz_path: str,
    config: ModelConfig | None = None,
    *,
    strict: bool = True,
    rngs: nnx.Rngs | None = None,
) -> VGGT:
    model = create_vggt_model(config, rngs=rngs)
    return load_nnx_weights_npz(model, npz_path, strict=strict)


__all__ = [
    "ModelConfig",
    "VGGT",
    "forward",
    "Transform",
    "NONE",
    "LINEAR_WEIGHT",
    "CONV_OIHW_TO_HWIO",
    "ConversionReport",
    "download_pretrained_weights",
    "create_vggt_model",
    "create_vggt_from_pretrained",
    "map_torch_key_to_nnx",
    "create_vggt_from_checkpoint",
    "create_vggt_from_torch_checkpoint",
    "default_orbax_checkpoint_dir",
    "save_nnx_weights_orbax",
    "load_nnx_weights_orbax",
    "create_vggt_from_orbax_checkpoint",
    "save_nnx_weights_npz",
    "load_nnx_weights_npz",
    "create_vggt_from_nnx_npz",
]
