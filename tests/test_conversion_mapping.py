import numpy as np
from flax import nnx

from vggt_jax.models.vggt.modeling import (
    LINEAR_WEIGHT,
    NONE,
    ModelConfig,
    VGGT,
    create_vggt_from_nnx_npz,
    map_torch_key_to_nnx,
    save_nnx_weights_npz,
)


def test_map_torch_key_block_and_camera():
    assert map_torch_key_to_nnx("aggregator.frame_blocks.3.attn.qkv.weight") == (
        "aggregator.frame_blocks.3.attn.qkv.kernel",
        LINEAR_WEIGHT,
    )
    assert map_torch_key_to_nnx("aggregator.global_blocks.12.norm2.weight") == (
        "aggregator.global_blocks.12.norm2.scale",
        NONE,
    )
    assert map_torch_key_to_nnx("camera_head.poseLN_modulation.1.weight") == (
        "camera_head.pose_ln_mod.kernel",
        LINEAR_WEIGHT,
    )


def test_map_torch_key_patch_embed_and_tracker():
    assert map_torch_key_to_nnx("aggregator.patch_embed.patch_embed.proj.weight") == (
        "aggregator.patch_embed_holder.patch_embed.proj.weight",
        NONE,
    )
    assert map_torch_key_to_nnx("track_head.tracker.updateformer.time_blocks.0.attn.in_proj_weight") == (
        "track_head.tracker.updateformer.time_blocks.0.attn.in_proj_weight",
        NONE,
    )
    assert map_torch_key_to_nnx("some.unknown.key") == (None, None)


def test_save_nnx_weights_npz(tmp_path):
    model = VGGT(ModelConfig.vggt_tiny(), rngs=nnx.Rngs(0))
    output = tmp_path / "weights.npz"
    save_nnx_weights_npz(model, str(output))
    data = np.load(output)
    assert len(data.files) > 0


def test_reload_nnx_weights_npz_roundtrip(tmp_path):
    cfg = ModelConfig.vggt_tiny()
    model = VGGT(cfg, rngs=nnx.Rngs(0))
    output = tmp_path / "weights.npz"
    save_nnx_weights_npz(model, str(output))

    reloaded = create_vggt_from_nnx_npz(str(output), cfg, rngs=nnx.Rngs(1), strict=True)

    _, state_a = nnx.split(model)
    _, state_b = nnx.split(reloaded)
    dict_a = nnx.to_pure_dict(state_a)
    dict_b = nnx.to_pure_dict(state_b)

    assert np.allclose(np.asarray(dict_a["aggregator"]["camera_token"]), np.asarray(dict_b["aggregator"]["camera_token"]))
