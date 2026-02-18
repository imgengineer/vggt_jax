import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from vggt.heads.head_act import activate_pose
from vggt.layers.block import Block, BlockConfig


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale) + shift


@dataclasses.dataclass(frozen=True)
class CameraHeadConfig:
    dim_in: int = 2048
    trunk_depth: int = 4
    num_heads: int = 16
    mlp_ratio: float = 4.0
    init_values: float = 0.01
    pose_encoding_type: str = "absT_quaR_FoV"
    trans_act: str = "linear"
    quat_act: str = "linear"
    fl_act: str = "relu"


class CameraHead(nnx.Module):
    def __init__(self, cfg: CameraHeadConfig, *, rngs: nnx.Rngs):
        if cfg.pose_encoding_type != "absT_quaR_FoV":
            raise ValueError(f"Unsupported pose encoding type: {cfg.pose_encoding_type}")
        self.cfg = cfg
        self.deterministic = False
        self.target_dim = 9
        block_cfg = BlockConfig(
            dim=cfg.dim_in,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            init_values=cfg.init_values,
            use_rope=False,
            qk_norm=False,
        )
        self.trunk = nnx.List([Block(block_cfg, rngs=rngs) for _ in range(cfg.trunk_depth)])
        self.token_norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-5, rngs=rngs)
        self.trunk_norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-5, rngs=rngs)
        self.empty_pose_tokens = nnx.Param(jnp.zeros((1, 1, self.target_dim), dtype=jnp.float32))
        self.embed_pose = nnx.Linear(self.target_dim, cfg.dim_in, rngs=rngs)
        self.pose_ln_mod = nnx.Linear(cfg.dim_in, 3 * cfg.dim_in, use_bias=True, rngs=rngs)
        self.adaln_norm = nnx.LayerNorm(cfg.dim_in, epsilon=1e-6, use_bias=False, use_scale=False, rngs=rngs)
        self.pose_branch_1 = nnx.Linear(cfg.dim_in, cfg.dim_in // 2, rngs=rngs)
        self.pose_branch_2 = nnx.Linear(cfg.dim_in // 2, self.target_dim, rngs=rngs)

    def _trunk_forward(
        self,
        x: jnp.ndarray,
        *,
        rngs: nnx.Rngs | None,
    ) -> jnp.ndarray:
        for block in self.trunk:
            x = block(x, pos=None, rngs=rngs)
        return x

    def _pose_branch(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.pose_branch_1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.pose_branch_2(x)
        return x

    def __call__(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        num_iterations: int = 4,
        *,
        rngs: nnx.Rngs | None,
    ) -> list[jnp.ndarray]:
        tokens = aggregated_tokens_list[-1]
        pose_tokens = self.token_norm(tokens[:, :, 0])

        batch_size, seq_len, _ = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list: list[jnp.ndarray] = []
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                pose_input = jnp.broadcast_to(self.empty_pose_tokens[...], (batch_size, seq_len, self.target_dim))
            else:
                pose_input = jax.lax.stop_gradient(pred_pose_enc)

            mod_input = self.embed_pose(pose_input)
            mod_input = jax.nn.silu(mod_input)
            shift, scale, gate = jnp.split(self.pose_ln_mod(mod_input), 3, axis=-1)

            pose_tokens_mod = gate * modulate(self.adaln_norm(pose_tokens), shift, scale)
            pose_tokens_mod = pose_tokens_mod + pose_tokens
            pose_tokens_mod = self._trunk_forward(pose_tokens_mod, rngs=rngs)

            delta = self._pose_branch(self.trunk_norm(pose_tokens_mod))
            pred_pose_enc = delta if pred_pose_enc is None else pred_pose_enc + delta

            activated = activate_pose(
                pred_pose_enc,
                trans_act=self.cfg.trans_act,
                quat_act=self.cfg.quat_act,
                fl_act=self.cfg.fl_act,
            )
            pred_pose_enc_list.append(activated)

        return pred_pose_enc_list


__all__ = [
    "CameraHeadConfig",
    "CameraHead",
    "modulate",
]
