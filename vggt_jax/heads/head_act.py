import jax
import jax.numpy as jnp


def inverse_log_transform(y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y) * jnp.expm1(jnp.abs(y))


def base_pose_act(x: jnp.ndarray, act_type: str = "linear") -> jnp.ndarray:
    if act_type == "linear":
        return x
    if act_type == "inv_log":
        return inverse_log_transform(x)
    if act_type == "exp":
        return jnp.exp(x)
    if act_type == "relu":
        return jax.nn.relu(x)
    raise ValueError(f"Unknown act_type: {act_type}")


def activate_pose(
    pred_pose_enc: jnp.ndarray, trans_act: str = "linear", quat_act: str = "linear", fl_act: str = "linear"
) -> jnp.ndarray:
    trans = base_pose_act(pred_pose_enc[..., :3], trans_act)
    quat = base_pose_act(pred_pose_enc[..., 3:7], quat_act)
    fov = base_pose_act(pred_pose_enc[..., 7:], fl_act)
    return jnp.concatenate([trans, quat, fov], axis=-1)


def activate_head(
    out_bchw: jnp.ndarray, activation: str = "norm_exp", conf_activation: str = "expp1"
) -> tuple[jnp.ndarray, jnp.ndarray]:
    fmap = jnp.transpose(out_bchw, (0, 2, 3, 1))
    xyz = fmap[..., :-1]
    conf = fmap[..., -1]

    if activation == "norm_exp":
        d = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        d = jnp.clip(d, a_min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * jnp.expm1(d)
    elif activation == "norm":
        pts3d = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    elif activation == "exp":
        pts3d = jnp.exp(xyz)
    elif activation == "relu":
        pts3d = jax.nn.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy = xyz[..., :2]
        z = inverse_log_transform(xyz[..., 2:3])
        pts3d = jnp.concatenate([xy * z, z], axis=-1)
    elif activation == "sigmoid":
        pts3d = jax.nn.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1.0 + jnp.exp(conf)
    elif conf_activation == "expp0":
        conf_out = jnp.exp(conf)
    elif conf_activation == "sigmoid":
        conf_out = jax.nn.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out
