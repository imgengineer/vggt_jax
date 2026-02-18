from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .train_utils.general import check_and_fix_inf_nan


EPS = 1e-8


def _matrix_to_quaternion_xyzw(rot: jnp.ndarray) -> jnp.ndarray:
    m00 = rot[..., 0, 0]
    m11 = rot[..., 1, 1]
    m22 = rot[..., 2, 2]
    qx = jnp.sign(rot[..., 2, 1] - rot[..., 1, 2]) * jnp.sqrt(jnp.clip(1.0 + m00 - m11 - m22, a_min=0.0)) * 0.5
    qy = jnp.sign(rot[..., 0, 2] - rot[..., 2, 0]) * jnp.sqrt(jnp.clip(1.0 - m00 + m11 - m22, a_min=0.0)) * 0.5
    qz = jnp.sign(rot[..., 1, 0] - rot[..., 0, 1]) * jnp.sqrt(jnp.clip(1.0 - m00 - m11 + m22, a_min=0.0)) * 0.5
    qw = jnp.sqrt(jnp.clip(1.0 + m00 + m11 + m22, a_min=0.0)) * 0.5
    quat = jnp.stack([qx, qy, qz, qw], axis=-1)
    return quat / jnp.clip(jnp.linalg.norm(quat, axis=-1, keepdims=True), a_min=EPS)


def extri_intri_to_pose_encoding(
    extrinsics: jnp.ndarray,
    intrinsics: jnp.ndarray,
    image_hw: tuple[int, int],
) -> jnp.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    t = extrinsics[..., :3, 3]
    quat = _matrix_to_quaternion_xyzw(extrinsics[..., :3, :3])
    fx = jnp.clip(intrinsics[..., 0, 0], a_min=EPS)
    fy = jnp.clip(intrinsics[..., 1, 1], a_min=EPS)
    fov_h = 2.0 * jnp.arctan((h / 2.0) / fy)
    fov_w = 2.0 * jnp.arctan((w / 2.0) / fx)
    return jnp.concatenate([t, quat, fov_h[..., None], fov_w[..., None]], axis=-1)


@dataclass(eq=False)
class MultitaskLoss:
    camera: dict | None = None
    depth: dict | None = None
    point: dict | None = None
    track: dict | None = None

    def __call__(self, predictions, batch):
        total = jnp.asarray(0.0, dtype=jnp.float32)
        loss_dict = {}

        if self.camera is not None and "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)
            weighted = camera_loss_dict["loss_camera"] * float(self.camera.get("weight", 1.0))
            total = total + weighted
            loss_dict.update(camera_loss_dict)

        if self.depth is not None and "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            weighted = (
                depth_loss_dict["loss_conf_depth"]
                + depth_loss_dict["loss_reg_depth"]
                + depth_loss_dict["loss_grad_depth"]
            ) * float(self.depth.get("weight", 1.0))
            total = total + weighted
            loss_dict.update(depth_loss_dict)

        if self.point is not None and "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            weighted = (
                point_loss_dict["loss_conf_point"]
                + point_loss_dict["loss_reg_point"]
                + point_loss_dict["loss_grad_point"]
            ) * float(self.point.get("weight", 1.0))
            total = total + weighted
            loss_dict.update(point_loss_dict)

        if self.track is not None and "track" in predictions:
            raise NotImplementedError("Track loss is not implemented in the JAX trainer yet.")

        loss_dict["objective"] = total
        return loss_dict

    forward = __call__


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type: str = "l1"):
    if loss_type == "l1":
        loss_t = jnp.abs(pred_pose_enc[..., :3] - gt_pose_enc[..., :3])
        loss_r = jnp.abs(pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7])
        loss_fl = jnp.abs(pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:])
    elif loss_type == "l2":
        loss_t = jnp.linalg.norm(pred_pose_enc[..., :3] - gt_pose_enc[..., :3], axis=-1, keepdims=True)
        loss_r = jnp.linalg.norm(pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7], axis=-1, keepdims=True)
        loss_fl = jnp.linalg.norm(pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:], axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_t = jnp.mean(jnp.clip(check_and_fix_inf_nan(loss_t), a_max=100.0))
    loss_r = jnp.mean(check_and_fix_inf_nan(loss_r))
    loss_fl = jnp.mean(check_and_fix_inf_nan(loss_fl))
    return loss_t, loss_r, loss_fl


def compute_camera_loss(
    pred_dict,
    batch_data,
    loss_type: str = "l1",
    gamma: float = 0.6,
    pose_encoding_type: str = "absT_quaR_FoV",
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    weight_focal: float = 0.5,
    **kwargs,
):
    _ = kwargs
    if pose_encoding_type != "absT_quaR_FoV":
        raise ValueError(f"Unsupported pose encoding type: {pose_encoding_type}")

    pred_pose_encodings = pred_dict["pose_enc_list"]
    if not isinstance(pred_pose_encodings, (list, tuple)):
        pred_pose_encodings = [pred_pose_encodings]

    point_masks = batch_data.get("point_masks", None)
    if point_masks is None:
        valid_frame_mask = jnp.ones(pred_pose_encodings[-1].shape[:2], dtype=bool)
    else:
        valid_frame_mask = jnp.sum(point_masks[:, 0], axis=(-1, -2)) > 100

    gt_extrinsics = batch_data["extrinsics"]
    gt_intrinsics = batch_data["intrinsics"]
    image_hw = batch_data["images"].shape[-3:-1]
    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsics, gt_intrinsics, image_hw)

    total_loss_t = jnp.asarray(0.0, dtype=jnp.float32)
    total_loss_r = jnp.asarray(0.0, dtype=jnp.float32)
    total_loss_fl = jnp.asarray(0.0, dtype=jnp.float32)
    n_stages = len(pred_pose_encodings)

    any_valid = jnp.any(valid_frame_mask)
    for stage_idx, pred_pose in enumerate(pred_pose_encodings):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_valid = pred_pose[valid_frame_mask]
        gt_valid = gt_pose_encoding[valid_frame_mask]

        loss_t_stage, loss_r_stage, loss_fl_stage = jnp.where(
            any_valid,
            jnp.asarray(camera_loss_single(pred_valid, gt_valid, loss_type=loss_type)),
            jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        )

        total_loss_t = total_loss_t + stage_weight * loss_t_stage
        total_loss_r = total_loss_r + stage_weight * loss_r_stage
        total_loss_fl = total_loss_fl + stage_weight * loss_fl_stage

    avg_loss_t = total_loss_t / max(n_stages, 1)
    avg_loss_r = total_loss_r / max(n_stages, 1)
    avg_loss_fl = total_loss_fl / max(n_stages, 1)
    total_camera_loss = avg_loss_t * weight_trans + avg_loss_r * weight_rot + avg_loss_fl * weight_focal

    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_t,
        "loss_R": avg_loss_r,
        "loss_FL": avg_loss_fl,
    }


def _ensure_channel_last(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 4:
        return x[..., None]
    return x


def filter_by_quantile(x: jnp.ndarray, q: float) -> jnp.ndarray:
    if x.size == 0:
        return x
    thr = jnp.quantile(x, q)
    return x[x <= thr]


def gradient_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    conf: jnp.ndarray | None = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
) -> jnp.ndarray:
    prediction = _ensure_channel_last(prediction)
    target = _ensure_channel_last(target)
    diff = jnp.linalg.norm(prediction - target, axis=-1, keepdims=True)
    mask = mask.astype(jnp.float32)[..., None]

    grad_x = jnp.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    grad_y = jnp.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]

    if conf is not None:
        conf = jnp.clip(conf.astype(jnp.float32), a_min=EPS)
        conf_x = (conf[:, :, 1:] + conf[:, :, :-1]) / 2.0
        conf_y = (conf[:, 1:, :] + conf[:, :-1, :]) / 2.0
        grad_x = gamma * grad_x * conf_x[..., None] - alpha * jnp.log(conf_x[..., None])
        grad_y = gamma * grad_y * conf_y[..., None] - alpha * jnp.log(conf_y[..., None])

    numer = jnp.sum(grad_x * mask_x) + jnp.sum(grad_y * mask_y)
    denom = jnp.maximum(jnp.sum(mask_x) + jnp.sum(mask_y), 1.0)
    return numer / denom


def normal_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    conf: jnp.ndarray | None = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
) -> jnp.ndarray:
    return gradient_loss(prediction, target, mask, conf=conf, gamma=gamma, alpha=alpha)


def gradient_loss_multi_scale_wrapper(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    scales: int = 4,
    gradient_loss_fn=gradient_loss,
    conf: jnp.ndarray | None = None,
) -> jnp.ndarray:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for scale in range(scales):
        step = 2**scale
        total = total + gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )
    return total / max(scales, 1)


def regression_loss(
    pred: jnp.ndarray,
    gt: jnp.ndarray,
    mask: jnp.ndarray,
    conf: jnp.ndarray | None = None,
    gradient_loss_fn: str | None = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
    valid_range: float = -1.0,
):
    pred = _ensure_channel_last(pred)
    gt = _ensure_channel_last(gt)
    mask = mask.astype(bool)

    finite_mask = jnp.all(jnp.isfinite(gt), axis=-1)
    valid = mask & finite_mask

    reg_map = jnp.linalg.norm(gt - pred, axis=-1)
    reg_vals = reg_map[valid]
    reg_vals = check_and_fix_inf_nan(reg_vals)

    if conf is not None:
        conf_vals = jnp.clip(conf[valid], a_min=EPS)
        conf_loss_vals = gamma * reg_vals * conf_vals - alpha * jnp.log(conf_vals)
    else:
        conf_loss_vals = reg_vals

    if valid_range > 0:
        reg_vals = filter_by_quantile(reg_vals, valid_range)
        conf_loss_vals = filter_by_quantile(conf_loss_vals, valid_range)

    loss_reg = jnp.mean(reg_vals) if reg_vals.size > 0 else jnp.asarray(0.0, dtype=jnp.float32)
    loss_conf = jnp.mean(conf_loss_vals) if conf_loss_vals.size > 0 else jnp.asarray(0.0, dtype=jnp.float32)

    loss_grad = jnp.asarray(0.0, dtype=jnp.float32)
    if gradient_loss_fn:
        b, s = pred.shape[:2]
        pred2 = pred.reshape(b * s, pred.shape[2], pred.shape[3], pred.shape[4])
        gt2 = gt.reshape(b * s, gt.shape[2], gt.shape[3], gt.shape[4])
        mask2 = valid.reshape(b * s, valid.shape[2], valid.shape[3])
        conf2 = conf.reshape(b * s, conf.shape[2], conf.shape[3]) if conf is not None else None

        fn = normal_loss if "normal" in gradient_loss_fn else gradient_loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred2,
            gt2,
            mask2,
            scales=3,
            gradient_loss_fn=fn,
            conf=conf2 if ("conf" in str(gradient_loss_fn)) else None,
        )
    return loss_conf, loss_grad, loss_reg


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, **kwargs):
    _ = kwargs
    pred_depth = predictions["depth"]
    pred_depth_conf = predictions["depth_conf"]
    gt_depth = batch["depths"]
    gt_depth_mask = batch["point_masks"]
    gt_depth = check_and_fix_inf_nan(gt_depth)

    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_depth,
        gt_depth,
        gt_depth_mask,
        conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
    )
    return {
        "loss_conf_depth": loss_conf,
        "loss_reg_depth": loss_reg,
        "loss_grad_depth": loss_grad,
    }


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, **kwargs):
    _ = kwargs
    pred_points = predictions["world_points"]
    pred_points_conf = predictions["world_points_conf"]
    gt_points = check_and_fix_inf_nan(batch["world_points"])
    gt_points_mask = batch["point_masks"]

    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_points,
        gt_points,
        gt_points_mask,
        conf=pred_points_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
    )
    return {
        "loss_conf_point": loss_conf,
        "loss_reg_point": loss_reg,
        "loss_grad_point": loss_grad,
    }
