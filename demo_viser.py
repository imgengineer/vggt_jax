from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from vggt.models import ModelConfig, VGGT
from vggt.models.vggt import create_vggt_from_nnx_npz, create_vggt_from_orbax_checkpoint
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.io import load_and_preprocess_images_np, resolve_image_paths
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _load_model(
    *,
    weights_orbax: str | None,
    weights_npz: str | None,
    from_pretrained: bool,
) -> VGGT:
    config = ModelConfig.vggt_base()
    rngs = nnx.Rngs(0)

    if from_pretrained:
        model = VGGT.from_pretrained(config=config, rngs=rngs, prefer_hf_weights=True)
    elif weights_orbax and Path(weights_orbax).exists():
        model = create_vggt_from_orbax_checkpoint(weights_orbax, config=config, rngs=rngs, strict=True)
    elif weights_npz and Path(weights_npz).exists():
        model = create_vggt_from_nnx_npz(weights_npz, config=config, rngs=rngs, strict=True)
    else:
        model = VGGT(config, rngs=rngs)
    model.eval()
    return model


def _run_predictions(model: VGGT, image_folder: str, target_size: int) -> dict[str, np.ndarray]:
    image_paths = resolve_image_paths(image_folder)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_folder}")

    images = load_and_preprocess_images_np(image_paths, target_size=target_size)
    preds = model(jnp.asarray(images)[None, ...])

    out: dict[str, np.ndarray] = {}
    for key, value in preds.items():
        if isinstance(value, list):
            continue
        array = np.asarray(value)
        if array.shape[0] == 1:
            array = array[0]
        out[key] = array

    out["images"] = images
    image_h, image_w = images.shape[-3], images.shape[-2]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(out["pose_enc"], image_size_hw=(image_h, image_w))
    out["extrinsic"] = extrinsic[0]
    out["intrinsic"] = intrinsic[0]
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(out["depth"], out["extrinsic"], out["intrinsic"])
    return out


def _start_viser(
    predictions: dict[str, np.ndarray],
    *,
    port: int,
    conf_threshold: float,
    use_point_map: bool,
) -> None:
    try:
        import viser
        import viser.transforms as viser_tf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: viser. Install it to run demo_viser.py.") from exc

    images = predictions["images"]
    world_points = predictions["world_points"] if use_point_map else predictions["world_points_from_depth"]
    confidence = predictions["world_points_conf"] if use_point_map else predictions["depth_conf"]
    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]

    colors = (images * 255.0).astype(np.uint8)
    points = world_points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)

    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center

    cam_to_world = closed_form_inverse_se3(extrinsic)[:, :3, :]
    cam_to_world[..., -1] -= scene_center

    s, h, w, _ = world_points.shape
    frame_indices = np.repeat(np.arange(s), h * w)

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    gui_show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_conf = server.gui.add_slider("Confidence Percent", min=0.0, max=100.0, step=0.1, initial_value=conf_threshold)
    gui_frame = server.gui.add_dropdown("Frame", options=["All"] + [str(i) for i in range(s)], initial_value="All")

    init_threshold = np.percentile(conf_flat, conf_threshold) if conf_flat.size else 0.0
    init_mask = (conf_flat >= init_threshold) & (conf_flat > 1e-6)

    cloud = server.scene.add_point_cloud(
        "pointcloud",
        points=points_centered[init_mask],
        colors=colors_flat[init_mask],
        point_size=0.0015,
        point_shape="circle",
    )

    frames = []
    frustums = []
    for idx in range(s):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :4] = cam_to_world[idx]
        se3 = viser_tf.SE3.from_matrix(transform)

        frame_handle = server.scene.add_frame(
            f"frame_{idx}",
            wxyz=se3.rotation().wxyz,
            position=se3.translation(),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )
        frames.append(frame_handle)

        img = colors[idx]
        fy = float(intrinsic[idx, 1, 1])
        fov = 2.0 * np.arctan2(img.shape[0] / 2.0, fy) if fy > 1e-6 else np.deg2rad(45.0)
        frustum = server.scene.add_camera_frustum(
            f"frame_{idx}/frustum",
            fov=float(fov),
            aspect=float(img.shape[1]) / float(img.shape[0]),
            scale=0.06,
            image=img,
            line_width=1.0,
        )
        frustums.append(frustum)

    def _update_cloud() -> None:
        threshold = np.percentile(conf_flat, gui_conf.value) if conf_flat.size else 0.0
        mask = (conf_flat >= threshold) & (conf_flat > 1e-6)
        if gui_frame.value != "All":
            selected = int(gui_frame.value)
            mask &= frame_indices == selected
        cloud.points = points_centered[mask]
        cloud.colors = colors_flat[mask]

    @gui_conf.on_update
    def _(_event) -> None:
        _update_cloud()

    @gui_frame.on_update
    def _(_event) -> None:
        _update_cloud()

    @gui_show_cameras.on_update
    def _(_event) -> None:
        visible = bool(gui_show_cameras.value)
        for handle in frames:
            handle.visible = visible
        for handle in frustums:
            handle.visible = visible

    print(f"Viser running on http://localhost:{port}")
    while True:
        time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT JAX/Flax demo with Viser")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--conf-threshold", type=float, default=50.0)
    parser.add_argument("--use-point-map", action="store_true")
    parser.add_argument("--weights-orbax", type=str, default=None)
    parser.add_argument("--weights-npz", type=str, default=None)
    parser.add_argument("--from-pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = _load_model(
        weights_orbax=args.weights_orbax,
        weights_npz=args.weights_npz,
        from_pretrained=args.from_pretrained,
    )
    preds = _run_predictions(model, args.image_folder, target_size=args.target_size)
    _start_viser(
        preds,
        port=args.port,
        conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
    )


if __name__ == "__main__":
    main()
