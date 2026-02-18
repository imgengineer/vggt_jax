from __future__ import annotations

import argparse
import copy
import glob
import os
import random

import jax
import jax.numpy as jnp
import numpy as np

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square_np
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

try:
    from scripts._common import add_model_loading_args, load_vggt_model, resolve_model_config, resolve_model_load_spec
except ModuleNotFoundError:  # pragma: no cover
    from _common import add_model_loading_args, load_vggt_model, resolve_model_config, resolve_model_load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT -> COLMAP (feed-forward, no BA)")
    parser.add_argument(
        "--scene-dir",
        "--scene_dir",
        dest="scene_dir",
        type=str,
        required=True,
        help="Directory containing an `images/` folder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    add_model_loading_args(parser, include_tiny=False)

    parser.add_argument("--resolution", type=int, default=518, help="Square resolution to run VGGT")
    parser.add_argument("--camera-type", type=str, default="PINHOLE", choices=("PINHOLE", "SIMPLE_PINHOLE"))
    parser.add_argument("--conf-thres-value", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=100000)
    return parser.parse_args()


def _load_model(args: argparse.Namespace):
    cfg = resolve_model_config(tiny=False)
    load_spec = resolve_model_load_spec(
        tiny=False,
        weights_orbax=args.weights_orbax,
        weights_npz=args.weights_npz,
        from_pretrained=args.from_pretrained,
        prefer_orbax=args.prefer_orbax,
    )
    return load_vggt_model(
        config=cfg,
        load_spec=load_spec,
        repo_id=args.repo_id,
        filename=args.filename,
        cache_dir=args.cache_dir,
    )


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_names: list[str],
    original_coords: np.ndarray,
    *,
    img_size: int,
    shift_point2d_to_original_res: bool = True,
    shared_camera: bool = False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_names[pyimageid - 1]

        real_image_size = original_coords[pyimageid - 1, -2:]
        resize_ratio = float(np.max(real_image_size)) / float(img_size)

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2.0
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("backend:", jax.default_backend())
    print("devices:", jax.devices())

    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise SystemExit(f"No images found in {image_dir}")

    print(f"found images: {len(image_path_list)}", flush=True)
    base_image_names = [os.path.basename(path) for path in image_path_list]

    print("loading VGGT model...", flush=True)
    model = _load_model(args)
    print("model loaded.", flush=True)

    print("preprocessing images...", flush=True)
    images_np, original_coords = load_and_preprocess_images_square_np(image_path_list, target_size=args.resolution)
    images = jnp.asarray(images_np)
    print(f"preprocessed image batch: {images.shape}", flush=True)

    print("running model forward...", flush=True)
    preds = model(images)
    print("model forward done.", flush=True)
    pose_enc = np.asarray(preds["pose_enc"][0])
    depth_map = np.asarray(preds["depth"][0])
    depth_conf = np.asarray(preds["depth_conf"][0])

    _, height, width, _ = images_np.shape
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(height, width))
    extrinsic = extrinsic[0]
    intrinsic = intrinsic[0]

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    points_rgb = (images_np * 255.0).astype(np.uint8)
    points_xyf = create_pixel_coordinate_grid(points_3d.shape[0], points_3d.shape[1], points_3d.shape[2])

    conf_mask = depth_conf >= float(args.conf_thres_value)
    conf_mask = randomly_limit_trues(conf_mask, int(args.max_points))

    points_3d_flat = points_3d[conf_mask]
    points_xyf_flat = points_xyf[conf_mask]
    points_rgb_flat = points_rgb[conf_mask]

    image_size = np.asarray([width, height], dtype=np.int32)

    print("converting predictions to COLMAP reconstruction...", flush=True)
    try:
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d_flat,
            points_xyf_flat,
            points_rgb_flat,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type=args.camera_type,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency for COLMAP export. Install: `pip install pycolmap trimesh` "
            "(or use your preferred environment manager)."
        ) from exc

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_names,
        original_coords,
        img_size=args.resolution,
        shift_point2d_to_original_res=True,
        shared_camera=False,
    )

    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    print(f"Saved COLMAP reconstruction to: {sparse_reconstruction_dir}")

    try:
        import trimesh  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: trimesh. Install: `pip install trimesh`") from exc

    trimesh.PointCloud(points_3d_flat, colors=points_rgb_flat).export(os.path.join(sparse_reconstruction_dir, "points.ply"))
    print(f"Saved point cloud to: {os.path.join(sparse_reconstruction_dir, 'points.ply')}")


if __name__ == "__main__":
    main()
