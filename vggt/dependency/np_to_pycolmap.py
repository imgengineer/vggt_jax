from __future__ import annotations

import numpy as np


def batch_np_matrix_to_pycolmap_wo_track(
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_size: np.ndarray,
    *,
    shared_camera: bool = False,
    camera_type: str = "PINHOLE",
):
    try:
        import pycolmap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing dependency: pycolmap") from exc

    points3d = np.asarray(points3d, dtype=np.float32)
    points_xyf = np.asarray(points_xyf, dtype=np.float32)
    points_rgb = np.asarray(points_rgb, dtype=np.uint8)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    image_size = np.asarray(image_size, dtype=np.int32)

    num_frames = len(extrinsics)
    num_points = len(points3d)

    reconstruction = pycolmap.Reconstruction()
    for vidx in range(num_points):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    for fidx in range(num_frames):
        if camera is None or (not shared_camera):
            params = _build_pycolmap_intri(fidx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=params,
                camera_id=fidx + 1,
            )
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3])
        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []
        point2D_idx = 0

        belong = points_xyf[:, 2].astype(np.int32) == fidx
        belong = np.nonzero(belong)[0]
        for point3D_batch_idx in belong:
            point3D_id = int(point3D_batch_idx) + 1
            point2D_xy = points_xyf[point3D_batch_idx, :2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except Exception:  # pragma: no cover
            image.registered = False

        reconstruction.add_image(image)

    return reconstruction


def _build_pycolmap_intri(fidx: int, intrinsics: np.ndarray, camera_type: str) -> np.ndarray:
    if camera_type == "PINHOLE":
        return np.asarray(
            [
                intrinsics[fidx][0, 0],
                intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
            ],
            dtype=np.float64,
        )
    if camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2.0
        return np.asarray([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]], dtype=np.float64)

    raise ValueError(f"Camera type {camera_type} is not supported yet")

