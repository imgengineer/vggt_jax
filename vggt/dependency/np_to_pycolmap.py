from __future__ import annotations

import numpy as np


def _is_new_pycolmap_api(pycolmap) -> bool:
    image = pycolmap.Image()
    return hasattr(image, "image_id") and not hasattr(image, "id")


def _set_image_points2d(pycolmap, image, points2d_list) -> None:
    if hasattr(pycolmap, "ListPoint2D"):
        image.points2D = pycolmap.ListPoint2D(points2d_list)
    else:
        image.points2D = points2d_list


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

    num_frames = int(len(extrinsics))
    num_points = int(len(points3d))

    reconstruction = pycolmap.Reconstruction()
    point3d_ids: list[int] = []
    for vidx in range(num_points):
        point3d_id = int(reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx]))
        point3d_ids.append(point3d_id)

    if _is_new_pycolmap_api(pycolmap):
        image_template = pycolmap.Image()
        sensor_cls = type(image_template.data_id.sensor_id)
        data_cls = type(image_template.data_id)

        camera = None
        for fidx in range(num_frames):
            if camera is None or (not shared_camera):
                params = _build_pycolmap_intri(fidx, intrinsics, camera_type)
                camera_id = 1 if shared_camera else fidx + 1
                camera = pycolmap.Camera(
                    model=camera_type,
                    width=int(image_size[0]),
                    height=int(image_size[1]),
                    params=params,
                    camera_id=camera_id,
                )
                if hasattr(reconstruction, "exists_camera") and (not reconstruction.exists_camera(camera.camera_id)):
                    reconstruction.add_camera(camera)
                elif not hasattr(reconstruction, "exists_camera"):
                    reconstruction.add_camera(camera)

            frame_id = fidx + 1
            rig_id = frame_id

            sensor = sensor_cls()
            sensor.type = pycolmap.SensorType.CAMERA
            sensor.id = int(camera.camera_id)

            rig = pycolmap.Rig()
            rig.rig_id = rig_id
            rig.add_ref_sensor(sensor)
            reconstruction.add_rig(rig)

            cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3])
            frame = pycolmap.Frame()
            frame.frame_id = frame_id
            frame.rig_id = rig_id
            frame.rig = reconstruction.rig(rig_id)

            data = data_cls()
            data.sensor_id = sensor
            data.id = frame_id
            frame.add_data_id(data)
            frame.set_cam_from_world(int(camera.camera_id), cam_from_world)
            reconstruction.add_frame(frame)

            image = pycolmap.Image()
            image.image_id = frame_id
            image.name = f"image_{frame_id}"
            image.camera_id = int(camera.camera_id)
            image.frame_id = frame_id

            points2d_list = []
            point2d_idx = 0
            belong = points_xyf[:, 2].astype(np.int32) == fidx
            belong = np.nonzero(belong)[0]
            for point3d_batch_idx in belong:
                point3d_id = point3d_ids[int(point3d_batch_idx)]
                point2d_xy = points_xyf[point3d_batch_idx, :2]
                points2d_list.append(pycolmap.Point2D(point2d_xy, point3d_id))

                track = reconstruction.points3D[point3d_id].track
                track.add_element(frame_id, point2d_idx)
                point2d_idx += 1

            _set_image_points2d(pycolmap, image, points2d_list)
            reconstruction.add_image(image)
            reconstruction.register_frame(frame_id)
    else:
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

            points2d_list = []
            point2d_idx = 0

            belong = points_xyf[:, 2].astype(np.int32) == fidx
            belong = np.nonzero(belong)[0]
            for point3d_batch_idx in belong:
                point3d_id = point3d_ids[int(point3d_batch_idx)]
                point2d_xy = points_xyf[point3d_batch_idx, :2]
                points2d_list.append(pycolmap.Point2D(point2d_xy, point3d_id))

                track = reconstruction.points3D[point3d_id].track
                track.add_element(fidx + 1, point2d_idx)
                point2d_idx += 1

            try:
                _set_image_points2d(pycolmap, image, points2d_list)
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
