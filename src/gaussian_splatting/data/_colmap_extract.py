import sys
from pathlib import Path

import numpy as np


def extract_colmap_arrays(model_dir: Path) -> dict[str, np.ndarray]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    camera_records = sorted(reconstruction.images.items())
    point_records = sorted(reconstruction.points3D.items())

    world_to_camera = np.empty((len(camera_records), 4, 4), dtype=np.float32)
    intrinsics = np.empty((len(camera_records), 3, 3), dtype=np.float32)
    widths = np.empty(len(camera_records), dtype=np.int64)
    heights = np.empty(len(camera_records), dtype=np.int64)
    image_ids = np.empty(len(camera_records), dtype=np.int64)
    camera_ids = np.empty(len(camera_records), dtype=np.int64)
    image_names: list[str] = []
    for index, (image_id, image) in enumerate(camera_records):
        camera = reconstruction.cameras[image.camera_id]
        world_to_camera[index] = np.eye(4, dtype=np.float32)
        world_to_camera[index, :3] = np.asarray(
            image.cam_from_world().matrix(), dtype=np.float32
        )
        intrinsics[index] = np.asarray(
            camera.calibration_matrix(), dtype=np.float32
        )
        widths[index] = int(camera.width)
        heights[index] = int(camera.height)
        image_ids[index] = int(image_id)
        camera_ids[index] = int(image.camera_id)
        image_names.append(image.name)

    points = np.asarray([point.xyz for _, point in point_records], dtype=np.float32)
    colors = np.asarray([point.color for _, point in point_records], dtype=np.float32)
    observation_point_indices: list[int] = []
    observation_image_ids: list[int] = []
    observation_xy: list[np.ndarray] = []
    for point_index, (_, point) in enumerate(point_records):
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            observation_point_indices.append(point_index)
            observation_image_ids.append(int(element.image_id))
            observation_xy.append(
                np.asarray(image.points2D[element.point2D_idx].xy, dtype=np.float32)
            )

    return {
        "world_to_camera": world_to_camera,
        "intrinsics": intrinsics,
        "widths": widths,
        "heights": heights,
        "image_ids": image_ids,
        "camera_ids": camera_ids,
        "image_names": np.asarray(image_names),
        "points": points.reshape(-1, 3),
        "colors": colors.reshape(-1, 3),
        "observation_point_indices": np.asarray(
            observation_point_indices, dtype=np.int64
        ),
        "observation_image_ids": np.asarray(observation_image_ids, dtype=np.int64),
        "observation_xy": np.asarray(observation_xy, dtype=np.float32).reshape(-1, 2),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: _colmap_extract MODEL_DIR OUTPUT_FILE")
    arrays = extract_colmap_arrays(Path(sys.argv[1]))
    np.savez_compressed(sys.argv[2], **arrays)


if __name__ == "__main__":
    main()
