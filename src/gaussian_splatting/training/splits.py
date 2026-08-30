from gaussian_splatting.data.colmap import ColmapScene


def partition_camera_indices(
    scene: ColmapScene,
    holdout_image_ids: tuple[int, ...] = (),
    *,
    train_image_ids: tuple[int, ...] = (),
    test_image_ids: tuple[int, ...] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve explicit image-ID splits into stable scene indices."""
    registered_ids = [camera.image_id for camera in scene.cameras]
    if any(image_id is None for image_id in registered_ids):
        raise ValueError("camera splitting requires registered COLMAP image IDs")
    if len(set(registered_ids)) != len(registered_ids):
        raise ValueError("registered COLMAP image IDs must be unique")

    indices_by_image_id = {
        camera.image_id: index for index, camera in enumerate(scene.cameras)
    }
    evaluation_image_ids = test_image_ids or holdout_image_ids
    requested_ids = set(train_image_ids) | set(evaluation_image_ids)
    missing = sorted(requested_ids - indices_by_image_id.keys())
    if missing:
        names = ", ".join(str(image_id) for image_id in missing)
        raise ValueError(f"configured image IDs are not registered in the scene: {names}")

    evaluation_set = set(evaluation_image_ids)
    if set(train_image_ids) & evaluation_set:
        raise ValueError("training and test image IDs must be disjoint")
    if train_image_ids:
        training_indices = tuple(
            indices_by_image_id[image_id] for image_id in train_image_ids
        )
    else:
        training_indices = tuple(
            index
            for index, camera in enumerate(scene.cameras)
            if camera.image_id not in evaluation_set
        )
    evaluation_indices = tuple(
        indices_by_image_id[image_id] for image_id in evaluation_image_ids
    )
    if not training_indices:
        raise ValueError("at least one registered camera must remain for training")
    return training_indices, evaluation_indices
