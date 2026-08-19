import torch


def covariance_3d(log_scales: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Build anisotropic 3D covariance matrices from scale and rotation."""
    # TODO(student): normalize quaternions and evaluate R @ S @ S.T @ R.T.
    raise NotImplementedError("Milestone 1: implement 3D covariance construction")


def project_points(
    points_world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to pixel coordinates and camera-space depth."""
    # TODO(student): homogeneous transform, perspective divide, then apply intrinsics.
    raise NotImplementedError("Milestone 1: implement pinhole projection")


def project_covariance_2d(
    means_camera: torch.Tensor,
    covariances_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D covariance through the local Jacobian of perspective projection."""
    # TODO(student): derive J and evaluate J @ covariance @ J.T.
    raise NotImplementedError("Milestone 1: implement covariance projection")

