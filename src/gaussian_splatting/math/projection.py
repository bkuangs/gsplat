import torch
import torch.nn.functional as F


def quaternion_to_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert WXYZ quaternions to rotation matrices."""
    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have shape (N, 4)")
    quat_norm = F.normalize(quaternions, dim=-1)
    w, x, y, z = quat_norm.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def covariance_3d(log_scales: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Build anisotropic 3D covariance matrices from scale and WXYZ rotation."""
    if log_scales.ndim != 2 or log_scales.shape[-1] != 3:
        raise ValueError("log_scales must have shape (N, 3)")
    if quaternions.shape != (log_scales.shape[0], 4):
        raise ValueError("quaternions must have shape (N, 4)")
    scales = log_scales.exp()
    covariances = torch.diag_embed(scales.square())
    rotation = quaternion_to_rotation(quaternions)
    return rotation @ covariances @ rotation.transpose(-1, -2)


def project_points(
    points_world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to pixel coordinates and camera-space depth."""

    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape (4, 4)")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")

    points_h = torch.cat(
        [points_world, torch.ones_like(points_world[:, :1])],
        dim=-1,
    )
    points_cam_h = points_h @ world_to_camera.T
    points_cam = points_cam_h[:, :3]
    depth = points_cam[:, 2]
    epsilon = torch.finfo(depth.dtype).eps
    safe_depth = torch.where(
        depth.abs() >= epsilon,
        depth,
        torch.where(depth < 0, -epsilon, epsilon),
    )
    normalized_h = torch.stack(
        [
            points_cam[:, 0] / safe_depth,
            points_cam[:, 1] / safe_depth,
            torch.ones_like(depth),
        ],
        dim=-1,
    )
    pixel_h = normalized_h @ intrinsics.T
    pixels = pixel_h[:, :2] / pixel_h[:, 2:3]
    return pixels, depth


def project_covariance_2d(
    means_camera: torch.Tensor,
    covariances_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D covariance through the local Jacobian of perspective projection."""
    if means_camera.ndim != 2 or means_camera.shape[-1] != 3:
        raise ValueError("means_camera must have shape (N, 3)")
    if covariances_camera.shape != (means_camera.shape[0], 3, 3):
        raise ValueError("covariances_camera must have shape (N, 3, 3)")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    x = means_camera[:, 0]
    y = means_camera[:, 1]
    z = means_camera[:, 2]
    zeros = torch.zeros_like(z)
    row_u = torch.stack(
        [fx / z, zeros, -(fx * x) / z.square()],
        dim=-1,
    )
    row_v = torch.stack(
        [zeros, fy / z, -(fy * y) / z.square()],
        dim=-1,
    )
    jacobian = torch.stack([row_u, row_v], dim=-2)
    return jacobian @ covariances_camera @ jacobian.transpose(-1, -2)
