import torch
import torch.nn.functional as F


def covariance_3d(log_scales: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Build anisotropic 3D covariance matrices from scale and WXYZ rotation."""

    scales = log_scales.exp()
    variances = scales.square()
    covariances = torch.diag_embed(variances)
    quat_norm = F.normalize(quaternions, dim=-1)

    # Separate WXYZ components
    w, x, y, z = quat_norm.unbind(dim=-1)

    # Convert each quaternion into a rotation matrix
    rotation = torch.stack(
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
    ).reshape(-1, 3, 3)     # (N, 9) -> (N, 3, 3)

    return rotation @ covariances @ rotation.transpose(-1, -2)


def project_points(
    points_world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to pixel coordinates and camera-space depth."""

    # Append homogenous 1s: (x, y, z) -> (x, y, z, 1) to allow for affine transformation
    points_h = torch.cat(
        [points_world, torch.ones_like(points_world[:, :1])],
        dim=-1,
    )

    # World -> camera. torch stores points as rows, so we have to take the transpose to get columns
    points_cam_h = points_h @ world_to_camera.T     # @ = matrix multiplication!

    # Remove homogenous 1s
    points_cam = points_cam_h[:, :3]

    # Extract depth
    depth = points_cam_h[:, 2]

    # Prevent division by 0.
    # `near` is the camera’s near clipping distance: the smallest positive 
    # camera-space depth that you consider safe and renderable.
    near = 1e-2
    valid = depth > near

    safe_depth = torch.where(
        valid,
        depth,
        torch.ones_like(depth),
    )

    x_normalized = points_cam[:, 0] / safe_depth
    y_normalized = points_cam[:, 1] / safe_depth

    # Perspective division
    normalized_h = torch.stack(
        [
            x_normalized,
            y_normalized,
            torch.tensor(1.0, device=points_cam.device, dtype=points_cam.dtype),
        ], dim=-1)

    # Apply intrinsics (multiply by K)
    pixel_h = normalized_h @ intrinsics.T
    pixels = pixel_h[:, :2] / pixel_h[:, 2:3]

    return pixels, depth

def project_covariance_2d(
    means_camera: torch.Tensor,
    covariances_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D covariance through the local Jacobian of perspective projection."""
    # TODO(student): derive J and evaluate J @ covariance @ J.T.
    raise NotImplementedError("Milestone 1: implement covariance projection")
