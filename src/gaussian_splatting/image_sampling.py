import torch
import torch.nn.functional as F


def sample_image_at_coordinates(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    source_width: int,
    source_height: int,
) -> torch.Tensor:
    """Sample source-image coordinates whose pixel centers are half-integers."""
    if image.ndim != 2:
        raise ValueError("sampled image must have shape (H, W)")
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("sample coordinates must have shape (N, 2)")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    normalized = torch.stack(
        [
            2.0 * coordinates[:, 0] / source_width - 1.0,
            2.0 * coordinates[:, 1] / source_height - 1.0,
        ],
        dim=-1,
    )
    grid = normalized.reshape(1, 1, -1, 2)
    return F.grid_sample(
        image[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0, 0]
