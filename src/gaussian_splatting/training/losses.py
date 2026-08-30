import torch
import torch.nn.functional as F


def _gaussian_kernel(
    channels: int,
    kernel_size: int,
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the Gaussian convolution weights."""
    coordinates = torch.arange(kernel_size, dtype=dtype, device=device)
    coordinates -= (kernel_size - 1) / 2

    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()

    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size)


def _gaussian_blur(
    image: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    unbatched = image.ndim == 3
    if unbatched:
        image = image.unsqueeze(0)  # add batch dimension (BCHW) if not there

    padding = kernel.shape[-1] // 2
    image = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    blurred = F.conv2d(image, kernel, groups=image.shape[1])

    return blurred.squeeze(0) if unbatched else blurred


def photometric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.2,
) -> torch.Tensor:
    """Compare encoded RGB tensors in CHW or BCHW layout against targets in [0, 1]."""

    c1 = 1e-4
    c2 = 9e-4
    kernel = _gaussian_kernel(
        channels=prediction.shape[-3],
        kernel_size=11,                 # 11 x 11 window size
        sigma=1.5,                      # how quickly weights decrease away from center
        dtype=prediction.dtype,
        device=prediction.device,
    )

    # L1

    l1 = torch.abs(prediction - target).mean()

    # SSIM

    mu_x = _gaussian_blur(prediction, kernel)
    mu_y = _gaussian_blur(target, kernel)

    var_x = _gaussian_blur(prediction.square(), kernel) - mu_x.square()
    var_y = _gaussian_blur(target.square(), kernel) - mu_y.square()
    cov_xy = _gaussian_blur(prediction * target, kernel) - mu_x * mu_y

    # c1 and c2 help prevent undefined (division by 0)
    ssim_map = (
        (2 * mu_x * mu_y + c1)
        * (2 * cov_xy + c2)
    ) / (
        (mu_x.square() + mu_y.square() + c1)
        * (var_x + var_y + c2)
    )

    ssim = ssim_map.mean()

    return (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim)


def depth_prior_loss(
    rendered_depth: torch.Tensor,
    alpha: torch.Tensor,
    prior_depth: torch.Tensor,
    prior_mask: torch.Tensor,
    *,
    alpha_threshold: float = 0.1,
    beta: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Robustly compare expected and aligned prior depth in log space."""
    if rendered_depth.shape != alpha.shape:
        raise ValueError("rendered depth and alpha must have matching shapes")
    if rendered_depth.ndim == 3 and rendered_depth.shape[0] == 1:
        rendered_depth = rendered_depth[0]
        alpha = alpha[0]
    if rendered_depth.ndim != 2:
        raise ValueError("rendered depth and alpha must have shape (H, W) or (1, H, W)")
    if prior_depth.shape != rendered_depth.shape or prior_mask.shape != rendered_depth.shape:
        raise ValueError("depth prior tensors must match the rendered image shape")
    if beta <= 0:
        raise ValueError("depth loss beta must be positive")
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("depth alpha threshold must be between zero and one")

    valid_prior = prior_mask & torch.isfinite(prior_depth) & (prior_depth > 0)
    if not valid_prior.any():
        raise ValueError("depth prior mask contains no finite positive values")
    valid = (
        valid_prior
        & torch.isfinite(rendered_depth)
        & (rendered_depth > 0)
        & (alpha > alpha_threshold)
    )
    coverage = float(
        valid.count_nonzero().item() / valid_prior.count_nonzero().item()
    )
    if not valid.any():
        return rendered_depth.sum() * 0.0, coverage
    residual = rendered_depth[valid].log() - prior_depth[valid].log()
    loss = F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=beta,
    )
    return loss, coverage