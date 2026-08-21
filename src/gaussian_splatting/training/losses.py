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
        image = image.unsqueeze(0)

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

    # 11 x 11 kernel size -> window size
    # Sigma: Gaussian kernel assigns a weight to each neighboring pixel; Sigma controls 
    #        how quickly those weights decrease as you move away from the center.
    #
    # We will use an isotropic sigma.
    c1 = 1e-4
    c2 = 9e-4
    kernel = _gaussian_kernel(
        channels=prediction.shape[-3],
        kernel_size=11,
        sigma=1.5,
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