import math

import torch


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mse)

