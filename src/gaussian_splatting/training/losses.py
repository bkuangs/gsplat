import torch


def photometric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.2,
) -> torch.Tensor:
    """Blend L1 and differentiable SSIM as used by 3DGS training."""
    # TODO(student): implement L1 + SSIM and document image range assumptions.
    raise NotImplementedError("Milestone 3: implement the photometric objective")

