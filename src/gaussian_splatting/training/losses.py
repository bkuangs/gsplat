import torch


def photometric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.2,
) -> torch.Tensor:
    """Compare encoded RGB tensors in CHW or BCHW layout against targets in [0, 1]."""
    # TODO(student): implement L1 + SSIM without silently changing layout or color space.
    raise NotImplementedError("Milestone 3: implement the photometric objective")
