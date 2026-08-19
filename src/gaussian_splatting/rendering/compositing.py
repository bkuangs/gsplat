import torch


def alpha_composite(
    colors: torch.Tensor,
    alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Front-to-back composite samples along rays."""
    # TODO(student): compute transmittance and alpha-weighted color without in-place ops.
    raise NotImplementedError("Milestone 1: implement differentiable alpha compositing")

