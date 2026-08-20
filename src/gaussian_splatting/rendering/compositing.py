import torch


def alpha_composite(
    colors: torch.Tensor,
    alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Front-to-back composite samples along rays."""

    transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones_like(alphas[:1]),
                1.0 - alphas[:-1],
            ]
        ),
        dim=0,
    )

    weights = transmittance * alphas
    composite_color = (weights[:, None] * colors).sum(dim=0)
    composite_alpha = weights.sum()

    return composite_color, composite_alpha

