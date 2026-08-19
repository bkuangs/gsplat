import torch
from torch import nn


class GaussianModel(nn.Module):
    """Trainable parameterization of a set of anisotropic 3D Gaussians."""

    def __init__(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacity_logits: torch.Tensor,
        sh_coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        count = means.shape[0]
        expected = {
            "means": (count, 3),
            "log_scales": (count, 3),
            "quaternions": (count, 4),
            "opacity_logits": (count, 1),
        }
        values = {
            "means": means,
            "log_scales": log_scales,
            "quaternions": quaternions,
            "opacity_logits": opacity_logits,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if sh_coefficients.ndim != 3 or sh_coefficients.shape[0] != count:
            raise ValueError("sh_coefficients must have shape (N, coefficients, channels)")
        if sh_coefficients.shape[-1] != 3:
            raise ValueError("spherical harmonic coefficients must have three color channels")

        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quaternions = nn.Parameter(quaternions)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.sh_coefficients = nn.Parameter(sh_coefficients)

    @classmethod
    def from_point_cloud(
        cls,
        points: torch.Tensor,
        colors: torch.Tensor,
        sh_degree: int,
        initial_opacity: float,
        initial_scale: float,
    ) -> "GaussianModel":
        """Initialize Gaussian parameters from a COLMAP sparse point cloud."""
        # TODO(student): choose stable inverse activations and initialize the SH DC term.
        raise NotImplementedError("Milestone 2: initialize Gaussians from sparse points")

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid()

    @property
    def normalized_quaternions(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quaternions, dim=-1)

