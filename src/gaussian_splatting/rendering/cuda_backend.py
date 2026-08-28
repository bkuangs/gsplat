import torch
from torch import nn

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import Camera, RenderOutput


class CudaRasterizer(nn.Module):
    """Adapter around the external gsplat CUDA rasterizer."""

    def __init__(self, sh_degree: int) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        self.sh_degree = sh_degree
        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError(
                "CUDA rendering requires an NVIDIA GPU and the cuda extra: uv sync --extra cuda"
            ) from error
        self._rasterization = rasterization

    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        background: tuple[float, float, float],
    ) -> RenderOutput:
        if model.means.device.type != "cuda":
            raise ValueError("CudaRasterizer requires model parameters on a CUDA device")
        if camera.world_to_camera.device.type != "cuda":
            raise ValueError("CudaRasterizer requires camera tensors on a CUDA device")

        rendered, alpha, metadata = self._rasterization(
            means=model.means,
            quats=model.normalized_quaternions,
            scales=model.scales,
            opacities=model.opacities.squeeze(-1),
            colors=model.sh_coefficients,
            viewmats=camera.world_to_camera.unsqueeze(0),
            Ks=camera.intrinsics.unsqueeze(0),
            width=camera.width,
            height=camera.height,
            sh_degree=self.sh_degree,
            backgrounds=torch.tensor(
                [background], device=model.means.device, dtype=model.means.dtype
            ),
            render_mode="RGB+ED",
        )
        means_2d = metadata["means2d"][0]
        if means_2d.requires_grad:
            means_2d.retain_grad()
        projected_radii = metadata["radii"][0]
        if projected_radii.ndim == 2:
            visibility = (projected_radii > 0).all(dim=-1)
            radii = projected_radii.max(dim=-1).values
        elif projected_radii.ndim == 1:
            radii = projected_radii
            visibility = radii > 0
        else:
            raise RuntimeError("gsplat returned radii with an unsupported shape")
        return RenderOutput(
            rgb=rendered[0, ..., :3].permute(2, 0, 1),
            alpha=alpha[0].permute(2, 0, 1),
            depth=rendered[0, ..., 3:].permute(2, 0, 1),
            radii=radii,
            means_2d=means_2d,
            visibility=visibility,
        )
