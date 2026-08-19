from torch import nn

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import Camera, RenderOutput


class TorchRasterizer(nn.Module):
    """Readable reference rasterizer. Correctness matters more than speed."""

    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        background: tuple[float, float, float],
    ) -> RenderOutput:
        # TODO(student): project, cull, tile/sort, evaluate Gaussians, and composite.
        raise NotImplementedError("Milestone 2: implement the PyTorch reference rasterizer")

