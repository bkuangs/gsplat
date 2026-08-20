from torch import nn

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import Camera, RenderOutput


class TorchRasterizer(nn.Module):
    """Readable reference rasterizer. Correctness matters more than speed."""

    def __init__(self, sh_degree: int) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        self.sh_degree = sh_degree

    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        background: tuple[float, float, float],
    ) -> RenderOutput:
        # TODO(student): project, cull, tile/sort, evaluate self.sh_degree, and composite.
        raise NotImplementedError("Milestone 2: implement the PyTorch reference rasterizer")
