import importlib.util
import os
from pathlib import Path

import torch
from torch import nn

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import Camera, RenderOutput


def _ensure_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"CUDA toolkit cache path already exists: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _configure_bundled_cuda_toolkit() -> None:
    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is None or nvidia_spec.submodule_search_locations is None:
        return

    toolkit_root = next(
        (
            Path(location) / "cu13"
            for location in nvidia_spec.submodule_search_locations
            if (Path(location) / "cu13" / "bin" / "nvcc").is_file()
        ),
        None,
    )
    if toolkit_root is None:
        return

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cuda_home = cache_home / "learn-3d-gaussian-splatting" / "cuda-13.0"
    cuda_home.mkdir(parents=True, exist_ok=True)
    for directory in ("bin", "include", "nvvm"):
        _ensure_symlink(toolkit_root / directory, cuda_home / directory)

    library_dir = cuda_home / "lib"
    library_dir.mkdir(exist_ok=True)
    for library in (toolkit_root / "lib").iterdir():
        if library.is_file():
            _ensure_symlink(library, library_dir / library.name)
    _ensure_symlink(
        toolkit_root / "lib" / "libcudart.so.13",
        library_dir / "libcudart.so",
    )

    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{os.environ['PATH']}"
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_home)


class CudaRasterizer(nn.Module):
    """Adapter around the external gsplat CUDA rasterizer."""

    def __init__(self, sh_degree: int) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        self.sh_degree = sh_degree
        _configure_bundled_cuda_toolkit()
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
        if len(background) != 3:
            raise ValueError("background must contain three values")

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
            backgrounds=None,
            packed=True,
            render_mode="RGB+ED",
        )
        projected_means = metadata["means2d"]
        gaussian_ids = metadata["gaussian_ids"]
        means_2d = model.means.new_zeros(
            (model.means.shape[0], 2),
            requires_grad=projected_means.requires_grad,
        )
        if projected_means.requires_grad:
            def retain_projected_gradient(gradient):
                dense_gradient = torch.zeros_like(means_2d)
                dense_gradient.index_add_(0, gaussian_ids, gradient)
                means_2d.grad = dense_gradient
                return gradient

            projected_means.register_hook(retain_projected_gradient)
        projected_radii = metadata["radii"]
        if projected_radii.ndim == 2:
            packed_radii = projected_radii.max(dim=-1).values
        elif projected_radii.ndim == 1:
            packed_radii = projected_radii
        else:
            raise RuntimeError("gsplat returned radii with an unsupported shape")
        radii = model.means.new_zeros(model.means.shape[0])
        radii.scatter_reduce_(
            0,
            gaussian_ids,
            packed_radii.to(radii.dtype),
            reduce="amax",
            include_self=False,
        )
        visibility = radii > 0
        rgb = rendered[0, ..., :3].permute(2, 0, 1)
        if any(value != 0.0 for value in background):
            background_tensor = model.means.new_tensor(background)[:, None, None]
            rgb = rgb + (1.0 - alpha[0].permute(2, 0, 1)) * background_tensor
        return RenderOutput(
            rgb=rgb,
            alpha=alpha[0].permute(2, 0, 1),
            depth=rendered[0, ..., 3:].permute(2, 0, 1),
            radii=radii,
            means_2d=means_2d,
            visibility=visibility,
        )
