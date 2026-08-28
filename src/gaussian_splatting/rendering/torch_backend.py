import torch
import torch.nn.functional as F
from torch import nn

from gaussian_splatting.math.projection import (
    covariance_3d,
    project_covariance_2d,
    project_points,
)
from gaussian_splatting.math.spherical_harmonics import evaluate_sh
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.types import Camera, RenderOutput


class TorchRasterizer(nn.Module):
    """Readable reference rasterizer. Correctness matters more than speed."""

    def __init__(self, sh_degree: int) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        if sh_degree > 3:
            raise ValueError("the reference renderer supports SH degrees through 3")
        self.sh_degree = sh_degree

    def forward(
        self,
        model: GaussianModel,
        camera: Camera,
        background: tuple[float, float, float],
    ) -> RenderOutput:
        if camera.world_to_camera.device != model.means.device:
            raise ValueError("camera and model must be on the same device")
        if len(background) != 3:
            raise ValueError("background must contain three values")

        dtype = model.means.dtype
        device = model.means.device
        view = camera.world_to_camera.to(dtype=dtype)
        intrinsics = camera.intrinsics.to(dtype=dtype)
        rotation = view[:3, :3]

        means_2d, depths = project_points(model.means, view, intrinsics)
        if means_2d.requires_grad:
            means_2d.retain_grad()
        means_h = torch.cat(
            [model.means, torch.ones_like(model.means[:, :1])], dim=-1
        )
        means_camera = (means_h @ view.T)[:, :3]

        covariance_world = covariance_3d(model.log_scales, model.quaternions)
        covariance_camera = rotation @ covariance_world @ rotation.T
        safe_depth = depths.clamp_min(1e-4)
        covariance_2d = project_covariance_2d(
            torch.stack(
                [means_camera[:, 0], means_camera[:, 1], safe_depth], dim=-1
            ),
            covariance_camera,
            intrinsics,
        )
        covariance_2d = covariance_2d + 0.3 * torch.eye(
            2, dtype=dtype, device=device
        )
        eigenvalues = torch.linalg.eigvalsh(covariance_2d)
        radii = 3.0 * eigenvalues[:, -1].clamp_min(0.0).sqrt()

        finite = torch.isfinite(means_2d).all(dim=-1) & torch.isfinite(radii)
        visibility = (
            (depths > 1e-2)
            & finite
            & (means_2d[:, 0] + radii >= 0)
            & (means_2d[:, 0] - radii < camera.width)
            & (means_2d[:, 1] + radii >= 0)
            & (means_2d[:, 1] - radii < camera.height)
        )
        output_radii = torch.where(visibility, radii, torch.zeros_like(radii))

        background_tensor = model.means.new_tensor(background)[:, None, None]
        if not visibility.any():
            zero = (
                means_2d.sum() + sum(parameter.sum() for parameter in model.parameters())
            ) * 0.0
            return RenderOutput(
                rgb=background_tensor.expand(3, camera.height, camera.width) + zero,
                alpha=model.means.new_zeros((1, camera.height, camera.width)) + zero,
                depth=model.means.new_zeros((1, camera.height, camera.width)) + zero,
                radii=output_radii.detach(),
                means_2d=means_2d,
                visibility=visibility,
            )

        camera_center = -(rotation.T @ view[:3, 3])
        directions = F.normalize(model.means - camera_center, dim=-1)
        colors = (evaluate_sh(model.sh_coefficients, directions, self.sh_degree) + 0.5)
        colors = colors.clamp_min(0.0)

        visible_indices = visibility.nonzero(as_tuple=False).squeeze(-1)
        visible_indices = visible_indices[depths[visible_indices].argsort()]
        centers = means_2d[visible_indices]
        covariance = covariance_2d[visible_indices]
        inverse_covariance = torch.linalg.inv(covariance)
        visible_radii = radii[visible_indices]

        ys = torch.arange(camera.height, dtype=dtype, device=device) + 0.5
        xs = torch.arange(camera.width, dtype=dtype, device=device) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack([grid_x, grid_y], dim=-1)
        offsets = pixels[None] - centers[:, None, None]
        mahalanobis = torch.einsum(
            "nhwi,nij,nhwj->nhw", offsets, inverse_covariance, offsets
        )
        in_footprint = (
            offsets[..., 0].abs() <= visible_radii[:, None, None]
        ) & (offsets[..., 1].abs() <= visible_radii[:, None, None])
        gaussian = torch.exp(-0.5 * mahalanobis) * in_footprint
        alphas = (
            model.opacities[visible_indices, 0, None, None] * gaussian
        ).clamp(max=0.999)

        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alphas[:1]), 1.0 - alphas[:-1]], dim=0),
            dim=0,
        )
        weights = transmittance * alphas
        accumulated_alpha = weights.sum(dim=0)
        rgb = torch.einsum("nhw,nc->chw", weights, colors[visible_indices])
        rgb = rgb + (1.0 - accumulated_alpha)[None] * background_tensor
        weighted_depth = torch.einsum("nhw,n->hw", weights, depths[visible_indices])
        expected_depth = torch.where(
            accumulated_alpha > 0,
            weighted_depth / accumulated_alpha.clamp_min(1e-8),
            torch.zeros_like(weighted_depth),
        )
        return RenderOutput(
            rgb=rgb,
            alpha=accumulated_alpha[None],
            depth=expected_depth[None],
            radii=output_radii.detach(),
            means_2d=means_2d,
            visibility=visibility,
        )
