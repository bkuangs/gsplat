import pytest

from gaussian_splatting.rendering.torch_backend import TorchRasterizer


def test_torch_rasterizer_tracks_active_sh_degree() -> None:
    rasterizer = TorchRasterizer(sh_degree=2)
    assert rasterizer.sh_degree == 2


def test_torch_rasterizer_rejects_negative_sh_degree() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TorchRasterizer(sh_degree=-1)
