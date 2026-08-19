import math

import pytest
import torch

from gaussian_splatting.metrics import psnr


def test_psnr_is_infinite_for_identical_images() -> None:
    image = torch.zeros(3, 4, 4)
    assert math.isinf(psnr(image, image))


def test_psnr_for_known_error() -> None:
    prediction = torch.zeros(1)
    target = torch.full((1,), 0.1)
    assert psnr(prediction, target) == pytest.approx(20.0)

