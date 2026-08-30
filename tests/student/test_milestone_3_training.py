import torch

from gaussian_splatting.training.losses import photometric_loss


def test_photometric_loss_is_zero_for_identical_images() -> None:
    image = torch.rand(1, 3, 16, 16)
    loss = photometric_loss(image, image)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_photometric_loss_backpropagates() -> None:
    prediction = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    photometric_loss(prediction, target).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
