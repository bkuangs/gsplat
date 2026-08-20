import torch

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.training.checkpoint import load_checkpoint, save_checkpoint


def _model(count: int) -> GaussianModel:
    quaternions = torch.zeros(count, 4)
    quaternions[:, 0] = 1.0
    return GaussianModel(
        means=torch.randn(count, 3),
        log_scales=torch.zeros(count, 3),
        quaternions=quaternions,
        opacity_logits=torch.zeros(count, 1),
        sh_coefficients=torch.zeros(count, 1, 3),
    )


def test_checkpoint_reconstructs_saved_gaussian_count_and_optimizer(tmp_path) -> None:
    model = _model(count=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.means.sum().backward()
    optimizer.step()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=12, metadata={"scene": "tiny"})

    loaded_model, loaded_optimizer, step, metadata = load_checkpoint(
        path,
        optimizer_factory=lambda restored: torch.optim.Adam(restored.parameters(), lr=0.01),
    )

    assert loaded_model.means.shape == (5, 3)
    torch.testing.assert_close(loaded_model.means, model.means)
    assert loaded_optimizer is not None
    assert len(loaded_optimizer.state) == len(optimizer.state)
    assert step == 12
    assert metadata == {"scene": "tiny"}
