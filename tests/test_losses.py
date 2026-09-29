import numpy as np
import pytest
import torch

from clamf.losses import freqmae, get_loss


def test_zero_for_perfect_prediction() -> None:
    target = torch.randn(3, 48)
    assert freqmae(target, target).item() == 0.0


def test_non_negative() -> None:
    torch.manual_seed(0)
    for _ in range(5):
        assert freqmae(torch.randn(4, 48), torch.randn(4, 48)).item() >= 0.0


def test_manual_small_cases() -> None:
    zeros = torch.zeros(1, 4)
    impulse = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # DFT = [1, 1, 1, 1]
    alternating = torch.tensor([[1.0, -1.0, 1.0, -1.0]])  # DFT = [0, 0, 4, 0]
    assert freqmae(impulse, zeros).item() == pytest.approx(4.0)
    assert freqmae(alternating, zeros).item() == pytest.approx(4.0)
    # batch mean of the per-series sums
    both = freqmae(torch.cat([impulse, 2 * alternating]), torch.zeros(2, 4))
    assert both.item() == pytest.approx(6.0)


def test_matches_eq_15_with_explicit_dft() -> None:
    rng = np.random.default_rng(0)
    y, y_hat = rng.normal(size=(2, 6)), rng.normal(size=(2, 6))
    n = np.arange(6)
    dft = np.exp(-2j * np.pi * np.outer(n, n) / 6)  # Eq. (14)
    expected = np.abs(y_hat @ dft.T - y @ dft.T).sum(axis=1).mean()
    got = freqmae(torch.tensor(y_hat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    assert got.item() == pytest.approx(expected, rel=1e-5)


def test_gradient_is_finite() -> None:
    target = torch.randn(2, 48)
    for start in (target.clone(), torch.randn(2, 48)):  # includes a perfect prediction
        pred = start.requires_grad_()
        freqmae(pred, target).backward()
        assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("name", ["freqmae", "mse", "mae"])
def test_get_loss_returns_float32(name: str) -> None:
    pred = torch.randn(2, 8, dtype=torch.bfloat16)
    target = torch.randn(2, 8)
    loss = get_loss(name)(pred, target)
    assert loss.dtype == torch.float32
    assert loss.ndim == 0


def test_get_loss_matches_torch_losses() -> None:
    pred, target = torch.randn(2, 8), torch.randn(2, 8)
    assert get_loss("mse")(pred, target).item() == pytest.approx(((pred - target) ** 2).mean())
    assert get_loss("mae")(pred, target).item() == pytest.approx((pred - target).abs().mean())


def test_get_loss_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="unknown loss"):
        get_loss("huber")
    with pytest.raises(ValueError, match="shapes differ"):
        get_loss("mse")(torch.zeros(2, 8), torch.zeros(2, 7))
