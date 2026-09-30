import pytest
import torch

from clamf.utils import device as device_mod
from clamf.utils.device import autocast, effective_amp, resolve_device


def fake_availability(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_mod.torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    fake_availability(monkeypatch, cuda, mps)
    assert resolve_device("auto") == torch.device(expected)


def test_cpu_is_always_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_availability(monkeypatch, cuda=False, mps=False)
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("name", ["cuda", "mps"])
def test_unavailable_accelerator_raises(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    fake_availability(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match=name):
        resolve_device(name)


@pytest.mark.parametrize(
    ("amp", "device", "expected"),
    [
        ("bf16", "cuda", "bf16"),
        ("bf16", "mps", "none"),
        ("bf16", "cpu", "none"),
        ("none", "cuda", "none"),
    ],
)
def test_bf16_only_on_cuda(amp: str, device: str, expected: str) -> None:
    assert effective_amp(amp, torch.device(device)) == expected


def test_autocast_runs_bf16_only_when_enabled() -> None:
    cpu = torch.device("cpu")
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    with autocast(cpu, "bf16"):
        assert (a @ b).dtype == torch.bfloat16
    with autocast(cpu, "none"):
        assert (a @ b).dtype == torch.float32
