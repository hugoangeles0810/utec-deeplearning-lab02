"""Device selection (AGENTS.md §4): ``auto`` picks cuda → mps → cpu; bf16 only on CUDA (D-004)."""

import contextlib
from typing import Literal

import torch

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
AmpName = Literal["bf16", "none"]


def resolve_device(name: DeviceName = "auto") -> torch.device:
    """Return the requested device, or the best available one for ``auto``.

    Raises ``RuntimeError`` if an explicitly requested accelerator is not available.
    """
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if name == "auto":
        return torch.device(next((d for d, ok in available.items() if ok), "cpu"))
    if name != "cpu" and not available[name]:
        raise RuntimeError(f"device '{name}' was requested but is not available")
    return torch.device(name)


def effective_amp(amp: AmpName, device: torch.device) -> AmpName:
    """Mixed precision actually used: bf16 autocast only on CUDA; MPS and CPU run float32 (D-004)."""
    return "bf16" if amp == "bf16" and device.type == "cuda" else "none"


def autocast(device: torch.device, amp: AmpName) -> contextlib.AbstractContextManager:
    """Context for the forward pass: bf16 autocast when ``amp`` (already effective) is ``"bf16"``,
    otherwise a no-op. Losses and metrics run outside it, in float32 (D-004)."""
    if amp == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
