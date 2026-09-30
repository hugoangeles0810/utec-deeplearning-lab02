"""Device selection (AGENTS.md §4): ``auto`` picks cuda → mps → cpu; bf16 only on CUDA (D-004)."""

from typing import Literal

import torch

DeviceName = Literal["auto", "cuda", "mps", "cpu"]


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


def effective_amp(amp: Literal["bf16", "none"], device: torch.device) -> Literal["bf16", "none"]:
    """Mixed precision actually used: bf16 autocast only on CUDA; MPS and CPU run float32 (D-004)."""
    return "bf16" if amp == "bf16" and device.type == "cuda" else "none"
