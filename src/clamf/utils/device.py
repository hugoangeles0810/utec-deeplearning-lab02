"""Device selection (AGENTS.md §4): ``auto`` picks cuda → mps → cpu."""

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
