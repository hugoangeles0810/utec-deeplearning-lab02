"""Resumable training checkpoints and best-model weights (D-004 §5).

Each run keeps two files in ``<train.checkpoint_dir>/<mlflow run id>/``:

- ``last.pt``: everything needed to resume after a finished epoch (model, optimizer, epoch, early
  stopping, the RNG states of ``random``, ``numpy`` and ``torch`` on every device, and the state of
  the train-loader generator that shuffles the batches) plus the resolved config.
- ``best.pt``: model weights of the best val epoch, with that epoch and the config.

Files are written to a temporary name and renamed, so a pod killed mid-write keeps the previous
checkpoint. Only tensors and plain Python types are stored, so they load with
``torch.load(weights_only=True)``.
"""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from clamf.utils.early_stopping import EarlyStopping

LAST = "last.pt"
BEST = "best.pt"


def run_checkpoint_dir(checkpoint_dir: str | Path, run_id: str) -> Path:
    """``<checkpoint_dir>/<run_id>``, one folder per MLflow run."""
    return Path(checkpoint_dir) / run_id


def save_training_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    early_stopping: EarlyStopping,
    generator: torch.Generator,
    config: dict[str, Any],
) -> None:
    """Write ``last.pt`` at the end of ``epoch`` (0-based, already finished)."""
    _atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "early_stopping": early_stopping.state_dict(),
            "rng": rng_state(generator),
            "config": config,
        },
        Path(path),
    )


def load_training_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    generator: torch.Generator,
    config: dict[str, Any],
) -> int:
    """Restore ``last.pt`` in place and return the last finished epoch.

    Raises ``ValueError`` if the checkpoint was written with a different config.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != config:
        raise ValueError(f"{path} was written with a different config; resume with the same YAML")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])  # moves the state to the parameters' device
    early_stopping.load_state_dict(state["early_stopping"])
    set_rng_state(state["rng"], generator)
    return state["epoch"]


def save_model(path: str | Path, model: nn.Module, epoch: int, config: dict[str, Any]) -> None:
    """Write ``best.pt``: the weights of ``model`` at ``epoch``."""
    _atomic_save({"model": model.state_dict(), "epoch": epoch, "config": config}, Path(path))


def load_model(path: str | Path, model: nn.Module, config: dict[str, Any] | None = None) -> int:
    """Load the weights of ``best.pt`` into ``model`` and return their epoch.

    ``config`` is a subset of the resolved config (e.g. ``{"model": {...}, "data":
    {"history_hours": 336}}``); raises ``ValueError`` if any of its keys differs from the config
    the checkpoint was written with. Keys left out are not checked.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if config is not None:
        differ = _mismatches(config, state["config"])
        if differ:
            raise ValueError(f"{path} was written with a different config: {', '.join(differ)}")
    model.load_state_dict(state["model"])
    return state["epoch"]


def rng_state(generator: torch.Generator) -> dict[str, Any]:
    """RNG states of ``random``, ``numpy``, ``torch`` (CPU, CUDA, MPS) and ``generator``."""
    _, keys, pos, has_gauss, cached_gaussian = np.random.get_state(legacy=True)
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "keys": torch.from_numpy(keys.astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
        "generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    """Inverse of :func:`rng_state`; device states missing on this machine are skipped."""
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state(
        (
            "MT19937",
            np_state["keys"].numpy().astype(np.uint32),
            np_state["pos"],
            np_state["has_gauss"],
            np_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    generator.set_state(state["generator"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def _atomic_save(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _mismatches(expected: dict[str, Any], actual: dict[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys of ``expected`` whose value differs in ``actual`` (nested dicts as subsets)."""
    out = []
    for key, value in expected.items():
        other = actual.get(key)
        if isinstance(value, dict) and isinstance(other, dict):
            out += _mismatches(value, other, f"{prefix}{key}.")
        elif other != value:
            out.append(prefix + key)
    return out
