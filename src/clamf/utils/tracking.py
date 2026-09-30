"""MLflow tracking (AGENTS.md §6; D-001, D-004).

The tracking URI comes from ``MLFLOW_TRACKING_URI`` (MLflow's default is ``sqlite:///mlflow.db``);
the experiment is ``logging.experiment`` and every training run is one MLflow run. At the start of a
run :func:`log_run_setup` records the flattened config as params, the device, GPU and effective AMP,
the git commit, and the YAML used; per-epoch values go through :func:`log_epoch`.
"""

import subprocess
from pathlib import Path
from typing import Any

import mlflow
import torch

from clamf.config import Config, to_dict

TAU_QUANTILES = {"p50": 0.5, "p90": 0.9}
_QUANTILE_MAX_ELEMENTS = 2**24  # torch.quantile input limit


def start_run(cfg: Config, run_id: str | None = None) -> mlflow.ActiveRun:
    """Start a run in ``logging.experiment``, or resume ``run_id`` (D-004: resumable training)."""
    mlflow.set_experiment(cfg.logging.experiment)
    return mlflow.start_run(run_id=run_id)


def flatten_config(cfg: Config) -> dict[str, Any]:
    """Config as ``{"model.d_model": 64, ...}``; tuples become lists so they log as ``[1, 24]``."""
    flat: dict[str, Any] = {}

    def visit(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                visit(value, f"{prefix}{key}.")
            else:
                flat[prefix + key] = list(value) if isinstance(value, tuple) else value

    visit(to_dict(cfg), "")
    return flat


def git_commit(repo: str | Path = ".") -> tuple[str, bool]:
    """HEAD commit and whether the working tree has uncommitted changes (``"unknown"`` outside
    a git repository)."""
    try:
        sha = _git(repo, "rev-parse", "HEAD")
        dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, dirty


def environment_info(device: torch.device, amp: str) -> dict[str, str]:
    """Device, GPU name, effective AMP and torch version of this run (D-004)."""
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    return {"device": device.type, "gpu": gpu, "amp": amp, "torch_version": torch.__version__}


def log_run_setup(cfg: Config, config_path: str | Path, device: torch.device, amp: str) -> None:
    """Log params (flattened config and environment), git tags and the YAML files of the run."""
    mlflow.log_params({**flatten_config(cfg), **environment_info(device, amp)})
    sha, dirty = git_commit()
    mlflow.set_tags({"git_commit": sha, "git_dirty": str(dirty).lower()})
    mlflow.log_artifact(str(config_path), artifact_path="config")
    mlflow.log_dict(_plain(to_dict(cfg)), "config/resolved.yaml")


def log_epoch(metrics: dict[str, float], epoch: int) -> None:
    """Per-epoch metrics (losses, seconds, tau summary) with ``epoch`` as the step."""
    mlflow.log_metrics(metrics, step=epoch)


def tau_summary(tau: torch.Tensor) -> dict[str, float]:
    """Distribution of the predicted lags (D-001): mean, p50, p90 and max, overall and per layer.

    ``tau`` is ``(decoder_layers, ...)`` in time steps (hours), e.g. ``(layers, B, L)`` from
    :attr:`~clamf.models.clamf_former.ModelOutput.tau` or several batches concatenated along ``B``.
    Keys look like ``tau/mean`` and ``tau/layer0/p90``.
    """
    per_layer = tau.detach().float().flatten(1).cpu()
    summary = _describe(per_layer.flatten(), "tau")
    for i, values in enumerate(per_layer):
        summary |= _describe(values, f"tau/layer{i}")
    return summary


def _describe(values: torch.Tensor, prefix: str) -> dict[str, float]:
    if values.numel() > _QUANTILE_MAX_ELEMENTS:  # evenly spaced subsample keeps the quantiles
        step = -(-values.numel() // _QUANTILE_MAX_ELEMENTS)
        sample = values[::step]
    else:
        sample = values
    quantiles = torch.quantile(sample, torch.tensor(list(TAU_QUANTILES.values())))
    return {
        f"{prefix}/mean": values.mean().item(),
        **{f"{prefix}/{k}": q.item() for k, q in zip(TAU_QUANTILES, quantiles, strict=True)},
        f"{prefix}/max": values.max().item(),
    }


def _git(repo: str | Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _plain(node: Any) -> Any:
    """Tuples to lists so the resolved config dumps as plain YAML."""
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    if isinstance(node, (tuple, list)):
        return [_plain(v) for v in node]
    return node
