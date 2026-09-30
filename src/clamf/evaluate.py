"""Evaluation entrypoint (paper Sec. 3.2 and Eq. 16-20; D-003, D-016).

    uv run python -m clamf.evaluate --config configs/experiments/clamf.yaml --run-id <mlflow run id>
    uv run python -m clamf.evaluate --config configs/experiments/clamf.yaml --run-id <id> --split test

Loads ``best.pt`` of a training run (from ``<train.checkpoint_dir>/<run id>/`` or, if it is not
there, from the run's artifacts), forecasts one split, denormalizes to mm/h and logs into that same
MLflow run:

- ``<split>/<metric>_{median,mean,n_excluded}``: main pooled metrics per basin with the D-003
  exclusion of degenerate basins, aggregated over basins;
- ``<split>_literal/...``: the same over every basin (thresholds at 0), as a reference;
- ``<split>/lead/{nse,rmse}_{median,mean}`` with the lead hour (1..H) as the step;
- ``<split>/checkpoint_epoch``, ``<split>/n_basins`` and ``<split>/n_windows``;
- artifacts in ``eval/<split>/``: ``predictions.parquet`` (one row per window and lead),
  ``basin_metrics.csv`` and ``lead_metrics.csv``.

Val is reported provisionally until test has its future meteorology (D-003, D-013); ``--split
test`` fails with :class:`~clamf.data.dataset.MissingFutureMeteoError` until then.
"""

import argparse
import dataclasses
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from clamf.config import DEFAULT_BASE, Config, EvalConfig, load_config, to_dict
from clamf.data.dataset import (
    DeviceLoader,
    batch_to_device,
    build_loader,
    load_scalers,
    meteo_channels,
)
from clamf.metrics import METRICS, basin_metrics, lead_metrics, summarize
from clamf.models.clamf_former import build_model
from clamf.utils.checkpoint import BEST, load_model, run_checkpoint_dir
from clamf.utils.device import AmpName, DeviceName, autocast, effective_amp, resolve_device
from clamf.utils.tracking import open_run

log = logging.getLogger(__name__)

Split = Literal["val", "test"]


@dataclass(frozen=True)
class Forecast:
    """Forecasts of a split in mm/h: ``obs``/``pred`` ``(N, H)``, ``basin_id``/``row_id`` ``(N,)``."""

    obs: np.ndarray
    pred: np.ndarray
    basin_id: np.ndarray
    row_id: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Long format: one row per (window, lead hour 1..H)."""
        n, horizon = self.obs.shape
        return pd.DataFrame(
            {
                "row_id": np.repeat(self.row_id, horizon),
                "basin_id": np.repeat(self.basin_id, horizon),
                "lead": np.tile(np.arange(1, horizon + 1, dtype=np.int16), n),
                "obs": self.obs.ravel(),
                "pred": self.pred.ravel(),
            }
        )


@dataclass(frozen=True)
class Evaluation:
    """Summary metrics (MLflow keys without the split prefix) and the per-basin/per-lead tables."""

    summary: dict[str, float]
    literal: dict[str, float]
    basins: pd.DataFrame
    leads: pd.DataFrame


def evaluate(
    cfg: Config, run_id: str, split: Split = "val", device_name: DeviceName | None = None
) -> dict[str, float]:
    """Evaluate the best checkpoint of ``run_id`` on ``split`` and log it into that run.

    ``model``, the window lengths and the normalization of ``cfg`` must match the ones the run was
    trained with; paths, batch sizes, ``eval`` and the rest may differ. Returns the logged scalar
    metrics.
    """
    device = resolve_device(device_name or cfg.device)
    amp = effective_amp(cfg.train.amp, device)
    loader = build_loader(cfg, split, device)
    model = build_model(cfg, n_meteo=meteo_channels(loader)).to(device)

    with open_run(run_id):
        path = checkpoint_path(cfg, run_id)
        epoch = load_model(path, model, _trained_with(cfg))
        log.info("run %s, epoch %d, split %s on %s (amp: %s)", run_id, epoch, split, device, amp)

        forecast = predict(model, loader, device, amp, cfg.data.processed_dir)
        result = score(forecast, cfg.eval)
        metrics = {
            f"{split}/checkpoint_epoch": epoch,
            f"{split}/n_basins": len(result.basins),
            f"{split}/n_windows": len(forecast.obs),
            **{f"{split}/{k}": v for k, v in result.summary.items()},
            **{f"{split}_literal/{k}": v for k, v in result.literal.items()},
        }
        mlflow.log_metrics(metrics)
        for lead, row in result.leads.iterrows():
            mlflow.log_metrics({f"{split}/lead/{k}": v for k, v in row.items()}, step=int(lead))
        _log_tables(forecast, result, split)
    _print_summary(result, split)
    return metrics


def checkpoint_path(cfg: Config, run_id: str) -> Path:
    """Local ``best.pt`` of ``run_id``, or the copy logged as an artifact (e.g. trained on a pod)."""
    local = run_checkpoint_dir(cfg.train.checkpoint_dir, run_id) / BEST
    if local.exists():
        return local
    log.info("%s not found; downloading it from the run artifacts", local)
    return Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"checkpoints/{BEST}")
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DeviceLoader | DataLoader,
    device: torch.device,
    amp: AmpName,
    processed_dir: str | Path,
) -> Forecast:
    """Forecast every window of ``loader`` and denormalize forecast and target to mm/h."""
    model.eval()
    parts: dict[str, list[torch.Tensor]] = {"pred": [], "target": [], "basin_id": [], "row_id": []}
    for batch in loader:
        batch = batch_to_device(batch, device)
        with autocast(device, amp):
            pred = model(batch["enc_x"], batch["dec_x"]).pred
        parts["pred"].append(pred.float())
        for key in ("target", "basin_id", "row_id"):
            parts[key].append(batch[key])
    arrays = {k: torch.cat(v).cpu().numpy() for k, v in parts.items()}
    scalers = load_scalers(processed_dir)
    basin_id = arrays["basin_id"]
    return Forecast(
        obs=scalers.denormalize_q(arrays["target"], basin_id),
        pred=scalers.denormalize_q(arrays["pred"], basin_id),
        basin_id=basin_id,
        row_id=arrays["row_id"],
    )


def score(forecast: Forecast, cfg: EvalConfig) -> Evaluation:
    """Main and literal pooled metrics per basin (D-003) and NSE/RMSE per lead hour."""
    literal_cfg = dataclasses.replace(cfg, min_obs_std=0.0, min_obs_mean=0.0)
    args = (forecast.obs, forecast.pred, forecast.basin_id)
    basins = basin_metrics(*args, cfg)
    literal = basin_metrics(*args, literal_cfg)
    return Evaluation(
        summary=summarize(basins),
        literal=summarize(literal),
        basins=basins.join(literal.add_suffix("_literal")),
        leads=lead_metrics(*args, cfg),
    )


def _trained_with(cfg: Config) -> dict[str, Any]:
    """Settings that must match the checkpoint: the model and what defines its inputs."""
    config = to_dict(cfg)
    data = {k: config["data"][k] for k in ("history_hours", "horizon_hours", "normalization")}
    return {"model": config["model"], "data": data}


def _log_tables(forecast: Forecast, result: Evaluation, split: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        forecast.to_frame().to_parquet(out / "predictions.parquet", index=False)
        result.basins.to_csv(out / "basin_metrics.csv")
        result.leads.to_csv(out / "lead_metrics.csv")
        mlflow.log_artifacts(str(out), f"eval/{split}")


def _print_summary(result: Evaluation, split: str) -> None:
    rows = {
        name: [result.summary[f"{name}_{s}"] for s in ("median", "mean")]
        + [result.summary[f"{name}_n_excluded"]]
        for name in METRICS
    }
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["median", "mean", "excluded"])
    print(f"{split} ({len(result.basins)} basins, D-003 exclusion):")
    print(table.to_string(float_format="{:.4f}".format))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CLAMF-Former run.")
    parser.add_argument("--config", required=True, help="experiment YAML the run was trained with")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="base YAML")
    parser.add_argument("--run-id", required=True, help="MLflow run id of the training run")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="default: val")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="override")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, base=args.base)
    evaluate(cfg, args.run_id, split=args.split, device_name=args.device)


if __name__ == "__main__":
    main()
