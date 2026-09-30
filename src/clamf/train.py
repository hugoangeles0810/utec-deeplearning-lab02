"""Training entrypoint (paper Sec. 3.2 and Table 3; D-004, D-012).

    uv run python -m clamf.train --config configs/experiments/clamf.yaml
    uv run python -m clamf.train --config configs/experiments/clamf.yaml --resume <mlflow run id>

One global model over every basin (D-004): Adam with a constant learning rate, the loss of
``train.loss`` on the normalized forecast, val loss after every epoch, early stopping on it, and
``best.pt``/``last.pt`` in ``<train.checkpoint_dir>/<run id>/``. ``--resume`` continues a run from
its ``last.pt`` with the same YAML. Everything is logged to MLflow (AGENTS.md §6).
"""

import argparse
import logging
import time
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from clamf.config import DEFAULT_BASE, Config, load_config, to_dict
from clamf.data.dataset import DeviceLoader, batch_to_device, build_dataloaders, meteo_channels
from clamf.data.prepare import SCALERS_FILE
from clamf.losses import LossFn, get_loss
from clamf.models.clamf_former import build_model
from clamf.utils.checkpoint import (
    BEST,
    LAST,
    load_training_state,
    run_checkpoint_dir,
    save_model,
    save_training_state,
)
from clamf.utils.device import AmpName, DeviceName, autocast, effective_amp, resolve_device
from clamf.utils.early_stopping import EarlyStopping
from clamf.utils.seed import seed_everything
from clamf.utils.tracking import log_epoch, log_run_setup, start_run, tau_summary

log = logging.getLogger(__name__)

Loader = DeviceLoader | DataLoader


def train(
    cfg: Config,
    config_path: str | Path,
    run_id: str | None = None,
    device_name: DeviceName | None = None,
) -> str:
    """Train ``cfg`` (or resume MLflow run ``run_id``) and return the run id.

    ``device_name`` overrides ``cfg.device`` without changing the config, so a run can resume on
    another machine.
    """
    device = resolve_device(device_name or cfg.device)
    amp = effective_amp(cfg.train.amp, device)
    seed_everything(cfg.seed)
    loaders = build_dataloaders(cfg, device)
    model = build_model(cfg, n_meteo=meteo_channels(loaders["train"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)  # Table 3, D-012
    loss_fn = get_loss(cfg.train.loss)
    early_stopping = EarlyStopping.from_config(cfg.train)
    generator = loaders["train"].generator
    config = to_dict(cfg)

    with start_run(cfg, run_id) as run:
        ckpt_dir = run_checkpoint_dir(cfg.train.checkpoint_dir, run.info.run_id)
        if run_id is None:
            log_run_setup(cfg, config_path, device, amp)
            mlflow.log_artifact(str(Path(cfg.data.processed_dir) / SCALERS_FILE), "data")
            start_epoch = 0
        else:
            start_epoch = 1 + load_training_state(
                ckpt_dir / LAST,
                model=model,
                optimizer=optimizer,
                early_stopping=early_stopping,
                generator=generator,
                config=config,
            )
            log.info("resuming run %s at epoch %d", run.info.run_id, start_epoch)
        log.info("run %s on %s (amp: %s)", run.info.run_id, device, amp)

        epochs_run = start_epoch
        for epoch in range(start_epoch, cfg.train.max_epochs):
            if early_stopping.should_stop:  # also when resuming a run that already stopped
                break
            start = time.perf_counter()
            train_loss = train_epoch(model, loaders["train"], optimizer, loss_fn, device, amp)
            val_loss, tau = validate(model, loaders["val"], loss_fn, device, amp)
            seconds = time.perf_counter() - start

            if early_stopping.step(val_loss, epoch):
                save_model(ckpt_dir / BEST, model, epoch, config)
            save_training_state(
                ckpt_dir / LAST,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                early_stopping=early_stopping,
                generator=generator,
                config=config,
            )
            metrics = {"loss/train": train_loss, "loss/val": val_loss, "time/epoch_s": seconds}
            if tau is not None:
                metrics |= tau_summary(tau)
            log_epoch(metrics, epoch)
            epochs_run = epoch + 1
            log.info(
                "epoch %d  train %.4f  val %.4f  best %.4f (epoch %d)  %.1f s",
                epoch, train_loss, val_loss, early_stopping.best_loss, early_stopping.best_epoch,
                seconds,
            )  # fmt: skip

        _log_summary(early_stopping, epochs_run, ckpt_dir / BEST)
        return run.info.run_id


def train_epoch(
    model: nn.Module,
    loader: Loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    amp: AmpName,
) -> float:
    """One pass over ``loader``; returns the sample-weighted mean training loss."""
    model.train()
    total = torch.zeros((), device=device)
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            pred = model(batch["enc_x"], batch["dec_x"]).pred
        loss = loss_fn(pred, batch["target"])  # float32, outside autocast (D-004)
        loss.backward()
        optimizer.step()
        total += loss.detach() * len(pred)
        n += len(pred)
    return total.item() / n  # a single device sync per epoch


@torch.no_grad()
def validate(
    model: nn.Module, loader: Loader, loss_fn: LossFn, device: torch.device, amp: AmpName
) -> tuple[float, torch.Tensor | None]:
    """Sample-weighted mean loss over ``loader`` and the predicted lags ``(layers, N, L)`` of every
    sample (``None`` without the LAAM; D-001)."""
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    taus = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        with autocast(device, amp):
            out = model(batch["enc_x"], batch["dec_x"])
        total += loss_fn(out.pred, batch["target"]) * len(out.pred)
        n += len(out.pred)
        if out.tau is not None:
            taus.append(out.tau)
    return total.item() / n, torch.cat(taus, dim=1) if taus else None


def _log_summary(early_stopping: EarlyStopping, epochs_run: int, best_path: Path) -> None:
    """Best epoch (0-based, like the metric steps), epochs run and best val loss (D-004 §6), and
    ``best.pt``."""
    mlflow.log_metrics(
        {
            "best_epoch": early_stopping.best_epoch,
            "epochs_run": epochs_run,
            "best_val_loss": early_stopping.best_loss,
        }
    )
    mlflow.set_tag("stopped_early", str(early_stopping.should_stop).lower())
    if best_path.exists():
        mlflow.log_artifact(str(best_path), "checkpoints")
    else:
        log.warning("no val loss improved (NaN?); there is no best checkpoint")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CLAMF-Former (or an ablation/baseline).")
    parser.add_argument("--config", required=True, help="experiment YAML (overrides the base)")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="base YAML")
    parser.add_argument("--resume", metavar="RUN_ID", help="MLflow run id to resume")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="override")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, base=args.base)
    run_id = train(cfg, args.config, run_id=args.resume, device_name=args.device)
    print(f"run id: {run_id}")


if __name__ == "__main__":
    main()
