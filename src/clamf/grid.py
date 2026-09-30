"""Run the experiment grid end to end, resuming whatever is left (D-004, D-018; docs/runpod.md).

    uv run python -m clamf.grid
    uv run python -m clamf.grid --configs configs/experiments/dev.yaml

For every YAML, in order, the run named ``logging.run_name`` in ``logging.experiment`` is:

- trained from scratch if it does not exist;
- resumed from its ``last.pt`` if it exists but has no ``epochs_run`` metric (training was cut
  short); a run cut before its first checkpoint is deleted and trained again;
- left as is if training finished;

and then evaluated on val (:func:`clamf.evaluate.evaluate`) unless it already has val metrics. Running
it again after a crash or a pod restart picks up where it stopped. A failing run is reported and the
next one still runs; the exit code is non-zero if any failed.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient

from clamf.config import DEFAULT_BASE, Config, load_config
from clamf.evaluate import evaluate
from clamf.train import train
from clamf.utils.checkpoint import LAST, run_checkpoint_dir
from clamf.utils.device import DeviceName

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("configs/experiments")
# The two runs of the main comparison first (CLAMF-Former vs baseline), then the ablations.
GRID = tuple(
    EXPERIMENTS_DIR / f"{name}.yaml"
    for name in ("clamf", "vanilla", "claam", "msfm", "cam", "laam")
)
DONE_METRIC = "epochs_run"  # logged when training ends (clamf.train._log_summary)
EVALUATED_METRIC = "val/nse_median"  # logged by clamf.evaluate on val


class DuplicateRunError(RuntimeError):
    """Raised when an experiment has more than one active run with the same name (D-018)."""


@dataclass(frozen=True)
class GridResult:
    """Outcome of one YAML: ``status`` is ``trained``, ``resumed``, ``done`` or ``failed``."""

    config: Path
    run_name: str
    status: str
    run_id: str | None = None
    error: str | None = None


def run_grid(
    configs: list[Path], base: Path | None = DEFAULT_BASE, device_name: DeviceName | None = None
) -> list[GridResult]:
    """Train (or resume) and evaluate every config in order; see the module docstring."""
    results = []
    for path in configs:
        try:
            results.append(_run_one(load_config(path, base=base), path, device_name))
        except Exception as exc:  # keep going with the next run; KeyboardInterrupt still stops
            log.exception("%s failed", path)
            results.append(GridResult(path, path.stem, "failed", error=repr(exc)))
    return results


def find_run(cfg: Config) -> Run | None:
    """The active run named ``logging.run_name`` in ``logging.experiment``, if any."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.logging.experiment)
    if experiment is None:
        return None
    name = cfg.logging.run_name.replace("'", "\\'")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{name}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    if len(runs) > 1:
        ids = ", ".join(r.info.run_id for r in runs)
        raise DuplicateRunError(
            f"{len(runs)} runs named {cfg.logging.run_name!r} in {cfg.logging.experiment!r} "
            f"({ids}); keep one per name (D-018)"
        )
    return runs[0] if runs else None


def _run_one(cfg: Config, path: Path, device_name: DeviceName | None) -> GridResult:
    if not cfg.logging.run_name:
        raise ValueError(f"{path}: the grid needs logging.run_name to find the run again")
    run = find_run(cfg)
    if run is None:
        status, run_id = "trained", train(cfg, path, device_name=device_name)
    elif DONE_METRIC in run.data.metrics:
        status, run_id = "done", run.info.run_id
    elif (run_checkpoint_dir(cfg.train.checkpoint_dir, run.info.run_id) / LAST).exists():
        log.info("resuming %s (%s)", cfg.logging.run_name, run.info.run_id)
        status, run_id = (
            "resumed",
            train(cfg, path, run_id=run.info.run_id, device_name=device_name),
        )
    else:
        log.warning("run %s stopped before its first checkpoint; starting over", run.info.run_id)
        MlflowClient().delete_run(run.info.run_id)
        status, run_id = "trained", train(cfg, path, device_name=device_name)

    if EVALUATED_METRIC not in MlflowClient().get_run(run_id).data.metrics:
        evaluate(cfg, run_id, split="val", device_name=device_name)
    return GridResult(path, cfg.logging.run_name, status, run_id)


def _print_summary(results: list[GridResult]) -> None:
    client = MlflowClient()
    print(f"{'run':<10} {'status':<8} {'run id':<32} {'best epoch':>10} {'val NSE med':>11}")
    for r in results:
        metrics = client.get_run(r.run_id).data.metrics if r.run_id else {}
        best = metrics.get("best_epoch", float("nan"))
        nse = metrics.get(EVALUATED_METRIC, float("nan"))
        print(f"{r.run_name:<10} {r.status:<8} {r.run_id or '-':<32} {best:>10.0f} {nse:>11.4f}")
        if r.error:
            print(f"  {r.error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the experiment grid.")
    parser.add_argument(
        "--configs", nargs="+", type=Path, default=list(GRID), help="YAMLs, in order"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="base YAML")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="override")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    results = run_grid(args.configs, base=args.base, device_name=args.device)
    _print_summary(results)
    if any(r.status == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
