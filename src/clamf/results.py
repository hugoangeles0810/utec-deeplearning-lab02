"""Read the experiment grid back from MLflow and build the result tables (AGENTS.md §6; D-003, D-018).

Every number of the result tables comes from the runs of ``clamf-grid``: the metrics that
:mod:`clamf.evaluate` logged (``<split>/...`` and ``<split>_literal/...``), the training history
that :mod:`clamf.train` logged, and the ``eval/<split>/`` artifacts. Nothing is copied by hand.

    grid = Grid.load(tracking_uri="sqlite:///results/runpod/mlflow.db")
    grid.paper_table("table5")                      # CLAMF-1..CLAMF on val
    paired_comparison(grid.basin_metrics(), [("clamf", "vanilla")])

The split is a parameter: val is reported provisionally until test has its future meteorology
(D-003, D-013); with ``split="test"`` the same tables are read from the ``test/`` metrics.
"""

import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from clamf.grid import DuplicateRunError
from clamf.metrics import METRICS

GRID_EXPERIMENT = "clamf-grid"
GRID_RUNS = ("vanilla", "cam", "laam", "claam", "msfm", "clamf")
FLAGS = {
    "MSFM": "model.use_msfm",
    "CAM": "model.use_causal_encoder",
    "LAAM": "model.use_lag_aware_cross_attn",
}
# Rows of the paper tables and the run each one reads (docs/experiments.md, D-018).
PAPER_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "table5": (
        ("CLAMF-1", "vanilla"),
        ("CLAMF-2", "msfm"),
        ("CLAMF-3", "claam"),
        ("CLAMF", "clamf"),
    ),
    "table6": (("CLAAM-1", "vanilla"), ("CLAAM-2", "cam"), ("CLAAM-3", "laam"), ("CLAAM", "claam")),
    "main": (("CLAMF-Former", "clamf"), ("Transformer vanilla", "vanilla")),
}
SUMMARY_STATS = ("median", "mean", "n_excluded")
# +1 if a larger value is better, -1 if a smaller one is; BIAS is compared by its absolute value.
_DIRECTION = {"nse": 1, "kge": 1, "rmse": -1, "tpe": -1, "bias": -1}


@dataclass(frozen=True)
class GridRun:
    """One finished run of the grid: its MLflow id and its latest params, metrics and tags."""

    name: str
    run_id: str
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]


class Grid:
    """The runs of the experiment grid, read from an MLflow tracking store."""

    def __init__(self, client: MlflowClient, runs: Mapping[str, GridRun]) -> None:
        self.client = client
        self.runs = dict(runs)

    @classmethod
    def load(
        cls,
        experiment: str = GRID_EXPERIMENT,
        names: Sequence[str] = GRID_RUNS,
        tracking_uri: str | None = None,
    ) -> "Grid":
        """The active run named after each of ``names`` in ``experiment``, in that order.

        ``tracking_uri`` defaults to ``MLFLOW_TRACKING_URI``. Raises if a run is missing or if a
        name has more than one active run (D-018).
        """
        client = MlflowClient(tracking_uri)
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            raise ValueError(f"experiment {experiment!r} not found in {client.tracking_uri}")
        found: dict[str, list] = {}
        for run in client.search_runs([exp.experiment_id], run_view_type=ViewType.ACTIVE_ONLY):
            found.setdefault(run.info.run_name, []).append(run)
        missing = [n for n in names if n not in found]
        if missing:
            raise ValueError(f"runs {missing} not found in experiment {experiment!r}")
        runs = {}
        for name in names:
            if len(found[name]) > 1:
                ids = ", ".join(r.info.run_id for r in found[name])
                raise DuplicateRunError(f"{len(found[name])} runs named {name!r} ({ids}; D-018)")
            run = found[name][0]
            runs[name] = GridRun(
                name, run.info.run_id, run.data.params, run.data.metrics, run.data.tags
            )
        return cls(client, runs)

    @property
    def names(self) -> list[str]:
        return list(self.runs)

    def __getitem__(self, name: str) -> GridRun:
        return self.runs[name]

    def flags(self) -> pd.DataFrame:
        """Which modules each run enables: one bool column per flag in :data:`FLAGS`."""
        rows = {
            name: {flag: run.params.get(key) == "True" for flag, key in FLAGS.items()}
            for name, run in self.runs.items()
        }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("run")

    def training(self) -> pd.DataFrame:
        """Training summary per run (D-004): best epoch and epochs run (0-based, as logged), best
        val loss, mean seconds per epoch, total training hours and whether it stopped early."""
        rows = {}
        for name, run in self.runs.items():
            times = self._history(run.run_id, "time/epoch_s")
            rows[name] = {
                "best_epoch": int(run.metrics["best_epoch"]),
                "epochs_run": int(run.metrics["epochs_run"]),
                "best_val_loss": run.metrics["best_val_loss"],
                "epoch_time_s": times.mean(),
                "train_time_h": times.sum() / 3600,
                "stopped_early": run.tags.get("stopped_early") == "true",
            }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("run")

    def summary(self, split: str = "val", literal: bool = False) -> pd.DataFrame:
        """Median, mean and number of excluded basins of each metric, one row per run.

        ``literal`` reads the version over every basin (``<split>_literal/``) instead of the main
        one with the D-003 exclusion (``<split>/``). Columns: ``(metric, stat)``.
        """
        prefix = f"{split}_literal" if literal else split
        columns = pd.MultiIndex.from_product([METRICS, SUMMARY_STATS], names=["metric", "stat"])
        rows = []
        for name, run in self.runs.items():
            keys = [f"{prefix}/{m}_{s}" for m, s in columns]
            missing = [k for k in keys if k not in run.metrics]
            if missing:
                raise KeyError(f"run {name!r} has no {missing[0]!r}: evaluate it on {split} first")
            rows.append([run.metrics[k] for k in keys])
        table = pd.DataFrame(rows, index=pd.Index(self.names, name="run"), columns=columns)
        for metric in METRICS:
            table[(metric, "n_excluded")] = table[(metric, "n_excluded")].astype(int)
        return table

    def paper_table(self, table: str, split: str = "val", literal: bool = False) -> pd.DataFrame:
        """A table of the paper (:data:`PAPER_TABLES`) from :meth:`summary`, with each row
        labeled by the paper name, the run it reads and its flags (✓/✗)."""
        rows = PAPER_TABLES[table]
        summary = self.summary(split, literal)
        flags = self.flags()
        index = pd.MultiIndex.from_tuples(
            [
                (label, run, *("✓" if flags.loc[run, f] else "✗" for f in FLAGS))
                for label, run in rows
            ],
            names=["row", "run", *FLAGS],
        )
        out = summary.loc[[run for _, run in rows]]
        out.index = index
        return out

    def history(self, key: str) -> pd.DataFrame:
        """A metric logged per step (e.g. ``loss/val``), indexed by step with one column per run;
        runs that did not log it are left out."""
        series = {name: self._history(run.run_id, key) for name, run in self.runs.items()}
        series = {name: s for name, s in series.items() if len(s)}
        return pd.DataFrame(series).rename_axis("step")

    def tau(self) -> pd.DataFrame:
        """Distribution of the LAAM lag ``τ`` (D-001) at the best epoch of each run that has it:
        rows ``(run, layer)`` with ``layer`` = ``all`` or the decoder layer, columns mean, p50,
        p90 and max, in hours."""
        stats = ("mean", "p50", "p90", "max")
        rows = {}
        for name, run in self.runs.items():
            if "tau/mean" not in run.metrics:
                continue
            best = int(run.metrics["best_epoch"])
            layers = sorted(
                {int(k.split("/")[1][5:]) for k in run.metrics if k.startswith("tau/layer")}
            )
            for layer in ["all", *layers]:
                prefix = "tau" if layer == "all" else f"tau/layer{layer}"
                rows[(name, str(layer))] = {
                    s: self._history(run.run_id, f"{prefix}/{s}").get(best, np.nan) for s in stats
                }
        index = pd.MultiIndex.from_tuples(list(rows), names=["run", "layer"])
        return pd.DataFrame(list(rows.values()), index=index, columns=list(stats))

    def basin_metrics(self, split: str = "val") -> dict[str, pd.DataFrame]:
        """``eval/<split>/basin_metrics.csv`` of each run, indexed by ``basin_id``: the metrics
        with the D-003 exclusion and their ``*_literal`` versions."""
        return {
            name: self._read_artifact(run.run_id, f"eval/{split}/basin_metrics.csv", "basin_id")
            for name, run in self.runs.items()
        }

    def lead_metrics(self, split: str = "val") -> dict[str, pd.DataFrame]:
        """``eval/<split>/lead_metrics.csv`` of each run, indexed by lead hour (1..H)."""
        return {
            name: self._read_artifact(run.run_id, f"eval/{split}/lead_metrics.csv", "lead")
            for name, run in self.runs.items()
        }

    def predictions(self, name: str, split: str = "val") -> pd.DataFrame:
        """``eval/<split>/predictions.parquet`` of one run: one row per (window, lead) in mm/h."""
        return self._read_artifact(self.runs[name].run_id, f"eval/{split}/predictions.parquet")

    def _history(self, run_id: str, key: str) -> pd.Series:
        history = self.client.get_metric_history(run_id, key)
        return pd.Series({m.step: m.value for m in history}, dtype=float).sort_index()

    def _read_artifact(self, run_id: str, path: str, index: str | None = None) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(self.client.download_artifacts(run_id, path, tmp))
            if local.suffix == ".parquet":
                return pd.read_parquet(local)
            return pd.read_csv(local, index_col=index)


def wilcoxon_signed_rank(diff: Iterable[float]) -> float:
    """Two-sided p-value of the Wilcoxon signed-rank test that ``diff`` is centered on 0.

    Zero differences are dropped, tied ``|diff|`` get their average rank and the variance is
    corrected for ties; the p-value uses the normal approximation without continuity correction,
    which is accurate for the hundreds of basins compared here. ``NaN`` with no non-zero
    difference.
    """
    d = np.asarray(list(diff), dtype=np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return math.nan
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    mean = n * (n + 1) / 4
    _, ties = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (ties**3 - ties).sum() / 48
    if var == 0:
        return math.nan
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def paired_comparison(
    basins: Mapping[str, pd.DataFrame], pairs: Iterable[tuple[str, str]], metric: str = "nse"
) -> pd.DataFrame:
    """Compare runs basin by basin on ``metric`` (per-basin tables of :meth:`Grid.basin_metrics`).

    For each pair ``(a, b)``, over the basins where both have the metric defined, the improvement
    of ``a`` over ``b`` is ``a − b`` for NSE and KGE, ``b − a`` for RMSE and TPE and
    ``|b| − |a|`` for BIAS (positive = ``a`` is better). Columns: number of basins, median and mean
    improvement, fraction of basins where ``a`` is better and the Wilcoxon p-value.
    """
    if metric not in _DIRECTION:
        raise ValueError(f"unknown metric {metric!r}; expected one of {list(_DIRECTION)}")
    rows = {}
    for a, b in pairs:
        pair = pd.concat({"a": basins[a][metric], "b": basins[b][metric]}, axis=1).dropna()
        if metric == "bias":
            pair = pair.abs()
        improvement = _DIRECTION[metric] * (pair["a"] - pair["b"])
        rows[f"{a} vs {b}"] = {
            "n_basins": len(improvement),
            "median_improvement": improvement.median(),
            "mean_improvement": improvement.mean(),
            "win_rate": (improvement > 0).mean(),
            "p_value": wilcoxon_signed_rank(improvement),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("comparison")
