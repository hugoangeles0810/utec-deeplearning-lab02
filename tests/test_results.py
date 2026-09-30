import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from clamf.grid import DuplicateRunError
from clamf.metrics import METRICS
from clamf.results import (
    FLAGS,
    GRID_RUNS,
    PAPER_TABLES,
    SUMMARY_STATS,
    Grid,
    paired_comparison,
    wilcoxon_signed_rank,
)

# (MSFM, CAM, LAAM) per run of the grid (docs/experiments.md)
RUN_FLAGS = {
    "vanilla": (False, False, False),
    "cam": (False, True, False),
    "laam": (False, False, True),
    "claam": (False, True, True),
    "msfm": (True, False, False),
    "clamf": (True, True, True),
}


def summary_value(k: int, metric: str, stat: str, literal: bool) -> float:
    """A distinct, recognizable value for every run/metric/stat of the fake grid."""
    if stat == "n_excluded":
        return 0 if literal else 2
    return k + METRICS.index(metric) / 10 + SUMMARY_STATS.index(stat) / 100 + literal / 1000


def log_run(client: MlflowClient, name: str, k: int, tmp_path: Path) -> str:
    """A finished grid run with the metrics, history and artifacts that train/evaluate log."""
    experiment = client.get_experiment_by_name("unit-test").experiment_id
    run_id = client.create_run(experiment, run_name=name).info.run_id
    for (_, key), on in zip(FLAGS.items(), RUN_FLAGS[name]):
        client.log_param(run_id, key, str(on))
    for literal, prefix in ((False, "val"), (True, "val_literal")):
        for metric in METRICS:
            for stat in SUMMARY_STATS:
                value = summary_value(k, metric, stat, literal)
                client.log_metric(run_id, f"{prefix}/{metric}_{stat}", value)
    for step in range(3):
        client.log_metric(run_id, "loss/train", 10 - step, step=step)
        client.log_metric(run_id, "loss/val", 12 - step + k, step=step)
        client.log_metric(run_id, "time/epoch_s", 60.0 * (step + 1), step=step)
        if RUN_FLAGS[name][2]:
            for prefix in ("tau", "tau/layer0", "tau/layer1"):
                for stat in ("mean", "p50", "p90", "max"):
                    client.log_metric(run_id, f"{prefix}/{stat}", 10 * step + k, step=step)
    client.log_metric(run_id, "best_epoch", 1)
    client.log_metric(run_id, "epochs_run", 3)
    client.log_metric(run_id, "best_val_loss", 11 + k)
    client.set_tag(run_id, "stopped_early", "true")

    out = tmp_path / "eval" / name
    out.mkdir(parents=True)
    basins = pd.DataFrame(
        {m: [0.5 + k / 10, 0.7, np.nan] for m in METRICS},
        index=pd.Index([3, 5, 9], name="basin_id"),
    )
    basins.to_csv(out / "basin_metrics.csv")
    leads = pd.DataFrame({"nse_median": [0.9, 0.8]}, index=pd.Index([1, 2], name="lead"))
    leads.to_csv(out / "lead_metrics.csv")
    pd.DataFrame(
        {"row_id": [7, 7], "basin_id": [3, 3], "lead": [1, 2], "obs": [1.0, 2.0], "pred": [k, k]}
    ).to_parquet(out / "predictions.parquet", index=False)
    for path in out.iterdir():
        client.log_artifact(run_id, str(path), "eval/val")
    client.set_terminated(run_id)
    return run_id


@pytest.fixture
def grid(tracking: MlflowClient, tmp_path: Path) -> Grid:
    for k, name in enumerate(GRID_RUNS):
        log_run(tracking, name, k, tmp_path)
    return Grid.load(experiment="unit-test")


def test_load_finds_every_run_in_grid_order(grid: Grid) -> None:
    assert grid.names == list(GRID_RUNS)
    flags = grid.flags()
    assert list(flags.columns) == list(FLAGS)
    for name, expected in RUN_FLAGS.items():
        assert tuple(flags.loc[name]) == expected


def test_load_raises_on_missing_or_duplicate_runs(tracking: MlflowClient, tmp_path: Path) -> None:
    log_run(tracking, "vanilla", 0, tmp_path / "a")
    with pytest.raises(ValueError, match="not found"):
        Grid.load(experiment="unit-test")
    with pytest.raises(ValueError, match="experiment"):
        Grid.load(experiment="nope")
    log_run(tracking, "vanilla", 1, tmp_path / "b")
    with pytest.raises(DuplicateRunError):
        Grid.load(experiment="unit-test", names=["vanilla"])


def test_summary_reads_the_logged_metrics(grid: Grid) -> None:
    for literal in (False, True):
        table = grid.summary("val", literal=literal)
        assert list(table.index) == list(GRID_RUNS)
        for k, name in enumerate(GRID_RUNS):
            for metric in METRICS:
                for stat in SUMMARY_STATS:
                    expected = summary_value(k, metric, stat, literal)
                    assert table.loc[name, (metric, stat)] == pytest.approx(expected)
    with pytest.raises(KeyError, match="evaluate it on test"):
        grid.summary("test")


def test_paper_tables_read_their_runs_with_labels_and_flags(grid: Grid) -> None:
    summary = grid.summary()
    for table, rows in PAPER_TABLES.items():
        out = grid.paper_table(table)
        assert list(out.index.get_level_values("row")) == [label for label, _ in rows]
        for (label, run, *flags), values in zip(out.index, out.to_numpy()):
            assert flags == ["✓" if on else "✗" for on in RUN_FLAGS[run]]
            np.testing.assert_array_equal(values, summary.loc[run].to_numpy())


def test_training_summary(grid: Grid) -> None:
    training = grid.training()
    assert training.loc["claam", "best_epoch"] == 1
    assert training.loc["claam", "epochs_run"] == 3
    assert training.loc["claam", "best_val_loss"] == 11 + GRID_RUNS.index("claam")
    assert training.loc["claam", "epoch_time_s"] == pytest.approx(120.0)
    assert training.loc["claam", "train_time_h"] == pytest.approx(0.1)
    assert training["stopped_early"].all()


def test_history_and_tau(grid: Grid) -> None:
    val = grid.history("loss/val")
    assert list(val.columns) == list(GRID_RUNS)
    assert list(val.index) == [0, 1, 2]
    assert val.loc[2, "cam"] == 12 - 2 + GRID_RUNS.index("cam")

    assert list(grid.history("tau/mean").columns) == ["laam", "claam", "clamf"]
    tau = grid.tau()
    assert list(tau.index.get_level_values("run").unique()) == ["laam", "claam", "clamf"]
    assert list(tau.loc["clamf"].index) == ["all", "0", "1"]
    # value at the best epoch (1): 10 * 1 + k
    assert tau.loc[("clamf", "0"), "p90"] == 10 + GRID_RUNS.index("clamf")


def test_eval_artifacts(grid: Grid) -> None:
    basins = grid.basin_metrics()
    assert basins["cam"].index.name == "basin_id"
    assert basins["cam"].loc[3, "nse"] == pytest.approx(0.6)
    assert grid.lead_metrics()["cam"].loc[2, "nse_median"] == 0.8
    pred = grid.predictions("cam")
    assert list(pred.columns) == ["row_id", "basin_id", "lead", "obs", "pred"]
    assert (pred["pred"] == GRID_RUNS.index("cam")).all()


def test_wilcoxon_matches_hand_computed_values() -> None:
    # ranks 1..6, W+ = 15, E = 10.5, Var = 22.75
    assert wilcoxon_signed_rank([1, 2, 3, 4, 5, -6]) == pytest.approx(0.345448, abs=1e-6)
    # the zero is dropped; |d| = 1, 1, 1, 2 -> ranks 2, 2, 2, 4; W+ = 8, E = 5, Var = 7.5 - 0.5
    assert wilcoxon_signed_rank([1, 1, -1, 2, 0]) == pytest.approx(0.256839, abs=1e-6)
    assert wilcoxon_signed_rank([2, 1, -1, -2]) == pytest.approx(1.0)
    assert math.isnan(wilcoxon_signed_rank([0, 0]))


def test_wilcoxon_matches_scipy_on_large_samples() -> None:
    stats = pytest.importorskip("scipy.stats")
    diff = np.random.default_rng(2025).normal(0.1, 1, 300).round(2)  # rounding makes ties
    expected = stats.wilcoxon(diff, method="approx").pvalue
    assert wilcoxon_signed_rank(diff) == pytest.approx(expected, rel=1e-6)


def test_paired_comparison_orients_each_metric() -> None:
    index = pd.Index([1, 2, 3, 4], name="basin_id")
    a = pd.DataFrame(
        {"nse": [0.9, 0.8, 0.7, np.nan], "rmse": [1.0, 1.0, 3.0, 1.0], "bias": [-0.1] * 4},
        index=index,
    )
    b = pd.DataFrame(
        {"nse": [0.8, 0.9, 0.5, 0.5], "rmse": [2.0, 2.0, 2.0, 2.0], "bias": [0.2] * 4},
        index=index,
    )
    nse = paired_comparison({"a": a, "b": b}, [("a", "b")], "nse").loc["a vs b"]
    assert nse["n_basins"] == 3  # basin 4 has no NSE for a
    assert nse["median_improvement"] == pytest.approx(0.1)
    assert nse["win_rate"] == pytest.approx(2 / 3)

    rmse = paired_comparison({"a": a, "b": b}, [("a", "b")], "rmse").loc["a vs b"]
    assert rmse["mean_improvement"] == pytest.approx((1 + 1 - 1 + 1) / 4)  # lower is better
    bias = paired_comparison({"a": a, "b": b}, [("a", "b")], "bias").loc["a vs b"]
    assert bias["mean_improvement"] == pytest.approx(0.1)  # |b| - |a|
    assert bias["win_rate"] == 1.0

    with pytest.raises(ValueError, match="unknown metric"):
        paired_comparison({"a": a, "b": b}, [("a", "b")], "r2")
