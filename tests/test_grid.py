import dataclasses
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

import clamf.train as train_mod
from clamf.config import Config
from clamf.grid import DuplicateRunError, find_run, main, run_grid
from tests.synthetic import write_yaml


def grid_yaml(cfg: Config, tmp_path: Path, name: str) -> Path:
    cfg = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, max_epochs=2),
        logging=dataclasses.replace(cfg.logging, run_name=name),
    )
    return write_yaml(cfg, tmp_path / f"{name or 'unnamed'}.yaml")


def runs(tracking: MlflowClient) -> list[Any]:
    experiment = tracking.get_experiment_by_name("unit-test")
    return tracking.search_runs([experiment.experiment_id])


def test_trains_evaluates_and_then_skips(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path
) -> None:
    configs = [grid_yaml(base_cfg, tmp_path, "a"), grid_yaml(base_cfg, tmp_path, "b")]
    first = run_grid(configs, base=None)
    assert [(r.run_name, r.status) for r in first] == [("a", "trained"), ("b", "trained")]
    for r in first:
        metrics = tracking.get_run(r.run_id).data.metrics
        assert metrics["epochs_run"] == 2
        assert "val/nse_median" in metrics

    second = run_grid(configs, base=None)
    assert [r.status for r in second] == ["done", "done"]
    assert [r.run_id for r in second] == [r.run_id for r in first]
    assert len(runs(tracking)) == 2
    assert len(tracking.get_metric_history(first[0].run_id, "loss/train")) == 2


def test_resumes_an_interrupted_run(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = grid_yaml(base_cfg, tmp_path, "a")

    def crash(metrics: dict[str, float], epoch: int) -> None:  # after last.pt of epoch 0
        raise RuntimeError("pod lost")

    with monkeypatch.context() as m:
        m.setattr(train_mod, "log_epoch", crash)
        (failed,) = run_grid([config], base=None)
    assert failed.status == "failed" and "pod lost" in failed.error

    (resumed,) = run_grid([config], base=None)
    (run,) = runs(tracking)
    assert resumed.status == "resumed"
    assert resumed.run_id == run.info.run_id
    assert run.data.metrics["epochs_run"] == 2
    assert "val/nse_median" in run.data.metrics


def test_restarts_a_run_without_checkpoint(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = grid_yaml(base_cfg, tmp_path, "a")

    def crash(*_: Any, **__: Any) -> float:  # before the first checkpoint
        raise RuntimeError("out of memory")

    with monkeypatch.context() as m:
        m.setattr(train_mod, "train_epoch", crash)
        (failed,) = run_grid([config], base=None)
    assert failed.status == "failed"
    (crashed,) = runs(tracking)

    (retried,) = run_grid([config], base=None)
    assert retried.status == "trained"
    assert retried.run_id != crashed.info.run_id
    assert [r.info.run_id for r in runs(tracking)] == [retried.run_id]  # the crashed one is deleted
    assert tracking.get_run(retried.run_id).data.metrics["epochs_run"] == 2


def test_duplicate_run_names_are_rejected(base_cfg: Config, tracking: MlflowClient) -> None:
    cfg = dataclasses.replace(base_cfg, logging=dataclasses.replace(base_cfg.logging, run_name="a"))
    experiment_id = tracking.get_experiment_by_name("unit-test").experiment_id
    for _ in range(2):
        tracking.create_run(experiment_id, run_name="a")
    with pytest.raises(DuplicateRunError, match="2 runs named 'a'"):
        find_run(cfg)


def test_cli_exits_non_zero_when_a_run_fails(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ok = grid_yaml(base_cfg, tmp_path, "a")
    unnamed = grid_yaml(base_cfg, tmp_path, "")  # the grid needs logging.run_name
    empty_base = tmp_path / "base.yaml"
    empty_base.write_text("")
    with pytest.raises(SystemExit) as exit_info:
        main(["--configs", str(ok), str(unnamed), "--base", str(empty_base)])
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "trained" in out and "failed" in out and "run_name" in out
