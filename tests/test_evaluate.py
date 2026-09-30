import dataclasses
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from clamf.config import Config
from clamf.data.dataset import MissingFutureMeteoError
from clamf.data.prepare import prepare
from clamf.evaluate import Forecast, evaluate, main, score
from clamf.metrics import METRICS
from clamf.train import train
from clamf.utils.checkpoint import run_checkpoint_dir
from tests.synthetic import HOR, N_VAL, RawData, tiny_training_config, write_yaml


@pytest.fixture
def trained(base_cfg: Config, tmp_path: Path) -> tuple[Config, str]:
    cfg = dataclasses.replace(base_cfg, train=dataclasses.replace(base_cfg.train, max_epochs=2))
    return cfg, train(cfg, write_yaml(cfg, tmp_path / "exp.yaml"))


def test_cli_logs_metrics_and_artifacts_into_the_run(
    trained: tuple[Config, str],
    raw: RawData,
    tracking: MlflowClient,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    cfg, run_id = trained
    empty_base = tmp_path / "base.yaml"
    empty_base.write_text("")
    config = write_yaml(cfg, tmp_path / "exp.yaml")
    main(["--config", str(config), "--base", str(empty_base), "--run-id", run_id])
    assert "nse" in capsys.readouterr().out

    metrics = tracking.get_run(run_id).data.metrics
    for name in METRICS:
        for prefix in ("val", "val_literal"):
            for stat in ("median", "mean", "n_excluded"):
                assert f"{prefix}/{name}_{stat}" in metrics
    assert metrics["val/checkpoint_epoch"] == metrics["best_epoch"]
    assert metrics["val/n_windows"] == N_VAL
    assert metrics["val/n_basins"] == len(np.unique(raw.basin_id[raw.split == 1]))
    lead_nse = tracking.get_metric_history(run_id, "val/lead/nse_median")
    assert sorted(m.step for m in lead_nse) == list(range(1, HOR + 1))

    out = Path(mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=tmp_path / "dl"))
    preds = pd.read_parquet(out / "eval" / "val" / "predictions.parquet")
    assert len(preds) == N_VAL * HOR
    obs = preds.pivot(index="row_id", columns="lead", values="obs")
    np.testing.assert_allclose(obs.to_numpy(), raw.y[obs.index], rtol=1e-5, atol=1e-6)  # mm/h
    assert np.isfinite(preds["pred"]).all()
    basins = pd.read_csv(out / "eval" / "val" / "basin_metrics.csv", index_col="basin_id")
    assert {*METRICS, *(f"{m}_literal" for m in METRICS)} == set(basins.columns)
    leads = pd.read_csv(out / "eval" / "val" / "lead_metrics.csv", index_col="lead")
    assert list(leads.index) == list(range(1, HOR + 1))


def test_checkpoint_is_downloaded_when_not_on_disk(
    trained: tuple[Config, str], tracking: MlflowClient
) -> None:
    cfg, run_id = trained
    shutil.rmtree(run_checkpoint_dir(cfg.train.checkpoint_dir, run_id))  # e.g. trained on a pod
    metrics = evaluate(cfg, run_id)
    assert metrics["val/checkpoint_epoch"] == tracking.get_run(run_id).data.metrics["best_epoch"]


def test_run_is_found_whatever_the_configured_experiment(
    trained: tuple[Config, str], tracking: MlflowClient
) -> None:
    cfg, run_id = trained
    other = dataclasses.replace(cfg, logging=dataclasses.replace(cfg.logging, experiment="other"))
    assert "val/nse_median" in evaluate(other, run_id)
    assert "val/nse_median" in tracking.get_run(run_id).data.metrics
    assert tracking.get_experiment_by_name("other") is None


def test_evaluation_is_deterministic(trained: tuple[Config, str]) -> None:
    cfg, run_id = trained
    first, second = evaluate(cfg, run_id), evaluate(cfg, run_id)
    assert first.keys() == second.keys()
    np.testing.assert_array_equal(list(first.values()), list(second.values()))


def test_rejects_a_different_model_config(trained: tuple[Config, str]) -> None:
    cfg, run_id = trained
    other = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dropout=0.3))
    with pytest.raises(ValueError, match=r"different config: model\.dropout$"):
        evaluate(other, run_id)


def test_paths_and_batch_sizes_can_change_after_training(trained: tuple[Config, str]) -> None:
    cfg, run_id = trained
    data = dataclasses.replace(cfg.data, eval_batch_size=3, preload_to_device=False)
    assert "val/nse_median" in evaluate(dataclasses.replace(cfg, data=data), run_id)


def test_eval_settings_can_change_after_training(trained: tuple[Config, str]) -> None:
    cfg, run_id = trained
    other = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, min_obs_std=1e6))
    metrics = evaluate(other, run_id)
    assert metrics["val/nse_n_excluded"] == metrics["val/n_basins"]  # every basin excluded
    assert np.isnan(metrics["val/nse_median"])
    assert np.isfinite(metrics["val_literal/nse_median"])


def test_test_split_needs_future_meteorology(trained: tuple[Config, str]) -> None:
    cfg, run_id = trained
    with pytest.raises(MissingFutureMeteoError, match="D-013"):
        evaluate(cfg, run_id, split="test")


def test_test_split_is_evaluated_once_it_has_y_aux(
    raw: RawData, cfg: Config, tracking: MlflowClient, tmp_path: Path
) -> None:
    raw.y_aux_test = np.zeros((len(raw.X_test), HOR, raw.y_aux.shape[-1]), np.float32)
    raw.write()
    prepare(cfg)
    tiny = tiny_training_config(cfg, tmp_path)
    tiny = dataclasses.replace(tiny, train=dataclasses.replace(tiny.train, max_epochs=1))
    run_id = train(tiny, write_yaml(tiny, tmp_path / "exp.yaml"))
    metrics = evaluate(tiny, run_id, split="test")
    assert metrics["test/n_windows"] == len(raw.X_test)
    assert "test_literal/rmse_mean" in metrics


def test_score_of_a_perfect_forecast() -> None:
    rng = np.random.default_rng(0)
    obs = rng.gamma(2.0, size=(12, 4)).astype(np.float32)
    forecast = Forecast(
        obs=obs, pred=obs.copy(), basin_id=np.repeat([3, 7], 6), row_id=np.arange(12)
    )
    result = score(forecast, Config().eval)
    assert result.summary["nse_median"] == pytest.approx(1.0)
    assert result.summary["kge_mean"] == pytest.approx(1.0)
    assert result.summary["rmse_mean"] == pytest.approx(0.0)
    assert result.summary["bias_median"] == pytest.approx(0.0)
    assert result.summary["tpe_mean"] == pytest.approx(0.0)
    assert list(result.basins.index) == [3, 7]
    np.testing.assert_allclose(result.leads["nse_median"], 1.0)

    frame = forecast.to_frame()
    assert len(frame) == 12 * 4
    assert frame.loc[5, ["row_id", "lead", "obs"]].tolist() == [1, 2, obs[1, 1]]
