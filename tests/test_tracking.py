import subprocess
from pathlib import Path

import mlflow
import pytest
import torch
import yaml
from mlflow.tracking import MlflowClient

from clamf.config import Config, LoggingConfig, ModelConfig
from clamf.utils.tracking import (
    environment_info,
    flatten_config,
    git_commit,
    log_epoch,
    log_run_setup,
    start_run,
    tau_summary,
)


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MlflowClient:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    client = MlflowClient()
    # artifacts under tmp_path too; set_experiment reuses this experiment by name
    client.create_experiment("unit-test", artifact_location=(tmp_path / "artifacts").as_uri())
    yield client
    if mlflow.active_run():
        mlflow.end_run()


def test_flatten_config_uses_dotted_keys() -> None:
    flat = flatten_config(Config(model=ModelConfig(msfm_scales=(1, 24))))
    assert flat["seed"] == 2025
    assert flat["model.d_model"] == 64
    assert flat["model.msfm_scales"] == [1, 24]
    assert flat["data.normalization.meteo"] == "global_zscore"
    assert flat["train.checkpoint_dir"] == "checkpoints"
    assert not any(isinstance(v, dict) for v in flat.values())


def test_git_commit_inside_and_outside_a_repo(tmp_path: Path) -> None:
    sha, dirty = git_commit(Path(__file__).parent)
    assert len(sha) == 40 and isinstance(dirty, bool)
    assert git_commit(tmp_path) == ("unknown", False)


def test_git_commit_detects_uncommitted_changes(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "f.txt").write_text("a")
    git("add", "f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    assert git_commit(tmp_path)[1] is False
    (tmp_path / "f.txt").write_text("b")
    assert git_commit(tmp_path)[1] is True


def test_environment_info_on_cpu() -> None:
    info = environment_info(torch.device("cpu"), "none")
    assert info == {
        "device": "cpu",
        "gpu": "none",
        "amp": "none",
        "torch_version": torch.__version__,
    }


def test_run_setup_logs_params_tags_and_config(tracking: MlflowClient, tmp_path: Path) -> None:
    cfg = Config(logging=LoggingConfig(experiment="unit-test"))
    config_path = tmp_path / "exp.yaml"
    config_path.write_text("model:\n  use_msfm: false\n")
    with start_run(cfg) as run:
        log_run_setup(cfg, config_path, torch.device("cpu"), "none")
        log_epoch({"loss/train": 1.5, "loss/val": 2.0}, epoch=0)
        log_epoch({"loss/train": 1.0, "loss/val": 1.8}, epoch=1)

    data = tracking.get_run(run.info.run_id).data
    assert tracking.get_experiment(run.info.experiment_id).name == "unit-test"
    assert data.params["model.d_model"] == "64"
    assert data.params["model.msfm_scales"] == "[1, 24, 96]"
    assert (data.params["device"], data.params["amp"]) == ("cpu", "none")
    assert data.tags["git_commit"] == git_commit()[0]
    assert data.tags["git_dirty"] in {"true", "false"}
    history = tracking.get_metric_history(run.info.run_id, "loss/val")
    assert [(m.step, m.value) for m in history] == [(0, 2.0), (1, 1.8)]

    artifacts = Path(
        mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=tmp_path / "dl")
    )
    assert (artifacts / "config" / "exp.yaml").read_text() == config_path.read_text()
    resolved = yaml.safe_load((artifacts / "config" / "resolved.yaml").read_text())
    assert resolved["model"]["msfm_scales"] == [1, 24, 96]


def test_start_run_resumes_an_existing_run(tracking: MlflowClient) -> None:
    cfg = Config(logging=LoggingConfig(experiment="unit-test"))
    with start_run(cfg) as run:
        log_epoch({"loss/val": 2.0}, epoch=0)
    with start_run(cfg, run_id=run.info.run_id) as resumed:
        log_epoch({"loss/val": 1.0}, epoch=1)
    assert resumed.info.run_id == run.info.run_id
    assert len(tracking.get_metric_history(run.info.run_id, "loss/val")) == 2


def test_tau_summary_overall_and_per_layer() -> None:
    tau = torch.stack([torch.arange(11.0), torch.arange(11.0) * 2])[:, None, :]  # (2, 1, 11)
    summary = tau_summary(tau)
    assert summary["tau/layer0/mean"] == pytest.approx(5.0)
    assert summary["tau/layer0/p50"] == pytest.approx(5.0)
    assert summary["tau/layer0/p90"] == pytest.approx(9.0)
    assert summary["tau/layer1/max"] == pytest.approx(20.0)
    assert summary["tau/mean"] == pytest.approx(7.5)
    assert summary["tau/max"] == pytest.approx(20.0)
    assert len(summary) == 4 * 3


def test_tau_summary_handles_more_than_the_quantile_limit() -> None:
    tau = torch.rand(1, 2**24 + 10)
    summary = tau_summary(tau)
    assert summary["tau/p50"] == pytest.approx(0.5, abs=1e-2)
