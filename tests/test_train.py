import dataclasses
import json
from pathlib import Path

import mlflow
import pytest
import torch
import yaml
from mlflow.tracking import MlflowClient

import clamf.train as train_mod
from clamf.config import Config, LoggingConfig, ModelConfig, TrainConfig, to_dict
from clamf.data.prepare import prepare
from clamf.train import main, train
from clamf.utils.checkpoint import BEST, LAST, run_checkpoint_dir
from tests.synthetic import RawData


@pytest.fixture
def base_cfg(raw: RawData, cfg: Config, tracking: MlflowClient, tmp_path: Path) -> Config:
    """Tiny model on the synthetic cache, CPU, logging to the ``unit-test`` experiment."""
    prepare(cfg)
    model = ModelConfig(
        d_model=8, d_fusion=8, n_heads=2, d_ff=16, encoder_layers=1, decoder_layers=2,
        msfm_scales=cfg.model.msfm_scales,
    )  # fmt: skip
    return dataclasses.replace(
        cfg,
        device="cpu",
        model=model,
        train=TrainConfig(lr=1e-2, max_epochs=3, checkpoint_dir=str(tmp_path / "ckpt")),
        logging=LoggingConfig(experiment="unit-test"),
    )


def write_yaml(cfg: Config, path: Path) -> Path:
    path.write_text(yaml.safe_dump(json.loads(json.dumps(to_dict(cfg)))))  # tuples -> lists
    return path


def history(client: MlflowClient, run_id: str, key: str) -> list[float]:
    return [m.value for m in client.get_metric_history(run_id, key)]


def weights(cfg: Config, run_id: str) -> dict[str, torch.Tensor]:
    ckpt = run_checkpoint_dir(cfg.train.checkpoint_dir, run_id) / LAST
    return torch.load(ckpt, weights_only=True)["model"]


def test_cli_trains_and_logs_everything(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = write_yaml(base_cfg, tmp_path / "exp.yaml")
    empty_base = tmp_path / "base.yaml"
    empty_base.write_text("")
    main(["--config", str(config), "--base", str(empty_base)])
    run_id = capsys.readouterr().out.split("run id: ")[1].strip()

    run = tracking.get_run(run_id)
    for key in ("loss/train", "loss/val", "time/epoch_s", "tau/mean", "tau/layer1/p90"):
        assert len(history(tracking, run_id, key)) == 3, key
    val = history(tracking, run_id, "loss/val")
    assert run.data.metrics["epochs_run"] == 3
    assert run.data.metrics["best_val_loss"] == min(val)
    assert run.data.metrics["best_epoch"] == val.index(min(val))
    assert run.data.tags["stopped_early"] == "false"
    assert (run.data.params["device"], run.data.params["amp"]) == ("cpu", "none")

    ckpt = run_checkpoint_dir(base_cfg.train.checkpoint_dir, run_id)
    assert torch.load(ckpt / LAST, weights_only=True)["epoch"] == 2
    assert torch.load(ckpt / BEST, weights_only=True)["epoch"] == val.index(min(val))
    artifacts = Path(mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=tmp_path / "dl"))
    for path in (
        "config/exp.yaml",
        "config/resolved.yaml",
        "data/scalers.json",
        "checkpoints/best.pt",
    ):
        assert (artifacts / path).exists(), path


def test_training_reduces_the_loss(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path
) -> None:
    cfg = dataclasses.replace(base_cfg, train=dataclasses.replace(base_cfg.train, max_epochs=5))
    run_id = train(cfg, write_yaml(cfg, tmp_path / "exp.yaml"))
    losses = history(tracking, run_id, "loss/train")
    assert losses[-1] < losses[0]


def test_without_laam_there_is_no_tau(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path
) -> None:
    model = dataclasses.replace(base_cfg.model, use_lag_aware_cross_attn=False, use_msfm=False)
    cfg = dataclasses.replace(base_cfg, model=model)
    run_id = train(cfg, write_yaml(cfg, tmp_path / "exp.yaml"))
    metrics = tracking.get_run(run_id).data.metrics
    assert "loss/val" in metrics
    assert not any(k.startswith("tau/") for k in metrics)


def test_resumed_run_matches_uninterrupted(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = dataclasses.replace(base_cfg, train=dataclasses.replace(base_cfg.train, max_epochs=4))
    config = write_yaml(cfg, tmp_path / "exp.yaml")
    straight = train(cfg, config)

    def crash_after_epoch_1(metrics: dict[str, float], epoch: int) -> None:
        mlflow.log_metrics(metrics, step=epoch)
        if epoch == 1:
            raise KeyboardInterrupt  # the pod dies after last.pt of epoch 1 was written

    with monkeypatch.context() as m:
        m.setattr(train_mod, "log_epoch", crash_after_epoch_1)
        with pytest.raises(KeyboardInterrupt):
            train(cfg, config)
    (interrupted,) = [
        r.info.run_id
        for r in tracking.search_runs([tracking.get_experiment_by_name("unit-test").experiment_id])
        if r.info.run_id != straight
    ]
    torch.manual_seed(123)  # a fresh process would start from another global RNG state
    assert train(cfg, config, run_id=interrupted) == interrupted

    for (name, a), b in zip(
        weights(cfg, straight).items(), weights(cfg, interrupted).values(), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    assert history(tracking, interrupted, "loss/val") == history(tracking, straight, "loss/val")
    assert tracking.get_run(interrupted).data.metrics["epochs_run"] == 4


def test_resume_rejects_a_different_config(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path
) -> None:
    run_id = train(base_cfg, write_yaml(base_cfg, tmp_path / "exp.yaml"))
    other = dataclasses.replace(base_cfg, train=dataclasses.replace(base_cfg.train, lr=1e-3))
    with pytest.raises(ValueError, match="different config"):
        train(other, write_yaml(other, tmp_path / "other.yaml"), run_id=run_id)


def test_early_stopping_stops_and_keeps_the_best_epoch(
    base_cfg: Config, tracking: MlflowClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_cfg = dataclasses.replace(base_cfg.train, max_epochs=10, early_stopping_patience=2)
    cfg = dataclasses.replace(base_cfg, train=train_cfg)
    val_losses = iter([3.0, 1.0, 2.0, 2.0, 0.5])  # epoch 1 is the best, then 2 bad epochs
    monkeypatch.setattr(train_mod, "validate", lambda *_: (next(val_losses), None))
    config = write_yaml(cfg, tmp_path / "exp.yaml")
    run_id = train(cfg, config)

    run = tracking.get_run(run_id)
    assert run.data.metrics["epochs_run"] == 4
    assert run.data.metrics["best_epoch"] == 1
    assert run.data.tags["stopped_early"] == "true"
    ckpt = run_checkpoint_dir(cfg.train.checkpoint_dir, run_id)
    assert torch.load(ckpt / BEST, weights_only=True)["epoch"] == 1

    assert train(cfg, config, run_id=run_id) == run_id  # a stopped run trains no more
    assert len(history(tracking, run_id, "loss/val")) == 4
