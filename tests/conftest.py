"""Shared fixtures: a tiny synthetic dataset (see ``tests/synthetic.py``) and a throwaway MLflow
tracking store."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from clamf.config import Config
from clamf.data.prepare import prepare
from tests.synthetic import RawData, make_config, make_raw, tiny_training_config


@pytest.fixture
def raw(tmp_path: Path) -> RawData:
    data = make_raw(tmp_path / "raw")
    data.write()
    return data


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MlflowClient]:
    """SQLite store in ``tmp_path`` with an ``unit-test`` experiment whose artifacts stay there."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    client = MlflowClient()
    # artifacts under tmp_path too; set_experiment reuses this experiment by name
    client.create_experiment("unit-test", artifact_location=(tmp_path / "artifacts").as_uri())
    yield client
    if mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def base_cfg(raw: RawData, cfg: Config, tracking: MlflowClient, tmp_path: Path) -> Config:
    """Tiny training setup on the prepared synthetic cache (see ``tiny_training_config``)."""
    prepare(cfg)
    return tiny_training_config(cfg, tmp_path)
