import shutil
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from clamf.utils.mlflow_store import main, relocate, snapshot


def make_store(root: Path, monkeypatch: pytest.MonkeyPatch, as_uri: bool) -> str:
    """A store in ``root`` like the pod's: ``mlflow.db`` plus ``mlruns/`` with one artifact."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{root / 'mlflow.db'}")
    client = MlflowClient()
    location = root / "mlruns"
    experiment_id = client.create_experiment(
        "grid", artifact_location=location.as_uri() if as_uri else str(location)
    )
    (root / "best.pt").write_text("weights")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_artifact(str(root / "best.pt"), "checkpoints")
    return run.info.run_id


@pytest.mark.parametrize("as_uri", [False, True], ids=["path", "file-uri"])
def test_copied_store_finds_its_artifacts_after_relocate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_uri: bool
) -> None:
    pod, mac = tmp_path / "workspace" / "repo", tmp_path / "mac" / "results"
    pod.mkdir(parents=True)
    run_id = make_store(pod, monkeypatch, as_uri)

    mac.mkdir(parents=True)
    snapshot(pod / "mlflow.db", mac / "mlflow.db")
    shutil.copytree(pod / "mlruns", mac / "mlruns")
    shutil.rmtree(pod)  # the pod is gone
    assert relocate(mac / "mlflow.db", str(pod), str(mac)) == 2  # experiment and run
    assert relocate(mac / "mlflow.db", str(pod), str(mac)) == 0

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{mac / 'mlflow.db'}")
    assert MlflowClient().get_run(run_id).info.artifact_uri.endswith(f"{run_id}/artifacts")
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(tmp_path / "dl")
    )
    assert Path(path).read_text() == "weights"


def test_relocate_leaves_other_prefixes_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_store(root, monkeypatch, as_uri=False)
    # ".../rep" is a string prefix of ".../repo/mlruns" but not a parent directory
    assert relocate(root / "mlflow.db", str(tmp_path / "rep"), str(tmp_path / "other")) == 0


def test_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_store(root, monkeypatch, as_uri=False)
    copy = tmp_path / "copy.db"
    main(["snapshot", "--db", str(root / "mlflow.db"), "--out", str(copy)])
    main(["relocate", "--db", str(copy), "--from", str(root), "--to", str(tmp_path / "new")])
    assert "relocated 2 artifact locations" in capsys.readouterr().out
