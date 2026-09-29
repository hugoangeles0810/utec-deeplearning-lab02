"""Tiny synthetic Rainfall-Runoff dataset with the same layout as the real files."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from clamf.config import Config, DataConfig, ModelConfig

HIST, HOR, N_CH, TARGET = 24, 6, 12, 11
N_TRAIN, N_VAL, N_TEST, N_BASINS = 40, 12, 10, 4


@dataclass
class RawData:
    """Arrays written to ``dir`` (kept in memory so tests can compare against them)."""

    dir: Path
    X: np.ndarray
    y: np.ndarray
    y_aux: np.ndarray
    split: np.ndarray
    basin_id: np.ndarray
    X_test: np.ndarray
    basin_test: np.ndarray
    test_targets: np.ndarray
    y_aux_test: np.ndarray | None = None

    def write(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.dir / "train.h5", "w") as f:
            f["X"], f["y"], f["y_aux"] = self.X, self.y, self.y_aux
            f["split"], f["basin_id"] = self.split, self.basin_id
        with h5py.File(self.dir / "test.h5", "w") as f:
            f["X"], f["basin_id"] = self.X_test, self.basin_test
            if self.y_aux_test is not None:
                f["y_aux"] = self.y_aux_test
        columns = {"Id": np.arange(N_TEST)}
        columns |= {f"q_{h:02d}": self.test_targets[:, h - 1] for h in range(1, HOR + 1)}
        pd.DataFrame(columns).to_csv(self.dir / "test_targets.csv", index=False)
        meta = {
            "history_hours": HIST,
            "forecast_hours": HOR,
            "target_channel": TARGET,
            "channels": [{"index": i, "name": f"ch{i}"} for i in range(N_CH)],
            "y_aux_channels": [i for i in range(N_CH) if i != TARGET],
            "samples": {"train": N_TRAIN, "validation": N_VAL, "test": N_TEST},
        }
        (self.dir / "metadata.json").write_text(json.dumps(meta))


def make_raw(root: Path, seed: int = 0) -> RawData:
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL
    split = rng.permutation(np.r_[np.zeros(N_TRAIN), np.ones(N_VAL)]).astype(np.uint8)
    basin_id = np.tile(np.arange(N_BASINS), n // N_BASINS + 1)[:n].astype(np.int32)
    scale = np.array([0.01, 0.1, 1.0, 5.0], dtype=np.float32)

    def meteo(*shape: int) -> np.ndarray:
        offsets = np.arange(N_CH - 1, dtype=np.float32) * 10
        return (rng.normal(size=(*shape, N_CH - 1)) + offsets).astype(np.float32)

    def discharge(basins: np.ndarray, hours: int) -> np.ndarray:
        q = rng.gamma(2.0, size=(len(basins), hours)) * scale[basins, None]
        return q.astype(np.float32)

    X = np.empty((n, HIST, N_CH), np.float32)
    X[..., :TARGET] = meteo(n, HIST)
    X[..., TARGET] = discharge(basin_id, HIST)
    basin_test = rng.integers(0, N_BASINS, N_TEST).astype(np.int32)
    X_test = np.empty((N_TEST, HIST, N_CH), np.float32)
    X_test[..., :TARGET] = meteo(N_TEST, HIST)
    X_test[..., TARGET] = discharge(basin_test, HIST)
    return RawData(
        dir=root,
        X=X,
        y=discharge(basin_id, HOR),
        y_aux=meteo(n, HOR),
        split=split,
        basin_id=basin_id,
        X_test=X_test,
        basin_test=basin_test,
        test_targets=discharge(basin_test, HOR),
    )


def make_config(tmp_path: Path, **data: object) -> Config:
    fields = {
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "history_hours": HIST,
        "horizon_hours": HOR,
        "batch_size": 8,
        "eval_batch_size": 8,
    }
    # Scales that divide HIST + HOR = 30, since the paper-adapted 24/96 do not.
    return Config(data=DataConfig(**(fields | data)), model=ModelConfig(msfm_scales=(1, 5, 15)))
