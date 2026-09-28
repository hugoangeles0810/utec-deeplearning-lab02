"""Load and validate the raw Rainfall-Runoff files (docs/data_preparation.md §1, §5).

Every check raises :class:`DataValidationError`; nothing is ever imputed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

TRAIN_FILE = "train.h5"
TEST_FILE = "test.h5"
TEST_TARGETS_FILE = "test_targets.csv"
METADATA_FILE = "metadata.json"

TRAIN_SPLIT = 0
VAL_SPLIT = 1


class DataValidationError(ValueError):
    """Raised when a raw file does not match the expected layout or holds non-finite values."""


@dataclass(frozen=True)
class DatasetSpec:
    """Layout of the dataset, read from ``metadata.json``."""

    history_hours: int
    horizon_hours: int
    target_channel: int
    meteo_channels: tuple[int, ...]
    channel_names: tuple[str, ...]
    n_train: int
    n_val: int
    n_test: int

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def meteo_names(self) -> tuple[str, ...]:
        return tuple(self.channel_names[i] for i in self.meteo_channels)


def load_metadata(raw_dir: str | Path) -> DatasetSpec:
    """Read ``metadata.json`` from ``raw_dir``."""
    meta: dict[str, Any] = json.loads((Path(raw_dir) / METADATA_FILE).read_text())
    channels = sorted(meta["channels"], key=lambda c: c["index"])
    spec = DatasetSpec(
        history_hours=int(meta["history_hours"]),
        horizon_hours=int(meta["forecast_hours"]),
        target_channel=int(meta["target_channel"]),
        meteo_channels=tuple(int(c) for c in meta["y_aux_channels"]),
        channel_names=tuple(c["name"] for c in channels),
        n_train=int(meta["samples"]["train"]),
        n_val=int(meta["samples"]["validation"]),
        n_test=int(meta["samples"]["test"]),
    )
    expected = tuple(i for i in range(spec.n_channels) if i != spec.target_channel)
    if spec.meteo_channels != expected:
        raise DataValidationError(
            f"y_aux_channels {spec.meteo_channels} must be every channel except the target"
        )
    return spec


def validate_train_h5(f: h5py.File, spec: DatasetSpec) -> None:
    """Check keys, shapes, dtypes, split counts and basin ids of ``train.h5``."""
    n = spec.n_train + spec.n_val
    _check_dataset(f, "X", (n, spec.history_hours, spec.n_channels), np.float32)
    _check_dataset(f, "y", (n, spec.horizon_hours), np.float32)
    _check_dataset(f, "y_aux", (n, spec.horizon_hours, len(spec.meteo_channels)), np.float32)
    _check_dataset(f, "split", (n,), None)
    _check_dataset(f, "basin_id", (n,), None)
    split = f["split"][:]
    counts = {code: int(np.count_nonzero(split == code)) for code in (TRAIN_SPLIT, VAL_SPLIT)}
    if counts != {TRAIN_SPLIT: spec.n_train, VAL_SPLIT: spec.n_val} or len(np.unique(split)) != 2:
        raise DataValidationError(
            f"train.h5: split counts {counts} do not match metadata "
            f"(train={spec.n_train}, val={spec.n_val})"
        )
    basins = f["basin_id"][:]
    missing = np.setdiff1d(basins[split == VAL_SPLIT], basins[split == TRAIN_SPLIT])
    if missing.size:
        raise DataValidationError(f"train.h5: val basins without train windows: {missing[:10]}")


def validate_test_h5(f: h5py.File, spec: DatasetSpec, train_basins: np.ndarray) -> None:
    """Check ``test.h5``; ``y_aux`` is optional (D-013) but validated when present."""
    n = spec.n_test
    _check_dataset(f, "X", (n, spec.history_hours, spec.n_channels), np.float32)
    _check_dataset(f, "basin_id", (n,), None)
    if "y_aux" in f:
        _check_dataset(f, "y_aux", (n, spec.horizon_hours, len(spec.meteo_channels)), np.float32)
    missing = np.setdiff1d(f["basin_id"][:], train_basins)
    if missing.size:
        raise DataValidationError(f"test.h5: basins without train windows: {missing[:10]}")


def load_test_targets(path: str | Path, spec: DatasetSpec) -> np.ndarray:
    """Read ``test_targets.csv`` as a ``(n_test, horizon)`` float32 array ordered by ``Id``."""
    df = pd.read_csv(path)
    columns = ["Id"] + [f"q_{h:02d}" for h in range(1, spec.horizon_hours + 1)]
    if list(df.columns) != columns:
        raise DataValidationError(f"{path}: expected columns Id, q_01..q_{spec.horizon_hours:02d}")
    if len(df) != spec.n_test or not np.array_equal(df["Id"].to_numpy(), np.arange(spec.n_test)):
        raise DataValidationError(f"{path}: Id must be 0..{spec.n_test - 1} in order")
    targets = df[columns[1:]].to_numpy(dtype=np.float32)
    check_finite(targets, "test_targets")
    return targets


def check_finite(array: np.ndarray, name: str) -> None:
    """Raise if ``array`` holds NaN or inf values."""
    bad = ~np.isfinite(array)
    if bad.any():
        raise DataValidationError(f"{name}: {int(bad.sum())} non-finite values")


def _check_dataset(
    f: h5py.File, key: str, shape: tuple[int, ...], dtype: type[np.generic] | None
) -> None:
    if key not in f:
        raise DataValidationError(f"{Path(f.filename).name}: missing dataset '{key}'")
    ds = f[key]
    if ds.shape != shape:
        raise DataValidationError(
            f"{Path(f.filename).name}: '{key}' has shape {ds.shape}, expected {shape}"
        )
    if dtype is not None and ds.dtype != dtype:
        raise DataValidationError(
            f"{Path(f.filename).name}: '{key}' has dtype {ds.dtype}, expected {np.dtype(dtype)}"
        )
