import h5py
import numpy as np
import pytest

from clamf.data.io import (
    DataValidationError,
    load_metadata,
    load_test_targets,
    validate_test_h5,
    validate_train_h5,
)
from tests.synthetic import RawData


def validate(raw: RawData) -> None:
    raw.write()
    spec = load_metadata(raw.dir)
    with h5py.File(raw.dir / "train.h5") as f:
        validate_train_h5(f, spec)
    with h5py.File(raw.dir / "test.h5") as f:
        validate_test_h5(f, spec, np.unique(raw.basin_id))
    load_test_targets(raw.dir / "test_targets.csv", spec)


def test_valid_data_passes(raw: RawData) -> None:
    validate(raw)


def test_metadata_is_parsed(raw: RawData) -> None:
    spec = load_metadata(raw.dir)
    assert (spec.history_hours, spec.horizon_hours, spec.target_channel) == (24, 6, 11)
    assert spec.meteo_channels == tuple(range(11))


def test_wrong_shape_raises(raw: RawData) -> None:
    raw.y = raw.y[:, :-1]
    with pytest.raises(DataValidationError, match="'y' has shape"):
        validate(raw)


def test_wrong_split_counts_raise(raw: RawData) -> None:
    raw.split[np.flatnonzero(raw.split == 1)[0]] = 0
    with pytest.raises(DataValidationError, match="split counts"):
        validate(raw)


def test_missing_key_raises(raw: RawData) -> None:
    raw.write()
    with h5py.File(raw.dir / "train.h5", "a") as f:
        del f["y_aux"]
    spec = load_metadata(raw.dir)
    with h5py.File(raw.dir / "train.h5") as f, pytest.raises(DataValidationError, match="y_aux"):
        validate_train_h5(f, spec)


def test_unknown_test_basin_raises(raw: RawData) -> None:
    raw.basin_test[0] = 99
    with pytest.raises(DataValidationError, match="basins without train"):
        validate(raw)


def test_non_finite_targets_raise(raw: RawData) -> None:
    raw.test_targets[2, 3] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        validate(raw)


def test_unordered_test_ids_raise(raw: RawData) -> None:
    path = raw.dir / "test_targets.csv"
    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DataValidationError, match="Id must be"):
        load_test_targets(path, load_metadata(raw.dir))
