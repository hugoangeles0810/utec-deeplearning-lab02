import dataclasses
import os

import numpy as np
import pytest

from clamf.config import Config
from clamf.data.io import DataValidationError
from clamf.data.prepare import StaleCacheError, hash_hours, prepare, verify_cache
from clamf.data.scalers import Scalers
from tests.synthetic import TARGET, RawData


def test_cache_matches_manual_normalization(raw: RawData, cfg: Config) -> None:
    out = prepare(cfg, chunk_rows=5)
    s = Scalers.from_json(out / "scalers.json")
    for name, code in (("train", 0), ("val", 1)):
        rows = np.flatnonzero(raw.split == code)
        d = out / name
        basins = raw.basin_id[rows]
        np.testing.assert_array_equal(np.load(d / "row_id.npy"), rows)
        np.testing.assert_array_equal(np.load(d / "basin_id.npy"), basins)
        np.testing.assert_allclose(
            np.load(d / "x_meteo.npy"), s.normalize_meteo(raw.X[rows][..., :TARGET]), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.load(d / "q_hist.npy"), s.normalize_q(raw.X[rows][..., TARGET], basins), rtol=1e-6
        )
        np.testing.assert_allclose(np.load(d / "y_aux.npy"), s.normalize_meteo(raw.y_aux[rows]))
        np.testing.assert_allclose(np.load(d / "target.npy"), s.normalize_q(raw.y[rows], basins))
    test_target = np.load(out / "test" / "target.npy")
    np.testing.assert_allclose(test_target, s.normalize_q(raw.test_targets, raw.basin_test))
    assert not (out / "test" / "y_aux.npy").exists()
    assert verify_cache(cfg)["splits"]["test"] == {"n": 10, "has_y_aux": False}


def test_test_y_aux_is_cached_when_present(raw: RawData, cfg: Config) -> None:
    raw.y_aux_test = np.ones((10, 6, 11), np.float32)
    raw.write()
    out = prepare(cfg)
    assert np.load(out / "test" / "y_aux.npy").shape == (10, 6, 11)


def test_rerun_is_a_noop_and_stale_cache_is_detected(raw: RawData, cfg: Config) -> None:
    out = prepare(cfg)
    stamp = (out / "manifest.json").stat().st_mtime_ns
    prepare(cfg)
    assert (out / "manifest.json").stat().st_mtime_ns == stamp

    other = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            normalization=dataclasses.replace(cfg.data.normalization, discharge_std_floor=0.5),
        ),
    )
    with pytest.raises(StaleCacheError):
        verify_cache(other)
    with pytest.raises(StaleCacheError):
        prepare(other)
    prepare(other, force=True)
    assert verify_cache(other)

    os.utime(raw.dir / "train.h5", ns=(0, 0))
    with pytest.raises(StaleCacheError):
        verify_cache(other)


def test_missing_cache_raises(cfg: Config) -> None:
    with pytest.raises(FileNotFoundError, match="clamf.data.prepare"):
        verify_cache(cfg)


def test_shared_hours_are_detected(raw: RawData, cfg: Config) -> None:
    train_row = np.flatnonzero(raw.split == 0)[0]
    raw.X_test[3, 5] = raw.X[train_row, 10]
    raw.write()
    with pytest.raises(DataValidationError, match="test shares 1 hourly rows"):
        prepare(cfg)
    assert not os.path.exists(cfg.data.processed_dir)
    prepare(cfg, check_overlap=False)


def test_non_finite_raw_values_raise(raw: RawData, cfg: Config) -> None:
    raw.y_aux[4, 2, 1] = np.inf
    raw.write()
    with pytest.raises(DataValidationError, match="non-finite"):
        prepare(cfg)


def test_hash_distinguishes_rows() -> None:
    x = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]], np.float32)
    h = hash_hours(x)
    assert h[0] == h[2] != h[1]
