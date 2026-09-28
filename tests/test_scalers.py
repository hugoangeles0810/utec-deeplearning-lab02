import h5py
import numpy as np
import pytest
import torch

from clamf.data.io import load_metadata
from clamf.data.scalers import Scalers, fit_scalers
from tests.synthetic import TARGET, RawData


def fit(raw: RawData, floor: float = 1e-3, chunk_rows: int = 7) -> Scalers:
    raw.write()
    with h5py.File(raw.dir / "train.h5") as f:
        return fit_scalers(f, load_metadata(raw.dir), floor, chunk_rows=chunk_rows)


def test_stats_match_train_rows(raw: RawData) -> None:
    scalers = fit(raw)
    train = raw.split == 0
    meteo = raw.X[train][..., :TARGET].reshape(-1, TARGET).astype(np.float64)
    np.testing.assert_allclose(scalers.meteo_mean, meteo.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(scalers.meteo_std, meteo.std(axis=0), rtol=1e-4)
    for b in scalers.basin_ids:
        q = raw.X[train & (raw.basin_id == b)][..., TARGET].astype(np.float64)
        np.testing.assert_allclose(scalers.q_mean[b], q.mean(), rtol=1e-5)
        np.testing.assert_allclose(scalers.q_std[b], q.std(), rtol=1e-4)


def test_val_rows_do_not_change_scalers(raw: RawData) -> None:
    before = fit(raw)
    raw.X[raw.split == 1] = 1e6
    after = fit(raw)
    for name in ("meteo_mean", "meteo_std", "q_mean", "q_std"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_std_floor_is_applied(raw: RawData) -> None:
    raw.X[raw.basin_id == 0, :, TARGET] = 0.5
    scalers = fit(raw, floor=0.25)
    assert scalers.q_std[0] == pytest.approx(0.25)
    assert (scalers.q_std >= 0.25).all()


def test_round_trip_numpy_and_torch(raw: RawData) -> None:
    scalers = fit(raw)
    q, basins = raw.y, raw.basin_id
    z = scalers.normalize_q(q, basins)
    np.testing.assert_allclose(scalers.denormalize_q(z, basins), q, rtol=1e-5, atol=1e-6)

    zt = scalers.normalize_q(torch.from_numpy(q), torch.from_numpy(basins))
    np.testing.assert_allclose(zt.numpy(), z, rtol=1e-6)
    back = scalers.denormalize_q(zt, torch.from_numpy(basins))
    assert back.dtype == torch.float32
    np.testing.assert_allclose(back.numpy(), q, rtol=1e-5, atol=1e-6)


def test_unknown_basin_raises(raw: RawData) -> None:
    scalers = fit(raw)
    with pytest.raises(KeyError):
        scalers.normalize_q(raw.y[:2], np.array([0, 99]))


def test_json_round_trip(raw: RawData, tmp_path) -> None:
    scalers = fit(raw)
    scalers.to_json(tmp_path / "s.json")
    loaded = Scalers.from_json(tmp_path / "s.json")
    np.testing.assert_array_equal(loaded.q_std, scalers.q_std)
    np.testing.assert_array_equal(loaded.meteo_mean, scalers.meteo_mean)
    assert loaded.meteo_names == scalers.meteo_names
