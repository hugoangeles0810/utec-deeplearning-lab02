import math
import warnings

import numpy as np
import pytest

from clamf.config import EvalConfig
from clamf.metrics import METRICS, basin_metrics, bias, kge, lead_metrics, nse, rmse, summarize, tpe

OBS = np.array([1.0, 2.0, 3.0, 4.0])


def test_perfect_prediction() -> None:
    assert nse(OBS, OBS) == 1.0
    assert kge(OBS, OBS) == pytest.approx(1.0)
    assert rmse(OBS, OBS) == 0.0
    assert tpe(OBS, OBS) == 0.0
    assert bias(OBS, OBS) == 0.0


def test_mean_prediction() -> None:
    pred = np.full_like(OBS, OBS.mean())
    assert nse(OBS, pred) == pytest.approx(0.0)
    # constant pred: rho = 0 (D-003), lambda = 0, gamma = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kge(OBS, pred) == pytest.approx(1 - math.sqrt(2))


def test_offset_by_hand() -> None:
    pred = OBS + 0.5  # SSE = 1, SST = 5, rho = 1, lambda = 1, gamma = 3 / 2.5
    assert nse(OBS, pred) == pytest.approx(0.8)
    assert rmse(OBS, pred) == pytest.approx(0.5)
    assert kge(OBS, pred) == pytest.approx(0.8)
    assert bias(OBS, pred) == pytest.approx(0.2)


def test_scaled_by_hand() -> None:
    pred = 2 * OBS  # rho = 1, lambda = 2, gamma = 2
    assert kge(OBS, pred) == pytest.approx(1 - math.sqrt(2))
    assert bias(OBS, pred) == pytest.approx(1.0)
    assert rmse(OBS, pred) == pytest.approx(math.sqrt(7.5))


def test_tpe_uses_top_observed_hours() -> None:
    obs = np.arange(1.0, 101.0)  # top 2 % = the hours with 99 and 100
    pred = obs.copy()
    pred[[98, 99]] += 10.0
    pred[0] += 1000.0  # errors outside the peaks do not count
    assert tpe(obs, pred, top_fraction=0.02) == pytest.approx(20 / 199)


def test_tpe_keeps_at_least_one_hour() -> None:
    assert tpe(OBS, OBS * 0.5, top_fraction=0.01) == pytest.approx(0.5)


def test_metrics_reduce_over_axis_0() -> None:
    obs = np.stack([OBS, 10 * OBS], axis=1)
    pred = obs + 0.5
    np.testing.assert_allclose(nse(obs, pred), [0.8, 1 - 4 * 0.25 / 500])
    np.testing.assert_allclose(rmse(obs, pred), [0.5, 0.5])


def test_degenerate_denominators_are_nan() -> None:
    flat, dry = np.full(4, 0.5), np.zeros(4)
    pred = np.ones(4)
    assert math.isnan(nse(flat, pred))
    assert math.isnan(kge(flat, pred))
    assert rmse(flat, pred) == pytest.approx(0.5)
    for fn in (bias, tpe, kge):
        assert math.isnan(fn(dry, pred))
    small = OBS * 1e-4  # std and mean below the thresholds
    assert math.isnan(nse(small, pred, min_obs_std=1e-3))
    assert math.isnan(bias(small, pred, min_obs_mean=1e-3))
    assert math.isnan(tpe(small, pred, min_obs_mean=1e-3))
    assert math.isnan(kge(small, pred, min_obs_std=1e-3))
    assert not math.isnan(nse(small, pred))


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        nse(OBS, OBS[:3])
    with pytest.raises(ValueError, match="finite"):
        rmse(OBS, np.array([1.0, np.nan, 3.0, 4.0]))


def windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three basins with 5 windows of 4 lead hours; basin 7 is flat (degenerate)."""
    rng = np.random.default_rng(0)
    basin_id = np.repeat([3, 5, 7], 5)
    obs = rng.gamma(2.0, 0.5, size=(15, 4))
    obs[basin_id == 7] = 0.2
    pred = obs + rng.normal(0, 0.1, size=obs.shape)
    return obs, pred, basin_id


def test_basin_metrics_pool_windows_and_leads() -> None:
    obs, pred, basin_id = windows()
    cfg = EvalConfig()
    table = basin_metrics(obs, pred, basin_id, cfg)
    assert list(table.index) == [3, 5, 7]
    assert list(table.columns) == list(METRICS)
    o, p = obs[basin_id == 5].ravel(), pred[basin_id == 5].ravel()
    assert table.loc[5, "nse"] == pytest.approx(nse(o, p))
    assert table.loc[5, "kge"] == pytest.approx(kge(o, p))
    assert table.loc[5, "tpe"] == pytest.approx(tpe(o, p, cfg.tpe_top_fraction))
    assert table.loc[7, ["nse", "kge"]].isna().all()
    assert table.loc[7, ["rmse", "tpe", "bias"]].notna().all()


def test_summarize_skips_excluded_basins() -> None:
    obs, pred, basin_id = windows()
    table = basin_metrics(obs, pred, basin_id, EvalConfig())
    summary = summarize(table)
    assert summary["nse_n_excluded"] == 1
    assert summary["rmse_n_excluded"] == 0
    assert summary["nse_mean"] == pytest.approx(table.loc[[3, 5], "nse"].mean())
    assert summary["rmse_median"] == pytest.approx(table["rmse"].median())


def test_lead_metrics_per_lead_hour() -> None:
    obs, pred, basin_id = windows()
    leads = lead_metrics(obs, pred, basin_id, EvalConfig())
    assert list(leads.index) == [1, 2, 3, 4]
    assert list(leads.columns) == ["nse_median", "nse_mean", "rmse_median", "rmse_mean"]
    per_basin = [nse(obs[basin_id == b][:, 1], pred[basin_id == b][:, 1]) for b in (3, 5)]
    assert leads.loc[2, "nse_mean"] == pytest.approx(np.mean(per_basin))


def test_basin_metrics_checks_shapes() -> None:
    obs, pred, basin_id = windows()
    with pytest.raises(ValueError, match="basin_id"):
        basin_metrics(obs, pred, basin_id[:-1], EvalConfig())
