"""Evaluation metrics (paper Eq. 16-20) and their per-basin aggregation (D-003, D-016).

All inputs are in original units (mm/h). Every metric reduces over axis 0 of ``obs``/``pred``, so a
``(n,)`` series gives a scalar and a ``(n, H)`` array gives one value per column (e.g. per lead).

A metric is ``NaN`` when its observed denominator is degenerate (D-003):

- ``std(obs) < min_obs_std`` -> NSE and KGE;
- ``mean(obs) < min_obs_mean`` -> BIAS, TPE and KGE;
- RMSE is always defined.

Denominators that are exactly zero are always ``NaN``, also with thresholds of 0 (literal version).
"""

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd

from clamf.config import EvalConfig

METRICS = ("nse", "kge", "rmse", "tpe", "bias")
LEAD_METRICS = ("nse", "rmse")


def nse(obs: np.ndarray, pred: np.ndarray, min_obs_std: float = 0.0) -> np.ndarray:
    """Nash-Sutcliffe efficiency, Eq. (16): ``1 - SSE / SST``."""
    obs, pred = _pair(obs, pred)
    sse = ((obs - pred) ** 2).sum(axis=0)
    sst = ((obs - obs.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - _ratio(sse, sst, _defined(obs.std(axis=0), min_obs_std))


def rmse(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Root mean squared error, Eq. (17)."""
    obs, pred = _pair(obs, pred)
    return np.sqrt(((obs - pred) ** 2).mean(axis=0))


def tpe(
    obs: np.ndarray, pred: np.ndarray, top_fraction: float = 0.02, min_obs_mean: float = 0.0
) -> float:
    """Top-peak error, Eq. (18): relative absolute error on the ``top_fraction`` highest ``obs``.

    Only for 1-D series ``(n,)``; the number of peak hours is ``max(1, round(top_fraction * n))``.
    """
    obs, pred = _pair(obs, pred)
    if obs.ndim != 1:
        raise ValueError(f"tpe expects 1-D series, got shape {obs.shape}")
    n_top = max(1, math.floor(top_fraction * obs.size + 0.5))
    top = np.argsort(obs, kind="stable")[::-1][:n_top]
    num = np.abs(pred[top] - obs[top]).sum()
    den = obs[top].sum()
    return float(_ratio(num, den, _defined(obs.mean(), min_obs_mean) & (den > 0)))


def kge(
    obs: np.ndarray, pred: np.ndarray, min_obs_std: float = 0.0, min_obs_mean: float = 0.0
) -> np.ndarray:
    """Kling-Gupta efficiency, Eq. (19): ``1 - sqrt((rho-1)^2 + (lambda-1)^2 + (gamma-1)^2)``.

    ``lambda = std(pred) / std(obs)`` and ``gamma = mean(pred) / mean(obs)``; ``rho`` is taken as 0
    when ``pred`` is constant (D-003).
    """
    obs, pred = _pair(obs, pred)
    obs_mean, pred_mean = obs.mean(axis=0), pred.mean(axis=0)
    obs_std, pred_std = obs.std(axis=0), pred.std(axis=0)
    ok = _defined(obs_std, min_obs_std) & _defined(obs_mean, min_obs_mean)
    cov = ((obs - obs_mean) * (pred - pred_mean)).mean(axis=0)
    varies = np.ptp(pred, axis=0) > 0
    rho = np.clip(_ratio(cov, obs_std * pred_std, ok & varies, fill=0.0), -1.0, 1.0)
    lam = _ratio(pred_std, obs_std, ok)
    gamma = _ratio(pred_mean, obs_mean, ok)
    return 1.0 - np.sqrt((rho - 1) ** 2 + (lam - 1) ** 2 + (gamma - 1) ** 2)


def bias(obs: np.ndarray, pred: np.ndarray, min_obs_mean: float = 0.0) -> np.ndarray:
    """Relative volume bias, Eq. (20): ``(sum(pred) - sum(obs)) / sum(obs)``."""
    obs, pred = _pair(obs, pred)
    ok = _defined(obs.mean(axis=0), min_obs_mean)
    return _ratio(pred.sum(axis=0) - obs.sum(axis=0), obs.sum(axis=0), ok)


def basin_metrics(
    obs: np.ndarray, pred: np.ndarray, basin_id: np.ndarray, cfg: EvalConfig
) -> pd.DataFrame:
    """Main (pooled) metrics per basin over all (window, lead) pairs (D-003).

    ``obs``/``pred``: ``(N, H)`` in mm/h; ``basin_id``: ``(N,)``. Returns a DataFrame indexed by
    ``basin_id`` with one column per metric in :data:`METRICS`.
    """
    rows = {}
    for b, (o, p) in _by_basin(obs, pred, basin_id):
        o, p = o.ravel(), p.ravel()
        rows[b] = {
            "nse": nse(o, p, cfg.min_obs_std),
            "kge": kge(o, p, cfg.min_obs_std, cfg.min_obs_mean),
            "rmse": rmse(o, p),
            "tpe": tpe(o, p, cfg.tpe_top_fraction, cfg.min_obs_mean),
            "bias": bias(o, p, cfg.min_obs_mean),
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(METRICS)).rename_axis(
        "basin_id"
    )


def lead_metrics(
    obs: np.ndarray, pred: np.ndarray, basin_id: np.ndarray, cfg: EvalConfig
) -> pd.DataFrame:
    """Secondary metrics per lead hour (D-003): NSE and RMSE per basin, aggregated over basins.

    Per basin and lead, NSE uses the series of that lead only (its own ``std(obs)`` for the
    threshold). Returns a DataFrame indexed by ``lead`` (1..H) with ``<metric>_median`` and
    ``<metric>_mean`` columns for each metric in :data:`LEAD_METRICS`.
    """
    per_basin = {"nse": [], "rmse": []}
    for _, (o, p) in _by_basin(obs, pred, basin_id):
        per_basin["nse"].append(nse(o, p, cfg.min_obs_std))
        per_basin["rmse"].append(rmse(o, p))
    out = {}
    for name in LEAD_METRICS:
        table = pd.DataFrame(np.stack(per_basin[name]))
        out[f"{name}_median"] = table.median(axis=0).to_numpy()
        out[f"{name}_mean"] = table.mean(axis=0).to_numpy()
    leads = pd.RangeIndex(1, obs.shape[1] + 1, name="lead")
    return pd.DataFrame(out, index=leads)


def summarize(per_basin: pd.DataFrame) -> dict[str, float]:
    """Median and mean over basins of each column, skipping ``NaN``, plus how many were excluded.

    Keys: ``<metric>_median``, ``<metric>_mean`` and ``<metric>_n_excluded``.
    """
    out: dict[str, float] = {}
    for name in per_basin.columns:
        col = per_basin[name]
        out[f"{name}_median"] = float(col.median())
        out[f"{name}_mean"] = float(col.mean())
        out[f"{name}_n_excluded"] = int(col.isna().sum())
    return out


def _pair(obs: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if obs.shape != pred.shape:
        raise ValueError(f"obs and pred shapes differ: {obs.shape} vs {pred.shape}")
    if obs.shape[0] == 0:
        raise ValueError("metrics need at least one observation")
    if not (np.isfinite(obs).all() and np.isfinite(pred).all()):
        raise ValueError("obs and pred must be finite")
    return obs, pred


def _defined(denominator: np.ndarray, threshold: float) -> np.ndarray:
    """Whether a denominator is usable: not below ``threshold`` and not zero."""
    return (denominator >= threshold) & (denominator != 0)


def _ratio(num: np.ndarray, den: np.ndarray, ok: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """``num / den`` where ``ok``, ``fill`` elsewhere (without division warnings)."""
    num, den, ok = np.broadcast_arrays(num, den, ok)
    out = np.full(num.shape, fill, dtype=np.float64)
    np.divide(num, den, out=out, where=ok)
    return out[()] if out.ndim == 0 else out


def _by_basin(
    obs: np.ndarray, pred: np.ndarray, basin_id: np.ndarray
) -> Iterator[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """Yield ``(basin_id, (obs_b, pred_b))`` with the ``(n_b, H)`` rows of each basin, sorted by id."""
    obs, pred = _pair(obs, pred)
    basin_id = np.asarray(basin_id)
    if obs.ndim != 2 or basin_id.shape != obs.shape[:1]:
        raise ValueError(
            f"expected obs (N, H) and basin_id (N,), got {obs.shape}, {basin_id.shape}"
        )
    ids, inverse = np.unique(basin_id, return_inverse=True)
    for k, b in enumerate(ids.tolist()):
        rows = inverse == k
        yield b, (obs[rows], pred[rows])
