"""Normalization statistics (D-007, docs/data_preparation.md §4).

- Meteorology: global z-score per channel.
- Discharge: per-basin z-score ``(q - mu_b) / sigma_b`` with ``sigma_b >= discharge_std_floor``.

Statistics are fitted on the history ``X`` of the train windows only (``split == 0``). The normalize
and denormalize helpers accept numpy arrays or torch tensors (on any device).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import h5py
import numpy as np
import torch

from clamf.data.io import TRAIN_SPLIT, DatasetSpec, check_finite

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class Scalers:
    """Fitted normalization tables.

    Shapes: ``meteo_mean``/``meteo_std`` ``(n_meteo,)``; ``basin_ids``/``q_mean``/``q_std``
    ``(n_basins,)`` with ``basin_ids`` sorted ascending.
    """

    meteo_names: tuple[str, ...]
    meteo_mean: np.ndarray
    meteo_std: np.ndarray
    basin_ids: np.ndarray
    q_mean: np.ndarray
    q_std: np.ndarray
    discharge_std_floor: float

    def normalize_meteo(self, x: ArrayT) -> ArrayT:
        """``(..., n_meteo)`` raw meteorology -> z-scores."""
        return (x - _like(self.meteo_mean, x)) / _like(self.meteo_std, x)

    def normalize_q(self, q: ArrayT, basin_id: ArrayT) -> ArrayT:
        """``q`` of shape ``(N, ...)`` in mm/h -> per-basin z-scores; ``basin_id`` is ``(N,)``."""
        mean, std = self._basin_stats(basin_id, q)
        return (q - mean) / std

    def denormalize_q(self, q: ArrayT, basin_id: ArrayT) -> ArrayT:
        """Inverse of :meth:`normalize_q`: per-basin z-scores -> mm/h."""
        mean, std = self._basin_stats(basin_id, q)
        return q * std + mean

    def basin_index(self, basin_id: ArrayT) -> ArrayT:
        """Row of each ``basin_id`` in the tables; raises for basins not seen in train."""
        table = _like(self.basin_ids, basin_id)
        if isinstance(basin_id, torch.Tensor):
            basin_id = basin_id.long()
            idx = torch.searchsorted(table, basin_id).clamp(max=len(self.basin_ids) - 1)
        else:
            idx = np.searchsorted(table, basin_id).clip(max=len(self.basin_ids) - 1)
        if bool((table[idx] != basin_id).any()):
            raise KeyError("basin_id not present in the train scalers")
        return idx

    def to_json(self, path: str | Path) -> None:
        payload = {
            "meteo": {
                "names": list(self.meteo_names),
                "mean": self.meteo_mean.tolist(),
                "std": self.meteo_std.tolist(),
            },
            "discharge": {
                "basin_ids": self.basin_ids.tolist(),
                "mean": self.q_mean.tolist(),
                "std": self.q_std.tolist(),
                "std_floor": self.discharge_std_floor,
            },
        }
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> "Scalers":
        payload = json.loads(Path(path).read_text())
        meteo, discharge = payload["meteo"], payload["discharge"]
        return cls(
            meteo_names=tuple(meteo["names"]),
            meteo_mean=np.asarray(meteo["mean"], dtype=np.float32),
            meteo_std=np.asarray(meteo["std"], dtype=np.float32),
            basin_ids=np.asarray(discharge["basin_ids"], dtype=np.int64),
            q_mean=np.asarray(discharge["mean"], dtype=np.float32),
            q_std=np.asarray(discharge["std"], dtype=np.float32),
            discharge_std_floor=float(discharge["std_floor"]),
        )

    def _basin_stats(self, basin_id: ArrayT, like: ArrayT) -> tuple[ArrayT, ArrayT]:
        idx = self.basin_index(basin_id)
        extra = (1,) * (like.ndim - 1)
        mean = _like(self.q_mean, like)[idx].reshape(-1, *extra)
        std = _like(self.q_std, like)[idx].reshape(-1, *extra)
        return mean, std


def fit_scalers(
    f: h5py.File, spec: DatasetSpec, discharge_std_floor: float, chunk_rows: int = 4096
) -> Scalers:
    """Fit :class:`Scalers` on the train windows of an open ``train.h5`` in one chunked pass.

    Sums are accumulated in float64. Hours shared by overlapping train windows are counted once
    per window (D-015).
    """
    split = f["split"][:]
    basin_all = f["basin_id"][:].astype(np.int64)
    basin_ids = np.unique(basin_all[split == TRAIN_SPLIT])
    meteo = list(spec.meteo_channels)
    n_meteo = len(meteo)

    m_sum = np.zeros(n_meteo)
    m_sq = np.zeros(n_meteo)
    q_sum = np.zeros(len(basin_ids))
    q_sq = np.zeros(len(basin_ids))
    q_count = np.zeros(len(basin_ids))
    n_hours = 0

    for start in range(0, len(split), chunk_rows):
        stop = min(start + chunk_rows, len(split))
        keep = split[start:stop] == TRAIN_SPLIT
        if not keep.any():
            continue
        x = f["X"][start:stop][keep].astype(np.float64)
        check_finite(x, f"train.h5 X[{start}:{stop}]")
        xm = x[..., meteo].reshape(-1, n_meteo)
        m_sum += xm.sum(axis=0)
        m_sq += np.square(xm).sum(axis=0)
        n_hours += xm.shape[0]

        q = x[..., spec.target_channel]
        idx = np.searchsorted(basin_ids, basin_all[start:stop][keep])
        q_sum += np.bincount(idx, weights=q.sum(axis=1), minlength=len(basin_ids))
        q_sq += np.bincount(idx, weights=np.square(q).sum(axis=1), minlength=len(basin_ids))
        q_count += np.bincount(idx, minlength=len(basin_ids)) * q.shape[1]

    m_mean = m_sum / n_hours
    m_std = np.sqrt(np.maximum(m_sq / n_hours - m_mean**2, 0.0))
    if (m_std == 0).any():
        constant = [spec.meteo_names[i] for i in np.flatnonzero(m_std == 0)]
        raise ValueError(f"constant meteorological channels in train: {constant}")
    q_mean = q_sum / q_count
    q_std = np.sqrt(np.maximum(q_sq / q_count - q_mean**2, 0.0))

    return Scalers(
        meteo_names=spec.meteo_names,
        meteo_mean=m_mean.astype(np.float32),
        meteo_std=m_std.astype(np.float32),
        basin_ids=basin_ids,
        q_mean=q_mean.astype(np.float32),
        q_std=np.maximum(q_std, discharge_std_floor).astype(np.float32),
        discharge_std_floor=discharge_std_floor,
    )


def _like(table: np.ndarray, ref: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(ref, torch.Tensor):
        dtype = torch.int64 if table.dtype.kind == "i" else ref.dtype
        return torch.as_tensor(table, dtype=dtype, device=ref.device)
    return table
