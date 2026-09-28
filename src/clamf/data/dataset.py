"""Datasets and DataLoaders over the normalized cache (docs/data_preparation.md §2, §6).

Each sample is a dict of tensors (``L = history_hours``, ``H = horizon_hours``):

- ``enc_x``    ``(L + H, n_meteo)`` float32: meteorology history followed by ``y_aux`` (D-006).
- ``dec_x``    ``(L + H, 1)`` float32: discharge history followed by ``H`` zeros (normalized space).
- ``target``   ``(H,)`` float32: normalized future discharge; only for the loss, never an input.
- ``basin_id`` ``()`` int64: to denormalize with :class:`~clamf.data.scalers.Scalers`.
- ``row_id``   ``()`` int64: row in the source file (``Id`` for test).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from clamf.config import Config
from clamf.data.prepare import SCALERS_FILE, SPLITS, verify_cache
from clamf.data.scalers import Scalers
from clamf.utils.seed import seed_worker

_ARRAYS = ("x_meteo", "q_hist", "y_aux", "target", "basin_id", "row_id")


class MissingFutureMeteoError(RuntimeError):
    """Raised for a split without ``y_aux`` (test until the professor provides it, D-013)."""


class RainfallRunoffDataset(Dataset[dict[str, torch.Tensor]]):
    """One split of the cache; memmaps are opened lazily so DataLoader workers do not copy them."""

    def __init__(
        self, processed_dir: str | Path, split: str, history_hours: int, horizon_hours: int
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.dir = Path(processed_dir) / split
        if not (self.dir / "y_aux.npy").exists():
            raise MissingFutureMeteoError(
                f"split '{split}' has no y_aux (future meteorology); see D-013 in docs/decisions.md"
            )
        header = np.load(self.dir / "x_meteo.npy", mmap_mode="r")
        horizon = np.load(self.dir / "target.npy", mmap_mode="r").shape[1]
        if not 0 < history_hours <= header.shape[1]:
            raise ValueError(f"history_hours must be in 1..{header.shape[1]}")
        if not 0 < horizon_hours <= horizon:
            raise ValueError(f"horizon_hours must be in 1..{horizon}")
        self.split = split
        self.history_hours = history_hours
        self.horizon_hours = horizon_hours
        self._len = header.shape[0]
        self._arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        a = self._open()
        hist, hor = self.history_hours, self.horizon_hours
        enc = np.concatenate([a["x_meteo"][i, -hist:], a["y_aux"][i, :hor]], axis=0)
        dec = np.concatenate([a["q_hist"][i, -hist:], np.zeros(hor, np.float32)])
        return {
            "enc_x": torch.from_numpy(enc),
            "dec_x": torch.from_numpy(dec).unsqueeze(-1),
            "target": torch.from_numpy(np.array(a["target"][i, :hor])),
            "basin_id": torch.tensor(a["basin_id"][i]),
            "row_id": torch.tensor(a["row_id"][i]),
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in _ARRAYS}
        return self._arrays


def load_scalers(processed_dir: str | Path) -> Scalers:
    """Scalers fitted by ``clamf.data.prepare`` (to denormalize predictions before metrics)."""
    return Scalers.from_json(Path(processed_dir) / SCALERS_FILE)


def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """DataLoaders for train (shuffled, seeded), val and, if it has ``y_aux``, test."""
    manifest = verify_cache(cfg)
    d = cfg.data
    loaders = {}
    for split in SPLITS:
        if not manifest["splits"][split]["has_y_aux"]:
            continue
        dataset = RainfallRunoffDataset(d.processed_dir, split, d.history_hours, d.horizon_hours)
        train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=d.batch_size if train else d.eval_batch_size,
            shuffle=train,
            generator=torch.Generator().manual_seed(cfg.seed) if train else None,
            num_workers=d.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=d.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
