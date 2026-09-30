"""Datasets and batch loaders over the normalized cache (docs/data_preparation.md §2, §6).

Each sample is a dict of tensors (``L = history_hours``, ``H = horizon_hours``):

- ``enc_x``    ``(L + H, n_meteo)`` float32: meteorology history followed by ``y_aux`` (D-006).
- ``dec_x``    ``(L + H, 1)`` float32: discharge history followed by ``H`` zeros (normalized space).
- ``target``   ``(H,)`` float32: normalized future discharge; only for the loss, never an input.
- ``basin_id`` ``()`` int64: to denormalize with :class:`~clamf.data.scalers.Scalers`.
- ``row_id``   ``()`` int64: row in the source file (``Id`` for test).

Batches add a leading ``B`` dim. With ``data.preload_to_device`` (D-004) each split is loaded once
onto the device and batched by :class:`DeviceLoader`; otherwise a ``DataLoader`` reads the memmaps.
"""

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from clamf.config import Config
from clamf.data.prepare import SCALERS_FILE, SPLITS, verify_cache
from clamf.data.scalers import Scalers
from clamf.utils.seed import seed_worker

_ARRAYS = ("x_meteo", "q_hist", "y_aux", "target", "basin_id", "row_id")

Batch = dict[str, torch.Tensor]


class MissingFutureMeteoError(RuntimeError):
    """Raised for a split without ``y_aux`` (test until the professor provides it, D-013)."""


class RainfallRunoffDataset(Dataset[Batch]):
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

    def __getitem__(self, i: int) -> Batch:
        return {k: torch.from_numpy(v) for k, v in self._windows(i).items()}

    def load_all(self, device: torch.device | str, chunk_rows: int = 8192) -> Batch:
        """Whole split as batched tensors on ``device``, read in chunks to bound host memory."""
        out: Batch = {}
        for start in range(0, len(self), chunk_rows):
            rows = slice(start, start + chunk_rows)
            for k, v in self._windows(rows).items():
                if k not in out:
                    shape = (len(self), *v.shape[1:])
                    out[k] = torch.empty(shape, dtype=torch.from_numpy(v).dtype, device=device)
                out[k][rows] = torch.from_numpy(v).to(device)
        return out

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _windows(self, rows: int | slice) -> dict[str, np.ndarray]:
        """Model inputs for one row (no batch dim) or a slice of rows (leading batch dim)."""
        a = self._open()
        hist, hor = self.history_hours, self.horizon_hours
        enc = np.concatenate([a["x_meteo"][rows, -hist:], a["y_aux"][rows, :hor]], axis=-2)
        q = a["q_hist"][rows, -hist:]
        dec = np.concatenate([q, np.zeros((*q.shape[:-1], hor), np.float32)], axis=-1)
        return {
            "enc_x": enc,
            "dec_x": dec[..., None],
            "target": np.array(a["target"][rows, :hor]),
            "basin_id": np.array(a["basin_id"][rows]),
            "row_id": np.array(a["row_id"][rows]),
        }

    def _open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in _ARRAYS}
        return self._arrays


class DeviceLoader:
    """Batches of a split already on the device (D-004): each batch is one index gather.

    Iterates like a ``DataLoader`` with ``drop_last=False``. With ``shuffle``, every pass draws a
    new permutation from ``generator`` (on the CPU), so saving ``generator.get_state()`` at the end
    of an epoch is enough to resume with the same order.
    """

    def __init__(
        self,
        tensors: Batch,
        batch_size: int,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        sizes = {len(t) for t in tensors.values()}
        if len(sizes) != 1:
            raise ValueError(f"tensors must share their first dim, got sizes {sorted(sizes)}")
        if shuffle and generator is None:
            raise ValueError("shuffle needs a seeded generator")
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.num_samples = sizes.pop()

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        device = next(iter(self.tensors.values())).device
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=self.generator).to(device)
            for idx in order.split(self.batch_size):
                yield {k: t[idx] for k, t in self.tensors.items()}
        else:
            for start in range(0, self.num_samples, self.batch_size):
                yield {k: t[start : start + self.batch_size] for k, t in self.tensors.items()}


def load_scalers(processed_dir: str | Path) -> Scalers:
    """Scalers fitted by ``clamf.data.prepare`` (to denormalize predictions before metrics)."""
    return Scalers.from_json(Path(processed_dir) / SCALERS_FILE)


def build_dataloaders(
    cfg: Config, device: torch.device | str
) -> dict[str, DeviceLoader | DataLoader]:
    """Loaders for train (shuffled, seeded), val and, if it has ``y_aux``, test.

    With ``data.preload_to_device`` the batches are already on ``device``; otherwise they are on the
    CPU and the caller moves them.
    """
    manifest = verify_cache(cfg)
    return {
        split: build_loader(cfg, split, device, verify=False)
        for split in SPLITS
        if manifest["splits"][split]["has_y_aux"]
    }


def build_loader(
    cfg: Config, split: str, device: torch.device | str, verify: bool = True
) -> DeviceLoader | DataLoader:
    """Loader for one split, as in :func:`build_dataloaders`; only train is shuffled.

    Raises :class:`MissingFutureMeteoError` for a split without ``y_aux`` (D-013).
    """
    if verify:
        verify_cache(cfg)
    d = cfg.data
    dataset = RainfallRunoffDataset(d.processed_dir, split, d.history_hours, d.horizon_hours)
    train = split == "train"
    batch_size = d.batch_size if train else d.eval_batch_size
    generator = torch.Generator().manual_seed(cfg.seed) if train else None
    if d.preload_to_device:
        tensors = dataset.load_all(device)
        return DeviceLoader(tensors, batch_size, shuffle=train, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=generator,
        num_workers=d.num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=d.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def meteo_channels(loader: DeviceLoader | DataLoader) -> int:
    """Meteorological channels of ``enc_x`` (11 in the dataset, D-006)."""
    if isinstance(loader, DeviceLoader):
        return loader.tensors["enc_x"].shape[-1]
    return loader.dataset[0]["enc_x"].shape[-1]


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """No-op for preloaded batches (D-004); moves ``DataLoader`` batches otherwise."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
