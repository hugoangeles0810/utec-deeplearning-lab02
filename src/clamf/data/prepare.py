"""Build the normalized ``.npy`` cache the datasets read (docs/data_preparation.md §6).

Usage:
    uv run python -m clamf.data.prepare --config configs/base.yaml [--force] [--skip-overlap-check]

Layout of ``processed_dir``::

    scalers.json, manifest.json
    {train,val,test}/x_meteo.npy  (N, 336, 11)  normalized meteorology history
                    q_hist.npy    (N, 336)      normalized discharge history
                    y_aux.npy     (N, 48, 11)   normalized future meteorology (test: only if given)
                    target.npy    (N, 48)       normalized future discharge
                    basin_id.npy  (N,)          int64
                    row_id.npy    (N,)          int64, row index in the source file (test ``Id``)
"""

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clamf.config import Config, NormalizationConfig, load_config
from clamf.data.io import (
    METADATA_FILE,
    TEST_FILE,
    TEST_TARGETS_FILE,
    TRAIN_FILE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    DatasetSpec,
    DataValidationError,
    check_finite,
    load_metadata,
    load_test_targets,
    validate_test_h5,
    validate_train_h5,
)
from clamf.data.scalers import Scalers, fit_scalers

MANIFEST_FILE = "manifest.json"
SCALERS_FILE = "scalers.json"
MANIFEST_VERSION = 1
SPLITS = ("train", "val", "test")


class StaleCacheError(RuntimeError):
    """Raised when ``processed_dir`` was built from other raw files or another normalization."""


def prepare(
    cfg: Config, force: bool = False, check_overlap: bool = True, chunk_rows: int = 4096
) -> Path:
    """Validate the raw files, fit the scalers and write the cache; return ``processed_dir``."""
    raw_dir = Path(cfg.data.raw_dir)
    out_dir = Path(cfg.data.processed_dir)
    expected = expected_manifest(raw_dir, cfg.data.normalization)
    current = _read_manifest(out_dir)
    if current is not None and not force:
        if _same_build(current, expected):
            print(f"cache up to date: {out_dir}")
            return out_dir
        raise StaleCacheError(f"{out_dir} was built from other inputs; rerun with --force")
    if out_dir.exists() and current is None and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty and has no {MANIFEST_FILE}; not touching it")

    spec = load_metadata(raw_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    with (
        h5py.File(raw_dir / TRAIN_FILE, "r") as train_f,
        h5py.File(raw_dir / TEST_FILE, "r") as test_f,
    ):
        validate_train_h5(train_f, spec)
        validate_test_h5(test_f, spec, train_basins=np.unique(train_f["basin_id"][:]))
        test_targets = load_test_targets(raw_dir / TEST_TARGETS_FILE, spec)

        print("fitting scalers on train ...", flush=True)
        scalers = fit_scalers(train_f, spec, cfg.data.normalization.discharge_std_floor, chunk_rows)
        scalers.to_json(tmp_dir / SCALERS_FILE)

        print("writing train/val ...", flush=True)
        hashes = _write_train_val(train_f, spec, scalers, tmp_dir, chunk_rows, check_overlap)
        print("writing test ...", flush=True)
        hashes["test"] = _write_test(
            test_f, test_targets, spec, scalers, tmp_dir, chunk_rows, check_overlap
        )

    if check_overlap:
        print("checking shared hours between splits ...", flush=True)
        check_no_shared_hours(hashes)

    expected["splits"] = {
        split: {
            "n": int(np.load(tmp_dir / split / "row_id.npy", mmap_mode="r").shape[0]),
            "has_y_aux": (tmp_dir / split / "y_aux.npy").exists(),
        }
        for split in SPLITS
    }
    (tmp_dir / MANIFEST_FILE).write_text(json.dumps(expected, indent=1))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    print(f"done: {out_dir.resolve()}")
    return out_dir


def expected_manifest(raw_dir: Path, normalization: NormalizationConfig) -> dict[str, Any]:
    """Fingerprint of the inputs a cache depends on (raw file size/mtime and normalization)."""
    sources = {}
    for name in (METADATA_FILE, TRAIN_FILE, TEST_FILE, TEST_TARGETS_FILE):
        stat = (raw_dir / name).stat()
        sources[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "version": MANIFEST_VERSION,
        "sources": sources,
        "normalization": dataclasses.asdict(normalization),
    }


def verify_cache(cfg: Config) -> dict[str, Any]:
    """Return the manifest of ``processed_dir``; raise if it is missing or stale."""
    out_dir = Path(cfg.data.processed_dir)
    current = _read_manifest(out_dir)
    if current is None:
        raise FileNotFoundError(
            f"no cache in {out_dir}; run: uv run python -m clamf.data.prepare --config <yaml>"
        )
    if not _same_build(current, expected_manifest(Path(cfg.data.raw_dir), cfg.data.normalization)):
        raise StaleCacheError(f"{out_dir} is stale; rerun clamf.data.prepare with --force")
    return current


def check_no_shared_hours(hashes: dict[str, np.ndarray]) -> None:
    """Raise if any hourly row of val or test also appears in train (row hashes, see below)."""
    train = np.unique(hashes["train"])
    for split in ("val", "test"):
        shared = int(np.isin(hashes[split], train).sum())
        if shared:
            raise DataValidationError(f"{split} shares {shared} hourly rows with train")


def hash_hours(x: np.ndarray) -> np.ndarray:
    """64-bit hash of each hourly row of ``x`` ``(..., n_channels)`` float32 -> ``(...)`` uint64."""
    words = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    h = np.full(words.shape[:-1], 0xCBF29CE484222325, dtype=np.uint64)
    for c in range(words.shape[-1]):
        h = (h ^ words[..., c]) * np.uint64(0x100000001B3)
    return h


def _write_train_val(
    f: h5py.File,
    spec: DatasetSpec,
    scalers: Scalers,
    out: Path,
    chunk_rows: int,
    collect_hashes: bool,
) -> dict[str, np.ndarray]:
    split = f["split"][:]
    basin = f["basin_id"][:].astype(np.int64)
    names = {"train": TRAIN_SPLIT, "val": VAL_SPLIT}
    writers = {
        name: _SplitWriter(out / name, int(np.count_nonzero(split == code)), spec, with_y_aux=True)
        for name, code in names.items()
    }
    hashes: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for start in range(0, len(split), chunk_rows):
        stop = min(start + chunk_rows, len(split))
        x, y, y_aux = f["X"][start:stop], f["y"][start:stop], f["y_aux"][start:stop]
        for arr, key in ((x, "X"), (y, "y"), (y_aux, "y_aux")):
            check_finite(arr, f"train.h5 {key}[{start}:{stop}]")
        for name, code in names.items():
            keep = split[start:stop] == code
            if not keep.any():
                continue
            rows = np.arange(start, stop)[keep]
            writers[name].write(
                scalers, spec, x[keep], basin[start:stop][keep], rows, y=y[keep], y_aux=y_aux[keep]
            )
            if collect_hashes:
                future = np.empty(
                    (int(keep.sum()), spec.horizon_hours, spec.n_channels), np.float32
                )
                future[..., list(spec.meteo_channels)] = y_aux[keep]
                future[..., spec.target_channel] = y[keep]
                hashes[name] += [hash_hours(x[keep]).ravel(), hash_hours(future).ravel()]
    for writer in writers.values():
        writer.close()
    return {name: np.concatenate(h) if h else np.empty(0, np.uint64) for name, h in hashes.items()}


def _write_test(
    f: h5py.File,
    targets: np.ndarray,
    spec: DatasetSpec,
    scalers: Scalers,
    out: Path,
    chunk_rows: int,
    collect_hashes: bool,
) -> np.ndarray:
    n = f["X"].shape[0]
    has_y_aux = "y_aux" in f
    basin = f["basin_id"][:].astype(np.int64)
    writer = _SplitWriter(out / "test", n, spec, with_y_aux=has_y_aux)
    hashes = []
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        x = f["X"][start:stop]
        check_finite(x, f"test.h5 X[{start}:{stop}]")
        y_aux = f["y_aux"][start:stop] if has_y_aux else None
        if y_aux is not None:
            check_finite(y_aux, f"test.h5 y_aux[{start}:{stop}]")
        writer.write(
            scalers,
            spec,
            x,
            basin[start:stop],
            np.arange(start, stop),
            y=targets[start:stop],
            y_aux=y_aux,
        )
        if collect_hashes:
            hashes.append(hash_hours(x).ravel())
    writer.close()
    return np.concatenate(hashes) if hashes else np.empty(0, np.uint64)


class _SplitWriter:
    """Sequentially fills the ``.npy`` memmaps of one split."""

    def __init__(self, out: Path, n: int, spec: DatasetSpec, with_y_aux: bool) -> None:
        out.mkdir(parents=True)
        n_meteo, hist, hor = len(spec.meteo_channels), spec.history_hours, spec.horizon_hours
        shapes: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
            "x_meteo": ((n, hist, n_meteo), np.float32),
            "q_hist": ((n, hist), np.float32),
            "target": ((n, hor), np.float32),
            "basin_id": ((n,), np.int64),
            "row_id": ((n,), np.int64),
        }
        if with_y_aux:
            shapes["y_aux"] = ((n, hor, n_meteo), np.float32)
        self.arrays = {
            name: np.lib.format.open_memmap(
                out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )
            for name, (shape, dtype) in shapes.items()
        }
        self.n = n
        self.pos = 0

    def write(
        self,
        scalers: Scalers,
        spec: DatasetSpec,
        x: np.ndarray,
        basin: np.ndarray,
        rows: np.ndarray,
        y: np.ndarray,
        y_aux: np.ndarray | None,
    ) -> None:
        sl = slice(self.pos, self.pos + len(x))
        a = self.arrays
        a["x_meteo"][sl] = scalers.normalize_meteo(x[..., list(spec.meteo_channels)])
        a["q_hist"][sl] = scalers.normalize_q(x[..., spec.target_channel], basin)
        a["target"][sl] = scalers.normalize_q(y, basin)
        a["basin_id"][sl] = basin
        a["row_id"][sl] = rows
        if y_aux is not None:
            a["y_aux"][sl] = scalers.normalize_meteo(y_aux)
        self.pos = sl.stop

    def close(self) -> None:
        if self.pos != self.n:
            raise RuntimeError(f"wrote {self.pos} rows, expected {self.n}")
        for arr in self.arrays.values():
            arr.flush()
        self.arrays.clear()


def _read_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def _same_build(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(current.get(key) == value for key, value in expected.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    parser.add_argument("--skip-overlap-check", action="store_true")
    args = parser.parse_args()
    prepare(load_config(args.config), force=args.force, check_overlap=not args.skip_overlap_check)


if __name__ == "__main__":
    main()
