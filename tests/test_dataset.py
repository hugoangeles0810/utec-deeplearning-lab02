import pickle

import numpy as np
import pytest
import torch

from clamf.config import Config
from clamf.data.dataset import (
    MissingFutureMeteoError,
    RainfallRunoffDataset,
    build_dataloaders,
    load_scalers,
)
from clamf.data.prepare import prepare
from tests.synthetic import HIST, HOR, TARGET, RawData


@pytest.fixture
def cache(raw: RawData, cfg: Config) -> str:
    return str(prepare(cfg))


def test_sample_shapes_and_zero_horizon(cache: str) -> None:
    sample = RainfallRunoffDataset(cache, "train", HIST, HOR)[0]
    assert sample["enc_x"].shape == (HIST + HOR, 11)
    assert sample["dec_x"].shape == (HIST + HOR, 1)
    assert sample["target"].shape == (HOR,)
    assert all(sample[k].dtype == torch.float32 for k in ("enc_x", "dec_x", "target"))
    assert sample["basin_id"].dtype == torch.int64
    assert torch.all(sample["dec_x"][-HOR:] == 0)


def test_sample_content_matches_raw(raw: RawData, cache: str) -> None:
    scalers = load_scalers(cache)
    ds = RainfallRunoffDataset(cache, "val", HIST, HOR)
    s = ds[2]
    row = int(s["row_id"])
    assert raw.split[row] == 1
    assert int(s["basin_id"]) == raw.basin_id[row]
    np.testing.assert_allclose(s["enc_x"][-HOR:], scalers.normalize_meteo(raw.y_aux[row]))
    np.testing.assert_allclose(s["enc_x"][:HIST], scalers.normalize_meteo(raw.X[row, :, :TARGET]))
    q = scalers.denormalize_q(s["dec_x"][:HIST, 0][None], s["basin_id"][None])
    np.testing.assert_allclose(q[0], raw.X[row, :, TARGET], rtol=1e-5, atol=1e-6)
    y = scalers.denormalize_q(s["target"][None], s["basin_id"][None])
    np.testing.assert_allclose(y[0], raw.y[row], rtol=1e-5, atol=1e-6)


def test_history_and_horizon_are_cropped(cache: str) -> None:
    full = RainfallRunoffDataset(cache, "train", HIST, HOR)[1]
    short = RainfallRunoffDataset(cache, "train", 10, 4)[1]
    assert short["enc_x"].shape == (14, 11)
    torch.testing.assert_close(short["enc_x"][:10], full["enc_x"][HIST - 10 : HIST])
    torch.testing.assert_close(short["enc_x"][10:], full["enc_x"][HIST : HIST + 4])
    torch.testing.assert_close(short["target"], full["target"][:4])
    with pytest.raises(ValueError, match="history_hours"):
        RainfallRunoffDataset(cache, "train", HIST + 1, HOR)


def test_target_does_not_leak_into_inputs(raw: RawData, cfg: Config, tmp_path) -> None:
    before = RainfallRunoffDataset(prepare(cfg), "train", HIST, HOR)[0]
    raw.y = raw.y * 3 + 1
    raw.write()
    after = RainfallRunoffDataset(prepare(cfg, force=True), "train", HIST, HOR)[0]
    torch.testing.assert_close(before["enc_x"], after["enc_x"])
    torch.testing.assert_close(before["dec_x"], after["dec_x"])
    assert not torch.allclose(before["target"], after["target"])


def test_test_split_without_y_aux_raises(cache: str) -> None:
    with pytest.raises(MissingFutureMeteoError, match="D-013"):
        RainfallRunoffDataset(cache, "test", HIST, HOR)


def test_dataloaders_skip_test_and_batch(cache: str, cfg: Config) -> None:
    loaders = build_dataloaders(cfg)
    assert set(loaders) == {"train", "val"}
    batch = next(iter(loaders["val"]))
    assert batch["enc_x"].shape == (8, HIST + HOR, 11)
    assert batch["dec_x"].shape == (8, HIST + HOR, 1)
    assert batch["target"].shape == (8, HOR)


def order(cfg: Config) -> list[int]:
    return torch.cat([b["row_id"] for b in build_dataloaders(cfg)["train"]]).tolist()


def test_train_order_is_reproducible(cache: str, cfg: Config) -> None:
    first = order(cfg)
    assert first == order(cfg)
    assert first != sorted(first)


def test_pickled_dataset_does_not_carry_arrays(cache: str) -> None:
    ds = RainfallRunoffDataset(cache, "train", HIST, HOR)
    ds[0]
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._arrays is None
    torch.testing.assert_close(clone[0]["enc_x"], ds[0]["enc_x"])
