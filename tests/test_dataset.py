import dataclasses
import pickle

import numpy as np
import pytest
import torch

from clamf.config import Config
from clamf.data.dataset import (
    DeviceLoader,
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


def with_preload(cfg: Config, preload: bool) -> Config:
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, preload_to_device=preload))


@pytest.mark.parametrize("preload", [True, False])
def test_dataloaders_skip_test_and_batch(cache: str, cfg: Config, preload: bool) -> None:
    loaders = build_dataloaders(with_preload(cfg, preload), "cpu")
    assert set(loaders) == {"train", "val"}
    batch = next(iter(loaders["val"]))
    assert batch["enc_x"].shape == (8, HIST + HOR, 11)
    assert batch["dec_x"].shape == (8, HIST + HOR, 1)
    assert batch["target"].shape == (8, HOR)


def order(loader: DeviceLoader | torch.utils.data.DataLoader) -> list[int]:
    return torch.cat([b["row_id"] for b in loader]).tolist()


@pytest.mark.parametrize("preload", [True, False])
def test_train_order_is_reproducible(cache: str, cfg: Config, preload: bool) -> None:
    cfg = with_preload(cfg, preload)
    first = order(build_dataloaders(cfg, "cpu")["train"])
    assert first == order(build_dataloaders(cfg, "cpu")["train"])
    assert first != sorted(first)
    assert sorted(first) == sorted(set(first))


@pytest.mark.parametrize("split", ["train", "val"])
def test_preloaded_batches_match_dataset(cache: str, cfg: Config, split: str) -> None:
    ds = RainfallRunoffDataset(cache, split, HIST, HOR)
    loader = build_dataloaders(with_preload(cfg, True), "cpu")[split]
    assert isinstance(loader, DeviceLoader)
    assert loader.num_samples == len(ds)
    batches = list(loader)
    assert len(batches) == len(loader)
    index = {int(ds[i]["row_id"]): i for i in range(len(ds))}
    for batch in batches[:2]:
        for j, row in enumerate(batch["row_id"].tolist()):
            for k, v in ds[index[row]].items():
                assert batch[k].dtype == v.dtype
                torch.testing.assert_close(batch[k][j], v)


def test_eval_loader_keeps_order_and_last_partial_batch(cache: str) -> None:
    ds = RainfallRunoffDataset(cache, "val", HIST, HOR)
    loader = DeviceLoader(ds.load_all("cpu"), batch_size=5)
    sizes = [len(b["row_id"]) for b in loader]
    assert sizes == [5, 5, len(ds) - 10]
    assert order(loader) == [int(ds[i]["row_id"]) for i in range(len(ds))]


def test_load_all_is_independent_of_chunking(cache: str) -> None:
    ds = RainfallRunoffDataset(cache, "train", HIST, HOR)
    whole, chunked = ds.load_all("cpu"), ds.load_all("cpu", chunk_rows=7)
    for k in whole:
        torch.testing.assert_close(whole[k], chunked[k])


def test_generator_state_resumes_the_next_epoch(cache: str) -> None:
    tensors = RainfallRunoffDataset(cache, "train", HIST, HOR).load_all("cpu")
    loader = DeviceLoader(tensors, 8, shuffle=True, generator=torch.Generator().manual_seed(1))
    epoch_1 = order(loader)
    state = loader.generator.get_state()
    epoch_2 = order(loader)
    assert epoch_1 != epoch_2
    resumed = DeviceLoader(tensors, 8, shuffle=True, generator=torch.Generator())
    resumed.generator.set_state(state)
    assert order(resumed) == epoch_2


def test_device_loader_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="first dim"):
        DeviceLoader({"a": torch.zeros(3), "b": torch.zeros(4)}, 2)
    with pytest.raises(ValueError, match="generator"):
        DeviceLoader({"a": torch.zeros(3)}, 2, shuffle=True)


def test_pickled_dataset_does_not_carry_arrays(cache: str) -> None:
    ds = RainfallRunoffDataset(cache, "train", HIST, HOR)
    ds[0]
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._arrays is None
    torch.testing.assert_close(clone[0]["enc_x"], ds[0]["enc_x"])
