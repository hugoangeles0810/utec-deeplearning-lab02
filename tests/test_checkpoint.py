import random
from pathlib import Path

import numpy as np
import pytest
import torch

from clamf.config import ModelConfig
from clamf.data.dataset import DeviceLoader
from clamf.losses import get_loss
from clamf.models.clamf_former import CLAMFFormer
from clamf.utils.checkpoint import (
    BEST,
    LAST,
    load_model,
    load_training_state,
    rng_state,
    run_checkpoint_dir,
    save_model,
    save_training_state,
    set_rng_state,
)
from clamf.utils.early_stopping import EarlyStopping
from clamf.utils.seed import seed_everything

HISTORY, HORIZON, N_METEO, N = 12, 4, 3, 24
CONFIG = {"seed": 7, "model": {"d_model": 8}}


def tiny_model() -> CLAMFFormer:
    cfg = ModelConfig(
        d_model=8, d_fusion=8, n_heads=2, d_ff=16, encoder_layers=1, decoder_layers=1,
        dropout=0.2, msfm_scales=(1, 4),
    )  # fmt: skip
    return CLAMFFormer(cfg, N_METEO, HISTORY, HORIZON)


def windows() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    length = HISTORY + HORIZON
    dec_x = torch.cat([torch.randn(N, HISTORY, 1, generator=g), torch.zeros(N, HORIZON, 1)], 1)
    return {
        "enc_x": torch.randn(N, length, N_METEO, generator=g),
        "dec_x": dec_x,
        "target": torch.randn(N, HORIZON, generator=g),
    }


class Run:
    """A tiny training loop that uses every RNG a real run touches (dropout, shuffling, numpy and
    random)."""

    def __init__(self) -> None:
        seed_everything(7)
        self.model = tiny_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.generator = torch.Generator().manual_seed(7)
        self.loader = DeviceLoader(windows(), batch_size=8, shuffle=True, generator=self.generator)
        self.early_stopping = EarlyStopping(patience=3)
        self.loss = get_loss("freqmae")
        self.noise: list[float] = []

    def epoch(self, epoch: int) -> None:
        self.model.train()
        for batch in self.loader:
            self.optimizer.zero_grad()
            self.loss(self.model(batch["enc_x"], batch["dec_x"]).pred, batch["target"]).backward()
            self.optimizer.step()
        self.noise.append(np.random.rand() + random.random())
        self.early_stopping.step(float(self.noise[-1]), epoch)

    def save(self, path: Path, epoch: int) -> None:
        save_training_state(
            path, model=self.model, optimizer=self.optimizer, epoch=epoch,
            early_stopping=self.early_stopping, generator=self.generator, config=CONFIG,
        )  # fmt: skip

    def load(self, path: Path, config: dict = CONFIG) -> int:
        return load_training_state(
            path, model=self.model, optimizer=self.optimizer,
            early_stopping=self.early_stopping, generator=self.generator, config=config,
        )  # fmt: skip


def scramble_rngs() -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)


def test_resumed_training_matches_uninterrupted(tmp_path: Path) -> None:
    straight = Run()
    for epoch in range(4):
        straight.epoch(epoch)

    first = Run()
    for epoch in range(2):
        first.epoch(epoch)
    first.save(tmp_path / LAST, epoch=1)

    scramble_rngs()
    resumed = Run()  # fresh process: new weights, optimizer and generator
    scramble_rngs()
    last_epoch = resumed.load(tmp_path / LAST)
    assert last_epoch == 1
    for epoch in range(last_epoch + 1, 4):
        resumed.epoch(epoch)

    for (name, a), b in zip(
        straight.model.state_dict().items(), resumed.model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    assert straight.noise[2:] == resumed.noise
    assert straight.early_stopping == resumed.early_stopping


def test_resume_rejects_a_different_config(tmp_path: Path) -> None:
    run = Run()
    run.save(tmp_path / LAST, epoch=0)
    with pytest.raises(ValueError, match="different config"):
        Run().load(tmp_path / LAST, config={**CONFIG, "seed": 8})


def test_checkpoint_loads_with_weights_only(tmp_path: Path) -> None:
    run = Run()
    run.epoch(0)
    run.save(tmp_path / LAST, epoch=0)
    state = torch.load(tmp_path / LAST, weights_only=True)
    assert set(state) == {"model", "optimizer", "epoch", "early_stopping", "rng", "config"}
    assert state["config"] == CONFIG


def test_save_is_atomic_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "run" / LAST  # parent folders are created
    run = Run()
    run.save(path, epoch=0)
    run.save(path, epoch=1)
    assert sorted(p.name for p in path.parent.iterdir()) == [LAST]
    assert torch.load(path, weights_only=True)["epoch"] == 1


def test_best_model_round_trip(tmp_path: Path) -> None:
    trained = Run()
    trained.epoch(0)
    save_model(tmp_path / BEST, trained.model, epoch=5, config=CONFIG)
    model = tiny_model()
    assert load_model(tmp_path / BEST, model) == 5
    for a, b in zip(trained.model.state_dict().values(), model.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)


def test_best_model_checks_only_the_given_config_keys(tmp_path: Path) -> None:
    config = {"model": {"d": 1}, "data": {"hours": 4, "path": "a"}, "eval": {"x": 1}}
    save_model(tmp_path / BEST, tiny_model(), epoch=0, config=config)
    subset = {"model": {"d": 1}, "data": {"hours": 4}}  # data.path and eval are not checked
    assert load_model(tmp_path / BEST, tiny_model(), subset) == 0
    with pytest.raises(ValueError, match=r"different config: model\.d, data\.hours$"):
        load_model(tmp_path / BEST, tiny_model(), {"model": {"d": 2}, "data": {"hours": 5}})


def test_rng_state_round_trip() -> None:
    generator = torch.Generator().manual_seed(1)
    seed_everything(1)
    state = rng_state(generator)
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )
    scramble_rngs()
    generator.manual_seed(99)
    set_rng_state(state, generator)
    assert (random.random(), np.random.rand()) == expected[:2]
    assert torch.equal(torch.rand(1), expected[2])
    assert torch.equal(torch.rand(1, generator=generator), expected[3])


def test_run_checkpoint_dir() -> None:
    assert run_checkpoint_dir("checkpoints", "abc123") == Path("checkpoints/abc123")
