"""The experiment grid in ``configs/experiments/`` (D-018)."""

import dataclasses
import itertools
from pathlib import Path

import pytest

from clamf.config import Config, load_config

REPO = Path(__file__).parents[1]
BASE = REPO / "configs" / "base.yaml"
EXPERIMENTS = sorted((REPO / "configs" / "experiments").glob("*.yaml"))
GRID = {"vanilla", "cam", "laam", "claam", "msfm", "clamf"}
FLAGS = ("use_msfm", "use_causal_encoder", "use_lag_aware_cross_attn")


def flags(cfg: Config) -> tuple[bool, ...]:
    return tuple(getattr(cfg.model, name) for name in FLAGS)


def grid_configs() -> dict[str, Config]:
    return {path.stem: load_config(path, base=BASE) for path in EXPERIMENTS if path.stem in GRID}


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_loads_and_is_named_after_its_file(path: Path) -> None:
    cfg = load_config(path, base=BASE)
    assert cfg.logging.run_name == path.stem


def test_grid_is_every_distinct_studied_combination() -> None:
    configs = grid_configs()
    assert set(configs) == GRID
    assert all(cfg.logging.experiment == "clamf-grid" for cfg in configs.values())
    combos = {flags(cfg) for cfg in configs.values()}
    assert len(combos) == len(configs)  # no two runs train the same model
    without_msfm = set(itertools.product([False], [False, True], [False, True]))  # Table 6
    assert combos == without_msfm | {(True, False, False), (True, True, True)}  # + Table 5


def test_grid_only_changes_flags_and_logging() -> None:
    """Same data, budget and hyperparameters for every run (D-004)."""
    base = load_config(BASE, base=None)
    for name, cfg in grid_configs().items():
        model = dataclasses.replace(cfg.model, **{f: getattr(base.model, f) for f in FLAGS})
        assert dataclasses.replace(cfg, model=model, logging=base.logging) == base, name
