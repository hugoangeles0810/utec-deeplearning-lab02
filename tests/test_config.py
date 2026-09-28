from pathlib import Path

import pytest

from clamf.config import Config, ConfigError, load_config

REPO_BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_repo_base_matches_dataclass_defaults() -> None:
    assert load_config(REPO_BASE, base=None) == Config()


def test_experiment_overrides_base(tmp_path: Path) -> None:
    base = write(tmp_path / "base.yaml", "seed: 1\ndata:\n  batch_size: 16\n  num_workers: 2\n")
    exp = write(
        tmp_path / "exp.yaml",
        "data:\n  batch_size: 32\n  normalization:\n    discharge_std_floor: 1\n",
    )

    cfg = load_config(exp, base=base)

    assert cfg.seed == 1
    assert cfg.data.batch_size == 32
    assert cfg.data.num_workers == 2
    assert cfg.data.normalization.discharge_std_floor == 1.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("data:\n  batch_sise: 3\n", "data.batch_sise"),
        ("foo: 1\n", "foo"),
        ("data:\n  batch_size: '8'\n", "expected int"),
        ("data:\n  batch_size: true\n", "expected int"),
        ("device: tpu\n", "not in"),
        ("data:\n  normalization:\n    discharge: log\n", "not in"),
        ("data:\n  history_hours: 0\n", "history_hours"),
        ("data:\n  normalization:\n    discharge_std_floor: 0\n", "discharge_std_floor"),
        ("data: 3\n", "expected a mapping"),
    ],
)
def test_invalid_configs_raise(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path / "exp.yaml", text), base=None)
