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


def test_repo_base_matches_paper_setup() -> None:
    cfg = load_config(REPO_BASE, base=None)

    assert (cfg.model.d_model, cfg.model.n_heads, cfg.model.d_ff) == (64, 4, 256)
    assert (cfg.model.encoder_layers, cfg.model.decoder_layers) == (4, 4)
    assert cfg.model.msfm_scales == (1, 24, 96)
    assert (cfg.train.lr, cfg.train.max_epochs, cfg.train.early_stopping_patience) == (
        1e-3,
        200,
        20,
    )
    assert cfg.train.loss == "freqmae"


def test_ablation_flags_and_scales_override(tmp_path: Path) -> None:
    exp = write(
        tmp_path / "exp.yaml",
        "model:\n  use_msfm: false\n  use_causal_encoder: false\n  msfm_scales: [1, 7]\n",
    )

    cfg = load_config(exp, base=REPO_BASE)

    assert not cfg.model.use_msfm and not cfg.model.use_causal_encoder
    assert cfg.model.use_lag_aware_cross_attn
    assert cfg.model.msfm_scales == (1, 7)


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
        ("model:\n  n_heads: 3\n", "divisible"),
        ("model:\n  dropout: 1\n", "dropout"),
        ("model:\n  use_msfm: 1\n", "expected bool"),
        ("model:\n  msfm_scales: 24\n", "expected a list"),
        ("model:\n  msfm_scales: [1, '24']\n", r"msfm_scales\[1\]"),
        ("model:\n  msfm_scales: [24, 96]\n", "start at 1"),
        ("model:\n  msfm_scales: [1, 96, 24]\n", "increasing"),
        ("model:\n  msfm_scales: [1, 7]\n", "divide history"),
        ("model:\n  lag_temperature: 0\n", "lag_temperature"),
        ("model:\n  lag_eps: 0.0\n", "lag_eps"),
        ("train:\n  loss: huber\n", "not in"),
        ("train:\n  amp: fp16\n", "not in"),
        ("train:\n  early_stopping_patience: 0\n", "early_stopping_patience"),
        ("eval:\n  tpe_top_fraction: 0\n", "tpe_top_fraction"),
        ("logging:\n  experiment: ''\n", "experiment"),
    ],
)
def test_invalid_configs_raise(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path / "exp.yaml", text), base=None)
