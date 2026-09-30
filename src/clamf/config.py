"""Typed experiment configuration loaded from YAML.

An experiment YAML overrides ``configs/base.yaml`` (deep merge). Unknown keys, wrong types and
invalid values raise :class:`ConfigError` at load time.
"""

import dataclasses
import itertools
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_BASE = Path("configs/base.yaml")


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization scheme (D-007): statistics are fitted on the train split only."""

    meteo: Literal["global_zscore"] = "global_zscore"
    discharge: Literal["basin_zscore"] = "basin_zscore"
    discharge_std_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.discharge_std_floor <= 0:
            raise ConfigError("data.normalization.discharge_std_floor must be > 0")


@dataclass(frozen=True)
class DataConfig:
    """Dataset location, window lengths and loader settings."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    history_hours: int = 336
    horizon_hours: int = 48
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 0  # DataLoader only (preload_to_device: false)
    preload_to_device: bool = True  # D-004: load each split onto the device, no DataLoader

    def __post_init__(self) -> None:
        for name in ("history_hours", "horizon_hours", "batch_size", "eval_batch_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"data.{name} must be > 0")
        if self.num_workers < 0:
            raise ConfigError("data.num_workers must be >= 0")


@dataclass(frozen=True)
class ModelConfig:
    """CLAMF-Former architecture (paper Table 2) and component flags for ablations/baseline.

    ``use_causal_encoder`` is CAM and ``use_lag_aware_cross_attn`` is LAAM (Sec. 2.2.2);
    ``use_msfm`` is the multi-scale fusion module (Sec. 2.2.3, D-005).
    """

    use_msfm: bool = True
    use_causal_encoder: bool = True
    use_lag_aware_cross_attn: bool = True
    d_model: int = 64
    d_fusion: int = 64
    n_heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    msfm_scales: tuple[int, ...] = (1, 24, 96)  # hours; D-002
    msfm_conv_kernel: int = 3  # odd, symmetric zero padding; D-005, D-010
    positional_encoding: Literal["sinusoidal"] = "sinusoidal"  # D-012
    lag_temperature: float = 1.0  # T of the soft lag-aware mask; D-001
    lag_margin: float = 0.5  # delta of the soft lag-aware mask; D-001
    lag_eps: float = 1e-6  # epsilon of the content aggregation, Eq. (2); D-009
    lag_tau_init_bias: float = -4.0  # initial b2 of Eq. (4), tau ~ 0.02 steps; D-001, D-009
    fused_attention: bool = True  # scaled_dot_product_attention unless weights are requested; D-004

    def __post_init__(self) -> None:
        for name in ("d_model", "d_fusion", "n_heads", "encoder_layers", "decoder_layers", "d_ff"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"model.{name} must be > 0")
        if self.d_model % self.n_heads:
            raise ConfigError("model.d_model must be divisible by model.n_heads")
        if self.d_fusion % self.n_heads:
            raise ConfigError("model.d_fusion must be divisible by model.n_heads")  # MSFM fusion
        if not 0 <= self.dropout < 1:
            raise ConfigError("model.dropout must be in [0, 1)")
        scales = self.msfm_scales
        if not scales or scales[0] != 1 or any(b <= a for a, b in itertools.pairwise(scales)):
            raise ConfigError("model.msfm_scales must start at 1 and be strictly increasing")
        if self.msfm_conv_kernel <= 0 or self.msfm_conv_kernel % 2 == 0:
            raise ConfigError("model.msfm_conv_kernel must be a positive odd integer")
        if self.lag_temperature <= 0:
            raise ConfigError("model.lag_temperature must be > 0")
        if self.lag_margin < 0:
            raise ConfigError("model.lag_margin must be >= 0")
        if self.lag_eps <= 0:
            raise ConfigError("model.lag_eps must be > 0")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization budget (paper Table 3, D-004)."""

    optimizer: Literal["adam"] = "adam"
    lr: float = 1e-3
    loss: Literal["freqmae", "mse", "mae"] = "freqmae"
    max_epochs: int = 200
    early_stopping_patience: int = 20  # epochs without val-loss improvement
    early_stopping_min_delta: float = 0.0  # improvement needed to reset patience; D-012
    amp: Literal["bf16", "none"] = "bf16"  # CUDA only; ignored on MPS/CPU (D-004)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ConfigError("train.lr must be > 0")
        for name in ("max_epochs", "early_stopping_patience"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"train.{name} must be > 0")
        if self.early_stopping_min_delta < 0:
            raise ConfigError("train.early_stopping_min_delta must be >= 0")


@dataclass(frozen=True)
class EvalConfig:
    """Metric settings (D-003); thresholds are in original units (mm/h)."""

    min_obs_std: float = 1e-3  # below it, NSE and KGE are NaN for that basin
    min_obs_mean: float = 1e-3  # below it, BIAS, TPE and KGE are NaN for that basin
    tpe_top_fraction: float = 0.02  # TPE-2 %: share of highest observed hours

    def __post_init__(self) -> None:
        for name in ("min_obs_std", "min_obs_mean"):
            if getattr(self, name) < 0:
                raise ConfigError(f"eval.{name} must be >= 0")
        if not 0 < self.tpe_top_fraction <= 1:
            raise ConfigError("eval.tpe_top_fraction must be in (0, 1]")


@dataclass(frozen=True)
class LoggingConfig:
    """MLflow settings; the tracking URI comes from ``MLFLOW_TRACKING_URI``."""

    experiment: str = "dev"

    def __post_init__(self) -> None:
        if not self.experiment:
            raise ConfigError("logging.experiment must not be empty")


@dataclass(frozen=True)
class Config:
    """Root configuration."""

    seed: int = 2025
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        # MaxPool with stride = k and no padding (D-002, D-005) needs k to divide the sequence.
        seq_len = self.data.history_hours + self.data.horizon_hours
        if self.model.use_msfm and any(seq_len % k for k in self.model.msfm_scales):
            raise ConfigError(
                f"every model.msfm_scales value must divide history + horizon = {seq_len}"
            )


def load_config(path: str | Path, base: str | Path | None = DEFAULT_BASE) -> Config:
    """Load ``path`` merged over ``base`` (skip the merge with ``base=None``)."""
    path = Path(path)
    raw: dict[str, Any] = {}
    if base is not None and Path(base).resolve() != path.resolve():
        raw = _read_yaml(Path(base))
    raw = _deep_merge(raw, _read_yaml(path))
    return _build(Config, raw, prefix="")


def to_dict(cfg: Config) -> dict[str, Any]:
    """Return the configuration as a nested plain dict."""
    return dataclasses.asdict(cfg)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return content


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(cls: type, raw: Any, prefix: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(prefix + k for k in unknown)}")
    kwargs = {name: _coerce(hints[name], value, prefix + name) for name, value in raw.items()}
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, name: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, prefix=name + ".")
    origin = typing.get_origin(hint)
    if origin is Literal:
        allowed = typing.get_args(hint)
        if value not in allowed:
            raise ConfigError(f"{name}: {value!r} not in {list(allowed)}")
        return value
    if origin is tuple:
        (item, _) = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{name}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(item, v, f"{name}[{i}]") for i, v in enumerate(value))
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
        raise ConfigError(f"{name}: expected {hint.__name__}, got {type(value).__name__}")
    return value
