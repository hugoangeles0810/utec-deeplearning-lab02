"""Typed experiment configuration loaded from YAML.

An experiment YAML overrides ``configs/base.yaml`` (deep merge). Unknown keys, wrong types and
invalid values raise :class:`ConfigError` at load time.
"""

import dataclasses
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
    num_workers: int = 0

    def __post_init__(self) -> None:
        for name in ("history_hours", "horizon_hours", "batch_size", "eval_batch_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"data.{name} must be > 0")
        if self.num_workers < 0:
            raise ConfigError("data.num_workers must be >= 0")


@dataclass(frozen=True)
class Config:
    """Root configuration."""

    seed: int = 2025
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    data: DataConfig = field(default_factory=DataConfig)


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
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
        raise ConfigError(f"{name}: expected {hint.__name__}, got {type(value).__name__}")
    return value
