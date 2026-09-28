"""Shared fixtures: a tiny synthetic dataset (see ``tests/synthetic.py``)."""

from pathlib import Path

import pytest

from clamf.config import Config
from tests.synthetic import RawData, make_config, make_raw


@pytest.fixture
def raw(tmp_path: Path) -> RawData:
    data = make_raw(tmp_path / "raw")
    data.write()
    return data


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path)
