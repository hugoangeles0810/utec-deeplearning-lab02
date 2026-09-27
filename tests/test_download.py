import io
from pathlib import Path

import pytest

from clamf.data import download as dl


@pytest.fixture
def fake_urlopen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def urlopen(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(b"payload")

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl, "FILES", {"a.bin": "id-a", "b.bin": "id-b"})
    return calls


def test_downloads_all_files(tmp_path: Path, fake_urlopen: list[str]) -> None:
    dl.download(tmp_path)

    assert (tmp_path / "a.bin").read_bytes() == b"payload"
    assert (tmp_path / "b.bin").read_bytes() == b"payload"
    assert not list(tmp_path.glob("*.part"))
    assert len(fake_urlopen) == 2


def test_skips_existing_files(tmp_path: Path, fake_urlopen: list[str]) -> None:
    (tmp_path / "a.bin").write_bytes(b"old")

    dl.download(tmp_path)

    assert (tmp_path / "a.bin").read_bytes() == b"old"
    assert fake_urlopen == [dl.URL.format(id="id-b")]
