"""Download the course dataset (Rainfall-Runoff) from its public Google Drive folder.

Folder: https://drive.google.com/drive/folders/1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w

Usage:
    uv run python -m clamf.data.download [--dest data/raw] [--force]
    python src/clamf/data/download.py        # on machines without uv (stdlib only)
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# File name -> Google Drive file id, smallest first.
FILES = {
    "metadata.json": "1TNmzf7Uk4qkzOkuYf0BmhhknDapPwcQO",
    "test_targets.csv": "1PFpyBgKK-Vpt1dWl5X9bSCxteQ01j5Xg",
    "test.h5": "1h2A-0tNqBWVuxk-zwNWmOK6-gLwnbWuJ",
    "train.h5": "1Ou6LEqX0eflNhf0AA_qp-lXd0SPlm6GG",
}


def download(dest: Path = Path("data/raw"), force: bool = False) -> None:
    """Download every file in ``FILES`` into ``dest``, skipping the ones already present."""
    dest.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        target = dest / name
        if target.exists() and not force:
            print(f"skip {name} (already exists)")
            continue
        print(f"downloading {name} ...", flush=True)
        part = target.with_name(name + ".part")
        with urllib.request.urlopen(URL.format(id=file_id)) as response, part.open("wb") as f:
            shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
        part.replace(target)
    print(f"done: {dest.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("data/raw"), help="output directory")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    download(args.dest, args.force)


if __name__ == "__main__":
    main()
