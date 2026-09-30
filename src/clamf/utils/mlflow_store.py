"""Copy a SQLite MLflow store between machines (docs/runpod.md).

    uv run python -m clamf.utils.mlflow_store snapshot --db mlflow.db --out mlflow.snapshot.db
    uv run python -m clamf.utils.mlflow_store relocate --db <db> --from <old root> --to <new root>

``snapshot`` makes a consistent copy with SQLite's online backup, so it can run while training
writes to the store. The store keeps absolute artifact paths (``/workspace/.../mlruns/1/...``);
after copying the store and its ``mlruns/`` to another machine, ``relocate`` rewrites that prefix so
MLflow finds the artifacts in their new place.
"""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

# (table, column) holding artifact locations; tables missing from a store are skipped.
ARTIFACT_COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
)


def snapshot(db: str | Path, out: str | Path) -> Path:
    """Consistent copy of the SQLite file ``db`` into ``out`` (replaced if it exists)."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    with (
        closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(tmp)) as dst,
    ):
        src.backup(dst)
    tmp.replace(out)
    return out


def relocate(db: str | Path, old: str, new: str) -> int:
    """Replace the ``old`` path prefix of every artifact location with ``new``, both as plain paths
    and as ``file://`` URIs; returns the number of rows changed. Both must be absolute and neither a
    prefix of the other, so running it twice is a no-op."""
    old, new = old.rstrip("/"), new.rstrip("/")
    prefixes = [(old, new), (Path(old).as_uri(), Path(new).as_uri())]
    changed = 0
    with closing(sqlite3.connect(db)) as conn, conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, column in ARTIFACT_COLUMNS:
            if table not in tables:
                continue
            for src, dst in prefixes:
                # prefix match only: the old root or anything below it
                cursor = conn.execute(
                    f"UPDATE {table} SET {column} = ? || substr({column}, ?) "
                    f"WHERE {column} = ? OR substr({column}, 1, ?) = ?",
                    (dst, len(src) + 1, src, len(src) + 1, src + "/"),
                )
                changed += cursor.rowcount
    return changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy a SQLite MLflow store between machines.")
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="consistent copy of the store")
    snap.add_argument("--db", default="mlflow.db", help="SQLite store to copy")
    snap.add_argument("--out", default="mlflow.snapshot.db", help="copy to write")
    move = sub.add_parser("relocate", help="rewrite the artifact path prefix")
    move.add_argument("--db", required=True, help="SQLite store to rewrite in place")
    move.add_argument("--from", dest="old", required=True, help="old root, e.g. /workspace/repo")
    move.add_argument("--to", dest="new", required=True, help="new root (absolute)")
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        print(f"snapshot: {snapshot(args.db, args.out)}")
    else:
        print(f"relocated {relocate(args.db, args.old, args.new)} artifact locations")


if __name__ == "__main__":
    main()
