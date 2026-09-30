#!/usr/bin/env bash
# Copy the pod's MLflow store, checkpoints and logs into results/runpod/ and point the store's
# artifact paths there (run it on the Mac; docs/runpod.md). Safe to run while the grid trains.
#
#   bash scripts/runpod/pull.sh root@<pod ip> <ssh port>
#   uv run mlflow ui --backend-store-uri sqlite:///results/runpod/mlflow.db
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: $0 <user@host> [ssh port]" >&2
  exit 2
fi
HOST="$1"
SSH="ssh -p ${2:-22} ${SSH_OPTS:-}"
REMOTE=/workspace/utec-deeplearning-lab02
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/results/runpod"
mkdir -p "$OUT"

echo "== snapshot on the pod"
$SSH "$HOST" "source $REMOTE/scripts/runpod/env.sh && cd $REMOTE && mkdir -p mlruns checkpoints logs \
  && uv run python -m clamf.utils.mlflow_store snapshot --db mlflow.db --out mlflow.snapshot.db"

echo "== copy"
rsync -az -e "$SSH" "$HOST:$REMOTE/mlflow.snapshot.db" "$OUT/mlflow.snapshot.db"
for dir in mlruns checkpoints logs; do
  rsync -az --delete -e "$SSH" "$HOST:$REMOTE/$dir/" "$OUT/$dir/"
done

echo "== relocate"
cp "$OUT/mlflow.snapshot.db" "$OUT/mlflow.db"
cd "$REPO"
uv run python -m clamf.utils.mlflow_store relocate --db "$OUT/mlflow.db" --from "$REMOTE" --to "$OUT"
echo "done: uv run mlflow ui --backend-store-uri sqlite:///$OUT/mlflow.db"
