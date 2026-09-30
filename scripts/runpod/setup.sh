#!/usr/bin/env bash
# Prepare a RunPod pod (docs/runpod.md). Run it after creating the pod and after every restart;
# steps already done are skipped.
#
#   git clone https://github.com/hugoangeles0810/utec-deeplearning-lab02 /workspace/utec-deeplearning-lab02
#   bash /workspace/utec-deeplearning-lab02/scripts/runpod/setup.sh [git ref, default main]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REF="${1:-main}"

echo "== tools"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$WORKSPACE/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux rsync  # rsync: pull.sh copies from the pod
fi

echo "== code ($REF)"
cd "$PROJECT"
git fetch --quiet origin
if git show-ref --verify --quiet "refs/remotes/origin/$REF"; then
  git checkout --quiet -B "$REF" "origin/$REF"
else
  git checkout --quiet --detach "$REF"
fi
git log -1 --oneline

echo "== environment"
uv sync --frozen
uv run python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(
        f"CUDA is not available: torch {torch.__version__} is built for CUDA {torch.version.cuda} "
        "and needs an NVIDIA driver >= 580; create the pod with the CUDA 13.0 filter"
    )
print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")
PY

echo "== data"
uv run python -m clamf.data.download
uv run python -m clamf.data.prepare --config configs/base.yaml

echo "== tests"
uv run pytest -q
echo "ready: bash scripts/runpod/run_grid.sh (see docs/runpod.md)"
