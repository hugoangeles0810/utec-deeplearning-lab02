#!/usr/bin/env bash
# Train and evaluate the grid on the pod (python -m clamf.grid), snapshot the MLflow store for
# pull.sh and stop the pod (docs/runpod.md). Relaunching it resumes whatever was left.
#
#   tmux new -d -s grid 'bash scripts/runpod/run_grid.sh'                      # the 6 grid runs
#   STOP_POD=0 bash scripts/runpod/run_grid.sh configs/experiments/dev.yaml   # smoke test
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT"
mkdir -p logs
log="logs/grid-$(date +%Y%m%d-%H%M%S).log"

args=()
if [ $# -gt 0 ]; then args=(--configs "$@"); fi
uv run python -m clamf.grid ${args[@]+"${args[@]}"} 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

uv run python -m clamf.utils.mlflow_store snapshot --db mlflow.db --out mlflow.snapshot.db
echo "clamf.grid exit code $status; log: $log"

if [ "${STOP_POD:-1}" = 1 ]; then
  echo "stopping pod $RUNPOD_POD_ID (STOP_POD=0 keeps it running)"
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
exit "$status"
