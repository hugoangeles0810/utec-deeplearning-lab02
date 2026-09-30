# Sourced by the RunPod scripts (docs/runpod.md). Everything lives on the /workspace volume, the
# only disk that survives a pod stop: code, uv, its cache and Python, data, MLflow and checkpoints.
export WORKSPACE="${WORKSPACE:-/workspace}"
export PROJECT="${PROJECT:-$WORKSPACE/utec-deeplearning-lab02}"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.local/share/uv/python"
export PATH="$WORKSPACE/.local/bin:$PATH"
export MLFLOW_TRACKING_URI="sqlite:///$PROJECT/mlflow.db"
export PYTHONUNBUFFERED=1
