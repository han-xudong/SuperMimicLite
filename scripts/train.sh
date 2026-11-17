#!/bin/bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$task_root/.cache/training/hf"
export ANY4HDMI_QPOS_CACHE_ROOT="$task_root/.cache/training/any4"
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONPATH="$task_root/training:$task_root/third_party/active-adaptation"
exec uv run --project "$task_root/environments/train" python "$task_root/training/train.py" "$@"
