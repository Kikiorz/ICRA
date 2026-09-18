#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/bin/python -u run_inference.py \
  --model /workspace/models/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --image examples/long_task01_ep0001.png \
  --output-dir outputs/cup_lift_bf16 2>&1 | tee inference.log
