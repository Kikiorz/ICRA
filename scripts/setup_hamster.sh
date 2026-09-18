#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/VILA/.git ]; then
  git clone --filter=blob:none --no-checkout https://github.com/NVlabs/VILA.git external/VILA
  git -C external/VILA checkout a5a380d6d09762d6f3fd0443aac6b475fba84f7e
fi
test "$(git -C external/VILA rev-parse HEAD)" = a5a380d6d09762d6f3fd0443aac6b475fba84f7e
if [ ! -x .venv-hamster/bin/python ]; then
  uv venv --python 3.12 .venv-hamster
fi
uv pip install --python .venv-hamster/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-hamster/bin/python -r hamster/requirements.txt
.venv-hamster/bin/python hamster/patch_vila.py external/VILA
PYTHONPATH="$PWD/external/VILA" .venv-hamster/bin/python -c 'from llava.model.builder import load_pretrained_model; print("VILA import OK")'
.venv-hamster/bin/python hamster/download.py
