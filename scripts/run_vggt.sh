#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/vggt/.git ]; then
  git clone https://github.com/facebookresearch/vggt.git external/vggt
  git -C external/vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce
fi
test "$(git -C external/vggt rev-parse HEAD)" = a288dd0f14786c93483e45524328726ab7b1b4ce
# Reuse the already validated CUDA 12.8 environment without changing its packages.
export PYTHONPATH="$PWD/external/vggt"
exec "${ICRA_GEOMETRY_PYTHON:-.venv-hamster/bin/python}" -m geometry.run_vggt --output-dir "${ICRA_VGGT_OUTPUT:-outputs/vggt_cup_single}" "$@"
