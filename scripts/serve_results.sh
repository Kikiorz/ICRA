#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory .
