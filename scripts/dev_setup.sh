#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
pip install -e ".[dev]"
pre-commit install --install-hooks || true
pre-commit install --hook-type pre-push || true
