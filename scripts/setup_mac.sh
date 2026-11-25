#!/usr/bin/env bash
# One-time setup on the Mac Mini (Apple silicon). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Python environment"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev,judge]"
pytest -q

if [ "${WITH_LEAN:-0}" = "1" ]; then
  echo "== Lean toolchain (elan) and Mathlib cache"
  ./scripts/setup_lean.sh
fi

echo "== Done. Next: ./scripts/fetch_data.sh"
