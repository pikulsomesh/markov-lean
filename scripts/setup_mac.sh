#!/usr/bin/env bash
# One-time setup on the Mac Mini (Apple silicon). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Python environment"
# Use the newest Python 3.10+ on the machine (macOS ships 3.9 as python3).
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for v in 3.13 3.12 3.11 3.10; do
    if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
  done
fi
PY="${PY:-python3}"
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "Need Python 3.10 or newer; found $("$PY" --version). Try: brew install python@3.12" >&2
  exit 1
fi
if [ -d .venv ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "Removing .venv built with an old Python"; rm -rf .venv
fi
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev,judge,search,figures]"
pytest -q

if [ "${WITH_LEAN:-0}" = "1" ]; then
  echo "== Lean toolchain (elan) and Mathlib cache"
  ./scripts/setup_lean.sh
fi

echo "== Done. Next: ./scripts/fetch_data.sh"
