#!/usr/bin/env bash
# The local Ollama judge on the Mac Mini under the same memory guard as the search
# (src/markov_lean/mac_guard.py): it stops cleanly when a GitHub runner job starts or memory
# gets tight, unloads the model from Ollama so the memory really comes back, and starts
# again (resuming the same sample) once there is room. It never runs at the same time as a
# guarded search: they take turns through one lock file.
#
#   scripts/mac_local_judge.sh --run results/pilot3 --sample-steps 500     # any `local-judge label` options
#   scripts/mac_local_judge.sh --status
#
# Settings: HOURS=48, RESUME_FREE_GB=8, MIN_FREE_GB=1, MAX_SWAP_GROWTH_GB=1.5, COOLDOWN=120,
# POLL=10, YIELD_TO_RUNNER=1, STATE=results/local_judge_guard.json. Nothing here calls an API.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
STATE="${STATE:-results/local_judge_guard.json}"
[ -x "$PY" ] || { echo "no $PY: run scripts/setup_mac.sh first" >&2; exit 1; }
if [ "${1:-}" = --status ]; then
  exec "$PY" -m markov_lean.mac_guard --status --state "$STATE"
fi

model=$("$PY" -c 'from markov_lean.ollama_judge import DEFAULT_MODEL; print(DEFAULT_MODEL)')
prev=""
for a in "$@"; do
  [ "$prev" = --model ] && model="$a"
  case "$a" in --model=*) model="${a#--model=}" ;; esac
  prev="$a"
done

guard=(-m markov_lean.mac_guard --hours "${HOURS:-48}" --state "$STATE"
       --min-free-gb "${MIN_FREE_GB:-1}" --resume-free-gb "${RESUME_FREE_GB:-8}"
       --max-swap-growth-gb "${MAX_SWAP_GROWTH_GB:-1.5}" --cooldown "${COOLDOWN:-120}" --poll "${POLL:-10}")
[ "${YIELD_TO_RUNNER:-1}" = 1 ] || guard+=(--no-runner-yield)
if command -v ollama >/dev/null 2>&1; then
  guard+=(--on-pause "ollama stop $(printf '%q' "$model")")   # Ollama would keep the model loaded for 30 min
fi
awake=()
command -v caffeinate >/dev/null 2>&1 && awake=(caffeinate -ims)

exec env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 ${awake[@]+"${awake[@]}"} \
  "$PY" "${guard[@]}" -- "$PY" -m markov_lean.ollama_judge label "$@"
