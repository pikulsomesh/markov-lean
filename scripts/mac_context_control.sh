#!/usr/bin/env bash
# The judge-context control on the Mac Mini (src/markov_lean/context_control.py, docs/context_control.md):
# the local Ollama judge labels the same proofs twice, steps in order and steps shuffled, then
# compares switching in the true step order. It runs under the same memory guard as the local judge
# (src/markov_lean/mac_guard.py), so it stops for GitHub runner jobs and resumes. No API call, no cost.
#
#   scripts/mac_context_control.sh            # label (resumes) and then compare
#   scripts/mac_context_control.sh --status
#
# Settings: RUN=results/pilot3, PROOFS=300, MODEL=qwen2.5:7b-instruct, PROMPT=v1,
# OUT=results/context_control/<model>, HOURS=48, MIN_FREE_GB=1, RESUME_FREE_GB=8,
# MAX_SWAP_GROWTH_GB=1.5, COOLDOWN=120, POLL=10, YIELD_TO_RUNNER=1,
# STATE=results/context_control_guard.json. If no Ollama server answers, it starts
# `ollama serve` for the run and stops it at the end; it pulls the model if it is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
RUN="${RUN:-results/pilot3}"
PROOFS="${PROOFS:-300}"
PROMPT="${PROMPT:-v1}"
STATE="${STATE:-results/context_control_guard.json}"
[ -x "$PY" ] || { echo "no $PY: run scripts/setup_mac.sh first" >&2; exit 1; }
MODEL="${MODEL:-$("$PY" -c 'from markov_lean.ollama_judge import DEFAULT_MODEL; print(DEFAULT_MODEL)')}"
OUT="${OUT:-$("$PY" -c 'import sys; from markov_lean.context_control import default_out; print(default_out(sys.argv[1], sys.argv[2]))' "$MODEL" "$PROMPT")}"

if [ "${1:-}" = --status ]; then
  "$PY" -m markov_lean.context_control status --out "$OUT"
  [ -f "$STATE" ] && "$PY" -m markov_lean.mac_guard --status --state "$STATE" || true
  exit 0
fi
[ -f "$RUN/judge/sample_steps.parquet" ] || { echo "no $RUN/judge/sample_steps.parquet" >&2; exit 1; }
mkdir -p "$OUT"

up() { "$PY" -c 'import urllib.request; urllib.request.urlopen("http://localhost:11434/api/version", timeout=3)' 2>/dev/null; }
served=""
if ! up; then
  command -v ollama >/dev/null 2>&1 || { echo "no Ollama server on localhost:11434 and no ollama command" >&2; exit 1; }
  nohup ollama serve >> "$OUT/ollama_serve.log" 2>&1 < /dev/null &
  served=$!
  trap '[ -n "$served" ] && kill "$served" 2>/dev/null || true' EXIT
  for _ in $(seq 30); do up && break; sleep 2; done
  up || { echo "ollama serve did not come up; see $OUT/ollama_serve.log" >&2; exit 1; }
  echo "started ollama serve (pid $served) for this run"
fi
if command -v ollama >/dev/null 2>&1 && ! ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
  echo "pulling $MODEL"; ollama pull "$MODEL"
fi

guard=(-m markov_lean.mac_guard --hours "${HOURS:-48}" --state "$STATE"
       --min-free-gb "${MIN_FREE_GB:-1}" --resume-free-gb "${RESUME_FREE_GB:-8}"
       --max-swap-growth-gb "${MAX_SWAP_GROWTH_GB:-1.5}" --cooldown "${COOLDOWN:-120}" --poll "${POLL:-10}")
[ "${YIELD_TO_RUNNER:-1}" = 1 ] || guard+=(--no-runner-yield)
command -v ollama >/dev/null 2>&1 && guard+=(--on-pause "ollama stop $(printf '%q' "$MODEL")")

env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 "$PY" "${guard[@]}" -- \
  "$PY" -m markov_lean.context_control label --run "$RUN" --proofs "$PROOFS" --model "$MODEL" \
  --prompt "$PROMPT" --out "$OUT"
if ! "$PY" -m markov_lean.context_control status --out "$OUT" --check; then
  echo "labelling is not finished (time limit or repeated failures); run again to resume"
  exit 0   # the queue marks the stage incomplete, and the next start resumes it
fi
env -u ANTHROPIC_API_KEY "$PY" -m markov_lean.context_control compare --out "$OUT" --run "$RUN"
