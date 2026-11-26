#!/usr/bin/env bash
# PFR case study: extract the tactic steps of teorth/pfr, then make free labels and a judge
# cost estimate. Spends nothing: no API key is used or needed.
#
#   bash scripts/pfr_extract.sh            # all stages; resumes where it stopped
#   bash scripts/pfr_extract.sh --status   # what is done
#
# Stages: 1 clone teorth/pfr at the pinned commit and fetch Mathlib's prebuilt cache;
# 2 `lake build PFR` (only PFR's 71 files; Mathlib comes from the cache);
# 3 run scripts/pfr/ExtractSteps.lean on each PFR file (one .jsonl per module, skipped once
#   written, so a stop loses at most the files in flight);
# 4 `python -m markov_lean.pfr convert` (steps table, free labels, summary);
# 5 `python -m markov_lean.judge estimate` on the steps table.
# Stages 2 and 3 run under the memory guard (src/markov_lean/mac_guard.py) with its shared
# lock, so they wait for another guarded job (the overnight search) to finish, and they stop
# and resume around GitHub runner jobs.
#
# Settings (environment): PFR_DIR (default ~/pfr), OUT (default results/pfr), JOBS (Lean
# files elaborated at once, default 2; each takes about 2 to 4 GB), RESUME_FREE_GB (default 6).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PFR_REV=1899ce8ed95fc310cce53d4c7ecfa2beb20ebec4
PFR_DIR=${PFR_DIR:-$HOME/pfr}
OUT=${OUT:-$ROOT/results/pfr}
JOBS=${JOBS:-2}
PY=${PY:-$ROOT/.venv/bin/python}
STATE=$OUT/state
EXTRACT=$ROOT/scripts/pfr/ExtractSteps.lean
export PATH="$HOME/.elan/bin:$PATH"

say() { printf '[pfr %s] %s\n' "$(date '+%H:%M')" "$*"; }
done_mark() { mkdir -p "$STATE"; date '+%Y-%m-%d %H:%M' > "$STATE/stage$1.done"; }
is_done() { [ -f "$STATE/stage$1.done" ]; }
modules() { (cd "$PFR_DIR" && find PFR -name '*.lean' | sort); }

if [ "${1:-}" = "--status" ]; then
  for i in 1 2 3 4 5; do is_done "$i" && echo "stage $i done $(cat "$STATE/stage$i.done")" || echo "stage $i not done"; done
  [ -d "$OUT/raw" ] && echo "extracted modules: $(ls "$OUT/raw" | grep -c '\.jsonl$' || true)"
  exit 0
fi

# --inner N: one guarded stage, run by the guard below (it may restart it after a stop)
if [ "${1:-}" = "--inner" ]; then
  cd "$PFR_DIR"
  case "$2" in
    build) lake build PFR ;;
    extract)
      mkdir -p "$OUT/raw"
      one() {  # file -> $OUT/raw/<module>.jsonl, written via .tmp so a stop leaves no partial file
        local f=$1 mod
        mod=$(printf '%s' "${f%.lean}" | tr '/' '.')
        [ -f "$OUT/raw/$mod.jsonl" ] && return 0
        lake env lean --run "$EXTRACT" "$f" "$mod" "$OUT/raw/$mod.jsonl.tmp" && mv "$OUT/raw/$mod.jsonl.tmp" "$OUT/raw/$mod.jsonl"
      }
      export -f one; export OUT EXTRACT
      modules | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {} ;;
  esac
  exit 0
fi

guarded() {  # name -- command...
  local name=$1; shift 2
  env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 "$PY" -m markov_lean.mac_guard --hours "${HOURS:-12}" \
    --state "$STATE/${name}_guard.json" --min-free-gb "${MIN_FREE_GB:-1}" \
    --resume-free-gb "${RESUME_FREE_GB:-6}" --max-swap-growth-gb "${MAX_SWAP_GROWTH_GB:-1.5}" \
    --cooldown "${COOLDOWN:-120}" --poll "${POLL:-10}" -- "$@"
}

mkdir -p "$STATE"
if ! is_done 1; then
  say "stage 1: teorth/pfr at ${PFR_REV:0:7} and Mathlib's cache"
  if [ ! -d "$PFR_DIR/.git" ]; then git clone https://github.com/teorth/pfr.git "$PFR_DIR"; fi
  (cd "$PFR_DIR" && { git checkout -q "$PFR_REV" 2>/dev/null || { git fetch -q origin "$PFR_REV" && git checkout -q "$PFR_REV"; }; } \
    && lake exe cache get)
  done_mark 1
fi
if ! is_done 2; then say "stage 2: lake build PFR (guarded)"; guarded build -- bash "$0" --inner build; done_mark 2; fi
if ! is_done 3; then
  say "stage 3: extracting steps from $(modules | wc -l | tr -d ' ') files, $JOBS at a time (guarded)"
  guarded extract -- bash "$0" --inner extract
  n=$(modules | wc -l | tr -d ' '); got=$(ls "$OUT/raw" | grep -c '\.jsonl$' || true)
  [ "$got" -eq "$n" ] || { say "only $got of $n modules extracted; rerun to resume"; exit 1; }
  done_mark 3
fi
if ! is_done 4; then
  say "stage 4: steps table and free labels"
  "$PY" -m markov_lean.pfr convert --raw "$OUT/raw" --out "$OUT" --pfr-root "$PFR_DIR"
  done_mark 4
fi
say "stage 5: judge cost estimate (spends nothing)"
env -u ANTHROPIC_API_KEY "$PY" -m markov_lean.judge estimate --steps "$OUT/pfr_steps.parquet" | tee "$OUT/estimate.txt"
done_mark 5
say "done. Send Claude: $OUT/summary.json and $OUT/estimate.txt"
