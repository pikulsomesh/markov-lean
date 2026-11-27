#!/usr/bin/env bash
# Cross-area case study: extract the tactic steps of one Lean 4 project (configs/cases/<name>.yaml),
# then make free labels and a judge cost estimate. Spends nothing: no API key is used or needed.
# The PFR route (scripts/pfr_extract.sh) for any project.
#
#   bash scripts/case_extract.sh configs/cases/pnt.yaml            # all stages; resumes where it stopped
#   bash scripts/case_extract.sh configs/cases/pnt.yaml --status   # what is done
#   bash scripts/case_extract.sh configs/cases/pnt.yaml --clean    # free its build files (after a failure)
#
# Stages: 1 clone the project at the config's pinned commit and fetch its Mathlib's prebuilt cache;
# 2 `lake build` the included modules only (Mathlib comes from the cache);
# 3 run scripts/pfr/ExtractSteps.lean on each included file (one .jsonl per module, skipped once
#   written, so a stop loses at most the files in flight);
# 4 `python -m markov_lean.case_study convert` (steps table, free labels, summary);
# 5 `python -m markov_lean.judge estimate` on the steps table;
# 6 delete what stage 1 added (the project's build, its Mathlib cache files, and its Lean toolchain
#   if this script installed it), keeping results/cases/<name>. KEEP_BUILD=1 skips it.
#
# Disk: stage 1 needs about 12 GB (a Lean toolchain ~3.3 GB if new, the project's Mathlib checkout
# and prebuilt files ~6 to 7 GB, the compressed cache ~2 GB); it stops early unless MIN_DISK_GB
# (default 15) are free. Stage 6 gives it all back, so run projects one after another.
# Stages 2 and 3 run under the memory guard (src/markov_lean/mac_guard.py) with its shared lock,
# so they wait for another guarded job (the overnight search) to finish, and they stop and resume
# around GitHub runner jobs.
#
# Settings (environment): CASE_DIR (default ~/cases/<name>), OUT (default results/cases/<name>),
# JOBS (Lean files elaborated at once, default 2; each takes about 2 to 4 GB), RESUME_FREE_GB
# (default 3), MIN_DISK_GB (default 15), KEEP_BUILD (default 0), HOURS (guard limit per stage, default 168; waiting for the
# lock counts, so a long search ahead of it must not use it up).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:?usage: bash scripts/case_extract.sh configs/cases/<name>.yaml [--status | --clean]}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
PY=${PY:-$ROOT/.venv/bin/python}
field() { "$PY" -m markov_lean.case_study field "$CONFIG" "$1"; }
NAME=$(field name); REPO=$(field repo); REV=$(field rev)
CASE_DIR=${CASE_DIR:-$HOME/cases/$NAME}
OUT=${OUT:-$ROOT/results/cases/$NAME}
JOBS=${JOBS:-2}
STATE=$OUT/state
EXTRACT=$ROOT/scripts/pfr/ExtractSteps.lean
export PATH="$HOME/.elan/bin:$PATH"

say() { printf '[%s %s] %s\n' "$NAME" "$(date '+%H:%M')" "$*"; }
done_mark() { mkdir -p "$STATE"; date '+%Y-%m-%d %H:%M' > "$STATE/stage$1.done"; }
is_done() { [ -f "$STATE/stage$1.done" ]; }
files() { "$PY" -m markov_lean.case_study files "$CONFIG" "$CASE_DIR"; }
toolchain_dir() { sed -e 's#/#--#' -e 's#:#---#' "$CASE_DIR/lean-toolchain" | tr -d '[:space:]'; }
MATHLIB_CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/mathlib
free_gb() { local d=$1; while [ ! -d "$d" ]; do d=$(dirname "$d"); done; df -Pk "$d" | awk 'NR==2 {print int($4 / 1048576)}'; }
need_disk() {  # gb dir what
  local f; f=$(free_gb "$2")
  if [ "$f" -lt "$1" ]; then
    say "stopping: only $f GB free on the disk holding $2; $3 needs about $1 GB."
    say "free some space (or set MIN_DISK_GB lower at your own risk) and rerun."
    exit 1
  fi
}

if [ "${2:-}" = "--status" ]; then
  for i in 1 2 3 4 5 6; do is_done "$i" && echo "stage $i done $(cat "$STATE/stage$i.done")" || echo "stage $i not done"; done
  for g in build extract; do
    [ -f "$STATE/${g}_guard.json" ] && echo "$g guard: $("$PY" -m markov_lean.mac_guard --status --state "$STATE/${g}_guard.json" | head -1)"
  done
  if [ -d "$OUT/raw" ] && [ -d "$CASE_DIR" ]; then
    echo "extracted modules: $(ls "$OUT/raw" | grep -c '\.jsonl$' || true) of $(files | wc -l | tr -d ' ')"
    [ -s "$OUT/raw/failed.txt" ] && echo "modules that failed to extract: $(wc -l < "$OUT/raw/failed.txt" | tr -d ' ')"
  fi
  exit 0
fi

clean() {  # stage 6: delete what stage 1 added; results/cases/<name> stays
  local before; before=$(free_gb "$HOME")
  rm -rf "$CASE_DIR/.lake"
  if [ -f "$STATE/mathlib_cache_before.txt" ] && [ -d "$MATHLIB_CACHE" ]; then
    # only the cache files stage 1 downloaded; markov-lean's own Mathlib files stay
    (cd "$MATHLIB_CACHE" && ls | grep -vxF -f "$STATE/mathlib_cache_before.txt" | while IFS= read -r f; do rm -f -- "$f"; done) || true
  fi
  if [ "$(cat "$STATE/toolchain_before" 2>/dev/null)" = new ]; then elan toolchain uninstall "$(cat "$STATE/toolchain")" || true; fi
  say "freed about $(( $(free_gb "$HOME") - before )) GB"
}

if [ "${2:-}" = "--clean" ]; then
  # after a failure: free the build files now (the next project needs the disk). A rerun
  # starts again from stage 1, keeping the steps already extracted to results/cases/<name>/raw.
  say "removing $NAME's build files after a failure (results/cases/$NAME stays)"
  clean
  rm -f "$STATE"/stage1.done "$STATE"/stage2.done
  exit 0
fi

# --inner STAGE: one guarded stage, run by the guard below (it may restart it after a stop)
if [ "${2:-}" = "--inner" ]; then
  cd "$CASE_DIR"
  case "$3" in
    build) files | sed -e 's#\.lean$##' -e 's#/#.#g' | xargs lake build ;;
    extract)
      mkdir -p "$OUT/raw"
      one() {  # file -> $OUT/raw/<module>.jsonl, written via .tmp so a stop leaves no partial file
        local f=$1 mod
        mod=$(printf '%s' "${f%.lean}" | tr '/' '.')
        [ -f "$OUT/raw/$mod.jsonl" ] && return 0
        if lake env lean --run "$EXTRACT" "$f" "$mod" "$OUT/raw/$mod.jsonl.tmp"; then
          mv "$OUT/raw/$mod.jsonl.tmp" "$OUT/raw/$mod.jsonl"
        else  # one bad file does not stop the others; the list is retried on a rerun
          rm -f "$OUT/raw/$mod.jsonl.tmp"; echo "$f" >> "$OUT/raw/failed.txt"
        fi
        return 0
      }
      rm -f "$OUT/raw/failed.txt"
      export -f one; export OUT EXTRACT
      files | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {} ;;
  esac
  exit 0
fi

guarded() {  # name -- command...
  local name=$1; shift 2
  env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 "$PY" -m markov_lean.mac_guard --hours "${HOURS:-168}" \
    --state "$STATE/${name}_guard.json" --min-free-gb "${MIN_FREE_GB:-1}" \
    --resume-free-gb "${RESUME_FREE_GB:-3}" --max-swap-growth-gb "${MAX_SWAP_GROWTH_GB:-1.5}" \
    --cooldown "${COOLDOWN:-120}" --poll "${POLL:-10}" -- "$@"
}

# the guard also exits 0 when its HOURS run out; only "done" in its state means the job finished
guard_done() {  # name
  "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "done" else 1)' \
    "$STATE/$1_guard.json" 2>/dev/null
}
stopped_early() { say "stage $1 stopped at its time limit (HOURS=${HOURS:-168}) with work left; rerun to resume"; exit 0; }

mkdir -p "$STATE"
if ! is_done 1 && ! is_done 3; then   # once the steps are extracted, no build is needed
  need_disk "${MIN_DISK_GB:-15}" "$HOME" "stage 1 ($NAME, its Mathlib and their prebuilt files)"
  say "stage 1: $REPO at ${REV:0:7} and its Mathlib cache"
  if [ ! -d "$CASE_DIR/.git" ]; then mkdir -p "$(dirname "$CASE_DIR")"; git clone "$REPO" "$CASE_DIR"; fi
  (cd "$CASE_DIR" && { git checkout -q "$REV" 2>/dev/null || { git fetch -q origin "$REV" && git checkout -q "$REV"; }; })
  # remember what was here before, so stage 6 removes only what this script added
  TC=$(toolchain_dir)
  [ -d "$HOME/.elan/toolchains/$TC" ] && echo keep > "$STATE/toolchain_before" || echo new > "$STATE/toolchain_before"
  echo "$TC" > "$STATE/toolchain"
  mkdir -p "$MATHLIB_CACHE"; ls "$MATHLIB_CACHE" > "$STATE/mathlib_cache_before.txt"
  (cd "$CASE_DIR" && lake exe cache get)
  done_mark 1
fi
is_done 2 || is_done 3 || need_disk 2 "$CASE_DIR" "stage 2 (building $NAME)"
is_done 3 || need_disk 1 "$OUT" "stage 3 (extracted steps)"
if ! is_done 2 && ! is_done 3; then say "stage 2: lake build of $(files | wc -l | tr -d ' ') modules (guarded)"; guarded build -- bash "$0" "$CONFIG" --inner build; guard_done build || stopped_early 2; done_mark 2; fi
if ! is_done 3; then
  say "stage 3: extracting steps from $(files | wc -l | tr -d ' ') files, $JOBS at a time (guarded)"
  guarded extract -- bash "$0" "$CONFIG" --inner extract
  guard_done extract || stopped_early 3
  n=$(files | wc -l | tr -d ' '); got=$(ls "$OUT/raw" | grep -c '\.jsonl$' || true)
  [ "$got" -gt 0 ] || { say "no module was extracted; see the log above"; exit 1; }
  [ "$got" -eq "$n" ] || say "$((n - got)) of $n modules failed to extract (listed in $OUT/raw/failed.txt); going on with $got"
  done_mark 3
fi
if ! is_done 4; then
  say "stage 4: steps table and free labels"
  "$PY" -m markov_lean.case_study convert --config "$CONFIG" --raw "$OUT/raw" --out "$OUT"
  done_mark 4
fi
say "stage 5: judge cost estimate (spends nothing)"
if env -u ANTHROPIC_API_KEY "$PY" -m markov_lean.judge estimate --steps "$OUT/steps.parquet" > "$OUT/estimate.txt" 2>&1; then
  cat "$OUT/estimate.txt"; done_mark 5
else
  say "the estimate failed (see $OUT/estimate.txt); the steps are saved, so this does not stop the run"
fi
if ! is_done 6 && [ "${KEEP_BUILD:-0}" != 1 ]; then
  say "stage 6: removing $NAME's build files (results/cases/$NAME stays)"
  clean
  done_mark 6
fi
say "done. Send Claude: $OUT/summary.json and $OUT/estimate.txt"
