#!/usr/bin/env bash
# After the case-study judge runs (docs/cases.md, "Judge plan"): one command that collects their labels,
# runs the judge-label switch test on each project, compares each with the Carleson control and rebuilds
# the paper. It never submits anything: the capped `judge submit` steps stay separate (docs/cases.md).
# `collect` only downloads the results of batches already submitted, which costs nothing.
#
#   bash scripts/after_case_judges.sh            # every step; a rerun skips what is done
#   bash scripts/after_case_judges.sh --status   # what is in and what is waiting
#
# Steps:
# 1 for each case with a submitted run (results/cases/<name>/api/batches.json): scripts/collect_batches.py
#   writes labels.parquet once its batches have ended, through a client that can only read batch status and
#   results (both free) and blocks every other call. Needs ANTHROPIC_API_KEY in this shell. A case that is
#   not submitted, or whose batches are still running, is listed and skipped; rerun later.
# 2 under the memory guard with its shared lock, so it waits for a running search or extraction to finish
#   and Mac runs stay one at a time (the key is removed from its environment):
#   a `judge agree` per case -> api/agreement.txt;
#   b `markov-lean switches` on judge labels per case -> <case>/switches (shuffle and Markov nulls, no
#     Mathlib folders, as for PFR);
#   c `case_study compare` of every case with a switches run (PFR too, from results/pfr/switches)
#     against Carleson -> results/cases/compare.json;
#   d `markov-lean paper` and `make -B -C paper`.
#
# Settings (environment): CASES (default "sphere_packing pnt carleson"), N_PERM (1000), N_BOOT (2000),
# HOURS (guard limit, default 48; waiting for the lock counts), MIN_FREE_GB (1), RESUME_FREE_GB (3),
# PAPER_ARGS (default "--pilot results/pilot3 --accuracy results/gold/accuracy"), SKIP_PDF=1 skips make,
# RESULTS (default results) and PAPER (default paper) folders, GUARD_LOCK (the guard's shared lock).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}
CASES=${CASES:-"sphere_packing pnt carleson"}
CONTROL=carleson
RESULTS=${RESULTS:-results}
PAPER=${PAPER:-paper}
OUT=$RESULTS/cases
STATE=$OUT/after_judges
PAPER_ARGS=${PAPER_ARGS:-"--pilot results/pilot3 --accuracy results/gold/accuracy"}
say() { printf '[cases %s] %s\n' "$(date '+%H:%M')" "$*"; }
newer() { [ -f "$1" ] && [ ! "$1" -ot "$2" ]; }   # $1 exists and is not older than $2

if [ "${1:-}" = "--status" ]; then
  for n in $CASES; do
    api=$OUT/$n/api
    if [ -f "$api/labels.parquet" ]; then l="labels in"
    elif [ -f "$api/batches.json" ]; then l="submitted, labels not collected"
    else l="not submitted"; fi
    s=$( [ -f "$OUT/$n/switches/switches.json" ] && echo "switch test done" || echo "no switch test yet")
    echo "$n: $l; $s"
  done
  [ -f "$OUT/compare.json" ] && echo "compare.json: $(date -r "$OUT/compare.json" '+%Y-%m-%d %H:%M')" \
    || echo "compare.json: not yet"
  [ -f "$STATE/guard.json" ] && echo "analysis guard: $("$PY" -m markov_lean.mac_guard --status --state "$STATE/guard.json" | head -1)"
  exit 0
fi

if [ "${1:-}" = "--analyse" ]; then   # step 2, run by the guard; resumable
  args=()
  for n in $CASES; do
    api=$OUT/$n/api
    [ -f "$api/labels.parquet" ] || continue
    if ! newer "$api/agreement.txt" "$api/labels.parquet"; then
      say "$n: judge agreement"
      "$PY" -m markov_lean.judge agree --labels "$api/labels.parquet" | tee "$api/agreement.txt.tmp"
      mv "$api/agreement.txt.tmp" "$api/agreement.txt"
    fi
    if ! newer "$OUT/$n/switches/switches.json" "$api/labels.parquet"; then
      say "$n: switch test on judge labels"
      "$PY" -m markov_lean.cli switches --run "$OUT/$n" --steps "$OUT/$n/steps.parquet" \
        --labels "$api/labels.parquet" --sources judge --out "$OUT/$n/switches" \
        --markov-group none --pooled-within all --nulls shuffle markov \
        --n-perm "${N_PERM:-1000}" --n-boot "${N_BOOT:-2000}" --no-figures
    fi
    args+=(--case "$n=$OUT/$n")
  done
  [ -f "$RESULTS/pfr/switches/proofs.csv" ] && args+=(--case "pfr=$RESULTS/pfr")
  if [ ${#args[@]} -gt 0 ]; then
    say "comparing with the $CONTROL control"
    "$PY" -m markov_lean.case_study compare "${args[@]}" --control "$CONTROL" --out "$OUT/compare.json"
  fi
  say "rebuilding the paper's tables and numbers"
  # shellcheck disable=SC2086
  "$PY" -m markov_lean.cli paper --results "$RESULTS" --paper "$PAPER" $PAPER_ARGS
  if [ "${SKIP_PDF:-0}" != 1 ]; then make -B -C "$PAPER"; fi
  date '+%Y-%m-%d %H:%M' > "$STATE/analyse.done"
  exit 0
fi

mkdir -p "$STATE"
waiting=()
for n in $CASES; do
  api=$OUT/$n/api
  [ -f "$OUT/$n/steps.parquet" ] || { say "$n: no steps table; run scripts/case_extract.sh first"; waiting+=("$n"); continue; }
  if [ -f "$api/labels.parquet" ]; then say "$n: labels already in"; continue; fi
  if [ ! -f "$api/batches.json" ]; then
    say "$n: not submitted yet (docs/cases.md, Judge plan); skipped"; waiting+=("$n"); continue
  fi
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    say "$n: set ANTHROPIC_API_KEY in this shell to collect its labels (reading results costs nothing)"
    waiting+=("$n"); continue
  fi
  say "$n: collecting labels"
  rc=0; "$PY" "$ROOT/scripts/collect_batches.py" --out "$api" || rc=$?
  if [ "$rc" = 3 ]; then say "$n: its batches are still running; rerun later"; waiting+=("$n")
  elif [ "$rc" != 0 ]; then say "$n: collecting failed (exit $rc); see above"; waiting+=("$n"); fi
done

rm -f "$STATE/analyse.done"
say "analysis (guarded: waits for any running search to finish first)"
env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 "$PY" -m markov_lean.mac_guard --hours "${HOURS:-48}" \
  --state "$STATE/guard.json" --min-free-gb "${MIN_FREE_GB:-1}" --resume-free-gb "${RESUME_FREE_GB:-3}" \
  ${GUARD_LOCK+--lock "$GUARD_LOCK"} -- bash "$0" --analyse
if [ ! -f "$STATE/analyse.done" ]; then
  say "the analysis stopped before it finished (guard limit HOURS=${HOURS:-48}); rerun to resume"
  exit 1
fi
[ ${#waiting[@]} -gt 0 ] && say "still waiting on: ${waiting[*]} (rerun once they are in)"
say "done. Send Claude: $OUT/compare.json and $PAPER/main.pdf"
