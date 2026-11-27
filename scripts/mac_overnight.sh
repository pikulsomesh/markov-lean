#!/usr/bin/env bash
# The paper's search comparison on the Mac Mini, with no GPU and no cloud, sharing the Mac
# with its GitHub Actions runners.
#
#   scripts/mac_overnight.sh                # PLAN=full: all 244 miniF2F problems, then report
#   PLAN=night scripts/mac_overnight.sh     # stop after 12 hours (rerun later to go on)
#   scripts/mac_overnight.sh --estimate     # print the plan and hours, run nothing
#   scripts/mac_overnight.sh --status       # what the run is doing now, and how far it got
#   scripts/mac_overnight.sh --report       # write the report from what has finished so far
#   scripts/mac_overnight.sh --dry-run      # print the commands, run nothing
#   scripts/mac_overnight.sh --agent-plist  # print an optional launchd agent (see the docs)
#
# The run is resumable: rerunning the same command skips finished problems. Conditions run
# interleaved per problem, so whenever it stops, the finished problems are paired.
#
# It gives memory back on its own (src/markov_lean/mac_guard.py): when a GitHub runner job
# starts (a Runner.Worker process), memory pressure reaches "critical", available memory falls
# under MIN_FREE_GB or swap grows, it stops the search cleanly, which frees its ~9 GB, and
# starts it again once no job has run for COOLDOWN seconds and RESUME_FREE_GB are free.
# It changes no system setting and never touches the runner.
#
# Settings, as environment variables:
#   PLAN=full                    full (up to HOURS=48, ends when all problems are done) or night (HOURS=12)
#   HOURS=                       stop after this many hours in all, pauses included
#   CONDITIONS="lm area_l0.3 random_l0.3"   conditions from the config
#   BENCHMARK=minif2f            or putnam (PutnamBench from scripts/fetch_gold_sources.sh; see
#                                docs/search_eval.md). Both go under the same OUT, so the paper reads both
#   LIMIT=                       first N problems of the fixed order (default: all)
#   OUT=results/mac_overnight    where the runs, logs and report go
#   MODEL=~/models/deepseek-prover-v1.5-rl_q4_k_m.gguf
#   MSM_DIR=                     saved Markov model (default: msm_dir in the config)
#   CONFIG=configs/search_eval_mac_overnight.yaml
#   MIN_FREE_GB=1                lower bound: stop the search below this many GB available
#                                (the loaded prover sits near 1.5 to 2 GB, so keep it under that)
#   RESUME_FREE_GB=3             upper bound: start or resume only with this many GB available
#                                (an idle Mac has about 7 to 9 GB; not under MIN_FREE_GB)
#   MAX_SWAP_GROWTH_GB=1.5  COOLDOWN=120  POLL=10
#   YIELD_TO_RUNNER=1            0 to ignore runner jobs (memory checks still apply)
#
# The prover runs in-process through llama.cpp. API and AWS credentials are removed from
# the run's environment, so nothing it starts can call a paid API or AWS.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

PLAN="${PLAN:-full}"
case "$PLAN" in
  full) HOURS="${HOURS:-48}" ;;
  night) HOURS="${HOURS:-12}" ;;
  *) echo "PLAN must be full or night, not $PLAN" >&2; exit 2 ;;
esac
CONDITIONS="${CONDITIONS:-lm area_l0.3 random_l0.3}"
BENCHMARK="${BENCHMARK:-minif2f}"
LIMIT="${LIMIT:-}"
OUT="${OUT:-results/mac_overnight}"
MODEL="${MODEL:-$HOME/models/deepseek-prover-v1.5-rl_q4_k_m.gguf}"
MSM_DIR="${MSM_DIR:-}"
CONFIG="${CONFIG:-configs/search_eval_mac_overnight.yaml}"
PY="${PY:-.venv/bin/python}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
RESUME_FREE_GB="${RESUME_FREE_GB:-3}"
MAX_SWAP_GROWTH_GB="${MAX_SWAP_GROWTH_GB:-1.5}"
COOLDOWN="${COOLDOWN:-120}"
POLL="${POLL:-10}"
YIELD_TO_RUNNER="${YIELD_TO_RUNNER:-1}"

mode=run
case "${1:-}" in
  "") ;;
  --estimate) mode=estimate ;;
  --dry-run) mode=dry ;;
  --status) mode=status ;;
  --report) mode=report ;;
  --agent-plist) mode=plist ;;
  -h|--help) sed -n '2,41p' "$0"; exit 0 ;;
  *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
esac

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'mac_overnight: %s\n' "$*" >&2; exit 1; }

search=(-m markov_lean.search_eval --config "$CONFIG" --out "$OUT" --benchmark "$BENCHMARK"
        --backend llama_cpp --model-path "$MODEL" --workers 1)
for c in $CONDITIONS; do search+=(--condition "$c"); done
[ -n "$LIMIT" ] && search+=(--limit "$LIMIT")
[ -n "$MSM_DIR" ] && search+=(--msm-dir "$MSM_DIR")
guard=(-m markov_lean.mac_guard --hours "$HOURS" --hours-flag=--max-hours --state "$OUT/guard_state.json"
       --min-free-gb "$MIN_FREE_GB" --resume-free-gb "$RESUME_FREE_GB"
       --max-swap-growth-gb "$MAX_SWAP_GROWTH_GB" --cooldown "$COOLDOWN" --poll "$POLL")
[ "$YIELD_TO_RUNNER" = 1 ] || guard+=(--no-runner-yield)
# The report covers every benchmark under OUT, so a Putnam run adds to the miniF2F report, not replaces it.
report=(-m markov_lean.search_report --runs "$OUT" --out "$OUT/report" --no-figures)

# No credentials reach the run. PYTHONUNBUFFERED keeps the progress lines live in the log.
clean_env=(env -u ANTHROPIC_API_KEY -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY
           -u AWS_SESSION_TOKEN -u AWS_PROFILE -u AWS_DEFAULT_PROFILE PYTHONUNBUFFERED=1)
awake=()
if command -v caffeinate >/dev/null 2>&1; then
  awake=(caffeinate -ims)   # no idle, disk or system sleep while the run lives; the display may sleep
fi

progress() {
  "$PY" - "$OUT/$BENCHMARK" $CONDITIONS <<'EOF'
import json, sys
from pathlib import Path
root, conds = Path(sys.argv[1]), sys.argv[2:]
for c in conds:
    f = root / c / "seed0" / "summary.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    print(f"  {c:<14} {len(rows):>4} problems done, {sum(bool(r.get('solved')) for r in rows):>4} solved")
EOF
}

case "$mode" in
  dry)
    echo "run:    ${clean_env[*]} ${awake[*]+${awake[*]} }$PY ${guard[*]} -- $PY ${search[*]} --yes --no-parquet"
    echo "report: $PY ${report[*]}"
    [ ${#awake[@]} -eq 0 ] && echo "note:   caffeinate not found (not a Mac?): the machine may sleep"
    exit 0 ;;
  status)
    "$PY" -m markov_lean.mac_guard --status --state "$OUT/guard_state.json"
    progress
    exit 0 ;;
  report)
    exec "$PY" "${report[@]}" ;;
  plist)
    label="com.markov-lean.search"
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Optional: starts scripts/mac_overnight.sh at login and again if it crashes (not when it finishes).
     Install:  scripts/mac_overnight.sh --agent-plist > ~/Library/LaunchAgents/$label.plist
               launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/$label.plist
     Remove:   launchctl bootout gui/\$(id -u)/$label && rm ~/Library/LaunchAgents/$label.plist -->
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/scripts/mac_overnight.sh</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PLAN</key><string>$PLAN</string>
    <key>CONDITIONS</key><string>$CONDITIONS</string>
    <key>BENCHMARK</key><string>$BENCHMARK</string>
    <key>OUT</key><string>$OUT</string>
    <key>MODEL</key><string>$MODEL</string>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.elan/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>600</integer>
  <key>StandardOutPath</key><string>$REPO/$OUT/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$REPO/$OUT/logs/launchd.log</string>
</dict>
</plist>
EOF
    exit 0 ;;
esac

# ---------------------------------------------------------------- checks before any compute
[ -x "$PY" ] || die "no $PY: run scripts/setup_mac.sh first (MAC_SMOKE.md step 1)"
[ -f "${MODEL/#\~/$HOME}" ] || die "model file $MODEL not found (MAC_SMOKE.md step 4), or set MODEL=..."
case " $CONDITIONS " in
  *" area_"*)
    msm="$MSM_DIR"
    [ -n "$msm" ] || msm=$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("msm_dir") or "")' "$CONFIG")
    [ -f "$msm/featurizer.pkl" ] || die "the area conditions need a saved Markov model: $msm/featurizer.pkl is missing.
  Fit one on this Mac (markov-lean pilot, see REPRODUCE.md) or set MSM_DIR=<a msm folder you produced>." ;;
esac
bench_path=$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["benchmarks"][sys.argv[2]]["path"])' "$CONFIG" "$BENCHMARK") \
  || die "benchmark $BENCHMARK is not in $CONFIG"
if [ ! -e "$bench_path" ]; then
  case "$bench_path" in
    data/raw/gold/*) die "$bench_path not found: run scripts/fetch_gold_sources.sh (pinned, about 11 MB for PutnamBench)" ;;
    *) die "$bench_path not found: run .venv/bin/markov-lean fetch $BENCHMARK" ;;
  esac
fi
if [ "$mode" = run ]; then
  "$PY" -c 'import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 3)' 2>/dev/null \
    || die "llama-cpp-python is missing or built without Metal (MAC_SMOKE.md step 1)"
fi

if [ "$mode" = estimate ]; then
  "${clean_env[@]}" "$PY" "${search[@]}" --estimate
  echo "Plan $PLAN: stops after $HOURS h in all, pauses for runner jobs included."
  exit 0
fi

# ---------------------------------------------------------------- the run
mkdir -p "$OUT/logs"
log="$OUT/logs/$(date '+%Y%m%d-%H%M%S').log"
{
  [ ${#awake[@]} -eq 0 ] && say "caffeinate not found: keep the machine from sleeping yourself"
  say "plan $PLAN: $BENCHMARK, conditions: $CONDITIONS, for up to $HOURS h. Log: $log"
  say "memory bounds: stops below MIN_FREE_GB=$MIN_FREE_GB GB available, starts at RESUME_FREE_GB=$RESUME_FREE_GB GB"
  say "One line per finished problem. Stops for runner jobs and low memory, then resumes on its own."
  say "Ctrl-C stops; rerun this command to resume. From another terminal: scripts/mac_overnight.sh --status"
} | tee -a "$log"

status=0
# tee -i ignores Ctrl-C, so the last lines still reach the log and the report is still written
"${clean_env[@]}" ${awake[@]+"${awake[@]}"} "$PY" "${guard[@]}" -- "$PY" "${search[@]}" --yes --no-parquet \
  2>&1 | tee -ai "$log" || status=$?

{
  say "rebuilding the trace Parquet files and writing the report"
  "$PY" "${search[@]}" --parquet-only || say "Parquet rebuild failed; the JSONL logs are complete"
  "$PY" "${report[@]}" || say "report failed (see above); the runs are saved"
  progress
  say "Report: $OUT/report/report.md. Rerun this command to go on with any problems left."
} 2>&1 | tee -ai "$log"
exit "$status"
