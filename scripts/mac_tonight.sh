#!/usr/bin/env bash
# Tonight on the Mac Mini, in one command: everything after `scripts/setup_mac.sh`
# (MAC_SMOKE.md step 1), ending with the overnight search. Checklist: MAC_TONIGHT.md.
#
#   scripts/mac_tonight.sh             # run every stage that has not finished yet
#   scripts/mac_tonight.sh --status    # which stages are done, and the search's progress
#   scripts/mac_tonight.sh --from N    # start at stage N (finished stages are still skipped)
#   scripts/mac_tonight.sh --only N    # run stage N alone, even if it finished before
#
# Stages (rerunning the script resumes: a finished stage is skipped, a stopped one goes on):
#   1  Lean and Mathlib: reuse the tag already pinned in lean/, or pick it as MAC_SMOKE.md
#      does; the tag is saved to results/mac/mathlib_tag.txt (use the same one on EC2)
#   2  the Lean REPL in ~/repl at the matching Lean version, and a 2 + 2 = 4 check
#   3  the prover model (4.22 GB, resumable download) and the miniF2F benchmark
#   4  search --check, then the 10-problem smoke search and its report
#   5  scripts/after_judges.sh (collecting finished batches is free) -> ~/markov_lean_results.zip
#   6  the local judge on the gold test half, prompt v1 and v2, and their accuracy
#   7  gold-set API scoring: prints the estimate, then asks y/N before submitting (cap $10)
#   8  scripts/mac_overnight.sh, the full search comparison (resumable, yields to runners)
#
# Stages 4, 6 and 8 run under the memory guard (src/markov_lean/mac_guard.py): they stop
# cleanly while a GitHub runner job is active or memory is tight, and go on afterwards.
# Only stage 7 can spend money, and only after you type y. The API key is never read or
# printed: stages 5 and 7 pass your shell's environment through, the rest remove the key.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR:-$HERE}"

PY="${PY:-.venv/bin/python}"
ML=("$PY" -m markov_lean.cli)
STATE="${STATE:-results/mac/tonight}"
TAG_FILE="${TAG_FILE:-results/mac/mathlib_tag.txt}"
LEAN_DIR="${LEAN_DIR:-lean}"
REPL_DIR="${REPL_DIR:-$HOME/repl}"
MODEL="${MODEL:-$HOME/models/deepseek-prover-v1.5-rl_q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/neopolita/deepseek-prover-v1.5-rl-gguf/resolve/main/deepseek-prover-v1.5-rl_q4_k_m.gguf}"
MODEL_MIN_BYTES="${MODEL_MIN_BYTES:-4100000000}"   # the file is 4.22 GB; less means a partial download
MINIF2F="${MINIF2F:-data/raw/deepseek_benchmarks/minif2f.jsonl}"
SMOKE_OUT="${SMOKE_OUT:-results/mac_smoke}"
GOLD="${GOLD:-results/gold}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5:7b-instruct}"
API_CAP="${API_CAP:-10}"
ASK_TIMEOUT="${ASK_TIMEOUT:-900}"   # seconds the y/N waits before it counts as no
SETUP_LEAN="${SETUP_LEAN:-$HERE/scripts/setup_lean.sh}"
AFTER_JUDGES="${AFTER_JUDGES:-$HERE/scripts/after_judges.sh}"
LOCAL_JUDGE="${LOCAL_JUDGE:-$HERE/scripts/mac_local_judge.sh}"
OVERNIGHT="${OVERNIGHT:-$HERE/scripts/mac_overnight.sh}"
NSTAGES=8

FROM=1
ONLY=""
MODE=run
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="${2:?--from needs a stage number}"; shift 2 ;;
    --only) ONLY="${2:?--only needs a stage number}"; shift 2 ;;
    --status) MODE=status; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
for n in "$FROM" ${ONLY:+"$ONLY"}; do
  case "$n" in [1-8]) ;; *) echo "stage must be 1 to $NSTAGES, not $n" >&2; exit 2 ;; esac
done

NAMES=("" "Lean and Mathlib" "Lean REPL" "prover model and miniF2F" "smoke search (10 problems)"
       "after_judges.sh and the results zip" "local judge v1 and v2 on the gold test half"
       "gold-set API scoring (asks first)" "overnight search (scripts/mac_overnight.sh)")
CURRENT=0
say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }
stage_line() { printf '\n========== stage %s of %s: %s\n' "$1" "$NSTAGES" "${NAMES[$1]}"; }
stop() {
  printf '\nSTOPPED at stage %s (%s): %s\n' "$CURRENT" "${NAMES[$CURRENT]:-}" "$*" >&2
  printf 'Fix that and rerun scripts/mac_tonight.sh to resume here.\n' >&2
  if [ "$CURRENT" -ge 1 ] && [ "$CURRENT" -lt "$NSTAGES" ]; then
    printf 'To skip this stage for tonight: scripts/mac_tonight.sh --from %s\n' "$((CURRENT + 1))" >&2
  fi
  exit 1
}
is_done() { [ -f "$STATE/stage$1.done" ]; }
mark() { mkdir -p "$STATE"; printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M')" "${2:-done}" > "$STATE/stage$1.done"; }
# Heavy jobs see no API or AWS credentials (stages 5 and 7 are the exceptions, on purpose).
no_keys=(env -u ANTHROPIC_API_KEY -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN
         -u AWS_PROFILE PYTHONUNBUFFERED=1)
guarded() {  # name hours -- command...: run under the memory guard, which yields to runner jobs
  local name=$1 hours=$2; shift 3
  local g=(-m markov_lean.mac_guard --hours "$hours" --state "$STATE/${name}_guard.json"
           --min-free-gb "${MIN_FREE_GB:-1}" --resume-free-gb "${RESUME_FREE_GB:-10}"
           --max-swap-growth-gb "${MAX_SWAP_GROWTH_GB:-1.5}" --cooldown "${COOLDOWN:-120}" --poll "${POLL:-10}")
  [ "${YIELD_TO_RUNNER:-1}" = 1 ] || g+=(--no-runner-yield)
  "${no_keys[@]}" "$PY" "${g[@]}" -- "$@"
}
export PATH="$HOME/.elan/bin:$PATH"
model_bytes() { if [ -f "$MODEL" ]; then wc -c < "$MODEL" | tr -d ' '; else echo 0; fi; }
pinned_tag() { sed -n 's/^rev = "\(.*\)"/\1/p' "$LEAN_DIR/lakefile.toml" 2>/dev/null | grep -v '^PINNED_BY_SETUP$' || true; }
lean_version() { cut -d: -f2 "$LEAN_DIR/lean-toolchain" 2>/dev/null || true; }

# ------------------------------------------------------------------------------ stages

stage1() {  # Lean, Mathlib and the saved tag
  if pgrep -f '(^|/)lake( |$)' >/dev/null 2>&1; then
    stop "a Lean build (lake) is still running, probably the setup you started earlier.
  Let it finish, or press Ctrl-C in that Terminal window, then rerun this script: it picks up the same tag."
  fi
  local tag
  tag=$(pinned_tag)
  if [ -n "$tag" ] && [ -f "$LEAN_DIR/lean-toolchain" ]; then
    say "lean/ is already pinned to Mathlib $tag: reusing it"
    if [ -f "$LEAN_DIR/lake-manifest.json" ]; then
      (cd "$LEAN_DIR" && lake exe cache get && lake build) \
        || stop "the Mathlib cache download or the Lean build failed (see above)."
    else
      MATHLIB_TAG="$tag" bash "$SETUP_LEAN" || stop "scripts/setup_lean.sh failed (see above)."
    fi
  else
    say "picking the newest Mathlib tag that the REPL also has (as in MAC_SMOKE.md step 2)"
    tag=$(comm -12 \
      <(git ls-remote --tags --refs https://github.com/leanprover-community/mathlib4.git | awk -F/ '{print $3}' | grep -E '^v4\.[0-9]+\.0$' | sort) \
      <(git ls-remote --tags --refs https://github.com/leanprover-community/repl.git | awk -F/ '{print $3}' | grep -E '^v4\.[0-9]+\.0$' | sort) \
      | sort -V | tail -1) || true
    [ -n "$tag" ] || stop "could not list the Mathlib and REPL tags from GitHub (network?)."
    say "Mathlib tag: $tag"
    MATHLIB_TAG="$tag" bash "$SETUP_LEAN" || stop "scripts/setup_lean.sh failed (see above)."
    tag=$(pinned_tag)
  fi
  [ -n "$tag" ] && [ -f "$LEAN_DIR/lean-toolchain" ] || stop "lean/ was not pinned (no rev in lakefile.toml or no lean-toolchain)."
  mkdir -p "$(dirname "$TAG_FILE")"
  printf '%s\n' "$tag" > "$TAG_FILE"
  say "saved the tag to $TAG_FILE ($tag, Lean $(lean_version)). On EC2: MATHLIB_TAG=$tag bash scripts/setup_ec2.sh"
  mark 1 "$tag"
}

stage2() {  # the REPL at the Lean version that lean/ uses
  local v
  v=$(lean_version)
  [ -n "$v" ] || stop "$LEAN_DIR/lean-toolchain is missing: run stage 1 first."
  [ -d "$REPL_DIR/.git" ] || git clone https://github.com/leanprover-community/repl.git "$REPL_DIR" \
    || stop "could not clone the REPL into $REPL_DIR."
  git -C "$REPL_DIR" fetch --tags -q || stop "could not fetch the REPL's tags."
  git -C "$REPL_DIR" checkout -q "$v" || stop "the REPL has no tag $v."
  (cd "$REPL_DIR" && lake build) || stop "the REPL build failed (see above)."
  say "checking the REPL with Mathlib: example : 2 + 2 = 4"
  local out
  out=$(printf '{"cmd": "example : 2 + 2 = 4 := by norm_num"}\n\n' \
        | (cd "$LEAN_DIR" && lake env "$REPL_DIR/.lake/build/bin/repl") 2>&1) || true
  printf '%s\n' "$out" | tail -5
  if ! printf '%s' "$out" | grep -q '"env"' || printf '%s' "$out" | grep -q '"severity": *"error"'; then
    stop "the REPL check did not answer with an env and no error (output above)."
  fi
  mark 2 "$v"
}

stage3() {  # the model and the benchmark
  if [ "$(model_bytes)" -ge "$MODEL_MIN_BYTES" ]; then
    say "model already downloaded: $MODEL"
  else
    say "downloading the prover model (4.22 GB) to $MODEL; Ctrl-C and a rerun resume it"
    mkdir -p "$(dirname "$MODEL")"
    curl -fL --retry 4 -C - -o "$MODEL" "$MODEL_URL" || stop "the model download stopped (rerun to resume it)."
    [ "$(model_bytes)" -ge "$MODEL_MIN_BYTES" ] || stop "the model file is only $(model_bytes) bytes; rerun to resume the download."
  fi
  if [ -f "$MINIF2F" ]; then say "miniF2F already fetched: $MINIF2F"; else
    "${ML[@]}" fetch minif2f || stop "markov-lean fetch minif2f failed."
    [ -f "$MINIF2F" ] || stop "fetch finished but $MINIF2F is missing."
  fi
  mark 3
}

stage4() {  # prover + Lean check, then the smoke search
  if [ -f "$STATE/check.ok" ]; then say "search --check passed earlier"; else
    say "search --check: loads the model, samples 4 tactics, imports Mathlib (a few minutes)"
    local out
    out=$("${no_keys[@]}" "$PY" -m markov_lean.search --check --config configs/search.yaml --backend llama_cpp \
          --model-path "$MODEL" 2>&1) || { printf '%s\n' "$out"; stop "search --check failed (output above)."; }
    printf '%s\n' "$out"
    printf '%s' "$out" | grep -q 'lean ok: True' || stop "search --check did not print 'lean ok: True' (output above)."
    mkdir -p "$STATE"; touch "$STATE/check.ok"
  fi
  local smoke=(-m markov_lean.cli search-eval --config configs/search_eval_mac.yaml --out "$SMOKE_OUT"
               --benchmark minif2f --limit 10 --condition lm --model-path "$MODEL")
  "${no_keys[@]}" "$PY" "${smoke[@]}" --estimate || stop "the smoke estimate failed."
  say "smoke search: 10 problems, at most about 30 minutes plus loading"
  guarded smoke 3 -- "$PY" "${smoke[@]}" --yes || stop "the smoke search stopped (rerun to resume it)."
  "${ML[@]}" search-report --runs "$SMOKE_OUT" --no-figures || stop "search-report failed."
  mark 4
}

stage5() {  # the free steps after the judge batches
  if [ -z "${ANTHROPIC_API_KEY+set}" ] && [ ! -f results/pilot3/judge/run/labels.parquet ]; then
    say "ANTHROPIC_API_KEY is not set in this Terminal, so after_judges.sh cannot download the batches."
    stop "run: export ANTHROPIC_API_KEY=... (in this Terminal only), then rerun this script. Collecting is free."
  fi
  bash "$AFTER_JUDGES" || stop "scripts/after_judges.sh stopped or some of its steps failed (see above;
  its log is results/pilot3/.after_judges/after_judges.log)."
  mark 5
}

stage6() {  # local judge, v1 and v2 prompts, on the gold test half (docs/judge_accuracy.md)
  command -v ollama >/dev/null 2>&1 || stop "ollama is not installed or not on PATH; open the Ollama app."
  ollama list 2>/dev/null | grep -q "$JUDGE_MODEL" \
    || stop "Ollama is not running or has no $JUDGE_MODEL. Open the Ollama app (and if needed: ollama pull $JUDGE_MODEL)."
  local steps="$GOLD/steps_test.parquet" local1 local2
  local1="$GOLD/local/$(printf '%s' "$JUDGE_MODEL" | tr ':' '_')"
  local2="${local1}_v2"
  if [ -f "$steps" ]; then say "gold test steps already written: $steps"; else
    "${ML[@]}" gold-set steps --strict-only --eval-split test --out "$steps" || stop "gold-set steps failed."
  fi
  local common=(--steps "$steps" --all --stratify-by none --model "$JUDGE_MODEL")
  say "local judge, prompt v1 (reuses problems already labelled)"
  bash "$LOCAL_JUDGE" "${common[@]}" --out "$local1" || stop "the v1 labelling stopped (rerun to resume it)."
  say "local judge, prompt v2"
  bash "$LOCAL_JUDGE" "${common[@]}" --prompt v2 --out "$local2" || stop "the v2 labelling stopped (rerun to resume it)."
  ollama stop "$JUDGE_MODEL" >/dev/null 2>&1 || true   # give the memory back before the search
  "${ML[@]}" judge-accuracy --eval-split test --out "$GOLD/accuracy_test" \
    --labels "$local1/labels.parquet" "$local2/labels.parquet" || stop "judge-accuracy failed."
  mark 6
}

stage7() {  # the only stage that can spend money, and only on a typed y
  local steps="$GOLD/steps_all.parquet" api="$GOLD/api"
  if [ -f "$api/batches.json" ]; then
    say "the gold set was already submitted ($api/batches.json): not submitting again"
    mark 7 submitted; return 0
  fi
  if [ -f "$steps" ]; then say "full gold steps already written: $steps"; else
    "${ML[@]}" gold-set steps --out "$steps" || stop "gold-set steps failed."
  fi
  say "estimate for the whole gold set (spends nothing):"
  "${no_keys[@]}" "$PY" -m markov_lean.judge estimate --steps "$steps" || stop "judge estimate failed."
  local answer=n
  if [ -t 0 ] || [ "${MAC_TONIGHT_TTY:-0}" = 1 ]; then
    printf '\nSubmit the gold set to the API judges now, capped at $%s? This spends API credit.\n' "$API_CAP"
    printf 'No answer in %s minutes counts as no, so the search still starts. [y/N] ' "$((ASK_TIMEOUT / 60))"
    read -r -t "$ASK_TIMEOUT" answer || { echo; answer=n; }
  else
    say "not at a terminal, so not asking: skipping the submit"
  fi
  case "$answer" in
    y|Y|yes|YES) ;;
    *) say "not submitted. To be asked again later: scripts/mac_tonight.sh --only 7"
       mark 7 declined; return 0 ;;
  esac
  if [ -z "${ANTHROPIC_API_KEY+set}" ]; then
    say "ANTHROPIC_API_KEY is not set in this Terminal: not submitted."
    say "Run: export ANTHROPIC_API_KEY=... then: scripts/mac_tonight.sh --only 7"
    mark 7 declined; return 0
  fi
  "$PY" -m markov_lean.judge submit --steps "$steps" --out "$api" --max-dollars "$API_CAP" \
    || stop "judge submit failed (see above). Nothing is resubmitted unless $api/batches.json is missing."
  say "submitted. Tomorrow: $PY -m markov_lean.judge collect --out $api, then bash scripts/after_judges.sh"
  mark 7 submitted
}

stage8() {  # the overnight search; it resumes and reports by itself
  local n
  for n in 1 2 3 4; do
    is_done "$n" || stop "stage $n (${NAMES[$n]}) has not finished; the search needs it. Run scripts/mac_tonight.sh."
  done
  for n in 5 6 7; do is_done "$n" || say "note: stage $n (${NAMES[$n]}) was skipped tonight"; done
  say "Mathlib tag for EC2: $(cat "$TAG_FILE" 2>/dev/null || echo unknown)"
  say "starting the full search. From another Terminal: scripts/mac_tonight.sh --status"
  mkdir -p "$STATE"; date '+%Y-%m-%d %H:%M' > "$STATE/stage8.started"
  MODEL="$MODEL" exec bash "$OVERNIGHT"
}

# ------------------------------------------------------------------------------ status

if [ "$MODE" = status ]; then
  for n in 1 2 3 4 5 6 7 8; do
    if is_done "$n"; then s="done   $(cat "$STATE/stage$n.done")"
    elif [ "$n" = 8 ] && [ -f "$STATE/stage8.started" ]; then s="started $(cat "$STATE/stage8.started")"
    else s="not yet"; fi
    printf '  %s %-48s %s\n' "$n" "${NAMES[$n]}" "$s"
  done
  printf '  Mathlib tag: %s   model: %s bytes\n' "$(cat "$TAG_FILE" 2>/dev/null || echo 'not saved yet')" "$(model_bytes)"
  if [ -f "$STATE/stage8.started" ]; then echo; bash "$OVERNIGHT" --status || true; fi
  exit 0
fi

# ------------------------------------------------------------------------------ run

[ -x "$PY" ] || { echo "no $PY: run scripts/setup_mac.sh first (MAC_SMOKE.md step 1)" >&2; exit 1; }
"$PY" -c 'import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 3)' 2>/dev/null || {
  echo "llama-cpp-python is missing or built without Metal: redo MAC_SMOKE.md step 1" >&2; exit 1; }
mkdir -p "$STATE/logs"
LOG="$STATE/logs/$(date '+%Y%m%d-%H%M%S').log"
exec > >(tee -ai "$LOG") 2>&1
say "log: $LOG"

if [ -n "$ONLY" ]; then first=$ONLY; last=$ONLY; else first=$FROM; last=$NSTAGES; fi
n=$first
while [ "$n" -le "$last" ]; do
  CURRENT=$n
  stage_line "$n"
  if [ -z "$ONLY" ] && [ "$n" != 8 ] && is_done "$n"; then
    say "already done ($(cat "$STATE/stage$n.done")), skipping"
  else
    "stage$n"
  fi
  n=$((n + 1))
done
say "finished stage $last."
