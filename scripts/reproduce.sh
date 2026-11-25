#!/usr/bin/env bash
# Reproduce every table and figure, and the dataset release, from a clean clone.
# REPRODUCE.md explains each stage, where it runs, how long it takes and what it costs.
#
#   bash scripts/reproduce.sh                  # all Mac stages, reusing cached judge labels
#   bash scripts/reproduce.sh --dry-run        # print the plan and every command, run nothing
#   bash scripts/reproduce.sh --stage pilot --stage judge
#   bash scripts/reproduce.sh --from score     # this stage and every one after it
#   bash scripts/reproduce.sh --labels path/to/labels.parquet   # use judge labels you already have
#   bash scripts/reproduce.sh --run-judges     # call the Anthropic API (asks first; ~$78 for pilot3)
#
# Stages, in order: setup data synthetic pilot judge score analysis full search report paper release
#
# Nothing here spends money unless you pass --run-judges (Anthropic API, after an estimate and
# a typed "yes") or run the search stage on a GPU instance you launched yourself. The script
# never creates, starts or stops cloud instances, never uploads anything, and never writes a key
# to disk: ANTHROPIC_API_KEY is read from your shell only.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_STAGES="setup data synthetic pilot judge score analysis full search report paper release"
STAGES=""
DRY_RUN=0
RUN_JUDGES=0
ASSUME_YES=0
RUN_SEARCH=0
LABELS=""
PILOT=${PILOT:-results/pilot3}
RELEASE=${RELEASE:-release}
SEARCH_OUT=${SEARCH_OUT:-results/search_eval}
PILOT_CONFIG=${PILOT_CONFIG:-configs/pilot.yaml}
PAPER_OUT=${PAPER_OUT:-paper}           # where paper_assets.py writes figures/ and tables/
MAX_DOLLARS=${MAX_DOLLARS:-85}
ROBUSTNESS_ARGS=${ROBUSTNESS_ARGS:-}    # e.g. "--jobs 4" or "--quick" for markov-lean robustness
PY=${PY:-}

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGES="$STAGES ${2:?--stage needs a name}"; shift 2 ;;
    --from)
      FROM=${2:?--from needs a name}; seen=0
      for s in $ALL_STAGES; do [ "$s" = "$FROM" ] && seen=1; [ "$seen" = 1 ] && STAGES="$STAGES $s"; done
      [ "$seen" = 1 ] || { echo "unknown stage: $FROM" >&2; exit 2; }
      shift 2 ;;
    --dry-run|-n) DRY_RUN=1; shift ;;
    --run-judges) RUN_JUDGES=1; shift ;;
    --run-search) RUN_SEARCH=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    --labels) LABELS=${2:?--labels needs a path}; shift 2 ;;
    --pilot) PILOT=${2:?}; shift 2 ;;
    --release) RELEASE=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
STAGES=${STAGES:-$ALL_STAGES}
for s in $STAGES; do
  case " $ALL_STAGES " in *" $s "*) ;; *) echo "unknown stage: $s (stages: $ALL_STAGES)" >&2; exit 2 ;; esac
done

if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
ML="$PY -m markov_lean.cli"

say() { printf '\n== %s\n' "$*"; }
run() {  # print a command, then run it unless --dry-run
  printf '+ %s\n' "$*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
have_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

LABELS_DST="$PILOT/judge/labels.parquet"

if want setup; then
  say "setup (Mac or EC2, free): Python venv, this package, tests"
  if [ -x .venv/bin/python ]; then echo "venv present: .venv"; else run bash scripts/setup_mac.sh; fi
  PY=.venv/bin/python; ML="$PY -m markov_lean.cli"
fi

if want data; then
  say "data (Mac, free): download LeanDojo Benchmark 4, Lean Workbook, miniF2F, ProofNet, PutnamBench"
  run bash scripts/fetch_data.sh
fi

if want synthetic; then
  say "synthetic (Mac, free): planted-area sanity check"
  run $ML pilot --config configs/pilot_synthetic.yaml --out results/pilot_synthetic
fi

if want pilot; then
  say "pilot pass 1 (Mac, free): fit the Markov model, write the judge sample"
  if [ -f "$PILOT/judge/sample_steps.parquet" ]; then
    echo "already done: $PILOT/judge/sample_steps.parquet exists (delete $PILOT to start over)"
  elif [ -f "$LABELS_DST" ]; then
    echo "$LABELS_DST exists but the judge sample does not; move the labels aside, run this stage," >&2
    echo "then put them back (pass 1 writes the sample only while no labels are present)." >&2
    exit 1
  else
    run $ML pilot --config "$PILOT_CONFIG" --out "$PILOT"
  fi
fi

if want judge; then
  say "judge labels"
  if [ -n "$LABELS" ]; then
    echo "using labels from --labels"
    run mkdir -p "$PILOT/judge"
    run cp "$LABELS" "$LABELS_DST"
  elif [ -f "$LABELS_DST" ]; then
    echo "reusing cached labels: $LABELS_DST (delete it to judge again)"
  elif [ -f "$PILOT/judge/run/labels.parquet" ]; then
    echo "reusing collected labels: $PILOT/judge/run/labels.parquet"
    run cp "$PILOT/judge/run/labels.parquet" "$LABELS_DST"
  elif [ "$RUN_JUDGES" = 1 ]; then
    J="$PY -m markov_lean.judge --config configs/judge.yaml"
    SAMPLE="$PILOT/judge/sample_steps.parquet"
    run $J estimate --steps "$SAMPLE"
    if [ "$DRY_RUN" = 0 ]; then
      if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "ANTHROPIC_API_KEY is not set in this shell. Export it (never put it in a file here) and rerun." >&2
        exit 1
      fi
      if [ "$ASSUME_YES" != 1 ]; then
        printf 'This submits Message Batches billed to your Anthropic account (cap $%s). Type yes to go on: ' "$MAX_DOLLARS"
        read -r answer
        [ "$answer" = "yes" ] || { echo "Stopped before any API call."; exit 1; }
      fi
    fi
    if [ -f "$PILOT/judge/run/batches.json" ]; then
      echo "batches already submitted for this run; collecting them instead of submitting again"
    else
      run $J submit --steps "$SAMPLE" --out "$PILOT/judge/run" --max-dollars "$MAX_DOLLARS"
    fi
    run $J collect --out "$PILOT/judge/run"
    run cp "$PILOT/judge/run/labels.parquet" "$LABELS_DST"
  else
    echo "No judge labels in $PILOT/judge/ and --run-judges not given."
    echo "Either pass --labels <labels.parquet> (for example from the released dataset's run),"
    echo "or rerun with --run-judges to call the Anthropic API (see REPRODUCE.md for the cost)."
    echo "Stages that need labels (score, paper tables, release) will be skipped."
  fi
fi

if want score; then
  say "pilot pass 2 (Mac, free): score macrostates against the consensus labels, gate 1"
  if [ -f "$LABELS_DST" ] || [ "$DRY_RUN" = 1 ]; then
    # exit 1 also means "gate 1 failed", which still writes a scored report: carry on only then
    status=0
    run $ML pilot --config "$PILOT_CONFIG" --out "$PILOT" || status=$?
    if [ "$status" != 0 ] && [ "$DRY_RUN" = 0 ]; then
      if "$PY" -c 'import json,sys; sys.exit(0 if "eval" in json.load(open(sys.argv[1])) else 1)' \
          "$PILOT/pilot_report.json" 2>/dev/null; then
        echo "gate 1 did not pass; see $PILOT/pilot_report.json"
      else
        echo "pilot pass 2 failed (exit $status)" >&2; exit "$status"
      fi
    fi
  else
    echo "skipped: no $LABELS_DST"
  fi
fi

if want analysis; then
  say "gate 1 analysis (Mac, free): robustness, significance, bridge steps, case studies"
  if [ -f "$LABELS_DST" ] || [ "$DRY_RUN" = 1 ]; then
    # robustness exits 1 when the robust verdict fails; its tables are still written, so carry on then
    status=0
    run $ML robustness --pilot "$PILOT" $ROBUSTNESS_ARGS || status=$?
    if [ "$status" != 0 ] && [ "$DRY_RUN" = 0 ]; then
      if [ -f "$PILOT/robustness/robustness.json" ]; then
        echo "robustness verdict did not pass; see $PILOT/robustness/robustness.md"
      else
        echo "robustness failed (exit $status)" >&2; exit "$status"
      fi
    fi
    run $ML significance --run "$PILOT" --config "$PILOT_CONFIG"
    run $ML bridges --run "$PILOT" --out "$PILOT/bridges"
    run $ML cases --run "$PILOT" --out "$PAPER_OUT/figures"
  else
    echo "skipped: no $LABELS_DST"
  fi
fi

if want full; then
  say "full fit and projection (Mac, free, 16 GB): every LeanDojo train proof, all datasets placed"
  run $ML full --config configs/full.yaml --out results/full
fi

if want search; then
  say "proof search (EC2 g6.xlarge with an L4 GPU; ~\$30 to \$46 of instance time; see docs/search_eval.md)"
  if [ -d "$SEARCH_OUT" ] && ls "$SEARCH_OUT"/*/*/seed*/summary.jsonl >/dev/null 2>&1; then
    echo "reusing search runs in $SEARCH_OUT (copied back from the instance)"
  elif [ "$RUN_SEARCH" = 1 ] && have_gpu; then
    E="$ML search-eval --config configs/search_eval.yaml --out $SEARCH_OUT"
    run bash scripts/serve_prover.sh
    run $PY -m markov_lean.search --check --config configs/search.yaml
    run $E --benchmark minif2f --condition lm --condition area_l0.3 --condition full --condition random_l0.3 --yes --max-usd 20
    run $E --benchmark minif2f --condition area_l0.1 --condition area_l0.6 --condition diversity --condition stuck_escape --yes --max-usd 20
    run $E --benchmark putnam --limit 150 --condition lm --condition full --condition random_l0.3 --yes --max-usd 15
  else
    echo "skipped: runs only on the GPU instance (with --run-search), after scripts/setup_ec2.sh."
    echo "On the Mac, copy the results back instead: rsync -av ubuntu@<ip>:markov-lean/$SEARCH_OUT/ $SEARCH_OUT/"
  fi
fi

if want report; then
  say "search report (Mac, free): pass@k tables, paired tests, figures"
  if ls "$SEARCH_OUT"/*/*/seed*/summary.jsonl >/dev/null 2>&1 || [ "$DRY_RUN" = 1 ]; then
    run $ML search-report --runs "$SEARCH_OUT"
  else
    echo "skipped: no search runs in $SEARCH_OUT"
  fi
fi

if want paper; then
  say "paper figures, tables and PDF (Mac, free)"
  # writes <paper>/figures: the figures, table_*.tex and results_macros.tex that main.tex reads
  run $ML paper --results results --paper "$PAPER_OUT" --pilot "$PILOT" --search "$SEARCH_OUT"
  if [ "$PAPER_OUT" != paper ]; then
    echo "PAPER_OUT is not paper/: not building the PDF"
  elif command -v latexmk >/dev/null 2>&1 || command -v pdflatex >/dev/null 2>&1; then
    run make -C paper
  else
    echo "no LaTeX on this machine: skip the PDF (brew install --cask mactex-no-gui)"
  fi
fi

if want release; then
  say "dataset release folder (Mac, free): export and check, nothing is uploaded"
  if [ -f "$LABELS_DST" ] || [ "$DRY_RUN" = 1 ]; then
    run $ML export-dataset --run "$PILOT" --out "$RELEASE" --force
    run $ML check-release "$RELEASE"
  else
    echo "skipped: no $LABELS_DST"
  fi
fi

say "done"
