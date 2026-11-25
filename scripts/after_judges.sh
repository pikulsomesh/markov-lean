#!/usr/bin/env bash
# Everything free that follows the judge Message Batches, in one command on the Mac.
#
#   export ANTHROPIC_API_KEY=...         # in this shell only; downloading finished results is free
#   bash scripts/after_judges.sh
#
# Steps, in order (each is skipped when it already finished on the same labels, so just rerun):
#   collect     download the finished batches into <pilot>/judge/run/labels.parquet (no waiting)
#   install     copy those labels to <pilot>/judge/labels.parquet, where every stage reads them
#   score       pilot pass 2: macrostates against the consensus labels, the gate 1 call
#   agreement   judge-agreement across Haiku, Sonnet and the local Qwen judge
#   switches    within-proof area switching against the nulls
#   significance  proof-level tests for gate 1, local Qwen labels as an extra judge
#   bridges     tactics and lemmas at area switches
#   cases       case-study figure and table for the paper
#   robustness  baselines and sweeps (only with --robustness; it is the slow one)
#   accuracy    gold-set accuracy of every judge whose gold labels are on disk
#   paper       paper/figures (tables, figures, results_macros.tex) and the PDF if LaTeX is installed
#   zip         ~/markov_lean_results.zip, to drag into the Claude Duet chat
#
# Options:
#   --dry-run        print the plan and every command, run nothing
#   --robustness     also run `markov-lean robustness` (ROBUSTNESS_ARGS="--jobs 4" passes options)
#   --recollect      download the batch results again even if labels are already collected
#   --force          rerun every analysis step even if it already finished
#   --no-zip         skip the zip
#   --pilot DIR      pilot run folder (default results/pilot3)
#   --zip PATH       where to write the zip (default ~/markov_lean_results.zip)
#
# This never submits, cancels or creates anything: the only API calls are reading the status and
# the results of batches you already submitted, through a client that blocks every other call.
# Nothing here launches cloud instances or uploads anything. The key is read from your shell only.
set -euo pipefail
cd "$(dirname "$0")/.."

PILOT=${PILOT:-results/pilot3}
PILOT_CONFIG=${PILOT_CONFIG:-configs/pilot.yaml}
PAPER_OUT=${PAPER_OUT:-paper}
RESULTS=${RESULTS:-results}
GOLD=${GOLD:-results/gold}
ZIP=${ZIP:-$HOME/markov_lean_results.zip}
ROBUSTNESS_ARGS=${ROBUSTNESS_ARGS:-}
MAX_ZIP_FILE_MB=${MAX_ZIP_FILE_MB:-60}
PY=${PY:-}
DRY_RUN=0
ROBUSTNESS=0
RECOLLECT=0
FORCE=0
DO_ZIP=1

usage() { sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|-n) DRY_RUN=1; shift ;;
    --robustness) ROBUSTNESS=1; shift ;;
    --recollect) RECOLLECT=1; shift ;;
    --force) FORCE=1; shift ;;
    --no-zip) DO_ZIP=0; shift ;;
    --pilot) PILOT=${2:?--pilot needs a folder}; shift 2 ;;
    --zip) ZIP=${2:?--zip needs a path}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
ML="$PY -m markov_lean.cli"
RUN_DIR="$PILOT/judge/run"
RUN_LABELS="$RUN_DIR/labels.parquet"
LABELS="$PILOT/judge/labels.parquet"
LOCAL_LABELS=${LOCAL_LABELS:-$PILOT/judge/local/qwen2.5_7b-instruct/labels.parquet}
STATE="$PILOT/.after_judges"
FAILED=""

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }
run() {  # print a command, then run it unless --dry-run
  printf '+ %s\n' "$*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}
sha() { "$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }

# A step is done when its marker holds the hash of the labels it ran on.
LABELS_SHA=""
done_already() {
  [ "$FORCE" = 0 ] && [ -n "$LABELS_SHA" ] && [ -f "$STATE/$1.done" ] && [ "$(cat "$STATE/$1.done")" = "$LABELS_SHA" ]
}
mark_done() { [ "$DRY_RUN" = 1 ] || { mkdir -p "$STATE"; echo "$LABELS_SHA" > "$STATE/$1.done"; }; }
# Run one analysis step; a failure is reported at the end and the remaining steps still run.
step() {
  local name=$1; shift
  if done_already "$name"; then echo "already done on these labels (--force to redo)"; return 0; fi
  if [ "$DRY_RUN" = 1 ]; then printf '+ %s\n' "$*"; return 0; fi
  printf '+ %s\n' "$*"
  if "$@"; then mark_done "$name"; else
    echo "FAILED: $name (exit $?). The other steps carry on; rerun this command to retry it." >&2
    FAILED="$FAILED $name"
  fi
}

# ----------------------------------------------------------------------------- preflight

say "preflight"
[ -f pyproject.toml ] && [ -d src/markov_lean ] || die "run this from a markov-lean clone"
"$PY" -c 'import markov_lean' 2>/dev/null || die "markov_lean is not importable with $PY; run: source .venv/bin/activate (or bash scripts/setup_mac.sh)"
[ -f "$PILOT_CONFIG" ] || die "pilot config $PILOT_CONFIG not found"
[ -f "$PILOT/judge/sample_steps.parquet" ] || die "$PILOT/judge/sample_steps.parquet is missing: this needs the pilot run that the batches judged (pass --pilot DIR if it lives elsewhere)"
[ -d "$PILOT/msm" ] || die "$PILOT/msm is missing: this needs the pilot run that the batches judged"
NEED_COLLECT=1
if [ -f "$RUN_LABELS" ] && [ "$RECOLLECT" = 0 ]; then NEED_COLLECT=0; fi
if [ "$NEED_COLLECT" = 1 ]; then
  [ -f "$RUN_DIR/batches.json" ] || die "$RUN_DIR/batches.json is missing, so there is nothing to collect. It is written when the batches are submitted; run this on the Mac that submitted them."
  [ -f "$RUN_DIR/manifest.json" ] || die "$RUN_DIR/manifest.json is missing; it is written next to batches.json at submit time."
  if [ "$DRY_RUN" = 0 ] && [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${AFTER_JUDGES_CLIENT_FACTORY:-}" ]; then
    die "ANTHROPIC_API_KEY is not set in this shell. Run: export ANTHROPIC_API_KEY=... (never in a file here), then rerun. Downloading finished results is free."
  fi
fi
if [ -f "$LOCAL_LABELS" ]; then echo "local judge labels: $LOCAL_LABELS"; else
  echo "no local judge labels at $LOCAL_LABELS: agreement and significance use Haiku and Sonnet only"; LOCAL_LABELS=""; fi
echo "pilot: $PILOT   python: $PY"

if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$STATE"
  LOG="$STATE/after_judges.log"
  exec > >(tee "$LOG") 2>&1
  echo "log: $LOG"
fi

# ----------------------------------------------------------------------------- collect and install

say "collect (free): download the finished batches, no waiting"
if [ "$NEED_COLLECT" = 0 ]; then
  echo "already collected: $RUN_LABELS (--recollect to download again)"
else
  status=0
  run "$PY" scripts/collect_batches.py --out "$RUN_DIR" || status=$?
  case "$status" in
    0) ;;
    3) die "some batches have not ended yet (status above). Nothing was written; rerun later." ;;
    *) die "collect failed (exit $status); see the message above" ;;
  esac
fi

say "install the labels where every stage reads them"
if [ "$DRY_RUN" = 0 ]; then
  [ -f "$RUN_LABELS" ] || die "$RUN_LABELS was not written"
  "$PY" - "$RUN_LABELS" "$RUN_DIR/issues.json" <<'EOF'
import json, sys
import pandas as pd
lab = pd.read_parquet(sys.argv[1])
if lab.empty:
    sys.exit("STOPPED: the collected labels are empty; check issues.json")
per = lab.groupby("judge").agg(rows=("proof_id", "size"), proofs=("proof_id", "nunique"))
print(per.to_string())
try:
    issues = json.load(open(sys.argv[2]))
except OSError:
    issues = []
counts = {}
for i in issues:
    counts[i.get("issue")] = counts.get(i.get("issue"), 0) + 1
print("issues:", counts or "none")
if lab["judge"].nunique() < 2:
    sys.exit("STOPPED: labels from fewer than two judges; the consensus needs Haiku and Sonnet")
EOF
  if [ -f "$LABELS" ] && ! cmp -s "$RUN_LABELS" "$LABELS"; then
    cp "$LABELS" "$PILOT/judge/labels.before_after_judges.parquet"
    echo "kept the previous $LABELS as $PILOT/judge/labels.before_after_judges.parquet"
  fi
  cp "$RUN_LABELS" "$LABELS"
  LABELS_SHA=$(sha "$LABELS")
  echo "$LABELS (sha256 ${LABELS_SHA:0:12})"
else
  printf '+ cp %s %s\n' "$RUN_LABELS" "$LABELS"
fi

# ----------------------------------------------------------------------------- analyses (no API calls)

say "score (free): pilot pass 2 and the gate 1 call"
# exit 1 from the pilot means "gate 1 failed", which still writes a scored report
gate_may_fail() {
  local s=0
  "$@" || s=$?
  [ "$s" = 0 ] && return 0
  "$PY" -c 'import json,sys; sys.exit(0 if "eval" in json.load(open(sys.argv[1])) else 1)' "$PILOT/pilot_report.json" \
    2>/dev/null && { echo "gate 1 did not pass (report written)"; return 0; }
  return "$s"
}
step score gate_may_fail $ML pilot --config "$PILOT_CONFIG" --out "$PILOT"

say "agreement (free): Haiku, Sonnet and the local judge"
step agreement $ML judge-agreement --run "$PILOT" --labels "$LABELS" ${LOCAL_LABELS:+"$LOCAL_LABELS"}

say "switches (free)"
step switches $ML switches --run "$PILOT" --config "$PILOT_CONFIG"

say "significance (free)"
step significance $ML significance --run "$PILOT" --config "$PILOT_CONFIG" ${LOCAL_LABELS:+--extra-labels "$LOCAL_LABELS"}

say "bridges (free)"
step bridges $ML bridges --run "$PILOT" --out "$PILOT/bridges"

say "cases (free): case-study figure and table"
step cases $ML cases --run "$PILOT" --out "$PAPER_OUT/figures"

say "robustness (free, slow)"
if [ "$ROBUSTNESS" = 1 ]; then
  verdict_may_fail() {  # exit 1 means the robust verdict failed; its tables are still written
    local s=0
    "$@" || s=$?
    [ "$s" = 0 ] || [ -f "$PILOT/robustness/robustness.json" ]
  }
  # shellcheck disable=SC2086
  step robustness verdict_may_fail $ML robustness --pilot "$PILOT" $ROBUSTNESS_ARGS
else
  echo "skipped (add --robustness; it runs the full sweep grid)"
fi

say "accuracy (free): judges on the gold set"
GOLD_LABELS=""
for f in "$GOLD"/local/*/labels.parquet "$GOLD"/api/labels.parquet; do
  [ -f "$f" ] && GOLD_LABELS="$GOLD_LABELS $f"
done
if [ -n "$GOLD_LABELS" ]; then
  # shellcheck disable=SC2086
  step accuracy $ML judge-accuracy --labels $GOLD_LABELS --out "$GOLD/accuracy"
else
  echo "skipped: no gold-set labels under $GOLD/local or $GOLD/api"
fi

say "paper (free): figures, tables, results_macros.tex"
ACC_ARGS=()   # bash 3.2 (macOS) safe expansion below
if [ -f "$GOLD/accuracy/judge_accuracy.json" ] || [ "$DRY_RUN" = 1 ]; then ACC_ARGS=(--accuracy "$GOLD/accuracy"); fi
FORCE_SAVED=$FORCE; FORCE=1   # always rebuild: it reads every step above
step paper $ML paper --results "$RESULTS" --paper "$PAPER_OUT" --pilot "$PILOT" ${ACC_ARGS[@]+"${ACC_ARGS[@]}"}
FORCE=$FORCE_SAVED
if [ "$PAPER_OUT" = paper ] && { command -v latexmk >/dev/null 2>&1 || command -v pdflatex >/dev/null 2>&1; }; then
  run make -C paper || echo "the PDF did not build; the figures and macros above are still fine" >&2
else
  echo "no PDF: LaTeX is not installed here (brew install --cask mactex-no-gui), or PAPER_OUT is not paper/"
fi

# ----------------------------------------------------------------------------- zip

say "zip the outputs for the chat"
if [ "$DO_ZIP" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    printf '+ (write %s)\n' "$ZIP"
  else
    "$PY" - "$ZIP" "$MAX_ZIP_FILE_MB" "$PILOT" "$PAPER_OUT" "$GOLD" "${LOCAL_LABELS:-}" <<'EOF'
import re, sys, zipfile
from pathlib import Path
zip_path, max_mb, pilot, paper, gold, local = sys.argv[1], float(sys.argv[2]), *map(Path, sys.argv[3:6]), sys.argv[6]
want = [pilot / "judge/run/labels.parquet", pilot / "judge/run/issues.json", pilot / "pilot_report.json",
        pilot / "judge/sample_steps.parquet", pilot / "msm/assignments.parquet", pilot / "msm/msm_report.json",
        pilot / ".after_judges/after_judges.log",
        pilot / "agreement", pilot / "switches", pilot / "significance", pilot / "bridges", pilot / "robustness",
        paper / "figures", paper / "main.pdf", gold / "accuracy"]
if local:
    want.append(Path(local))
KEY = re.compile(rb"sk-ant-[A-Za-z0-9_\-]{8,}")
files, skipped = [], []
for w in want:
    for f in ([w] if w.is_file() else sorted(p for p in w.rglob("*") if p.is_file()) if w.is_dir() else []):
        if "data/raw" in f.as_posix() or f.name.startswith(".env"):
            skipped.append((f, "excluded")); continue
        if f.stat().st_size > max_mb * 1e6:
            skipped.append((f, f"over {max_mb:g} MB")); continue
        if KEY.search(f.read_bytes()):
            sys.exit(f"STOPPED: {f} looks like it holds an API key; not zipping. Remove the key and rerun.")
        files.append(f)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:  # repo-relative names; an absolute path outside the repo keeps no leading /
        try:
            name = f.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            name = f.as_posix().lstrip("/")
        z.write(f, name)
for f, why in skipped:
    print(f"left out {f} ({why})")
print(f"{len(files)} files, {Path(zip_path).stat().st_size / 1e6:.1f} MB: {zip_path}")
EOF
  fi
else
  echo "skipped (--no-zip)"
fi

# ----------------------------------------------------------------------------- verdict

say "gate 1"
if [ "$DRY_RUN" = 1 ]; then
  echo "(dry run: no verdict)"
else
  "$PY" - "$PILOT/pilot_report.json" <<'EOF' || true
import json, sys
try:
    g = json.load(open(sys.argv[1]))["eval"]["gate1"]
except (OSError, KeyError, ValueError):
    print("GATE 1: no verdict (the score step did not finish)"); sys.exit(0)
word = "PASSED" if g["passed"] else "FAILED"
print(f"GATE 1 {word} at k={g['k']}: macrostate NMI {g['macrostate_nmi']:.3f} vs folder baseline "
      f"{g['folder_nmi']:.3f} (needs >= {g['threshold']} and above the baseline)")
EOF
fi
if [ -n "$FAILED" ]; then
  echo
  echo "Some steps failed:$FAILED. The zip still has the log; rerun this command to retry only those."
fi
if [ "$DO_ZIP" = 1 ] && [ "$DRY_RUN" = 0 ]; then
  echo
  echo ">>> Drag $ZIP into the Claude Duet chat."
fi
[ -z "$FAILED" ]
