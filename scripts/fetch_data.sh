#!/usr/bin/env bash
# Download the datasets into data/raw/ and convert them to Parquet in data/processed/.
# Run on the Mac Mini or EC2 (cloud Claude sessions can't reach these hosts).
# Safe to re-run: anything already downloaded is skipped.
#
#   ./scripts/fetch_data.sh                 # everything
#   ./scripts/fetch_data.sh leandojo minif2f  # a subset
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"

ALL="leandojo workbook minif2f proofnet putnam"
SOURCES="${*:-$ALL}"

md5_of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

fetch_leandojo() {
  # Zenodo record 12740403 (v10); URL and MD5 as given in ReProver's scripts/download_data.py.
  local url="https://zenodo.org/records/12740403/files/leandojo_benchmark_4.tar.gz?download=1"
  local md5="25e1ee60cd8925b9d2e8673ddcc34b4c"
  if [ -d "$RAW/leandojo_benchmark_4" ]; then echo "leandojo: already present"; return; fi
  local tgz="$RAW/leandojo_benchmark_4.tar.gz"
  curl -fL --retry 4 -o "$tgz" "$url"
  [ "$(md5_of "$tgz")" = "$md5" ] || { echo "leandojo: MD5 mismatch" >&2; exit 1; }
  tar -xzf "$tgz" -C "$RAW" && rm "$tgz"
  echo "leandojo: ok"
}

fetch_workbook() {
  # Hugging Face dataset internlm/Lean-Workbook: wkbk_1009.parquet holds the tactic-level
  # rows; lean_workbook.json (about 95 MB) holds about 140k problems as statements only.
  local dir="$RAW/lean_workbook"; mkdir -p "$dir"
  local base="https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main"
  for f in wkbk_1009.parquet lean_workbook.json; do
    if [ -s "$dir/$f" ]; then echo "workbook $f: already present"; continue; fi
    curl -fL --retry 4 -o "$dir/$f" "$base/$f"
    echo "workbook $f: ok ($(md5_of "$dir/$f"))"
  done
}

fetch_deepseek() {  # fetch_deepseek <minif2f|proofnet>
  # DeepSeek-Prover-V1.5's Lean 4 copies, which carry the valid/test split and goal states.
  local dir="$RAW/deepseek_benchmarks"; mkdir -p "$dir"
  if [ -s "$dir/$1.jsonl" ]; then echo "$1: already present"; return; fi
  curl -fL --retry 4 -o "$dir/$1.jsonl" \
    "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/main/datasets/$1.jsonl"
  echo "$1: ok ($(wc -l < "$dir/$1.jsonl" | tr -d ' ') rows)"
}

clone() {  # clone <name> <url>
  if [ -d "$RAW/$1/.git" ]; then echo "$1: already present"; return; fi
  git clone --depth 1 "$2" "$RAW/$1"
  echo "$1: ok at $(git -C "$RAW/$1" rev-parse --short HEAD)"
}

for s in $SOURCES; do
  case "$s" in
    leandojo) fetch_leandojo ;;
    workbook) fetch_workbook ;;
    minif2f)  fetch_deepseek minif2f ;;
    proofnet) fetch_deepseek proofnet ;;
    putnam)   clone PutnamBench https://github.com/trishullab/PutnamBench ;;
    *) echo "unknown source: $s (choose from: $ALL)" >&2; exit 2 ;;
  esac
done

echo "== Converting to Parquet"
if [ -d .venv ]; then source .venv/bin/activate; fi
python -c "from markov_lean.data import build_all; build_all('data/raw', 'data/processed')"
