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
  # Hugging Face dataset internlm/Lean-Workbook, file lean_workbook.json (about 95 MB).
  local dir="$RAW/lean_workbook"; mkdir -p "$dir"
  if [ -s "$dir/lean_workbook.json" ]; then echo "workbook: already present"; return; fi
  curl -fL --retry 4 -o "$dir/lean_workbook.json" \
    "https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main/lean_workbook.json"
  echo "workbook: ok ($(md5_of "$dir/lean_workbook.json"))"
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
    minif2f)  clone miniF2F-lean4 https://github.com/yangky11/miniF2F-lean4 ;;
    proofnet) clone ProofNet-lean4 https://github.com/rahul3613/ProofNet-lean4 ;;
    putnam)   clone PutnamBench https://github.com/trishullab/PutnamBench ;;
    *) echo "unknown source: $s (choose from: $ALL)" >&2; exit 2 ;;
  esac
done

echo "== Converting to Parquet"
if [ -d .venv ]; then source .venv/bin/activate; fi
python -c "from markov_lean.data import build_all; build_all('data/raw', 'data/processed')"
