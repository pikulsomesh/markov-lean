#!/usr/bin/env bash
# Download the benchmark files the gold MSC set is built from, pinned to the commits checked
# on 2026-11-25, into data/raw/gold/. Everything comes from GitHub (no Zenodo).
# Safe to re-run: anything already present is skipped.
#
#   ./scripts/fetch_gold_sources.sh
#   markov-lean gold-set build          # rewrites data/gold/msc_gold.csv from these files
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw/gold
mkdir -p "$RAW"

DS15=2c4ba9119eef74d0d611f494261b2c5bae98c69a   # deepseek-ai/DeepSeek-Prover-V1.5
PROOFNET=509ad79710ed4f46ff5c282ed5640c1aa9ac3f30 # zhangir-azerbayev/ProofNet
PUTNAM=a635813f5edf297d928f9bd5ef8324a26bdde09c   # trishullab/PutnamBench
DS2=e598a57ea3284997d4a2a168a069fdd5064afbc8      # deepseek-ai/DeepSeek-Prover-V2

get() {  # get <url> <dest>
  if [ -s "$2" ]; then echo "$(basename "$2"): already present"; return; fi
  curl -fL --retry 4 -o "$2" "$1"
  echo "$(basename "$2"): ok"
}

clone_at() {  # clone_at <name> <url> <sha>
  local d="$RAW/$1"
  if [ -d "$d/.git" ]; then echo "$1: already present"; return; fi
  git init -q "$d"
  git -C "$d" fetch -q --depth 1 "$2" "$3"
  git -C "$d" checkout -q FETCH_HEAD
  echo "$1: ok at $(git -C "$d" rev-parse --short HEAD)"
}

get "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/$DS15/datasets/minif2f.jsonl" "$RAW/minif2f.jsonl"
get "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/$DS15/datasets/proofnet.jsonl" "$RAW/proofnet.jsonl"
mkdir -p "$RAW/ProofNet/benchmark"
for s in valid test; do
  get "https://raw.githubusercontent.com/zhangir-azerbayev/ProofNet/$PROOFNET/benchmark/$s.jsonl" \
      "$RAW/ProofNet/benchmark/$s.jsonl"
done
clone_at PutnamBench https://github.com/trishullab/PutnamBench "$PUTNAM"
# whole miniF2F proofs; read by `gold-set steps --proofs`, never copied into the gold table
get "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V2/$DS2/minif2f-solutions.zip" \
    "$RAW/minif2f-solutions.zip"
