#!/usr/bin/env bash
# One-time setup ON an already-running EC2 g6.xlarge (NVIDIA L4 24 GB, Ubuntu Deep Learning
# AMI) for the search experiments. Run it from the repo root, on the instance:
#
#   git clone https://github.com/pikulsomesh/markov-lean && cd markov-lean
#   bash scripts/setup_ec2.sh
#
# It installs vLLM into ~/vllm-env, this package (with the search extra) into .venv, Lean
# and Mathlib via scripts/setup_lean.sh, and builds leanprover-community/repl at the tag
# that matches lean/lean-toolchain. It never calls the AWS API and never creates,
# starts or stops instances.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
VLLM_ENV=${VLLM_ENV:-$HOME/vllm-env}
VLLM_VERSION=${VLLM_VERSION:-}          # empty = newest release on PyPI
REPL_DIR=${REPL_DIR:-$HOME/repl}

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "No working NVIDIA GPU found (nvidia-smi). Run this on the g6.xlarge instance." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# git, curl and a venv-capable Python (present on the DLAMI; this is a no-op there).
if ! python3 -m venv --help >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y git curl python3-venv
fi

echo "== vLLM into $VLLM_ENV"
python3 -m venv "$VLLM_ENV"
"$VLLM_ENV/bin/pip" install -q --upgrade pip
"$VLLM_ENV/bin/pip" install -q "vllm${VLLM_VERSION:+==$VLLM_VERSION}"
"$VLLM_ENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"

echo "== markov-lean into $REPO/.venv"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev,search]"

echo "== Lean + Mathlib"
bash scripts/setup_lean.sh
# shellcheck disable=SC1091
source "$HOME/.elan/env"

echo "== leanprover-community/repl into $REPL_DIR"
TOOLCHAIN=$(cat lean/lean-toolchain)            # e.g. leanprover/lean4:v4.24.0
LEAN_TAG=${TOOLCHAIN##*:}
if [ ! -d "$REPL_DIR/.git" ]; then
  git clone https://github.com/leanprover-community/repl.git "$REPL_DIR"
fi
git -C "$REPL_DIR" fetch --tags -q
if ! git -C "$REPL_DIR" rev-parse -q --verify "refs/tags/$LEAN_TAG" >/dev/null; then
  echo "repl has no tag $LEAN_TAG to match lean/lean-toolchain ($TOOLCHAIN)." >&2
  echo "Pick a Mathlib tag that repl also has, e.g. MATHLIB_TAG=<tag> bash scripts/setup_lean.sh" >&2
  exit 1
fi
git -C "$REPL_DIR" checkout -q "$LEAN_TAG"
(cd "$REPL_DIR" && lake build)

echo "== smoke test: REPL inside the Mathlib project"
printf '{"cmd": "example : 2 + 2 = 4 := by norm_num"}\n\n' \
  | (cd lean && timeout 900 lake env "$REPL_DIR/.lake/build/bin/repl") | head -20

cat <<MSG

Setup done.
  Start the prover:  bash scripts/serve_prover.sh
  Check everything:  .venv/bin/python -m markov_lean.search --check --config configs/search.yaml
MSG
