#!/usr/bin/env bash
# Installs elan (the Lean version manager) if missing, pins the Lean project in lean/
# to the newest stable Mathlib release tag, and downloads Mathlib's prebuilt cache.
# Needed only for the search phase. Works on macOS and Linux (EC2).
set -euo pipefail
cd "$(dirname "$0")/../lean"

if ! command -v elan >/dev/null 2>&1; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none
  # shellcheck disable=SC1091
  source "$HOME/.elan/env"
fi

if [ -z "${MATHLIB_TAG:-}" ]; then
  # Newest tag of the form v4.N.0 (stable releases only, no -rc).
  MATHLIB_TAG=$(git ls-remote --tags --refs https://github.com/leanprover-community/mathlib4.git \
    | awk -F/ '{print $3}' | grep -E '^v4\.[0-9]+\.0$' | sort -V | tail -1)
fi
echo "Pinning Mathlib to $MATHLIB_TAG"
sed -i.bak -E "s/^rev = .*/rev = \"$MATHLIB_TAG\"/" lakefile.toml && rm -f lakefile.toml.bak
curl -sSfL "https://raw.githubusercontent.com/leanprover-community/mathlib4/$MATHLIB_TAG/lean-toolchain" -o lean-toolchain

lake update
lake exe cache get
lake build
echo "Lean project ready. Commit lean/lakefile.toml, lean/lean-toolchain and lean/lake-manifest.json."
