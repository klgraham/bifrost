#!/usr/bin/env bash
# Keep the native benchmark compiler choice consistent in every shell.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
if [[ "$(uname -s)" == Linux ]]; then
  export CC=gcc CXX=g++
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C linker=gcc"
fi
exec cargo "$@"
