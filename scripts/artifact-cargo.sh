#!/usr/bin/env bash
# Run the isolated benchmark-artifact crate with its Rust 1.89 toolchain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/artifacts"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
exec cargo "$@"
