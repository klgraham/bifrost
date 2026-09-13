#!/usr/bin/env bash
# Shared by local worktrees, cloud setup, and cached-container maintenance.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

case "$(uname -s)" in
  Linux)
    if ! command -v gcc >/dev/null || ! command -v g++ >/dev/null || ! command -v python3 >/dev/null || ! command -v curl >/dev/null; then
      elevate=()
      if [[ "$(id -u)" != 0 ]]; then elevate=(sudo); fi
      "${elevate[@]}" apt-get update
      "${elevate[@]}" apt-get install -y build-essential curl ca-certificates python3
    fi
    ;;
  Darwin)
    xcrun --find clang++ >/dev/null || { echo 'Install Xcode Command Line Tools: xcode-select --install' >&2; exit 1; }
    command -v python3 >/dev/null || { echo 'Install Python 3 for the verification helper.' >&2; exit 1; }
    ;;
  *) echo 'Supported setup platforms: macOS and Debian/Ubuntu Linux.' >&2; exit 1 ;;
esac

if ! command -v rustup >/dev/null; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o "$installer"
  sh "$installer" -y --profile minimal --default-toolchain none --no-modify-path
fi
# Read the shared pin without depending on a TOML library.
toolchain="$(sed -n 's/^channel = "\([^"]*\)"$/\1/p' rust-toolchain.toml)"
[[ -n "$toolchain" ]] || { echo 'Missing Rust toolchain pin' >&2; exit 1; }
rustup toolchain install "$toolchain" --profile minimal --component rustfmt --component clippy

# Fetch includes optional dependencies, allowing fiqa-prep to compile offline.
bash scripts/cargo.sh fetch --locked
bash scripts/cargo.sh fetch --locked --manifest-path benchmarks/competitors/Cargo.toml
bash scripts/cargo.sh test --locked --all-features --no-run
bash scripts/cargo.sh build --locked --release --benches
bash scripts/cargo.sh build --locked --release --manifest-path benchmarks/competitors/Cargo.toml
bash scripts/cargo.sh build --locked --release --all-features --manifest-path benchmarks/competitors/Cargo.toml
echo 'Bifrost setup complete. Run: bash scripts/verify-environment.sh'
