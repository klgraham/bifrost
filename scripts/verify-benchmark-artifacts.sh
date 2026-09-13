#!/usr/bin/env bash
# Verify the shared fixture contract, artifact tool, and benchmark consumer offline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy

for manifest in \
  benchmarks/fixture/Cargo.toml \
  benchmarks/artifacts/Cargo.toml \
  benchmarks/competitors/Cargo.toml
do
  bash scripts/cargo.sh fmt --check --manifest-path "$manifest"
  bash scripts/cargo.sh clippy --offline --locked --manifest-path "$manifest" \
    --all-targets -- -D warnings
  bash scripts/cargo.sh test --offline --locked --manifest-path "$manifest"
done
