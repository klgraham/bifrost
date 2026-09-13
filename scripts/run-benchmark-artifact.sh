#!/usr/bin/env bash
# Fetch (or resolve offline) one immutable fixture, then benchmark it separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 1 || $# -gt 2 || -z "$1" ]]; then
  echo 'usage: scripts/run-benchmark-artifact.sh ARTIFACT_ID [OUTPUT_DIR]' >&2
  exit 2
fi

artifact_id="$1"
output_directory="${2:-target/benchmark-runs/$artifact_id}"
unset OPENAI_API_KEY
unset HNSW_BENCH_VECTORS HNSW_BENCH_QUERIES HNSW_BENCH_DIMENSIONS
unset HNSW_BENCH_ARTIFACT_REVISION
fixture_path="$(bash scripts/prefetch-benchmark-artifact.sh "$artifact_id")"

export HNSW_BENCH_FIXTURE="$fixture_path"
export HNSW_BENCH_OUTPUT_DIR="$output_directory"
exec bash scripts/cargo.sh run --offline --locked --release \
  --manifest-path benchmarks/competitors/Cargo.toml
