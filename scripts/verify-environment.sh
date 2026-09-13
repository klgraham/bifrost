#!/usr/bin/env bash
# Offline validation plus bounded, synthetic benchmark execution.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/cargo.sh fmt --check
bash scripts/cargo.sh clippy --offline --locked --all-targets --all-features -- -D warnings
bash scripts/cargo.sh test --offline --locked --all-features
bash scripts/cargo.sh clippy --offline --locked --manifest-path benchmarks/competitors/Cargo.toml --all-targets --all-features -- -D warnings
bash scripts/cargo.sh test --offline --locked --manifest-path benchmarks/competitors/Cargo.toml --all-features
BIFROST_BENCH_ITERATIONS=10000 bash scripts/cargo.sh bench --offline --locked --bench distance
# Ignore fixture/size overrides so this remains a bounded synthetic smoke run.
unset HNSW_BENCH_FIXTURE HNSW_BENCH_EF_SEARCHES
export HNSW_BENCH_VECTORS=500 HNSW_BENCH_DIMENSIONS=64 HNSW_BENCH_QUERIES=10
export HNSW_BENCH_REPETITIONS=1 HNSW_BENCH_K=10 HNSW_BENCH_M=16
export HNSW_BENCH_EF_CONSTRUCTION=100 HNSW_BENCH_EF_SEARCH=50 HNSW_BENCH_SEED=42
bash scripts/cargo.sh run --offline --locked --release --manifest-path benchmarks/competitors/Cargo.toml
