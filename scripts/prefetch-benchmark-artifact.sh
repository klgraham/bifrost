#!/usr/bin/env bash
# Resolve one catalog artifact into Bifrost's complete, verified fixture cache.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -ne 1 || -z "$1" ]]; then
  echo 'usage: scripts/prefetch-benchmark-artifact.sh ARTIFACT_ID' >&2
  exit 2
fi

cargo_args=(run --locked --release)
fetch_args=(fetch --artifact "$1")

if [[ "${BIFROST_BENCH_OFFLINE:-0}" == 1 ]]; then
  cargo_args+=(--offline)
  fetch_args+=(--offline)
elif [[ "${BIFROST_BENCH_OFFLINE:-0}" != 0 ]]; then
  echo 'BIFROST_BENCH_OFFLINE must be 0 or 1' >&2
  exit 2
fi

if [[ "${BIFROST_BENCH_FETCH_REPAIR:-0}" == 1 ]]; then
  fetch_args+=(--repair)
elif [[ "${BIFROST_BENCH_FETCH_REPAIR:-0}" != 0 ]]; then
  echo 'BIFROST_BENCH_FETCH_REPAIR must be 0 or 1' >&2
  exit 2
fi

[[ -z "${BIFROST_BENCH_FIXTURE_ROOT:-}" ]] || \
  fetch_args+=(--destination-root "$BIFROST_BENCH_FIXTURE_ROOT")
[[ -z "${BIFROST_HF_CACHE_DIR:-}" ]] || \
  fetch_args+=(--cache-dir "$BIFROST_HF_CACHE_DIR")
[[ -z "${BIFROST_HF_ENDPOINT:-}" ]] || \
  fetch_args+=(--endpoint "$BIFROST_HF_ENDPOINT")
[[ -z "${BIFROST_HF_TOKEN_ENV:-}" ]] || \
  fetch_args+=(--token-env "$BIFROST_HF_TOKEN_ENV")

exec bash scripts/artifact-cargo.sh "${cargo_args[@]}" -- "${fetch_args[@]}"
