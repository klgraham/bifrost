# Benchmark artifacts

Benchmark artifacts let maintainers define versioned embedding fixtures once,
verify their identity and contents, cache them read-only, and feed identical
vectors, IDs, and qrels to every competitor without embedding work in the
benchmark process.

## Sub-features

- `artifact-catalog` validates unique dataset/model/dimension entries and
  distinguishes planned artifacts from immutable published revisions.
- `fixture-v2` validates manifest metadata, file allowlists, counts, shapes,
  normalized little-endian f32 vectors, IDs, qrels, SHA-256 values, and the
  optional resolution sidecar.
- `artifact-prepare-resume` reuses complete fixtures and verified embedding
  batches; tests inject an in-memory source and embedder.
- `artifact-fetch-cache` installs a complete read-only fixture atomically,
  rejects corrupt or partial state, and proves an explicit offline second run;
  tests inject a filesystem-backed fake downloader.
- `artifact-publish-contract` proves the upload allowlist, immutable identity,
  metadata, corruption checks, and idempotence through a fake publisher only.
- `artifact-benchmark-consumer` loads v2 and legacy v1 fixtures, gives all three
  HNSW implementations identical evaluation inputs, reports resolved artifact
  identity, and writes indexes only to a non-overlapping output directory.

## How to get to it (user POV)

- Read `benchmarks/artifacts/catalog.json` through
  `bifrost_benchmark_fixture::Catalog::load` and select an artifact ID.
- Use the Rust artifact tool's `fetch --offline` route for an already cached
  published fixture or `verify --fixture PATH --artifact ID` for local
  validation. Preparation and publication are separate, explicit maintainer
  workflows.
- Set `HNSW_BENCH_FIXTURE` to the verified fixture and
  `HNSW_BENCH_OUTPUT_DIR` to a separate writable run directory, then run the
  competitor binary. A resolved v2 report includes repository, 40-hex revision,
  and manifest SHA-256.

## Driving it with verify-bifrost

Preconditions:

- `scripts/verify-bifrost.sh doctor` exited 0 and found all three benchmark
  manifests, the catalog, and `scripts/verify-benchmark-artifacts.sh`.
- `scripts/verify-bifrost.sh launch --run-id $RUN_ID` created the isolated
  scratch directory. Dependencies must already be cached because every
  benchmark Cargo test runs with `--offline --locked`.
- No credential cleanup is required in the caller's shell. The harness removes
  OpenAI/Hugging Face credentials and proxy variables from the child.

- **Offline contract.** Run
  `scripts/verify-bifrost.sh drive benchmark-artifacts --run-id $RUN_ID`.
  The helper invokes `bash scripts/verify-benchmark-artifacts.sh`, which runs
  format, Clippy with warnings denied, and tests for the fixture, artifact-tool,
  and competitor manifests. Exit code `0`; the transcript starts the step with
  `ok safety external_credentials=unset proxy_environment=unset test_entrypoints_only=true`.
- **Catalog and fixture proof.** The transcript contains
  `catalog_loads_two_dimensions_and_distinguishes_publication_states ... ok`,
  `checked_in_catalog_contains_both_independent_dimensions ... ok`, and
  `verified_fixture_loads_vectors_ids_qrels_and_resolution ... ok`. It also
  contains `fixture_rejects_semantically_invalid_qrels_after_hashing ... ok`,
  proving qrels contents are validated after their file hash is accepted.
- **No-paid-work/cache proof.** The transcript contains
  `completed_fixture_makes_no_source_or_embedding_calls ... ok`,
  `empty_cache_fails_offline_without_taking_network_path ... ok`, and
  `online_then_offline_rebuild_uses_only_cached_files ... ok`.
- **Publication-contract proof.** The transcript contains
  `publication_uses_exact_artifact_and_metadata_allowlists ... ok` and
  `repeated_publication_is_idempotent ... ok`. These use `FakePublisher`; no
  external repository operation occurs.
- **Consumer proof.** The transcript contains
  `v2_local_and_resolved_fixtures_have_identical_inputs_and_results ... ok`,
  `read_only_fixture_benchmark_writes_only_to_separate_output ... ok`, and
  `rejects_output_paths_that_overlap_fixture ... ok`.
- **Evidence.** Inspect
  `artifacts/$RUN_ID/benchmark-artifacts/cargo-test.txt` and
  `artifacts/$RUN_ID/benchmark-artifacts/meta.json`. Require step exit `0`,
  `feature_id` `benchmark-artifacts`, and `safety_policy`
  `benchmark-artifacts`.

## Gotchas

- Cargo `--offline` prevents dependency downloads; the stronger runtime safety
  comes from test-only entry points, injected local doubles, removed credentials
  and proxies. Production `fetch --offline` passes `local_files_only(true)` on
  each file request, and its cache-only behavior is covered through the fake
  source. Do not replace this recipe with a direct artifact CLI invocation.
- The tests simulate download and publication semantics. They do not prove that
  a planned catalog entry has been published or that a remote Hugging Face
  artifact exists.
- `prepare-fiqa` and `publish` are intentional manual workflows. Never pass
  real tokens, an OpenAI endpoint, or the external-publication flag during this
  verification feature.
- USearch builds a native backend, so the first offline drive can compile for
  longer than the root library features. Compilation time is not benchmark
  timing.
- Fixture paths are read-only inputs. `HNSW_BENCH_OUTPUT_DIR` must not equal,
  contain, or be contained by the fixture path.
