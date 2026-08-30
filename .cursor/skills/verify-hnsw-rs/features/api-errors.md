# API errors

API errors let a caller distinguish bad configuration, wrong vector width, and
reused external IDs via typed `hnsw_rs::Error` values instead of panics.

## Sub-features

- `error-dim-insert` returns `Error::DimensionMismatch { expected, actual }` from `insert`.
- `error-dim-search` returns the same error from `search` without mutating the index.
- `error-duplicate-id` returns `Error::DuplicateExternalId(id)` on a second insert of the same `u32`.
- `error-config` rejects `dim == 0`, non-finite/`level_mult` outside `0.0..=1.0`, and `max_level` of `0` or `255`.

## How to get to it (user POV)

- Call `HnswIndex::new` with a `Config` that fails `validate` (`dim: 0`,
  `level_mult: NAN`, `max_level: u8::MAX`).
- Call `insert` or `search` with a slice whose length is not `Config::dim`.
- Insert the same external id twice (README: "Duplicate external IDs are rejected").

## Driving it with verify-hnsw-rs

Preconditions:

- `scripts/verify-hnsw-rs.sh doctor` exited 0.
- `scripts/verify-hnsw-rs.sh launch --run-id $RUN_ID` compiled tests.
- No extra env vars. Do not pass `--features fiqa-prep`.

- **Dimension mismatch.** Drive the error tests. Run
  `scripts/verify-hnsw-rs.sh drive api-errors --run-id $RUN_ID`.
  The helper runs
  `cargo test --all-features --lib -- --nocapture dimension_mismatches_return_errors sparse_and_duplicate_external_ids invalid_configs_are_rejected`.
  Exit code `0`. Transcript contains
  `test index::tests::dimension_mismatches_return_errors ... ok`.
- **Duplicate id.** Same drive. Transcript contains
  `test index::tests::sparse_and_duplicate_external_ids ... ok`
  (ids `100` and `5_000` insert; second `insert(100, …)` is
  `DuplicateExternalId(100)`; search of `[0.0, 1.0]` still returns `5_000`).
- **Invalid config.** Same drive. Transcript contains
  `test config::tests::invalid_configs_are_rejected ... ok`.
- **Proof.** Artifacts
  `artifacts/$RUN_ID/api-errors/cargo-test.txt` and
  `artifacts/$RUN_ID/api-errors/meta.json` with `exit_code` `0`.
  The transcript must show typed-error tests `ok`, not a panic backtrace.

## Gotchas

- `vector::dot` and `cosine_similarity` still `assert_eq!` on length. Those
  asserts are not this feature; only `HnswIndex::insert`/`search` and
  `Config::validate` are in scope.
- Sparse IDs are allowed; only **duplicate** IDs error. A gap between `100` and
  `5_000` is success.
- `InvalidConfig` messages are `&'static str`. Assert on the `Error` variant,
  not on panic text.
- Capacity errors (`CapacityExceeded`) are not practical to drive in this
  harness; do not claim them verified.
