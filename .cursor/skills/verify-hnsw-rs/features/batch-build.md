# Batch build

Batch build lets a caller insert a slice of unit-normalized vectors in order
and receive dense external IDs starting at zero, then search the resulting index.

## Sub-features

- `build-dense-ids` assigns external id `i` to `vectors[i]` for `i` in `0..n`.
- `build-search` returns `k` hits from the built index.
- `build-count` leaves `index.len()` equal to the number of input slices.

## How to get to it (user POV)

- Construct `HnswIndex::new(Config { dim, rng_seed: Some(seed), ..Config::default() })?`.
- Call `index.build(&vectors)?` where `vectors` is `&[&[f32]]` (README-adjacent
  public method; ids are `0..vectors.len()` as `u32`).
- Call `index.search(&query, k)?` on the same index.

## Driving it with verify-hnsw-rs

Preconditions:

- `scripts/verify-hnsw-rs.sh doctor` exited 0.
- `scripts/verify-hnsw-rs.sh launch --run-id $RUN_ID` compiled tests.
- Input vectors in the mapped test are length-3 slices (so `Config.dim` is `3`).

- **Build then search.** Drive the batch test. Run
  `scripts/verify-hnsw-rs.sh drive batch-build --run-id $RUN_ID`.
  The helper runs
  `cargo test --all-features --lib -- --nocapture build_batch_and_search`.
  Exit code `0`. Transcript contains
  `test index::tests::build_batch_and_search ... ok`
  (four vectors; `search([0.9, 0.9, 0.0], 2)` returns length `2`).
- **Proof.** Artifacts
  `artifacts/$RUN_ID/batch-build/cargo-test.txt` and
  `artifacts/$RUN_ID/batch-build/meta.json` with `exit_code` `0` and
  `feature_id` `batch-build`.

## Gotchas

- `build` is sequential `insert`. A dimension error on a later row leaves
  earlier rows inserted; the mapped test uses uniformly 3-d slices and does
  not cover partial failure.
- Dense IDs start at zero. Do not combine `build` with earlier `insert(0, …)`
  in the same index — that is a duplicate-id error (`api-errors`).
- The fourth example vector `[1.0, 1.0, 0.0]` is **not** unit-normalized.
  The test only asserts result length, not ranking; do not tighten this proof
  to cosine order without normalizing first.
- `build` is not persistence. Saving the built index is `persist-v3`.
