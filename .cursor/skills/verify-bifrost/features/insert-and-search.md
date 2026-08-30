# Insert and search

Insert and search let a caller add unit-normalized vectors under sparse `u32`
external IDs and retrieve at most `k` nearest neighbors by cosine distance
(`1 - dot`) without rebuilding the index.

## Sub-features

- `insert-sparse-id` inserts a vector under a caller-chosen `u32` id.
- `search-k` returns at most `k` `SearchHit { id, distance }` values sorted by
  distance then id.
- `search-empty` returns an empty `Vec` for an empty index or `k == 0`.
- `search-ranking` ranks an identical unit vector first and the opposite vector last.

## How to get to it (user POV)

- Construct `HnswIndex::new(Config { dim, rng_seed: Some(seed), ..Config::default() })?`.
- Call `index.insert(id, &vector)?` then `index.search(&query, k)?`.
- Follow the README usage example (`dim: 4`, ids `100` and `5_000`, query
  `[0.98, 0.02, 0.0, 0.0]` expecting nearest id `100`).

## Driving it with verify-bifrost

Preconditions:

- `scripts/verify-bifrost.sh doctor` exited 0 for this checkout.
- `scripts/verify-bifrost.sh launch --run-id $RUN_ID` compiled `--all-features` tests.
- Scratch `TMPDIR` is the launch scratch dir for `$RUN_ID`.

- **Insert and nearest neighbor.** Drive the lib tests that insert three
  4-d basis vectors and search `k=2`. Run
  `scripts/verify-bifrost.sh drive insert-and-search --run-id $RUN_ID`.
  The helper runs
  `cargo test --all-features --lib -- --nocapture insert_and_search cosine_distance_ranking_is_correct empty_and_single_vector_indexes`.
  Exit code `0`. Transcript contains
  `test index::tests::insert_and_search ... ok`.
- **Ranking.** Same drive. Transcript contains
  `test index::tests::cosine_distance_ranking_is_correct ... ok`
  (nearest id `0` for query `[1,0,0,0]`, farthest id `3` for the opposite vector).
- **Empty and singleton.** Same drive. Transcript contains
  `test index::tests::empty_and_single_vector_indexes ... ok`
  (empty search returns `[]`; id `7` is recovered at distance near 0).
- **Proof.** Artifacts
  `artifacts/$RUN_ID/insert-and-search/cargo-test.txt` and
  `artifacts/$RUN_ID/insert-and-search/meta.json`.
  `meta.json` `exit_code` is `0` and `feature_id` is `insert-and-search`.
  The transcript lists all three tests as `ok` and does not mention
  `benchmarks/competitors` or FiQA.

## Gotchas

- Vectors must already be unit-normalized. The index will store whatever slice
  it is given; unnormalized inputs silently skew cosine distance.
- `insert`/`search` return `Error::DimensionMismatch` instead of panicking when
  the slice length differs from `Config::dim` — that is a different feature
  (`api-errors`).
- `k == 0` is a successful empty result, not an error.
- Duplicate IDs are rejected; this feature's tests use distinct ids `0..=3` and `7`.
- Do not assert on internal node indexes. Proof is `SearchHit.id` (external id)
  and relative distances.
