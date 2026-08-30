# Changelog

## Unreleased

- Wrap AVX2 and NEON init/arithmetic intrinsics in `unsafe { }` so
  `#![forbid(unsafe_op_in_unsafe_fn)]` compiles on rustc 1.85. Those calls
  stay unsafe operations on the crate MSRV even inside `unsafe fn`.
- The crates.io package is `bifrost-index`. The Rust crate path stays `bifrost`,
  so callers still `use bifrost::{Config, HnswIndex}`.
- Rename the Cargo package from `hnsw-rs` to `bifrost` and the Rust crate path
  from `hnsw_rs` to `bifrost`. The verification skill is now `verify-bifrost`.
  The default checkout path is `/Users/klogram/dev/klogram_labs/bifrost`.
  `HnswIndex`, `LoadedHnsw`, `.hnsw` snapshots, and the `HNSW` magic are
  unchanged.
- The node-capacity check accepts the last legal `NodeIndex`
  (`u32::MAX - 1`), so a graph may hold `u32::MAX` nodes. Index `u32::MAX`
  is unused: a count of `2^32` cannot be stored in `Graph::node_count` or
  the on-disk header, and search sizes the visited list from that count.
  Unit tests cover that fencepost without allocating.
- `Graph::node_count` no longer `expect`s if the `u32` insertion cap is
  bypassed. `insert_node` returns `Error::CapacityExceeded` at that length,
  and the public count saturates at `u32::MAX`.
- CI clippy-checks and unit-tests the competitor crate (no FiQA download) and
  runs library tests on Windows as well as Ubuntu and macOS. The competitor
  `Config` init uses `..Config::default()` so new fields such as
  `check_vectors` do not break the bench. Integration-test temp paths use a
  numeric nonce so Windows does not treat `SystemTime`'s Debug form as an
  invalid path.
- Drop the `documentation` crate metadata URL. It pointed at docs.rs/hnsw-rs,
  which is not this unpublished crate (`publish = false`); crates.io `hnsw_rs`
  is a different project.
- Search hits copy the already-computed candidate distance instead of launching
  another cosine kernel. Live and mmap search share `hits_from_candidates`;
  equal distances still sort by external id.
- Search marks visited nodes with a generation-stamped `u32` list reused across
  layers of one query (and across inserts on a live `HnswIndex`) and keeps the
  candidate frontier in a binary heap. Ranking and node-index ties match the
  previous full-bitmap / full-sort search. Incrementing the generation is an
  O(1) clear; wraparound fills the buffer and resumes at 1. One stamp array
  sized to `node_count` is still allocated per `search_knn` (or grown on the
  index during insert) so membership stays O(1) without hashing; neighbors
  `>=` that length are skipped.
- Document that `HnswIndex::build` assigns IDs `0..n-1` and is a convenience
  for an empty index. A second `build`, or `build` after an insert that
  already used one of those IDs, returns `Error::DuplicateExternalId` and
  does not replace the graph. Use `insert` to append with caller-chosen IDs.
- Public `dot`, `cosine_distance`, and `cosine_similarity` return
  `Error::DimensionMismatch` instead of asserting equal lengths. Inner
  kernels `debug_assert` after insert/search `check_dimension`.
- `debug_assert` that insert and search vectors are finite and near-unit
  (`||v||` within `0.01` of `1`). `Config::check_vectors` (default `false`)
  makes `HnswIndex` insert/search and `LoadedHnsw` search return
  `Error::InvalidVector` for the same failures. The flag is not stored in
  snapshots; `LoadedHnsw::set_check_vectors` enables query checks after load.
- Default `level_mult` to `1.0 - 1.0 / m` (`Config::level_mult_for_m`) so
  `Config::default()` matches the HNSW paper, hnswlib, and the competitor
  benchmark crate (`P(level >= L) = m^{-L}`). Existing snapshots that stored
  `0.5` still load.
- Write `.hnsw` snapshots through the same temp + `sync_all` + `rename` path
  as v2→v3 migration, so a crash mid-save cannot truncate a previous good
  file. Leftover `.{name}.tmp-{pid}-{nonce}` files do not block a later save.
- Open mapped snapshots read-only and document that callers must not mutate
  the file while `LoadedHnsw` lives. Keep a shared read mapping rather than
  `map_copy`: `MAP_PRIVATE` still has unspecified visibility of later writes,
  and this crate's own replace is a rename onto a new inode.
- Make `insert` transactional after the node is appended: a failed link or
  reverse-link prune restores adjacency, drops the isolated node, and leaves
  the external ID free to retry. `add_bidirectional_edge` is atomic.
- Select new-node neighbors and prune reverse edges with the paper / hnswlib
  diversity heuristic (Alg. 4), not simple nearest-`M`.
- Prune reverse edges after insertion so a node's outgoing degree stays at
  most `2M` on layer 0 (`Mmax0`) and `M` on upper layers (`Mmax`). Dropped
  reverse edges stay directed: the peer keeps its link.
- Reject `m == 0` in `Config::validate` and `.hnsw` headers.
- Reject `ef_construction == 0` and `ef_search == 0` in `Config::validate`,
  `.hnsw` headers, and `HnswIndex::set_ef_search` instead of clamping to 1.
- Keep `HnswIndex` construction parameters private; expose `config()`,
  `set_ef_search`, and `set_level_mult` so prune caps cannot change mid-index.
- Keep the construction graph private; expose `graph()`, `edges`, `node`,
  `layer_count`, `has_edge`, and `degree` so callers can inspect adjacency
  without replacing it or desyncing vectors.
- Add query-only search on a memory-mapped `.hnsw` snapshot via
  `LoadedHnsw::search`, `LoadedHnsw::search_with_ef`, `LoadedHnsw::open`, and
  `load_file`, so a saved index can be queried without re-inserting every
  vector.
- Search uses `max(ef_search, k)` so requesting more neighbors than the
  configured candidate width still returns up to `k` hits.
- Add `HnswIndex::search_with_ef` so live search can override candidate width
  the same way as `LoadedHnsw::search_with_ef`.
- Reject snapshots whose entry node does not exist at `entry_level`.
- Reject snapshots with duplicate `external_id`s, overlapping or gapped
  `vector_offset`s, or unsorted/duplicate edge lists. The writer emits
  unique IDs, packed vector rows, and sorted unique adjacency.

## 0.2.0 - 2026-07-18

- Add an isolated competitor benchmark crate for comparing hnsw-rs with
  `hnsw_rs` and USearch without adding those dependencies to the published
  library.
- Add reproducible 1,536-dimensional synthetic and BEIR FiQA-2018 evaluations
  covering build cost, query latency, exact ANN recall, nDCG, and qrels recall.
- Persist validated local `.hnsw` benchmark indexes after timed runs while
  keeping generated embeddings, indexes, and results out of version control.

## 0.1.0 - 2026-07-14

- Introduce HNSW graph construction and cosine-distance search for normalized
  vectors.
- Use the maintained `rand` crate for seeded and operating-system-backed random
  level generation.
- Add AVX2, NEON, and scalar distance kernels on stable Rust.
- Add `.hnsw` v3 save/load, CRC validation, and atomic v2 migration.
- Add a deterministic v3 persistence fixture and byte-for-byte writer test.

## API behavior

- Dimension mismatches and invalid configuration return typed Rust errors
  rather than relying on debug assertions. Public `vector::dot`,
  `cosine_distance`, and `cosine_similarity` return
  `Error::DimensionMismatch` when the arguments have different lengths.
- Insert and search `debug_assert` finite, near-unit vectors. With
  `Config::check_vectors` (or `set_check_vectors`), those failures are
  `Error::InvalidVector`. The flag is not persisted in `.hnsw` snapshots.
- Search results are owned `Vec<SearchHit>` values.
- Mmap accessors decode checked little-endian views rather than exposing typed
  slices cast directly from file bytes. The mapping is read-only; do not
  mutate the file while `LoadedHnsw` lives.
- `save_file` and v2→v3 migration replace the destination with temp +
  `sync_all` + `rename`.
- `LoadedHnsw::search` / `LoadedHnsw::open` query a saved snapshot without
  reconstructing a mutable index.
- Search candidate width is `max(ef_search, k)` on `search` and
  `search_with_ef` for both `HnswIndex` and `LoadedHnsw`.
- Neighbor selection uses the diversity heuristic for both insert and prune.
- Reverse-link pruning caps outgoing degree at `2M` (layer 0) and `M` (upper
  layers) and leaves the opposite directed edge in place.
- `Config::default` sets `level_mult` to `1.0 - 1.0 / m`. Snapshots may still
  store any finite value in `[0, 1]`.
- `Config::level_mult_for_m` returns the paper / hnswlib stop probability.
- `Config::max_degree` and `Config::new_node_neighbors` expose those derived
  limits. `Config::m`, `Config::ef_construction`, and `Config::ef_search` must
  be greater than zero.
- `HnswIndex::config` returns a copy; `set_ef_search` updates query width and
  rejects `0`. `set_check_vectors` toggles `Error::InvalidVector` on insert
  and search.
- `HnswIndex::graph` is a shared reference; `edges`, `node`, `layer_count`,
  `has_edge`, and `degree` inspect adjacency without allowing replacement.
- `Graph::node_count` does not panic if the node list exceeds `u32::MAX`;
  the value saturates. `insert_node` rejects that length with
  `Error::CapacityExceeded`.
- `HnswIndex::insert` is transactional: failure after the node is appended
  rolls back storage, adjacency (including prune), and the external-ID map.
- `HnswIndex::build` assigns dense IDs starting at zero. It is a convenience
  for an empty index; a second `build` or any colliding ID returns
  `Error::DuplicateExternalId`. Use `insert` to append.
