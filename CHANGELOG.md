# Changelog

## Unreleased

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
  rather than relying on debug assertions.
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
- `Config::max_degree` and `Config::new_node_neighbors` expose those derived
  limits. `Config::m`, `Config::ef_construction`, and `Config::ef_search` must
  be greater than zero.
- `HnswIndex::config` returns a copy; `set_ef_search` updates query width and
  rejects `0`.
- `HnswIndex::graph` is a shared reference; `edges`, `node`, `layer_count`,
  `has_edge`, and `degree` inspect adjacency without allowing replacement.
- `HnswIndex::insert` is transactional: failure after the node is appended
  rolls back storage, adjacency (including prune), and the external-ID map.
