# Changelog

## Unreleased

- Prune reverse edges after insertion so a node's degree stays at most `2M`
  on layer 0 (`Mmax0`) and `M` on upper layers (`Mmax`).
- Add query-only search on a memory-mapped `.hnsw` snapshot via
  `LoadedHnsw::search`, `LoadedHnsw::search_with_ef`, `LoadedHnsw::open`, and
  `load_file`, so a saved index can be queried without re-inserting every
  vector.
- Search uses `max(ef_search, k)` so requesting more neighbors than the
  configured candidate width still returns up to `k` hits.
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
  slices cast directly from file bytes.
- `LoadedHnsw::search` / `LoadedHnsw::open` query a saved snapshot without
  reconstructing a mutable index.
- Search candidate width is `max(ef_search, k)`.
- Reverse-link pruning caps outgoing degree at `2M` (layer 0) and `M` (upper
  layers). `Config::max_degree` and `Config::new_node_neighbors` expose those
  derived limits.
