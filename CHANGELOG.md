# Changelog

## Unreleased

- Add query-only search on a memory-mapped `.hnsw` snapshot via
  `LoadedHnsw::search` and `HnswIndex::load`, so a saved index can be queried
  without re-inserting every vector.

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
- `LoadedHnsw::search` and `HnswIndex::load` query a saved snapshot without
  reconstructing a mutable index.
