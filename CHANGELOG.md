# Changelog

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
