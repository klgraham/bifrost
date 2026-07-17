# hnsw-rs

A compact Rust implementation of Hierarchical Navigable Small World (HNSW)
search for normalized vector embeddings.

It supports incremental insertion, cosine-distance search, stable Rust SIMD
acceleration, and `.hnsw` v3 persistence.

## Features

- Dynamic insertion without rebuilding the index
- Sparse caller-facing `u32` IDs backed by dense internal node indexes
- Per-layer sorted, duplicate-free adjacency
- Cosine distance optimized for pre-normalized vectors
- AVX2 on detected `x86_64` CPUs, NEON on `aarch64`, and a scalar fallback
- Memory-mapped, bounds-checked file access
- CRC-32 integrity checking and atomic v2-to-v3 migration
- Deterministic v3 persistence with byte-for-byte fixture coverage

Deletion, updates, concurrent mutation, filtering, and quantization are not
currently supported.

## Usage

```rust
use hnsw_rs::{Config, HnswIndex};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut index = HnswIndex::new(Config {
    dim: 4,
    rng_seed: Some(42), // omit for an operating-system seed
    ..Config::default()
})?;

index.insert(100, &[1.0, 0.0, 0.0, 0.0])?;
index.insert(5_000, &[0.0, 1.0, 0.0, 0.0])?;

let hits = index.search(&[0.98, 0.02, 0.0, 0.0], 10)?;
assert_eq!(hits[0].id, 100);
# Ok(())
# }
```

`insert` and `search` return a dimension error instead of panicking when a
slice does not match `Config::dim`. Duplicate external IDs are rejected.
Random levels come from the maintained `rand` crate. A configured seed is
repeatable with the pinned dependency version.

### Normalized-vector contract

HNSW construction and search use `1 - dot(a, b)`. Inputs must therefore be
unit-normalized. This is the fast form of cosine distance and ranges from 0 for
identical unit vectors to 2 for opposite vectors. Use
`hnsw_rs::vector::cosine_similarity` when working with arbitrary vectors, but
normalize them before inserting them into an index.

### Level multiplier

`Config::level_mult` is the probability of stopping at the current level.
`1.0` always assigns level zero; `0.0` always assigns `max_level`. The default
is `0.5`.

## Persistence

```rust
use hnsw_rs::{load_file, Config, HnswIndex};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let path = std::env::temp_dir().join("example.hnsw");
let mut index = HnswIndex::new(Config {
    dim: 2,
    rng_seed: Some(1),
    ..Config::default()
})?;
index.insert(7, &[1.0, 0.0])?;
index.save(&path)?;

let loaded = load_file(&path)?;
assert_eq!(loaded.header().node_count, 1);
assert_eq!(loaded.node(0).unwrap().external_id, 7);
assert_eq!(
    loaded.vector(0).unwrap().iter().collect::<Vec<_>>(),
    [1.0, 0.0]
);
# std::fs::remove_file(path)?;
# Ok(())
# }
```

Loaded files remain memory-mapped. `node`, `vector`, and `edges` expose checked
views tied to the mapping's lifetime; values are decoded from explicit
little-endian bytes rather than obtained through unaligned pointer casts.

The v3 format contains:

1. A 64-byte header, including configuration, entry-point metadata, and CRC.
2. Row-major `f32` vector data.
3. Twelve-byte node records containing external ID, level, and vector offset.
4. Per-layer edge offset and length tables.
5. Flattened dense `u32` neighbor indexes.

The CRC covers every byte after the header. Valid v2 files are rewritten to v3
through a flushed temporary file and same-directory atomic rename before being
returned to the caller.

Serde is intentionally not used for this file format. `.hnsw` is a fixed
cross-language byte layout, not general Rust object serialization; Serde would
still require a custom serializer/deserializer for its offsets, CRC, mmap
views, and migration rules. Integer and float fields instead use the standard
library's explicit little-endian byte conversions.

## SIMD

Portable `std::simd` is not stable in Rust 1.96, so this crate uses stable
`std::arch` intrinsics. Runtime AVX2 detection is used on `x86_64`; NEON is part
of the baseline `aarch64` architecture. Every platform retains the safe scalar
kernel, which also serves as the test oracle. Fused multiply-add is avoided so
distance accumulation uses eight partial sums in every kernel.

Unsafe code is confined to:

- Read-only `memmap2` mapping after file-length checks
- Bounds-checked architecture loads and stores in `src/vector.rs`

## Verification

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo bench --bench distance
```

The persistence suite loads a deterministic v3 golden file and requires the
writer to reproduce all 220 bytes exactly. Its SHA-256 is documented in
`tests/fixtures/README.md`.

## Design notes

Construction uses owned vectors and mutable per-layer adjacency. Persistence
flattens edges only while saving. This keeps mutation straightforward while
retaining the original mmap-friendly snapshot representation.

Insertion selects the nearest `M` candidates and does not prune older nodes'
adjacency lists after adding reverse edges.

## License

MIT
