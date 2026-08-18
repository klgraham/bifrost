# hnsw-rs

A compact Rust implementation of Hierarchical Navigable Small World (HNSW)
search for normalized vector embeddings.

It supports incremental insertion, cosine-distance search, stable Rust SIMD
acceleration, and `.hnsw` v3 persistence.

## Features

- Dynamic insertion without rebuilding the index
- Sparse caller-facing `u32` IDs backed by dense internal node indexes
- Per-layer sorted, duplicate-free adjacency with HNSW `Mmax` / `Mmax0` degree caps
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
`Config::m`, `Config::ef_construction`, and `Config::ef_search` must be greater
than zero. Construction parameters are captured at `HnswIndex::new`; `config()`
returns a copy, and `set_ef_search` is the supported way to change the query
candidate width (`0` is rejected). `Config::dim` and the graph cannot be mutated
after construction: inspect neighbors with `edges` / `degree` / `layer_count`.
Search uses a layer-0 candidate width of `max(ef_search, k)`, so asking for more
hits than `Config::ef_search` still returns up to `k` neighbors when the graph
contains them. `search_with_ef` uses `max(ef, k)` for one query without changing the
stored width. A new node keeps at most `M` neighbors at layer 0 and
`max(M / 2, 1)` at upper layers, chosen with the Malkov & Yashunin / hnswlib
diversity heuristic (keep a candidate if it is closer to the new node than to
any already chosen neighbor). After each reverse link, the neighbor's
outgoing adjacency is re-selected with the same heuristic and capped at `2M`
at layer 0 (`Mmax0`) and `M` at upper layers (`Mmax`). Pruning is one-sided:
a dropped `A → B` edge is not removed from `B`, so the graph can become
directed after shrink. Random levels come from the maintained `rand` crate.
A configured seed is repeatable with the pinned dependency version.

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
let hits = loaded.search(&[1.0, 0.0], 1)?;
assert_eq!(hits[0].id, 7);
# std::fs::remove_file(path)?;
# Ok(())
# }
```

`LoadedHnsw::open` and `load_file` are the primary mapping constructors;
`HnswIndex::load` is the same query-only mapping and does not rebuild a
mutable index. Loaded files remain memory-mapped: `search` walks the on-disk
graph and vectors without re-inserting them, using `max(ef_search, k)` as the
candidate width, matching live `HnswIndex::search`. `search_with_ef`
overrides the stored `ef_search` on both the builder and the mapped snapshot.
`node`,
`vector`, and `edges` expose checked views tied to the mapping's lifetime;
values are decoded from explicit little-endian bytes rather than obtained
through unaligned pointer casts.

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
cargo clippy --manifest-path benchmarks/competitors/Cargo.toml --all-targets --all-features -- -D warnings
```

The persistence suite loads a deterministic v3 golden file and requires the
writer to reproduce all 220 bytes exactly. Its SHA-256 is documented in
`tests/fixtures/README.md`.

### Competitor benchmark

`benchmarks/competitors` is an independent crate that compares this project
with pure-Rust `hnsw_rs` and the Rust bindings for USearch. Keeping it separate
prevents the competing libraries and USearch's native backend from becoming
development dependencies of the library crate. The benchmark generates
deterministic unit-normalized `f32` vectors, builds each index through its
public Rust API, and reports sequential build throughput, query latency, query
throughput, and recall@k against an exact inner-product scan.

The default comparison uses 10,000 384-dimensional vectors, 100 queries,
`k=10`, `M=16`, `ef_construction=200`, and `ef_search=100`. The three indexes
use equivalent inner-product rankings: this crate's normalized cosine
distance, a non-negative `max(0, 1 - dot)` custom distance through `hnsw_rs`'s
public distance interface, and USearch's `MetricKind::IP` with `f32` storage.
The custom upstream distance avoids `hnsw_rs::DistDot` panicking when normal
`f32` rounding makes a unit vector's self-dot slightly greater than one; the
zero clamp affects only that rounding error. Data generation, exact ground
truth, and dependency setup are outside the reported timings.

Every workload setting can be overridden for quick smoke runs or larger tests:

```bash
HNSW_BENCH_VECTORS=100000 \
HNSW_BENCH_DIMENSIONS=768 \
HNSW_BENCH_QUERIES=1000 \
HNSW_BENCH_REPETITIONS=10 \
HNSW_BENCH_K=10 \
HNSW_BENCH_M=16 \
HNSW_BENCH_EF_CONSTRUCTION=200 \
HNSW_BENCH_EF_SEARCH=100 \
HNSW_BENCH_SEED=42 \
cargo run --release --manifest-path benchmarks/competitors/Cargo.toml
```

Set `HNSW_BENCH_EF_SEARCHES` to a comma-separated list to build each index
once and measure a controlled query-width sweep over the same graph:

```bash
HNSW_BENCH_DIMENSIONS=1536 \
HNSW_BENCH_EF_SEARCHES=200,400,800 \
cargo run --release --manifest-path benchmarks/competitors/Cargo.toml
```

The list overrides `HNSW_BENCH_EF_SEARCH`. This avoids rebuilding randomized
graphs between points on the latency-versus-recall curve.

#### BEIR FiQA-2018 with OpenAI embeddings

The independent benchmark crate can prepare and consume a real
[BEIR FiQA-2018](https://github.com/beir-cellar/beir) retrieval fixture. The
preparer embeds the 57,600 non-empty corpus documents and the 648 queries in
the test qrels with OpenAI `text-embedding-3-small` at its default 1,536
dimensions. (The source archive contains 38 empty corpus rows that are not
referenced by the test qrels; these are skipped.)
Generated source data, request caches, and vectors live under
`benchmarks/competitors/data/`, which is ignored by Git.

First, download and validate the public dataset without making an OpenAI API
request:

```bash
cargo run --release \
  --manifest-path benchmarks/competitors/Cargo.toml \
  --features fiqa-prep \
  --bin prepare-fiqa \
  -- --download-only
```

Then review the expected API usage, set `OPENAI_API_KEY`, and generate the
fixture:

```bash
OPENAI_API_KEY=... cargo run --release \
  --manifest-path benchmarks/competitors/Cargo.toml \
  --features fiqa-prep \
  --bin prepare-fiqa
```

Embedding responses are cached one batch at a time, so an interrupted run can
resume without repeating completed requests. The tool validates response
indexes, dimensions, and vector norms before atomically packing little-endian
`f32` files. Use `--max-corpus` and `--max-queries` for a lower-cost fixture;
`--help` lists the model, dimension, batch-size, and output overrides. OpenAI's
[embeddings API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
documents the request used by the preparer.

Run the comparison against the prepared vectors with:

```bash
HNSW_BENCH_FIXTURE=benchmarks/competitors/data/fiqa-text-embedding-3-small \
HNSW_BENCH_EF_SEARCHES=100,200,400 \
cargo run --release --manifest-path benchmarks/competitors/Cargo.toml
```

After the timed comparison, the runner saves this crate's built index under
the fixture directory as
`indexes/hnsw-rs-m<M>-efc<EF_CONSTRUCTION>-seed<SEED>.hnsw`. The raw `f32`
files remain the canonical cross-implementation fixture; the `.hnsw` file is a
derived, memory-mapped index for later query-only use.

For a fixture, `HNSW_BENCH_VECTORS` and `HNSW_BENCH_QUERIES` optionally limit
the loaded prefix, while its manifest supplies the vector dimension. In
addition to build and query performance, the report distinguishes exact ANN
recall from retrieval quality: nDCG@k and qrels recall@k use only test queries
that retain a relevant document in the loaded corpus subset.

Run benchmark comparisons on an otherwise idle machine. The suite deliberately
uses one caller for insertion and search so library-internal behavior is being
compared rather than different caller-side parallelization strategies.

## Design notes

Construction uses owned vectors and mutable per-layer adjacency. Persistence
flattens edges only while saving. This keeps mutation straightforward while
retaining the original mmap-friendly snapshot representation.

Insertion and reverse-link pruning share the paper / hnswlib diversity
heuristic (Alg. 4), not simple nearest-`M` (Alg. 3). A candidate is kept only
when it is closer to the query node than to any already chosen neighbor, up
to `M` at layer 0 (`max(M / 2, 1)` at upper layers) for a new node and `2M` /
`M` when shrinking an existing list. The peer keeps its link back to the hub;
search follows outgoing edges only.

## License

MIT
