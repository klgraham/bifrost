# Bifrost

Bifrost navigates vector space.

Bifrost is a Rust crate for approximate nearest-neighbor search. It keeps
unit-normalized embeddings in an HNSW graph and returns the closest caller IDs
it can find. Use it when you want ANN inside a Rust process: incremental insert,
cosine-distance search, and memory-mapped `.hnsw` snapshots. There is no server,
CLI, or network protocol.

The algorithm is Hierarchical Navigable Small World,
HNSW. Call `HnswIndex`. Tune `m`, `ef_construction`, and `ef_search`. Save
`.hnsw` files that still carry the `HNSW` magic. Those names are the algorithm,
not a second product name.

## Install

The crates.io package is `bifrost-index`. The Rust crate path stays `bifrost`.

```bash
cargo add bifrost-index
```

```rust
use bifrost::{Config, HnswIndex};
```

`bifrost` is already taken on crates.io. The GitHub repo is still
[klgraham/bifrost](https://github.com/klgraham/bifrost). To depend on git
instead:

```toml
[dependencies]
bifrost-index = { git = "https://github.com/klgraham/bifrost" }
```

Rust 1.87 or newer.

## Usage

```rust
use bifrost::{Config, HnswIndex};

fn main() -> bifrost::Result<()> {
    let mut index = HnswIndex::new(Config {
        dim: 4,
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        rng_seed: Some(42),
        ..Config::default()
    })?;

    index.insert(100, &[1.0, 0.0, 0.0, 0.0])?;
    index.insert(5_000, &[0.0, 1.0, 0.0, 0.0])?;

    let hits = index.search(&[0.9998, 0.02, 0.0, 0.0], 10)?;
    assert_eq!(hits[0].id, 100);
    Ok(())
}
```

`m`, `ef_construction`, and `ef_search` must be greater than zero.
`HnswIndex::new` captures the construction parameters. `set_ef_search` is the
supported way to change the stored query width later. `search` uses
`max(ef_search, k)`, so a request larger than `ef_search` can still return up
to `k` hits. `search_with_ef` overrides the width for one query.

`insert` takes a caller-chosen `u32` id. A second insert of the same id returns
`Error::DuplicateExternalId`. A slice that does not match `Config::dim` returns
`Error::DimensionMismatch` instead of panicking. `HnswIndex::build` fills an
empty index and assigns dense IDs `0..n-1`. A second `build`, or `build` after
an insert that already used one of those IDs, returns `DuplicateExternalId`.
Append later with `insert`.

Distance is `1 - dot(a, b)`. Insert and search vectors must be unit-normalized.
That is the fast cosine form: 0 for identical unit vectors, 2 for opposites.
`bifrost::vector::cosine_similarity` accepts arbitrary vectors. Normalize
before the index sees them. Debug builds `debug_assert` finite coordinates and
an L2 norm within `0.01` of 1 (`vector::UNIT_NORM_TOLERANCE`). Set
`Config::check_vectors` so release builds return `Error::InvalidVector` for the
same failures. The flag is not stored in `.hnsw` snapshots. Call
`LoadedHnsw::set_check_vectors` after mapping if queries should fail that way.

`Config::default()` sets `dim = 384`, `m = 16`, `ef_construction = 200`,
`ef_search = 100`, `max_level = 16`, `level_mult = 1 - 1/m` (`0.9375` when
`m = 16`), no `rng_seed`, and `check_vectors = false`.

## Persistence

`HnswIndex::save` writes a version-3 `.hnsw` snapshot. `load_file`,
`LoadedHnsw::open`, and `HnswIndex::load` memory-map that file for query-only
search. They do not rebuild a mutable `HnswIndex`. Further inserts still need a
live builder.

The file magic is `HNSW`. `bifrost::MAGIC` is `0x484E5357`. Current writes are
version 3. A valid version 2 file is rewritten to v3 through a flushed
temporary file and a same-directory `rename`, then returned. `save` uses that
same temp, `sync_all`, and `rename` path, so a crash mid-write cannot replace a
good file with a truncated one.

```rust
use bifrost::{load_file, Config, HnswIndex};

fn main() -> bifrost::Result<()> {
    let path = std::env::temp_dir().join("example.hnsw");
    let mut index = HnswIndex::new(Config {
        dim: 2,
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        rng_seed: Some(1),
        ..Config::default()
    })?;
    index.insert(7, &[1.0, 0.0])?;
    index.save(&path)?;

    let loaded = load_file(&path)?;
    assert_eq!(loaded.header().node_count, 1);
    assert_eq!(loaded.node(0).unwrap().external_id, 7);
    let hits = loaded.search(&[1.0, 0.0], 1)?;
    assert_eq!(hits[0].id, 7);
    Ok(())
}
```

The mapping is read-only. Do not truncate, overwrite in place, or otherwise
mutate a mapped `.hnsw` file while a `LoadedHnsw` for that path is alive.

## The algorithm

HNSW is the graph search of Yu. A. Malkov and D. A. Yashunin,
[Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs](https://arxiv.org/abs/1603.09320). Neighbor
selection follows that paper and [hnswlib](https://github.com/nmslib/hnswlib):
the diversity heuristic (paper Alg. 4), level multiplier `1 - 1/m`
(`Config::level_mult_for_m`), at most `M` neighbors for a new node at layer 0
and `max(M / 2, 1)` above, then outgoing caps of `2M` at layer 0 (`Mmax0`) and
`M` on upper layers (`Mmax`). Reverse-link pruning is one-sided. A dropped
outgoing edge can remain as an incoming edge on the peer.

Bifrost implements that algorithm. It does not rename it.

## Limits

Deletion, in-place updates, concurrent mutation, filtering, and quantization
are not supported.

## Verify

```bash
cargo test --all-features
```

CI also runs `cargo fmt --check` and
`cargo clippy --all-targets --all-features -- -D warnings` on Ubuntu and macOS.

## License

MIT
