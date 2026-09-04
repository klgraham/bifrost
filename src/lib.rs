//! Compact Hierarchical Navigable Small World (HNSW) index for unit `f32` vectors.
//!
//! The crates.io package is `bifrost-index`. The library crate path is `bifrost`,
//! so callers and doctests import `bifrost::{Config, HnswIndex, ...}`.
//!
//! # Overview
//!
//! [`HnswIndex`] builds a graph by incremental [`HnswIndex::insert`] (or a
//! one-shot [`HnswIndex::build`] on an empty index). [`HnswIndex::search`]
//! returns approximate nearest neighbors under cosine distance. Construction
//! parameters such as [`Config::m`] and [`Config::dim`] are fixed at
//! [`HnswIndex::new`]. Query width is `max(ef_search, k)` on both
//! [`HnswIndex::search`] and [`LoadedHnsw::search`]; [`HnswIndex::search_with_ef`]
//! and [`LoadedHnsw::search_with_ef`] override the stored candidate width for
//! one query.
//!
//! # Distance contract
//!
//! Insert and search treat vectors as unit-normalized `f32`. Cosine distance
//! is `1 - dot` and does not renormalize. Public [`vector::dot`],
//! [`vector::cosine_distance`], and [`vector::cosine_similarity`] return
//! [`Error::DimensionMismatch`] when the arguments have different lengths.
//! Insert and search `debug_assert` finite, near-unit vectors;
//! [`Config::check_vectors`] returns [`Error::InvalidVector`] for the same
//! failures.
//!
//! # Persistence
//!
//! [`HnswIndex::save`] writes a `.hnsw` v3 snapshot (temp + `sync_all` +
//! `rename`). [`load_file`] / [`LoadedHnsw::open`] map that file and
//! [`LoadedHnsw::search`] walks it in place. Valid v2 files are rewritten to
//! v3 on load. Do not mutate a mapped file while [`LoadedHnsw`] lives.
//!
//! # Limits
//!
//! There is no delete, update, filter, quantization, or concurrent-mutation
//! API. [`HnswIndex::build`] assigns IDs `0..n-1` and is a convenience for an
//! empty index; a second `build` or any colliding ID returns
//! [`Error::DuplicateExternalId`].
//!
//! # Examples
//!
//! ```
//! use bifrost::{load_file, Config, HnswIndex};
//!
//! let mut index = HnswIndex::new(Config {
//!     dim: 4,
//!     rng_seed: Some(42),
//!     ..Config::default()
//! })?;
//! index.insert(100, &[1.0, 0.0, 0.0, 0.0])?;
//! index.insert(200, &[0.0, 1.0, 0.0, 0.0])?;
//! let hits = index.search(&[1.0, 0.0, 0.0, 0.0], 10)?;
//! assert_eq!(hits[0].id, 100);
//!
//! let path = std::env::temp_dir().join(format!(
//!     "bifrost-overview-{}.hnsw",
//!     std::process::id()
//! ));
//! index.save(&path)?;
//! let loaded = load_file(&path)?;
//! assert_eq!(loaded.search(&[1.0, 0.0, 0.0, 0.0], 1)?[0].id, 100);
//! # let _ = std::fs::remove_file(&path);
//! # Ok::<(), bifrost::Error>(())
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]

mod config;
mod error;
mod graph;
mod index;
mod layer;
pub mod serialize;
pub mod vector;

pub use config::Config;
pub use error::{Error, Result};
pub use graph::{ExternalId, Graph, NodeIndex, NodeMeta};
pub use index::{HnswIndex, SearchHit};
pub use serialize::{
    HEADER_SIZE, Header, LoadedHnsw, MAGIC, MIGRATABLE_VERSION, NODE_META_SIZE, VERSION, load_file,
    save_file,
};
