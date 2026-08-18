//! A compact Hierarchical Navigable Small World (HNSW) vector index.
//!
//! It supports incremental insertion, approximate nearest-neighbor search with
//! cosine distance, and mmap-friendly `.hnsw` v3 persistence. Saved snapshots
//! can be mapped with [`load_file`] / [`LoadedHnsw::open`] and searched in place
//! through [`LoadedHnsw::search`]. Snapshots are replaced atomically (temp +
//! `sync_all` + `rename`); do not mutate a mapped file while [`LoadedHnsw`]
//! lives. Query width is `max(ef_search, k)` on both
//! [`HnswIndex::search`] and [`LoadedHnsw::search`]; `search_with_ef` overrides
//! the stored candidate width for a single query. Insert and search
//! `debug_assert` finite, near-unit vectors; [`Config::check_vectors`]
//! returns [`Error::InvalidVector`] for the same failures. Public
//! [`vector::dot`], [`vector::cosine_distance`], and
//! [`vector::cosine_similarity`] return [`Error::DimensionMismatch`] when
//! the arguments have different lengths.

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
