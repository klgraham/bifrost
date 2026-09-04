use std::{fmt, io};

/// Fallible public API result. The error type is always [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Recoverable failure from configuration, insert, search, or `.hnsw` I/O.
///
/// Match on the variant. The `Display` text is a diagnostic, not a stable
/// protocol.
#[derive(Debug)]
pub enum Error {
    /// A [`crate::Config`] field failed validation, or a later setter received
    /// an out-of-range value.
    ///
    /// Returned by [`crate::HnswIndex::new`], [`crate::HnswIndex::set_ef_search`],
    /// and [`crate::HnswIndex::set_level_mult`]. The payload names the failed
    /// rule (`dim`, `m`, `ef_construction`, and `ef_search` must be greater
    /// than zero; `max_level` must be in `1..=254`; `level_mult` must be
    /// finite and in `[0, 1]`). Fix the parameter and retry.
    InvalidConfig(&'static str),
    /// A vector's length does not match the expected dimension.
    ///
    /// Insert and search compare against [`crate::Config::dim`].
    /// [`crate::vector::dot`], [`crate::vector::cosine_distance`], and
    /// [`crate::vector::cosine_similarity`] compare the two argument lengths
    /// (`expected` from the first slice, `actual` from the second). Supply a
    /// slice of the expected length.
    DimensionMismatch { expected: usize, actual: usize },
    /// A coordinate is non-finite, or the L2 norm is more than
    /// [`crate::vector::UNIT_NORM_TOLERANCE`] from `1`.
    ///
    /// Returned only when [`crate::Config::check_vectors`] is set (or after
    /// [`crate::HnswIndex::set_check_vectors`] /
    /// [`crate::LoadedHnsw::set_check_vectors`]). Debug builds `debug_assert`
    /// the same contract even when the flag is off. Normalize the vector and
    /// drop NaN/Inf before retrying.
    InvalidVector(&'static str),
    /// [`crate::HnswIndex::insert`] or [`crate::HnswIndex::build`] reused an
    /// external ID already in the index.
    ///
    /// The payload is the colliding ID. [`crate::HnswIndex::build`] assigns
    /// `0..n-1` and is for an empty index; a second `build` or `build` after
    /// `insert(0, …)` fails here. Choose a free ID, or call `insert` to append.
    DuplicateExternalId(u32),
    /// An in-memory or on-disk count cannot be stored in `u32`.
    ///
    /// The graph may hold [`u32::MAX`] nodes (`0..=u32::MAX - 1`). A further
    /// insert, a `build` ID that does not fit in `u32`, or a vector/edge table
    /// that would overflow `u32` returns this. The payload names the quantity.
    /// The index is unchanged; split the data or use another index.
    CapacityExceeded(&'static str),
    /// A node index is missing from the graph, or is not a valid search entry.
    ///
    /// Public [`crate::Graph`] accessors return empty/`None` instead of this
    /// error. Insert and search can return it when the entry point is absent
    /// or a mapped vector is missing. A failed [`crate::HnswIndex::insert`] is
    /// rolled back, so the ID can be retried. Treat a persistent occurrence as
    /// a corrupt in-memory graph or snapshot.
    InvalidNode(u32),
    /// A layer is above a node's assigned level, or the layer does not exist.
    ///
    /// Public [`crate::Graph::edges`] returns an empty slice for this case.
    /// Insert and search can return it when descent starts from an entry that
    /// does not occupy the requested layer. Same recovery as
    /// [`Error::InvalidNode`].
    InvalidLayer(u8),
    /// The file does not start with [`crate::MAGIC`].
    ///
    /// The path is not a Bifrost `.hnsw` snapshot, or it is corrupt before the
    /// magic. Open a file written by [`crate::HnswIndex::save`].
    InvalidMagic,
    /// The snapshot version is neither [`crate::VERSION`] nor
    /// [`crate::MIGRATABLE_VERSION`].
    ///
    /// `expected` is the current writer version. `actual` is the file's
    /// version. Valid v2 files are rewritten to v3 on load; other versions
    /// cannot be opened. Rebuild the index with this crate, or use a matching
    /// reader.
    UnsupportedVersion { expected: u16, actual: u16 },
    /// The v3 data-section CRC32 does not match [`crate::Header::stored_crc`].
    ///
    /// `expected` is the header value; `actual` is the hash of bytes after
    /// [`crate::HEADER_SIZE`]. The file was truncated or overwritten in place.
    /// Restore a good snapshot; do not search this mapping.
    CrcMismatch { expected: u32, actual: u32 },
    /// The snapshot failed layout or writer-invariant checks after a valid
    /// magic and version.
    ///
    /// Typical causes: truncated sections, invalid header fields, duplicate
    /// external IDs, gapped vector offsets, or unsorted edge lists. The
    /// payload is a static reason. The file cannot be searched; rewrite it
    /// from a live [`crate::HnswIndex`].
    InvalidFile(&'static str),
    /// A filesystem or mmap error from save, load, or rename.
    ///
    /// Implements [`std::error::Error::source`]. Retry transient I/O; check
    /// permissions and disk space for persistent failures.
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidVector(message) => write!(f, "invalid vector: {message}"),
            Self::DuplicateExternalId(id) => write!(f, "duplicate external ID {id}"),
            Self::CapacityExceeded(what) => write!(f, "{what} exceeds the on-disk u32 capacity"),
            Self::InvalidNode(node) => write!(f, "invalid node index {node}"),
            Self::InvalidLayer(layer) => write!(f, "invalid graph layer {layer}"),
            Self::InvalidMagic => write!(f, "invalid HNSW file magic"),
            Self::UnsupportedVersion { expected, actual } => write!(
                f,
                "unsupported HNSW file version {actual}; expected {expected}"
            ),
            Self::CrcMismatch { expected, actual } => write!(
                f,
                "HNSW data CRC mismatch: expected {expected:#010x}, got {actual:#010x}"
            ),
            Self::InvalidFile(message) => write!(f, "invalid HNSW file: {message}"),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
