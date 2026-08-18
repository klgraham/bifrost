use std::{fmt, io};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    InvalidConfig(&'static str),
    DimensionMismatch { expected: usize, actual: usize },
    InvalidVector(&'static str),
    DuplicateExternalId(u32),
    CapacityExceeded(&'static str),
    InvalidNode(u32),
    InvalidLayer(u8),
    InvalidMagic,
    UnsupportedVersion { expected: u16, actual: u16 },
    CrcMismatch { expected: u32, actual: u32 },
    InvalidFile(&'static str),
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
