//! HNSW `.hnsw` v2/v3 persistence.
//!
//! This module deliberately does not use Serde. The format is a fixed,
//! cross-language byte layout with exact offsets, CRC coverage, mmap-backed
//! views, and an in-place version migration. A Serde format would still need
//! custom implementations for those rules while risking byte incompatibility.

use std::{
    cell::RefCell,
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    ops::Range,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crc32fast::Hasher;
use memmap2::{Mmap, MmapOptions};

use crate::{
    Config, Error, HnswIndex, NodeIndex, NodeMeta, Result, SearchHit,
    index::hits_from_candidates,
    layer::{SearchGraph, SearchVectors, search_knn},
    vector::cosine_distance,
};

pub const MAGIC: u32 = 0x484e_5357;
pub const VERSION: u16 = 3;
pub const MIGRATABLE_VERSION: u16 = 2;
pub const HEADER_SIZE: usize = 64;
pub const NODE_META_SIZE: usize = 12;

const CRC_OFFSET: usize = 38;

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub magic: u32,
    pub version: u16,
    pub dim: u16,
    pub node_count: u32,
    pub m: u8,
    pub ef_construction: u16,
    pub ef_search: u16,
    pub max_level: u8,
    pub level_mult: f64,
    pub entry_point: u32,
    pub entry_level: u8,
    pub layer_count: u8,
    pub reserved: [u8; 24],
}

impl Header {
    #[must_use]
    pub fn config(&self) -> Config {
        Config {
            dim: self.dim,
            m: self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            max_level: self.max_level,
            level_mult: self.level_mult,
            rng_seed: None,
            check_vectors: false,
        }
    }

    #[must_use]
    pub fn stored_crc(&self) -> u32 {
        u32::from_le_bytes(self.reserved[..4].try_into().expect("four CRC bytes"))
    }
}

#[derive(Clone, Debug)]
struct Sections {
    vectors: Range<usize>,
    nodes: Range<usize>,
    edge_offsets: Range<usize>,
    edge_lengths: Range<usize>,
    edges: Range<usize>,
}

/// Validated, memory-mapped HNSW file.
///
/// The mapping is query-only: [`LoadedHnsw::search`] walks the on-disk graph
/// and vectors in place. Construction and further inserts stay on
/// [`HnswIndex`].
///
/// The snapshot is opened read-only. Do not truncate, overwrite in place, or
/// otherwise mutate the mapped file (or its inode) while this value lives:
/// a shared `[u8]` mapping is unsound if those bytes change. This crate's
/// [`save_file`] and v2→v3 migration write a sibling temporary file and
/// `rename` it over the destination, so an existing mapping stays attached
/// to the previous inode. A shared read mapping is used instead of
/// `map_copy` because `MAP_PRIVATE` still has unspecified visibility of
/// later file writes, and the mmap-friendly path is meant to share pages.
pub struct LoadedHnsw {
    mmap: Mmap,
    header: Header,
    sections: Sections,
    check_vectors: bool,
}

impl std::fmt::Debug for LoadedHnsw {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadedHnsw")
            .field("header", &self.header)
            .field("file_size", &self.mmap.len())
            .field("check_vectors", &self.check_vectors)
            .finish()
    }
}

impl LoadedHnsw {
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    #[must_use]
    pub fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
        if node_index >= self.header.node_count {
            return None;
        }
        let offset = self.sections.nodes.start + node_index as usize * NODE_META_SIZE;
        Some(read_node(&self.mmap[offset..offset + NODE_META_SIZE]))
    }

    #[must_use]
    pub fn vector(&self, node_index: NodeIndex) -> Option<VectorView<'_>> {
        let node = self.node(node_index)?;
        let relative_start = (node.vector_offset as usize).checked_mul(4)?;
        let start = self.sections.vectors.start.checked_add(relative_start)?;
        let byte_length = usize::from(self.header.dim).checked_mul(4)?;
        let end = start.checked_add(byte_length)?;
        Some(VectorView {
            bytes: self.mmap.get(start..end)?,
        })
    }

    /// Memory-maps and validates a `.hnsw` snapshot for query-only search.
    ///
    /// This is the primary constructor; [`load_file`] is the same function.
    /// The file must not be mutated while the returned value lives; see the
    /// type-level contract on [`LoadedHnsw`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        load_file(path)
    }

    /// Enables or disables Result-returning checks on search queries.
    ///
    /// Debug builds assert finiteness and near-unit norm regardless. The flag
    /// is not read from the snapshot header; see [`Config::check_vectors`].
    pub fn set_check_vectors(&mut self, check_vectors: bool) {
        self.check_vectors = check_vectors;
    }

    /// Searches the mapped snapshot for at most `k` nearest neighbors.
    ///
    /// Uses the stored `ef_search`, expanded to `max(ef_search, k)` so a
    /// request larger than the saved candidate width still returns up to `k`
    /// hits. Call [`LoadedHnsw::search_with_ef`] to override the stored width.
    /// Query vectors are `debug_assert`ed finite and near-unit; set
    /// [`LoadedHnsw::set_check_vectors`] to return [`Error::InvalidVector`].
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.search_with_ef(query, k, self.header.ef_search)
    }

    /// Searches using an explicit layer-0 candidate width.
    ///
    /// The search width is `max(ef, k)`, matching [`HnswIndex::search_with_ef`].
    /// The graph and vectors stay mapped; nothing is copied into an owned
    /// [`HnswIndex`].
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: u16) -> Result<Vec<SearchHit>> {
        self.check_vector(query)?;
        let store = MappedVectors {
            loaded: self,
            scratch: RefCell::new(vec![0.0; usize::from(self.header.dim)]),
        };
        let entry_point = (self.header.node_count > 0).then_some(self.header.entry_point);
        let candidates = search_knn(
            self,
            &store,
            query,
            k,
            ef,
            entry_point,
            self.header.entry_level,
        )?;
        Ok(hits_from_candidates(self, candidates))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.header.node_count as usize
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.header.node_count == 0
    }

    fn check_dimension(&self, vector: &[f32]) -> Result<()> {
        let expected = usize::from(self.header.dim);
        if vector.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                actual: vector.len(),
            });
        }
        Ok(())
    }

    fn check_vector(&self, vector: &[f32]) -> Result<()> {
        self.check_dimension(vector)?;
        crate::vector::validate_input_vector(vector, self.check_vectors)
    }

    #[must_use]
    pub fn edges(&self, level: u8, node_index: NodeIndex) -> EdgeView<'_> {
        let Some(node) = self.node(node_index) else {
            return EdgeView { bytes: &[] };
        };
        if level >= self.header.layer_count || node.level < level {
            return EdgeView { bytes: &[] };
        }
        let slot = usize::from(level) * self.header.node_count as usize + node_index as usize;
        let offset = read_table_u32(&self.mmap, &self.sections.edge_offsets, slot) as usize;
        let length = read_table_u32(&self.mmap, &self.sections.edge_lengths, slot) as usize;
        let Some(relative_start) = offset.checked_mul(4) else {
            return EdgeView { bytes: &[] };
        };
        let Some(start) = self.sections.edges.start.checked_add(relative_start) else {
            return EdgeView { bytes: &[] };
        };
        let Some(byte_length) = length.checked_mul(4) else {
            return EdgeView { bytes: &[] };
        };
        let Some(end) = start.checked_add(byte_length) else {
            return EdgeView { bytes: &[] };
        };
        EdgeView {
            bytes: self.mmap.get(start..end).unwrap_or(&[]),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VectorView<'a> {
    bytes: &'a [u8],
}

impl VectorView<'_> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len() / 4
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<f32> {
        let start = index.checked_mul(4)?;
        let end = start.checked_add(4)?;
        let bytes = self.bytes.get(start..end)?;
        Some(f32::from_bits(u32::from_le_bytes(bytes.try_into().ok()?)))
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.bytes.chunks_exact(4).map(|bytes| {
            f32::from_bits(u32::from_le_bytes(
                bytes.try_into().expect("four-byte vector element"),
            ))
        })
    }

    /// Copies decoded little-endian `f32` values into `dest`.
    ///
    /// Returns `None` if `dest` does not have the same length as this view.
    #[must_use]
    pub fn copy_into(&self, dest: &mut [f32]) -> Option<()> {
        if dest.len() != self.len() {
            return None;
        }
        for (slot, value) in dest.iter_mut().zip(self.iter()) {
            *slot = value;
        }
        Some(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeView<'a> {
    bytes: &'a [u8],
}

impl<'a> EdgeView<'a> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len() / 4
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<NodeIndex> {
        let start = index.checked_mul(4)?;
        let end = start.checked_add(4)?;
        let bytes = self.bytes.get(start..end)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = NodeIndex> + 'a {
        self.bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte edge element")))
    }
}

struct MappedVectors<'a> {
    loaded: &'a LoadedHnsw,
    scratch: RefCell<Vec<f32>>,
}

impl SearchGraph for LoadedHnsw {
    fn node_count(&self) -> NodeIndex {
        self.header.node_count
    }

    fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
        LoadedHnsw::node(self, node_index)
    }

    fn neighbors(&self, level: u8, node_index: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.edges(level, node_index).iter()
    }
}

impl SearchVectors for MappedVectors<'_> {
    fn distance(&self, node_index: NodeIndex, query: &[f32]) -> Result<f32> {
        let Some(view) = self.loaded.vector(node_index) else {
            return Err(Error::InvalidNode(node_index));
        };
        let mut scratch = self.scratch.borrow_mut();
        view.copy_into(&mut scratch).ok_or(Error::InvalidFile(
            "mapped vector length does not match dim",
        ))?;
        Ok(cosine_distance(&scratch, query))
    }
}

/// Writes a validated `.hnsw` snapshot for later query-only mapping.
///
/// The snapshot is written to a same-directory temporary file named
/// `.{name}.tmp-{pid}-{nonce}`, flushed with [`File::sync_all`], then
/// atomically [`rename`](fs::rename)d over `path`. A crash during the write
/// cannot replace a previous good file with a truncated one. This is the
/// same replace path used by v2→v3 migration.
pub fn save_file(index: &HnswIndex, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let config = index.config();
    config.validate()?;
    let (edge_offsets, edge_lengths, edge_data) = build_edge_tables(index)?;
    let mut hasher = Hasher::new();
    update_data_crc(&mut hasher, index, &edge_offsets, &edge_lengths, &edge_data);

    let mut header = Header {
        magic: MAGIC,
        version: VERSION,
        dim: config.dim,
        node_count: index.graph().node_count(),
        m: config.m,
        ef_construction: config.ef_construction,
        ef_search: config.ef_search,
        max_level: config.max_level,
        level_mult: config.level_mult,
        entry_point: index.entry_point.unwrap_or(0),
        entry_level: index.entry_level,
        layer_count: index.layer_count(),
        reserved: [0; 24],
    };
    header.reserved[..4].copy_from_slice(&hasher.finalize().to_le_bytes());

    replace_file(path, |writer| {
        writer.write_all(&encode_header(&header))?;
        for &value in &index.vector_data {
            writer.write_all(&value.to_bits().to_le_bytes())?;
        }
        for &node in &index.graph().node_data {
            writer.write_all(&encode_node(node))?;
        }
        write_u32_values(writer, &edge_offsets)?;
        write_u32_values(writer, &edge_lengths)?;
        write_u32_values(writer, &edge_data)?;
        Ok(())
    })
}

/// Memory-maps a `.hnsw` file and validates magic, version, CRC, layout, and
/// writer invariants (unique IDs, packed vector offsets, sorted unique edges).
///
/// Equivalent to [`LoadedHnsw::open`]. The mapping can be searched in place.
/// Valid v2 files are rewritten to v3 before the mapping is returned.
///
/// The file is opened read-only. Do not mutate it while the returned
/// [`LoadedHnsw`] is alive; see that type for the mapping contract.
pub fn load_file(path: impl AsRef<Path>) -> Result<LoadedHnsw> {
    let path = path.as_ref();
    loop {
        let file = File::open(path)?;
        if file.metadata()?.len() < HEADER_SIZE as u64 {
            return Err(Error::InvalidFile("file is shorter than the header"));
        }
        // SAFETY: `file` is opened read-only, the mapping is a shared read
        // view retained by LoadedHnsw for every exposed slice's lifetime, and
        // all ranges are validated before access. Callers must not mutate the
        // file while the mapping lives; this crate replaces snapshots with
        // temp + rename so an existing mapping keeps its original inode.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let header = decode_header(&mmap[..HEADER_SIZE]);
        if header.magic != MAGIC {
            return Err(Error::InvalidMagic);
        }
        if header.version != VERSION && header.version != MIGRATABLE_VERSION {
            return Err(Error::UnsupportedVersion {
                expected: VERSION,
                actual: header.version,
            });
        }
        let sections = parse_sections(&mmap, &header)?;

        match header.version {
            VERSION => {
                let actual = crc32fast::hash(&mmap[HEADER_SIZE..]);
                let expected = header.stored_crc();
                if actual != expected {
                    return Err(Error::CrcMismatch { expected, actual });
                }
                validate(&mmap, &header, &sections)?;
                return Ok(LoadedHnsw {
                    mmap,
                    header,
                    sections,
                    check_vectors: false,
                });
            }
            MIGRATABLE_VERSION => {
                validate(&mmap, &header, &sections)?;
                let temporary = write_v3_migration(path, &mmap, header)?;
                drop(mmap);
                commit_temporary(temporary, path)?;
            }
            _ => unreachable!("version was checked before section parsing"),
        }
    }
}

fn build_edge_tables(index: &HnswIndex) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let node_count = index.len();
    let layer_count = usize::from(index.layer_count());
    let slots = node_count
        .checked_mul(layer_count)
        .ok_or(Error::CapacityExceeded("edge table"))?;
    let mut offsets = Vec::with_capacity(slots);
    let mut lengths = Vec::with_capacity(slots);
    let mut edges = Vec::new();
    for level in 0..layer_count {
        for node in 0..node_count {
            offsets.push(
                u32::try_from(edges.len()).map_err(|_| Error::CapacityExceeded("edge data"))?,
            );
            let node_edges = index.edges(level as u8, node as u32);
            lengths.push(
                u32::try_from(node_edges.len())
                    .map_err(|_| Error::CapacityExceeded("node edge list"))?,
            );
            edges.extend_from_slice(node_edges);
        }
    }
    Ok((offsets, lengths, edges))
}

fn update_data_crc(
    hasher: &mut Hasher,
    index: &HnswIndex,
    offsets: &[u32],
    lengths: &[u32],
    edges: &[u32],
) {
    for &value in &index.vector_data {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    for &node in &index.graph().node_data {
        hasher.update(&encode_node(node));
    }
    for values in [offsets, lengths, edges] {
        for &value in values {
            hasher.update(&value.to_le_bytes());
        }
    }
}

fn write_u32_values(writer: &mut impl Write, values: &[u32]) -> Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn encode_header(header: &Header) -> [u8; HEADER_SIZE] {
    let mut bytes = [0_u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&header.magic.to_le_bytes());
    bytes[4..6].copy_from_slice(&header.version.to_le_bytes());
    bytes[6..8].copy_from_slice(&header.dim.to_le_bytes());
    bytes[8..12].copy_from_slice(&header.node_count.to_le_bytes());
    bytes[12] = header.m;
    bytes[14..16].copy_from_slice(&header.ef_construction.to_le_bytes());
    bytes[16..18].copy_from_slice(&header.ef_search.to_le_bytes());
    bytes[18] = header.max_level;
    bytes[24..32].copy_from_slice(&header.level_mult.to_bits().to_le_bytes());
    bytes[32..36].copy_from_slice(&header.entry_point.to_le_bytes());
    bytes[36] = header.entry_level;
    bytes[37] = header.layer_count;
    bytes[CRC_OFFSET..CRC_OFFSET + 24].copy_from_slice(&header.reserved);
    bytes
}

fn decode_header(bytes: &[u8]) -> Header {
    let mut reserved = [0; 24];
    reserved.copy_from_slice(&bytes[CRC_OFFSET..CRC_OFFSET + 24]);
    Header {
        magic: read_u32(&bytes[0..4]),
        version: read_u16(&bytes[4..6]),
        dim: read_u16(&bytes[6..8]),
        node_count: read_u32(&bytes[8..12]),
        m: bytes[12],
        ef_construction: read_u16(&bytes[14..16]),
        ef_search: read_u16(&bytes[16..18]),
        max_level: bytes[18],
        level_mult: f64::from_bits(u64::from_le_bytes(
            bytes[24..32].try_into().expect("eight-byte f64"),
        )),
        entry_point: read_u32(&bytes[32..36]),
        entry_level: bytes[36],
        layer_count: bytes[37],
        reserved,
    }
}

fn encode_node(node: NodeMeta) -> [u8; NODE_META_SIZE] {
    let mut bytes = [0; NODE_META_SIZE];
    bytes[..4].copy_from_slice(&node.external_id.to_le_bytes());
    bytes[4] = node.level;
    bytes[8..12].copy_from_slice(&node.vector_offset.to_le_bytes());
    bytes
}

fn read_node(bytes: &[u8]) -> NodeMeta {
    NodeMeta {
        external_id: read_u32(&bytes[..4]),
        level: bytes[4],
        vector_offset: read_u32(&bytes[8..12]),
    }
}

fn parse_sections(data: &[u8], header: &Header) -> Result<Sections> {
    validate_header(header)?;
    let mut offset = HEADER_SIZE;
    let vector_bytes = checked_product(&[header.node_count as usize, usize::from(header.dim), 4])?;
    let vectors = take_range(data.len(), &mut offset, vector_bytes)?;
    let node_bytes = checked_product(&[header.node_count as usize, NODE_META_SIZE])?;
    let nodes = take_range(data.len(), &mut offset, node_bytes)?;
    let slots = checked_product(&[header.node_count as usize, usize::from(header.layer_count)])?;
    let table_bytes = checked_product(&[slots, 4])?;
    let edge_offsets = take_range(data.len(), &mut offset, table_bytes)?;
    let edge_lengths = take_range(data.len(), &mut offset, table_bytes)?;
    if (data.len() - offset) % 4 != 0 {
        return Err(Error::InvalidFile("edge data is not u32-aligned"));
    }
    let edges = offset..data.len();
    Ok(Sections {
        vectors,
        nodes,
        edge_offsets,
        edge_lengths,
        edges,
    })
}

fn validate_header(header: &Header) -> Result<()> {
    if header.dim == 0 || header.max_level == 0 {
        return Err(Error::InvalidFile("invalid dimension or maximum level"));
    }
    if header.m == 0 {
        return Err(Error::InvalidFile("invalid neighbor count"));
    }
    if header.ef_construction == 0 || header.ef_search == 0 {
        return Err(Error::InvalidFile("invalid candidate search width"));
    }
    if !header.level_mult.is_finite() || !(0.0..=1.0).contains(&header.level_mult) {
        return Err(Error::InvalidFile("invalid level multiplier"));
    }
    if header.node_count == 0 {
        if header.layer_count != 0 || header.entry_level != 0 {
            return Err(Error::InvalidFile("invalid empty-index header"));
        }
        return Ok(());
    }
    if header.layer_count == 0
        || header.entry_point >= header.node_count
        || header.entry_level >= header.layer_count
        || header.entry_level > header.max_level
    {
        return Err(Error::InvalidFile("invalid entry point or layer count"));
    }
    Ok(())
}

fn validate(data: &[u8], header: &Header, sections: &Sections) -> Result<()> {
    let node_count = header.node_count as usize;
    let layer_count = usize::from(header.layer_count);
    let vector_count = sections.vectors.len() / 4;
    let dim = usize::from(header.dim);
    let mut seen_ids = HashSet::with_capacity(node_count);
    let mut vector_offsets = Vec::with_capacity(node_count);
    for node_index in 0..node_count {
        let start = sections.nodes.start + node_index * NODE_META_SIZE;
        let node = read_node(&data[start..start + NODE_META_SIZE]);
        if node.level >= header.layer_count || node.level > header.max_level {
            return Err(Error::InvalidFile("node level is out of range"));
        }
        if !seen_ids.insert(node.external_id) {
            return Err(Error::InvalidFile("duplicate external id"));
        }
        let end = (node.vector_offset as usize)
            .checked_add(dim)
            .ok_or(Error::InvalidFile("vector offset overflow"))?;
        if end > vector_count {
            return Err(Error::InvalidFile("vector offset is out of range"));
        }
        vector_offsets.push(node.vector_offset);
    }

    if header.node_count > 0 {
        let start = sections.nodes.start + header.entry_point as usize * NODE_META_SIZE;
        let entry = read_node(&data[start..start + NODE_META_SIZE]);
        if entry.level < header.entry_level {
            return Err(Error::InvalidFile(
                "entry point does not exist at entry_level",
            ));
        }
    }

    // The writer stores one packed vector per node: offsets tile [0, n*dim)
    // with no gaps or overlaps. A permutation of {0, dim, 2*dim, ...} is ok.
    vector_offsets.sort_unstable();
    for (index, offset) in vector_offsets.into_iter().enumerate() {
        let expected = index
            .checked_mul(dim)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(Error::InvalidFile("vector offset overflow"))?;
        if offset != expected {
            return Err(Error::InvalidFile("overlapping or gapped vector offsets"));
        }
    }

    let edge_count = sections.edges.len() / 4;
    for level in 0..layer_count {
        for node_index in 0..node_count {
            let slot = level * node_count + node_index;
            let edge_offset = read_table_u32(data, &sections.edge_offsets, slot) as usize;
            let edge_length = read_table_u32(data, &sections.edge_lengths, slot) as usize;
            let end = edge_offset
                .checked_add(edge_length)
                .ok_or(Error::InvalidFile("edge span overflow"))?;
            if end > edge_count {
                return Err(Error::InvalidFile("edge span is out of range"));
            }
            let node_start = sections.nodes.start + node_index * NODE_META_SIZE;
            let node = read_node(&data[node_start..node_start + NODE_META_SIZE]);
            if level > usize::from(node.level) && edge_length != 0 {
                return Err(Error::InvalidFile("node has edges above its level"));
            }
            let mut previous = None;
            for edge_index in edge_offset..end {
                let byte_offset = sections.edges.start + edge_index * 4;
                let neighbor = read_u32(&data[byte_offset..byte_offset + 4]);
                if neighbor >= header.node_count {
                    return Err(Error::InvalidFile("edge references an invalid node"));
                }
                if previous.is_some_and(|prev| neighbor <= prev) {
                    return Err(Error::InvalidFile("unsorted or duplicate edge list"));
                }
                previous = Some(neighbor);
            }
        }
    }
    Ok(())
}

fn read_table_u32(data: &[u8], range: &Range<usize>, index: usize) -> u32 {
    let offset = range.start + index * 4;
    read_u32(&data[offset..offset + 4])
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(Error::InvalidFile("section size overflow"))
    })
}

fn take_range(file_size: usize, offset: &mut usize, length: usize) -> Result<Range<usize>> {
    let end = offset
        .checked_add(length)
        .ok_or(Error::InvalidFile("section offset overflow"))?;
    if end > file_size {
        return Err(Error::InvalidFile("truncated data section"));
    }
    let range = *offset..end;
    *offset = end;
    Ok(range)
}

fn write_v3_migration(path: &Path, data: &[u8], mut header: Header) -> Result<PathBuf> {
    header.version = VERSION;
    header.reserved = [0; 24];
    header.reserved[..4].copy_from_slice(&crc32fast::hash(&data[HEADER_SIZE..]).to_le_bytes());
    write_temporary(path, |writer| {
        writer.write_all(&encode_header(&header))?;
        writer.write_all(&data[HEADER_SIZE..])?;
        Ok(())
    })
}

/// Writes `write` to `.{name}.tmp-{pid}-{nonce}` next to `path`, then
/// [`File::sync_all`]s. The destination is unchanged until
/// [`commit_temporary`].
fn write_temporary(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<PathBuf> {
    let temporary = temporary_path(path);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let write_result = (|| -> Result<()> {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(temporary)
}

fn commit_temporary(temporary: PathBuf, path: &Path) -> Result<()> {
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn replace_file(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    let temporary = write_temporary(path, write)?;
    commit_temporary(temporary, path)
}

fn temporary_path(path: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("index.hnsw");
    path.with_file_name(format!(".{name}.tmp-{}-{nonce}", std::process::id()))
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes.try_into().expect("two-byte integer"))
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte integer"))
}

// Miri does not emulate OS-backed memory mapping; the safe core tests still
// run under Miri while persistence is covered on both CI operating systems.
#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    fn fixture_index() -> HnswIndex {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            max_level: 2,
            level_mult: 0.0,
            rng_seed: Some(1),
            ..Config::default()
        })
        .unwrap();
        index.insert(100, &[1.0, 0.0]).unwrap();
        index.set_level_mult(1.0).unwrap();
        index.insert(200, &[0.0, 1.0]).unwrap();
        index.insert(300, &[-1.0, 0.0]).unwrap();
        index
    }

    fn temporary_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hnsw-rs-{name}-{}-{}.hnsw",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn refresh_crc(bytes: &mut [u8]) {
        let crc = crc32fast::hash(&bytes[HEADER_SIZE..]);
        bytes[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    }

    #[test]
    fn header_offsets_match_v3_wire_format() {
        let mut reserved = [0_u8; 24];
        reserved[..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            dim: 384,
            node_count: 7,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_level: 16,
            // Historical default; still a valid stored value.
            level_mult: 0.5,
            entry_point: 3,
            entry_level: 2,
            layer_count: 4,
            reserved,
        };
        let bytes = encode_header(&header);
        assert_eq!(bytes.len(), 64);
        assert_eq!(read_u16(&bytes[14..16]), 200);
        assert_eq!(
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            0.5_f64.to_bits()
        );
        assert_eq!(read_u32(&bytes[CRC_OFFSET..CRC_OFFSET + 4]), 0x1234_5678);
        assert_eq!(decode_header(&bytes), header);
    }

    #[test]
    fn save_and_load_round_trip() {
        let path = temporary_file("round-trip");
        let index = fixture_index();
        save_file(&index, &path).unwrap();
        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.header().version, VERSION);
        assert_eq!(loaded.header().node_count, 3);
        for node in 0..index.len() as NodeIndex {
            assert_eq!(loaded.node(node), index.node(node));
            assert_eq!(
                loaded.vector(node).unwrap().iter().collect::<Vec<_>>(),
                index.vector_store().get(node)
            );
            for level in 0..index.layer_count() {
                assert_eq!(
                    loaded.edges(level, node).iter().collect::<Vec<_>>(),
                    index.edges(level, node)
                );
            }
        }
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    fn four_node_index(ef_search: u16) -> HnswIndex {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            ef_search,
            rng_seed: Some(1),
            ..Config::default()
        })
        .unwrap();
        for (id, vector) in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
            .iter()
            .enumerate()
        {
            index.insert(id as u32, vector).unwrap();
        }
        index
    }

    #[test]
    fn save_load_search_matches_live_index() {
        let path = temporary_file("search-round-trip");
        let index = fixture_index();
        let query = [0.9998_f32, 0.02];
        let live = index.search(&query, 3).unwrap();
        index.save(&path).unwrap();

        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.search(&query, 3).unwrap(), live);
        assert_eq!(loaded.search(&query, 1).unwrap(), live[..1]);
        assert!(loaded.search(&query, 0).unwrap().is_empty());
        assert!(matches!(
            loaded.search(&[1.0], 1),
            Err(Error::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn loaded_search_returns_k_when_larger_than_ef_search() {
        let path = temporary_file("search-k-gt-ef");
        let index = four_node_index(2);
        let live = index.search(&[1.0, 0.0], 4).unwrap();
        assert_eq!(live.len(), 4);
        index.save(&path).unwrap();

        let loaded = LoadedHnsw::open(&path).unwrap();
        assert_eq!(loaded.search(&[1.0, 0.0], 4).unwrap().len(), 4);
        assert_eq!(loaded.search(&[1.0, 0.0], 4).unwrap(), live);
        assert_eq!(loaded.search_with_ef(&[1.0, 0.0], 4, 2).unwrap(), live);
        assert_eq!(loaded.search_with_ef(&[1.0, 0.0], 1, 2).unwrap().len(), 1);
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn empty_snapshot_search_is_empty() {
        let path = temporary_file("empty-search");
        let index = HnswIndex::new(Config {
            dim: 2,
            rng_seed: Some(1),
            ..Config::default()
        })
        .unwrap();
        index.save(&path).unwrap();
        let loaded = load_file(&path).unwrap();
        assert!(loaded.is_empty());
        assert!(loaded.search(&[1.0, 0.0], 5).unwrap().is_empty());
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn migrated_v2_snapshot_is_searchable() {
        let path = temporary_file("migrated-search");
        let index = fixture_index();
        let expected = index.search(&[0.0, 1.0], 2).unwrap();
        save_file(&index, &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[4..6].copy_from_slice(&MIGRATABLE_VERSION.to_le_bytes());
        bytes[CRC_OFFSET..CRC_OFFSET + 24].fill(0);
        fs::write(&path, bytes).unwrap();
        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.header().version, VERSION);
        assert_eq!(loaded.search(&[0.0, 1.0], 2).unwrap(), expected);
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn save_load_search_matches_live_ranking() {
        let path = temporary_file("search-ranking");
        let mut index = HnswIndex::new(Config {
            dim: 4,
            rng_seed: Some(1234),
            ..Config::default()
        })
        .unwrap();
        for (id, vector) in [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
        .iter()
        .enumerate()
        {
            index.insert(id as u32, vector).unwrap();
        }
        let query = [1.0, 0.0, 0.0, 0.0];
        let live = index.search(&query, 4).unwrap();
        index.save(&path).unwrap();
        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.search(&query, 4).unwrap(), live);
        assert_eq!(live.first().unwrap().id, 0);
        assert_eq!(live.last().unwrap().id, 3);
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn vector_view_copy_into_rejects_length_mismatch() {
        let path = temporary_file("copy-into");
        save_file(&fixture_index(), &path).unwrap();
        let loaded = load_file(&path).unwrap();
        let view = loaded.vector(0).unwrap();
        let mut dest = [0.0_f32; 3];
        assert!(view.copy_into(&mut dest).is_none());
        let mut dest = [0.0_f32; 2];
        assert_eq!(view.copy_into(&mut dest), Some(()));
        assert_eq!(dest, [1.0, 0.0]);
        drop(loaded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn corrupted_data_fails_crc() {
        let path = temporary_file("crc");
        save_file(&fixture_index(), &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[HEADER_SIZE] ^= 0xff;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(load_file(&path), Err(Error::CrcMismatch { .. })));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn valid_v2_is_migrated_to_v3() {
        let path = temporary_file("migration");
        save_file(&fixture_index(), &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[4..6].copy_from_slice(&MIGRATABLE_VERSION.to_le_bytes());
        bytes[CRC_OFFSET..CRC_OFFSET + 24].fill(0);
        fs::write(&path, bytes).unwrap();
        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.header().version, VERSION);
        drop(loaded);
        let migrated = fs::read(&path).unwrap();
        assert_eq!(read_u16(&migrated[4..6]), VERSION);
        assert_eq!(
            read_u32(&migrated[CRC_OFFSET..CRC_OFFSET + 4]),
            crc32fast::hash(&migrated[HEADER_SIZE..])
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unsupported_versions_report_both_versions() {
        let path = temporary_file("version");
        save_file(&fixture_index(), &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[4..6].copy_from_slice(&99_u16.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            load_file(&path),
            Err(Error::UnsupportedVersion {
                expected: VERSION,
                actual: 99
            })
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_magic_and_truncated_sections_are_rejected() {
        let magic_path = temporary_file("magic");
        save_file(&fixture_index(), &magic_path).unwrap();
        let mut bytes = fs::read(&magic_path).unwrap();
        bytes[..4].fill(0);
        fs::write(&magic_path, bytes).unwrap();
        assert!(matches!(load_file(&magic_path), Err(Error::InvalidMagic)));
        fs::remove_file(magic_path).unwrap();

        let truncated_path = temporary_file("truncated");
        save_file(&fixture_index(), &truncated_path).unwrap();
        let mut bytes = fs::read(&truncated_path).unwrap();
        bytes.truncate(HEADER_SIZE + 1);
        fs::write(&truncated_path, bytes).unwrap();
        assert!(matches!(
            load_file(&truncated_path),
            Err(Error::InvalidFile(_))
        ));
        fs::remove_file(truncated_path).unwrap();
    }

    #[test]
    fn zero_m_header_is_rejected() {
        let path = temporary_file("zero-m");
        save_file(&fixture_index(), &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[12] = 0;
        refresh_crc(&mut bytes);
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            load_file(&path),
            Err(Error::InvalidFile("invalid neighbor count"))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn zero_ef_header_is_rejected() {
        let construction_path = temporary_file("zero-ef-construction");
        save_file(&fixture_index(), &construction_path).unwrap();
        let mut bytes = fs::read(&construction_path).unwrap();
        bytes[14..16].copy_from_slice(&0_u16.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&construction_path, bytes).unwrap();
        assert!(matches!(
            load_file(&construction_path),
            Err(Error::InvalidFile("invalid candidate search width"))
        ));
        fs::remove_file(construction_path).unwrap();

        let search_path = temporary_file("zero-ef-search");
        save_file(&fixture_index(), &search_path).unwrap();
        let mut bytes = fs::read(&search_path).unwrap();
        bytes[16..18].copy_from_slice(&0_u16.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&search_path, bytes).unwrap();
        assert!(matches!(
            load_file(&search_path),
            Err(Error::InvalidFile("invalid candidate search width"))
        ));
        fs::remove_file(search_path).unwrap();
    }

    fn node_section_start(header: &Header) -> usize {
        HEADER_SIZE + header.node_count as usize * usize::from(header.dim) * 4
    }

    fn first_multi_edge_list(bytes: &[u8]) -> Range<usize> {
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let node_count = header.node_count as usize;
        let slots = node_count * usize::from(header.layer_count);
        let offsets = node_section_start(&header) + node_count * NODE_META_SIZE;
        let lengths = offsets + slots * 4;
        let edges = lengths + slots * 4;
        for slot in 0..slots {
            let length = read_u32(&bytes[lengths + slot * 4..lengths + slot * 4 + 4]) as usize;
            if length >= 2 {
                let offset = read_u32(&bytes[offsets + slot * 4..offsets + slot * 4 + 4]) as usize;
                let start = edges + offset * 4;
                return start..start + length * 4;
            }
        }
        panic!("writer-produced fixture should contain a multi-edge list");
    }

    #[test]
    fn entry_point_below_entry_level_is_rejected() {
        let path = temporary_file("low-entry");
        save_file(&fixture_index(), &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        assert!(header.entry_level > 0);
        let nodes = node_section_start(&header);
        let entry = nodes + header.entry_point as usize * NODE_META_SIZE;
        bytes[entry + 4] = header.entry_level.saturating_sub(1);
        refresh_crc(&mut bytes);
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            load_file(&path),
            Err(Error::InvalidFile(
                "entry point does not exist at entry_level"
            ))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn duplicate_external_ids_are_rejected() {
        let path = temporary_file("duplicate-id");
        save_file(&fixture_index(), &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = node_section_start(&header);
        let first_id = read_u32(&bytes[nodes..nodes + 4]);
        bytes[nodes + NODE_META_SIZE..nodes + NODE_META_SIZE + 4]
            .copy_from_slice(&first_id.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            HnswIndex::load(&path),
            Err(Error::InvalidFile("duplicate external id"))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn overlapping_or_gapped_vector_offsets_are_rejected() {
        let overlap_path = temporary_file("vector-overlap");
        save_file(&fixture_index(), &overlap_path).unwrap();
        let mut bytes = fs::read(&overlap_path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = node_section_start(&header);
        bytes[nodes + NODE_META_SIZE + 8..nodes + NODE_META_SIZE + 12]
            .copy_from_slice(&0_u32.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&overlap_path, bytes).unwrap();
        assert!(matches!(
            load_file(&overlap_path),
            Err(Error::InvalidFile("overlapping or gapped vector offsets"))
        ));
        fs::remove_file(overlap_path).unwrap();

        let gap_path = temporary_file("vector-gap");
        save_file(&fixture_index(), &gap_path).unwrap();
        let mut bytes = fs::read(&gap_path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = node_section_start(&header);
        bytes[nodes + NODE_META_SIZE + 8..nodes + NODE_META_SIZE + 12]
            .copy_from_slice(&1_u32.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&gap_path, bytes).unwrap();
        assert!(matches!(
            load_file(&gap_path),
            Err(Error::InvalidFile("overlapping or gapped vector offsets"))
        ));
        fs::remove_file(gap_path).unwrap();
    }

    #[test]
    fn unsorted_or_duplicate_edge_lists_are_rejected() {
        let unsorted_path = temporary_file("unsorted-edges");
        save_file(&fixture_index(), &unsorted_path).unwrap();
        let mut bytes = fs::read(&unsorted_path).unwrap();
        let list = first_multi_edge_list(&bytes);
        let first = bytes[list.start..list.start + 4].to_vec();
        let second = bytes[list.start + 4..list.start + 8].to_vec();
        bytes[list.start..list.start + 4].copy_from_slice(&second);
        bytes[list.start + 4..list.start + 8].copy_from_slice(&first);
        refresh_crc(&mut bytes);
        fs::write(&unsorted_path, bytes).unwrap();
        assert!(matches!(
            load_file(&unsorted_path),
            Err(Error::InvalidFile("unsorted or duplicate edge list"))
        ));
        fs::remove_file(unsorted_path).unwrap();

        let duplicate_path = temporary_file("duplicate-edges");
        save_file(&fixture_index(), &duplicate_path).unwrap();
        let mut bytes = fs::read(&duplicate_path).unwrap();
        let list = first_multi_edge_list(&bytes);
        bytes.copy_within(list.start..list.start + 4, list.start + 4);
        refresh_crc(&mut bytes);
        fs::write(&duplicate_path, bytes).unwrap();
        assert!(matches!(
            load_file(&duplicate_path),
            Err(Error::InvalidFile("unsorted or duplicate edge list"))
        ));
        fs::remove_file(duplicate_path).unwrap();
    }

    #[test]
    fn invalid_node_metadata_is_rejected_after_crc_verification() {
        let level_path = temporary_file("node-level");
        save_file(&fixture_index(), &level_path).unwrap();
        let mut bytes = fs::read(&level_path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = HEADER_SIZE + header.node_count as usize * usize::from(header.dim) * 4;
        bytes[nodes + 4] = header.layer_count;
        refresh_crc(&mut bytes);
        fs::write(&level_path, bytes).unwrap();
        assert!(matches!(load_file(&level_path), Err(Error::InvalidFile(_))));
        fs::remove_file(level_path).unwrap();

        let offset_path = temporary_file("vector-offset");
        save_file(&fixture_index(), &offset_path).unwrap();
        let mut bytes = fs::read(&offset_path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = HEADER_SIZE + header.node_count as usize * usize::from(header.dim) * 4;
        bytes[nodes + 8..nodes + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&offset_path, bytes).unwrap();
        assert!(matches!(
            load_file(&offset_path),
            Err(Error::InvalidFile(_))
        ));
        fs::remove_file(offset_path).unwrap();
    }

    #[test]
    fn invalid_edge_spans_and_levels_are_rejected() {
        let span_path = temporary_file("edge-span");
        save_file(&fixture_index(), &span_path).unwrap();
        let mut bytes = fs::read(&span_path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = HEADER_SIZE + header.node_count as usize * usize::from(header.dim) * 4;
        let offsets = nodes + header.node_count as usize * NODE_META_SIZE;
        bytes[offsets..offsets + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&span_path, bytes).unwrap();
        assert!(matches!(load_file(&span_path), Err(Error::InvalidFile(_))));
        fs::remove_file(span_path).unwrap();

        let level_path = temporary_file("edge-level");
        save_file(&fixture_index(), &level_path).unwrap();
        let mut bytes = fs::read(&level_path).unwrap();
        let header = decode_header(&bytes[..HEADER_SIZE]);
        let nodes = HEADER_SIZE + header.node_count as usize * usize::from(header.dim) * 4;
        let slots = header.node_count as usize * usize::from(header.layer_count);
        let offsets = nodes + header.node_count as usize * NODE_META_SIZE;
        let lengths = offsets + slots * 4;
        let slot = header.node_count as usize + 1;
        bytes[offsets + slot * 4..offsets + slot * 4 + 4].copy_from_slice(&0_u32.to_le_bytes());
        bytes[lengths + slot * 4..lengths + slot * 4 + 4].copy_from_slice(&1_u32.to_le_bytes());
        refresh_crc(&mut bytes);
        fs::write(&level_path, bytes).unwrap();
        assert!(matches!(load_file(&level_path), Err(Error::InvalidFile(_))));
        fs::remove_file(level_path).unwrap();
    }

    fn sibling_temps(path: &Path) -> Vec<PathBuf> {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("index.hnsw");
        let prefix = format!(".{name}.tmp-");
        let directory = path.parent().unwrap_or(Path::new("."));
        let mut temps = fs::read_dir(directory)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|candidate| {
                candidate
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&prefix))
            })
            .collect::<Vec<_>>();
        temps.sort();
        temps
    }

    #[test]
    fn failed_temporary_write_leaves_destination_and_cleans_temp() {
        let path = temporary_file("atomic-failed-write");
        save_file(&fixture_index(), &path).unwrap();
        let original = fs::read(&path).unwrap();

        let error = write_temporary(&path, |writer| {
            writer.write_all(b"partial")?;
            Err(Error::InvalidFile("injected write failure"))
        })
        .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidFile("injected write failure")
        ));
        assert_eq!(fs::read(&path).unwrap(), original);
        assert!(sibling_temps(&path).is_empty());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn destination_is_unchanged_until_rename() {
        let path = temporary_file("atomic-until-rename");
        save_file(&fixture_index(), &path).unwrap();
        let original = fs::read(&path).unwrap();

        let temporary = write_temporary(&path, |writer| {
            writer.write_all(b"not-yet-committed")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(
            sibling_temps(&path).as_slice(),
            std::slice::from_ref(&temporary)
        );
        assert_eq!(fs::read(&temporary).unwrap(), b"not-yet-committed");

        commit_temporary(temporary, &path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"not-yet-committed");
        assert!(sibling_temps(&path).is_empty());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn leftover_temp_does_not_block_atomic_save() {
        let path = temporary_file("atomic-leftover");
        let first = fixture_index();
        save_file(&first, &path).unwrap();
        let leftover = path.with_file_name(format!(
            ".{}.tmp-1-1",
            path.file_name().and_then(|name| name.to_str()).unwrap()
        ));
        fs::write(&leftover, b"crashed mid-write").unwrap();

        let mut replacement = HnswIndex::new(Config {
            dim: 2,
            rng_seed: Some(2),
            ..Config::default()
        })
        .unwrap();
        replacement.insert(1, &[0.0, 1.0]).unwrap();
        save_file(&replacement, &path).unwrap();

        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.header().node_count, 1);
        assert_eq!(loaded.node(0).unwrap().external_id, 1);
        drop(loaded);
        assert_eq!(
            sibling_temps(&path).as_slice(),
            std::slice::from_ref(&leftover)
        );
        assert_eq!(fs::read(&leftover).unwrap(), b"crashed mid-write");
        fs::remove_file(leftover).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn successful_save_leaves_no_temporary_files() {
        let path = temporary_file("atomic-no-leftover");
        save_file(&fixture_index(), &path).unwrap();
        save_file(&fixture_index(), &path).unwrap();
        assert!(sibling_temps(&path).is_empty());
        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.header().node_count, 3);
        drop(loaded);
        fs::remove_file(path).unwrap();
    }
}
