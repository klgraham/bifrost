use std::{collections::HashMap, path::Path};

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    Config, Error, ExternalId, Graph, LoadedHnsw, NodeIndex, NodeMeta, Result,
    layer::{
        Candidate, SearchGraph, VectorStore, search_knn, search_layer, search_layer_excluding,
        select_neighbors_heuristic,
    },
    vector::cosine_distance_unchecked,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    pub id: ExternalId,
    pub distance: f32,
}

/// Mutable HNSW index supporting incremental insertion and nearest-neighbor search.
///
/// Persist a built index with [`HnswIndex::save`] and later reopen it for
/// query-only search with [`HnswIndex::load`]. Construction parameters such as
/// [`Config::m`] and [`Config::dim`] are fixed at [`HnswIndex::new`]; use
/// [`HnswIndex::set_ef_search`] to change the stored query candidate width, or
/// [`HnswIndex::search_with_ef`] to override it for one query. Search uses
/// `max(ef_search, k)` so a request larger than the stored width still returns
/// up to `k` hits. Insert and search `debug_assert` finite, near-unit vectors;
/// [`Config::check_vectors`] / [`HnswIndex::set_check_vectors`] make those
/// failures [`Error::InvalidVector`]. [`HnswIndex::build`] assigns IDs
/// `0..n-1` and is a convenience for an empty index; a second `build` or any
/// colliding ID returns [`Error::DuplicateExternalId`]. The graph
/// is inspectable through read-only accessors and cannot be replaced. Neighbor
/// lists are chosen with the paper / hnswlib diversity heuristic. After
/// reverse-link pruning, adjacency may be directed: a dropped outgoing edge is
/// not removed from the peer.
///
/// # Examples
///
/// Query-time candidate width can be changed after construction:
///
/// ```
/// # use hnsw_rs::{Config, HnswIndex};
/// let mut index = HnswIndex::new(Config {
///     dim: 2,
///     rng_seed: Some(1),
///     ..Config::default()
/// })?;
/// index.insert(0, &[1.0, 0.0])?;
/// index.set_ef_search(8)?;
/// assert_eq!(index.config().ef_search, 8);
/// let _ = index.search(&[1.0, 0.0], 1)?;
/// # Ok::<(), hnsw_rs::Error>(())
/// ```
///
/// Replacing the graph or mutating `dim` through the old public fields does
/// not compile:
///
/// ```compile_fail
/// # use hnsw_rs::{Config, Graph, HnswIndex};
/// let mut index = HnswIndex::new(Config {
///     dim: 2,
///     ..Config::default()
/// })
/// .unwrap();
/// index.graph = Graph::new();
/// ```
///
/// ```compile_fail
/// # use hnsw_rs::{Config, HnswIndex};
/// let mut index = HnswIndex::new(Config {
///     dim: 2,
///     ..Config::default()
/// })
/// .unwrap();
/// index.config.dim = 8;
/// ```
#[derive(Debug)]
pub struct HnswIndex {
    config: Config,
    graph: Graph,
    pub(crate) vector_data: Vec<f32>,
    pub(crate) vector_offsets: Vec<u32>,
    external_to_internal: HashMap<ExternalId, NodeIndex>,
    pub(crate) entry_point: Option<NodeIndex>,
    pub(crate) entry_level: u8,
    rng: StdRng,
    #[cfg(test)]
    fail_insert_after_append: bool,
    #[cfg(test)]
    fail_insert_after_first_link: bool,
}

pub(crate) fn hits_from_candidates<G: SearchGraph>(
    graph: &G,
    candidates: Vec<Candidate>,
) -> Vec<SearchHit> {
    let mut hits = candidates
        .into_iter()
        .map(|candidate| {
            let node = graph
                .node(candidate.node_index)
                .expect("search candidates refer to existing nodes");
            SearchHit {
                id: node.external_id,
                distance: candidate.distance,
            }
        })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits
}

impl HnswIndex {
    pub fn new(config: Config) -> Result<Self> {
        let config = config.validate()?;
        Ok(Self {
            config,
            graph: Graph::new(),
            vector_data: Vec::new(),
            vector_offsets: Vec::new(),
            external_to_internal: HashMap::new(),
            entry_point: None,
            entry_level: 0,
            rng: config
                .rng_seed
                .map_or_else(rand::make_rng, StdRng::seed_from_u64),
            #[cfg(test)]
            fail_insert_after_append: false,
            #[cfg(test)]
            fail_insert_after_first_link: false,
        })
    }

    /// Construction and search parameters captured at [`HnswIndex::new`].
    ///
    /// The returned value is a copy; mutating it does not change the index.
    /// [`Config::m`] and other construction caps stay fixed so reverse-link
    /// pruning cannot be retargeted mid-build.
    #[must_use]
    pub fn config(&self) -> Config {
        self.config
    }

    /// Sets the layer-0 candidate width used by [`HnswIndex::search`].
    ///
    /// `ef_search` must be greater than zero. [`HnswIndex::search_with_ef`]
    /// overrides this value for a single query without mutating the stored
    /// config. Both methods still search with `max(ef, k)`.
    pub fn set_ef_search(&mut self, ef_search: u16) -> Result<()> {
        if ef_search == 0 {
            return Err(Error::InvalidConfig("ef_search must be greater than zero"));
        }
        self.config.ef_search = ef_search;
        Ok(())
    }

    /// Enables or disables Result-returning vector checks on insert and search.
    ///
    /// Debug builds assert finiteness and near-unit norm regardless. See
    /// [`Config::check_vectors`].
    pub fn set_check_vectors(&mut self, check_vectors: bool) {
        self.config.check_vectors = check_vectors;
    }

    /// Sets the random-level stop probability used by later [`HnswIndex::insert`]
    /// calls. Construction caps such as [`Config::m`] are unchanged.
    pub fn set_level_mult(&mut self, level_mult: f64) -> Result<()> {
        if !level_mult.is_finite() || !(0.0..=1.0).contains(&level_mult) {
            return Err(Error::InvalidConfig(
                "level_mult must be finite and between zero and one",
            ));
        }
        self.config.level_mult = level_mult;
        Ok(())
    }

    /// Inserts a normalized vector associated with a caller-facing external ID.
    ///
    /// The slice must match [`Config::dim`]. Debug builds assert that every
    /// coordinate is finite and that `||v||` is within
    /// [`crate::vector::UNIT_NORM_TOLERANCE`] of `1`. When
    /// [`Config::check_vectors`] is set, the same failures return
    /// [`Error::InvalidVector`] instead of being accepted in release builds.
    ///
    /// New nodes keep up to [`Config::new_node_neighbors`] links chosen with
    /// the paper / hnswlib diversity heuristic. Each reverse link is then
    /// pruned with the same selector, so a dropped outgoing edge can remain as
    /// an incoming edge on the peer.
    ///
    /// Insertion is transactional. If greedy descent, linking, or reverse-link
    /// pruning returns an error, the caller can retry the same ID: the node is
    /// not visible to [`HnswIndex::search`], and adjacency matches the
    /// pre-insert graph (including lists that prune had already rewritten).
    pub fn insert(&mut self, id: ExternalId, vector: &[f32]) -> Result<()> {
        self.check_vector(vector)?;
        if self.external_to_internal.contains_key(&id) {
            return Err(Error::DuplicateExternalId(id));
        }

        if self.graph.node_data.len() >= u32::MAX as usize {
            return Err(Error::CapacityExceeded("node count"));
        }
        let node_index = self.graph.node_data.len() as u32;
        let vector_offset = u32::try_from(self.vector_data.len())
            .map_err(|_| Error::CapacityExceeded("vector data"))?;
        self.vector_data
            .len()
            .checked_add(vector.len())
            .and_then(|length| u32::try_from(length).ok())
            .ok_or(Error::CapacityExceeded("vector data"))?;

        let level = random_level(&mut self.rng, self.config.max_level, self.config.level_mult);
        let mut rollback = InsertRollback::capture(self, id);
        match self.insert_committed(id, vector, node_index, level, vector_offset, &mut rollback) {
            Ok(()) => Ok(()),
            Err(error) => {
                rollback.restore(self);
                Err(error)
            }
        }
    }

    fn insert_committed(
        &mut self,
        id: ExternalId,
        vector: &[f32],
        node_index: NodeIndex,
        level: u8,
        vector_offset: u32,
        rollback: &mut InsertRollback,
    ) -> Result<()> {
        self.vector_data.extend_from_slice(vector);
        self.vector_offsets.push(vector_offset);
        self.graph
            .insert_node(node_index, id, level, vector_offset)?;

        #[cfg(test)]
        if self.fail_insert_after_append {
            self.fail_insert_after_append = false;
            return Err(Error::InvalidNode(node_index));
        }

        let Some(mut entry_point) = self.entry_point else {
            self.external_to_internal.insert(id, node_index);
            self.entry_point = Some(node_index);
            self.entry_level = level;
            return Ok(());
        };
        let previous_entry_level = self.entry_level;

        let mut current_level = previous_entry_level;
        while current_level > level {
            let result = {
                let store = self.vector_store();
                search_layer(&self.graph, &store, entry_point, current_level, vector, 1)?
            };
            entry_point = result.nearest;
            current_level -= 1;
        }

        current_level = level.min(previous_entry_level);
        loop {
            let required_entry_level = current_level.saturating_sub(1);
            entry_point = self.insert_layer(
                entry_point,
                node_index,
                current_level,
                required_entry_level,
                vector,
                rollback,
            )?;
            if current_level == 0 {
                break;
            }
            current_level -= 1;
        }

        self.external_to_internal.insert(id, node_index);
        if level > previous_entry_level {
            self.entry_point = Some(node_index);
            self.entry_level = level;
        }
        Ok(())
    }

    /// Searches for at most `k` nearest neighbors to a normalized query vector.
    ///
    /// Layer 0 uses a candidate width of `max(ef_search, k)`, matching HNSW /
    /// hnswlib, so asking for more hits than [`Config::ef_search`] still
    /// returns up to `k` results when the graph contains them. Call
    /// [`HnswIndex::search_with_ef`] to override the stored width for one query.
    /// Query vectors are checked the same way as [`HnswIndex::insert`].
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Searches using an explicit layer-0 candidate width.
    ///
    /// The search width is `max(ef, k)`. The stored [`Config::ef_search`] is
    /// unchanged; use [`HnswIndex::set_ef_search`] to persist a new default.
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: u16) -> Result<Vec<SearchHit>> {
        self.check_vector(query)?;
        let store = self.vector_store();
        let candidates = search_knn(
            &self.graph,
            &store,
            query,
            k,
            ef,
            self.entry_point,
            self.entry_level,
        )?;
        Ok(hits_from_candidates(&self.graph, candidates))
    }

    /// Inserts a batch and assigns dense external IDs starting at zero.
    ///
    /// This is a convenience for an **empty** index. The `i`th vector is
    /// inserted as ID `i`, so IDs are always `0..vectors.len()`. A second
    /// [`HnswIndex::build`], or `build` after any [`HnswIndex::insert`] that
    /// already used one of those IDs (including `insert(0, …)`), returns
    /// [`Error::DuplicateExternalId`]. When the collision is on the first
    /// assigned ID, the graph is unchanged. Use [`HnswIndex::insert`] to
    /// append with caller-chosen IDs.
    ///
    /// Vectors are checked the same way as [`HnswIndex::insert`].
    ///
    /// # Examples
    ///
    /// A second `build` is rejected and does not replace the existing graph:
    ///
    /// ```
    /// # use hnsw_rs::{Config, Error, HnswIndex};
    /// let mut index = HnswIndex::new(Config {
    ///     dim: 2,
    ///     rng_seed: Some(1),
    ///     ..Config::default()
    /// })?;
    /// index.build(&[&[1.0, 0.0], &[0.0, 1.0]])?;
    /// assert_eq!(index.len(), 2);
    /// assert!(matches!(
    ///     index.build(&[&[1.0, 0.0]]),
    ///     Err(Error::DuplicateExternalId(0))
    /// ));
    /// assert_eq!(index.len(), 2);
    /// # Ok::<(), hnsw_rs::Error>(())
    /// ```
    pub fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        for (index, vector) in vectors.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| Error::CapacityExceeded("external ID"))?;
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Writes a validated `.hnsw` snapshot that can be searched later without
    /// re-inserting vectors.
    ///
    /// The write uses a same-directory temporary file, `sync_all`, and
    /// `rename`, so a crash mid-save cannot truncate a previous good file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_file(self, path)
    }

    /// Memory-maps a previously saved `.hnsw` snapshot for query-only search.
    ///
    /// Prefer [`LoadedHnsw::open`] or [`crate::load_file`]; this is the same
    /// mapping constructor and does **not** rebuild a mutable [`HnswIndex`].
    /// Further inserts still require a live builder. Do not mutate the file
    /// while the returned mapping lives.
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedHnsw> {
        LoadedHnsw::open(path)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.node_data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.node_data.is_empty()
    }

    #[must_use]
    pub fn entry_point(&self) -> Option<NodeIndex> {
        self.entry_point
    }

    #[must_use]
    pub fn entry_level(&self) -> u8 {
        self.entry_level
    }

    /// Read-only view of the construction graph.
    ///
    /// The returned reference cannot replace the graph stored on this index.
    /// [`Graph`] mutation methods are crate-private, so external callers cannot
    /// desync adjacency from the vector store.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Number of allocated layers in the construction graph.
    #[must_use]
    pub fn layer_count(&self) -> u8 {
        self.graph.layer_count()
    }

    /// Metadata for a dense internal node, if it exists.
    #[must_use]
    pub fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
        self.graph.node(node_index)
    }

    /// Outgoing neighbors of `node_index` at `level`.
    #[must_use]
    pub fn edges(&self, level: u8, node_index: NodeIndex) -> &[NodeIndex] {
        self.graph.edges(level, node_index)
    }

    /// Whether a directed edge exists from `source` to `destination` at `level`.
    #[must_use]
    pub fn has_edge(&self, level: u8, source: NodeIndex, destination: NodeIndex) -> bool {
        self.graph.has_edge(level, source, destination)
    }

    /// Outgoing degree of `node_index` at `level`.
    #[must_use]
    pub fn degree(&self, level: u8, node_index: NodeIndex) -> usize {
        self.graph.edges(level, node_index).len()
    }

    fn check_dimension(&self, vector: &[f32]) -> Result<()> {
        let expected = usize::from(self.config.dim);
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
        crate::vector::validate_input_vector(vector, self.config.check_vectors)
    }

    pub(crate) fn vector_store(&self) -> VectorStore<'_> {
        VectorStore {
            data: &self.vector_data,
            offsets: &self.vector_offsets,
            dim: self.config.dim,
        }
    }

    fn insert_layer(
        &mut self,
        entry_point: NodeIndex,
        node_index: NodeIndex,
        level: u8,
        required_entry_level: u8,
        vector: &[f32],
        rollback: &mut InsertRollback,
    ) -> Result<NodeIndex> {
        let max_neighbors = usize::from(self.config.new_node_neighbors(level));
        let candidates = {
            let store = self.vector_store();
            search_layer_excluding(
                &self.graph,
                &store,
                entry_point,
                level,
                vector,
                u32::from(self.config.ef_construction),
                Some(node_index),
            )?
            .candidates
        };
        let selected = {
            let store = self.vector_store();
            select_neighbors_heuristic(&store, &candidates, max_neighbors)
        };
        for candidate in selected {
            rollback.record(&self.graph, level, node_index);
            rollback.record(&self.graph, level, candidate.node_index);
            self.graph
                .add_bidirectional_edge(level, node_index, candidate.node_index)?;
            self.prune_neighbors(level, candidate.node_index)?;
            #[cfg(test)]
            if self.fail_insert_after_first_link {
                self.fail_insert_after_first_link = false;
                return Err(Error::InvalidNode(candidate.node_index));
            }
        }
        Ok(select_entry_point_for_level(
            &self.graph,
            &candidates,
            entry_point,
            required_entry_level,
        ))
    }

    fn prune_neighbors(&mut self, level: u8, node_index: NodeIndex) -> Result<()> {
        let max_degree = self.config.max_degree(level);
        let neighbors = self.graph.edges(level, node_index);
        // Never shrink to empty: `m == 0` is rejected, but a zero cap would
        // otherwise wipe an upper-layer reverse list after adding one edge.
        if max_degree == 0 || neighbors.len() <= max_degree {
            return Ok(());
        }

        let neighbors = neighbors.to_vec();
        let store = self.vector_store();
        let query = store.get(node_index);
        let scored = neighbors
            .into_iter()
            .map(|neighbor| Candidate {
                node_index: neighbor,
                distance: cosine_distance_unchecked(store.get(neighbor), query),
            })
            .collect::<Vec<_>>();
        let kept = select_neighbors_heuristic(&store, &scored, max_degree)
            .into_iter()
            .map(|candidate| candidate.node_index);
        self.graph.set_edges(level, node_index, kept)
    }
}

/// Snapshots index state so a failed insert can restore adjacency, storage,
/// and the external-ID map after the node has already been appended.
struct InsertRollback {
    id: ExternalId,
    vector_len: usize,
    node_count: usize,
    layer_count: u8,
    entry_point: Option<NodeIndex>,
    entry_level: u8,
    previous_edges: HashMap<(u8, NodeIndex), Vec<NodeIndex>>,
}

impl InsertRollback {
    fn capture(index: &HnswIndex, id: ExternalId) -> Self {
        Self {
            id,
            vector_len: index.vector_data.len(),
            node_count: index.graph.node_data.len(),
            layer_count: index.graph.layer_count(),
            entry_point: index.entry_point,
            entry_level: index.entry_level,
            previous_edges: HashMap::new(),
        }
    }

    fn record(&mut self, graph: &Graph, level: u8, node: NodeIndex) {
        self.previous_edges
            .entry((level, node))
            .or_insert_with(|| graph.edges(level, node).to_vec());
    }

    fn restore(self, index: &mut HnswIndex) {
        for ((level, node), edges) in self.previous_edges {
            index.graph.replace_edges(level, node, edges);
        }
        if index.graph.node_data.len() > self.node_count {
            index.graph.pop_last_node(self.layer_count);
        }
        index.vector_data.truncate(self.vector_len);
        index.vector_offsets.truncate(self.node_count);
        index.external_to_internal.remove(&self.id);
        index.entry_point = self.entry_point;
        index.entry_level = self.entry_level;
    }
}

fn random_level(rng: &mut StdRng, max_level: u8, level_mult: f64) -> u8 {
    let mut level = 0;
    while level < max_level {
        if rng.random_bool(level_mult) {
            break;
        }
        level += 1;
    }
    level
}

fn select_entry_point_for_level(
    graph: &Graph,
    candidates: &[Candidate],
    fallback: NodeIndex,
    required_level: u8,
) -> NodeIndex {
    candidates
        .iter()
        .find_map(|candidate| {
            graph
                .node(candidate.node_index)
                .filter(|meta| meta.level >= required_level)
                .map(|_| candidate.node_index)
        })
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::cosine_distance_unchecked;

    fn config(dim: u16) -> Config {
        Config {
            dim,
            rng_seed: Some(1234),
            ..Config::default()
        }
    }

    fn unit_at(degrees: f32) -> [f32; 2] {
        let radians = degrees.to_radians();
        [radians.cos(), radians.sin()]
    }

    fn unit_diagonal() -> [f32; 2] {
        [
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ]
    }

    fn heuristic_index() -> HnswIndex {
        HnswIndex::new(Config {
            dim: 2,
            m: 2,
            ef_construction: 16,
            max_level: 1,
            level_mult: 1.0,
            rng_seed: Some(1),
            ..Config::default()
        })
        .unwrap()
    }

    #[test]
    fn insert_selects_neighbors_with_diversity_heuristic() {
        let mut index = heuristic_index();
        index.insert(0, &unit_at(5.0)).unwrap();
        index.insert(1, &unit_at(6.0)).unwrap();
        index.insert(2, &unit_at(-30.0)).unwrap();
        index.insert(3, &unit_at(0.0)).unwrap();
        assert_eq!(
            index.edges(0, 3),
            &[0, 2],
            "new node should keep diverse A and C, not nearest A and B"
        );
    }

    #[test]
    fn prune_selects_neighbors_with_the_same_heuristic() {
        let mut index = heuristic_index();
        index.insert(0, &unit_at(0.0)).unwrap();
        index.insert(1, &unit_at(5.0)).unwrap();
        index.insert(2, &unit_at(6.0)).unwrap();
        index.insert(3, &unit_at(7.0)).unwrap();
        index.insert(4, &unit_at(8.0)).unwrap();
        index.insert(5, &unit_at(-30.0)).unwrap();
        // Mmax0 = 4. After the sixth nearby insert the hub must shrink; the
        // shared Alg. 4 selector should prefer the opposite-side node over
        // keeping only the tight positive-side cluster.
        let hub = index.edges(0, 0);
        assert!(hub.len() <= index.config().max_degree(0));
        assert!(
            hub.contains(&5),
            "prune should keep diverse node 5; hub={hub:?}"
        );
    }

    #[test]
    fn insert_and_search() {
        let mut index = HnswIndex::new(config(4)).unwrap();
        index.insert(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 0.0, 1.0, 0.0]).unwrap();
        let results = index.search(&[0.9998, 0.02, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn build_batch_and_search() {
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let vectors: [&[f32]; 4] = [
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[diagonal, diagonal, 0.0],
        ];
        let mut index = HnswIndex::new(config(3)).unwrap();
        index.build(&vectors).unwrap();
        assert_eq!(
            index.search(&[diagonal, diagonal, 0.0], 2).unwrap().len(),
            2
        );
    }

    /// Documented contract: `build` always assigns `0..n-1`, so a second
    /// `build` or `build` after `insert(0, …)` is `DuplicateExternalId` and
    /// leaves the graph unchanged. Callers who need to append use `insert`.
    #[test]
    fn build_on_non_empty_index_rejects_colliding_ids() {
        let mut index = HnswIndex::new(config(3)).unwrap();
        index.insert(0, &[1.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            index.build(&[&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]),
            Err(Error::DuplicateExternalId(0))
        ));
        assert_eq!(index.len(), 1);
        assert_eq!(index.search(&[1.0, 0.0, 0.0], 1).unwrap()[0].id, 0);

        let mut built = HnswIndex::new(config(3)).unwrap();
        built.build(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]).unwrap();
        assert!(matches!(
            built.build(&[&[0.0, 0.0, 1.0]]),
            Err(Error::DuplicateExternalId(0))
        ));
        assert_eq!(built.len(), 2);
        assert_eq!(built.search(&[0.0, 1.0, 0.0], 1).unwrap()[0].id, 1);
    }

    #[test]
    fn empty_and_single_vector_indexes() {
        let mut index = HnswIndex::new(config(4)).unwrap();
        assert!(index.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap().is_empty());
        index.insert(7, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = index.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(result[0].id, 7);
        assert!(result[0].distance.abs() < 0.001);
    }

    #[test]
    fn cosine_distance_ranking_is_correct() {
        let mut index = HnswIndex::new(config(4)).unwrap();
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
        let result = index.search(&[1.0, 0.0, 0.0, 0.0], 4).unwrap();
        assert_eq!(result.first().unwrap().id, 0);
        assert_eq!(result.last().unwrap().id, 3);
    }

    #[test]
    fn sparse_and_duplicate_external_ids() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.insert(100, &[1.0, 0.0]).unwrap();
        index.insert(5_000, &[0.0, 1.0]).unwrap();
        assert_eq!(index.search(&[0.0, 1.0], 1).unwrap()[0].id, 5_000);
        assert!(matches!(
            index.insert(100, &[1.0, 0.0]),
            Err(Error::DuplicateExternalId(100))
        ));
    }

    #[test]
    fn seeded_indexes_have_identical_node_levels() {
        let mut left = HnswIndex::new(config(2)).unwrap();
        let mut right = HnswIndex::new(config(2)).unwrap();
        for (id, vector) in [[1.0, 0.0], [0.0, 1.0], unit_diagonal(), [-1.0, 0.0]]
            .iter()
            .enumerate()
        {
            left.insert(id as u32, vector).unwrap();
            right.insert(id as u32, vector).unwrap();
        }
        assert_eq!(left.len(), right.len());
        for node in 0..left.len() as NodeIndex {
            assert_eq!(left.node(node), right.node(node));
        }
    }

    #[test]
    fn level_multiplier_extremes_are_deterministic() {
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(random_level(&mut rng, 8, 0.0), 8);
        assert_eq!(random_level(&mut rng, 8, 1.0), 0);
    }

    #[test]
    fn popular_node_degree_stays_within_mmax() {
        let config = Config {
            dim: 2,
            m: 4,
            ef_construction: 32,
            max_level: 4,
            level_mult: 1.0,
            rng_seed: Some(1),
            ..Config::default()
        };
        let mut index = HnswIndex::new(config).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        for id in 1..=48 {
            let angle = 0.02 * id as f32;
            index.insert(id, &[angle.cos(), angle.sin()]).unwrap();
        }

        let max0 = config.max_degree(0);
        assert_eq!(max0, 8);
        let hub_neighbors = index.edges(0, 0);
        assert!(
            hub_neighbors.len() <= max0,
            "hub degree {} exceeded Mmax0 {max0}",
            hub_neighbors.len()
        );
        assert!(
            !hub_neighbors.is_empty(),
            "hub should remain connected after nearby inserts"
        );

        for node in 0..index.len() as NodeIndex {
            let meta = index.node(node).unwrap();
            for level in 0..=meta.level {
                let degree = index.degree(level, node);
                let cap = config.max_degree(level);
                assert!(
                    degree <= cap,
                    "node {node} level {level} degree {degree} exceeded {cap}"
                );
            }
        }

        #[cfg(not(miri))]
        {
            let path = std::env::temp_dir().join(format!(
                "hnsw-rs-hub-degree-{}-{}.hnsw",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            index.save(&path).unwrap();
            let loaded = LoadedHnsw::open(&path).unwrap();
            let loaded_hub: Vec<_> = loaded.edges(0, 0).iter().collect();
            assert_eq!(loaded_hub, hub_neighbors);
            assert!(loaded_hub.len() <= max0);
            drop(loaded);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn prune_does_not_remove_reverse_of_dropped_edge() {
        const DIM: usize = 8;
        let config = Config {
            dim: DIM as u16,
            m: 2,
            ef_construction: 32,
            max_level: 1,
            level_mult: 1.0,
            rng_seed: Some(1),
            ..Config::default()
        };
        let mut index = HnswIndex::new(config).unwrap();
        let mut hub = vec![0.0; DIM];
        hub[0] = 1.0;
        index.insert(0, &hub).unwrap();
        // Near-orthogonal perturbations keep every insert closer to the hub
        // than to the other leaves, so reverse edges concentrate on node 0.
        for id in 1..=12 {
            let mut vector = vec![0.0; DIM];
            vector[0] = 1.0;
            vector[1 + ((id as usize - 1) % (DIM - 1))] = 0.02 * id as f32;
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            for value in &mut vector {
                *value /= norm;
            }
            index.insert(id, &vector).unwrap();
        }

        let hub_neighbors = index.edges(0, 0);
        assert!(hub_neighbors.len() <= config.max_degree(0));
        let dropped = (1..index.len() as NodeIndex)
            .find(|&node| !hub_neighbors.contains(&node) && index.has_edge(0, node, 0));
        assert!(
            dropped.is_some(),
            "expected a dropped peer that still points at the hub; hub neighbors={hub_neighbors:?}"
        );
    }

    #[test]
    fn zero_m_is_rejected_and_m_one_does_not_wipe_neighbors() {
        assert!(matches!(
            HnswIndex::new(Config {
                dim: 2,
                m: 0,
                rng_seed: Some(1),
                ..Config::default()
            }),
            Err(Error::InvalidConfig(_))
        ));

        let config = Config {
            dim: 2,
            m: 1,
            ef_construction: 16,
            max_level: 1,
            level_mult: 0.0,
            rng_seed: Some(1),
            ..Config::default()
        };
        let mut index = HnswIndex::new(config).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        for id in 1..=12 {
            let angle = 0.1 * id as f32;
            index.insert(id, &[angle.cos(), angle.sin()]).unwrap();
        }

        let mut connected = 0;
        for node in 0..index.len() as NodeIndex {
            assert!(index.degree(0, node) <= config.max_degree(0));
            let upper = index.degree(1, node);
            assert!(upper <= config.max_degree(1));
            connected += usize::from(upper > 0);
        }
        assert!(
            connected >= 2,
            "m=1 reverse links should be kept, not wiped to empty"
        );
    }

    #[test]
    fn config_accessor_copy_cannot_change_prune_cap() {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            m: 4,
            level_mult: 1.0,
            rng_seed: Some(3),
            ..Config::default()
        })
        .unwrap();
        let mut snapshot = index.config();
        snapshot.m = 1;
        index.set_ef_search(8).unwrap();
        assert_eq!(snapshot.m, 1);
        assert_eq!(index.config().m, 4);
        assert_eq!(index.config().ef_search, 8);

        index.insert(0, &[1.0, 0.0]).unwrap();
        for id in 1..=20 {
            let angle = 0.04 * id as f32;
            index.insert(id, &[angle.cos(), angle.sin()]).unwrap();
        }
        assert!(index.degree(0, 0) <= 8);
        assert!(index.degree(0, 0) > 2);
    }

    #[test]
    fn upper_layer_degree_is_capped_at_m() {
        let config = Config {
            dim: 2,
            m: 4,
            ef_construction: 32,
            max_level: 1,
            level_mult: 0.0,
            rng_seed: Some(2),
            ..Config::default()
        };
        let mut index = HnswIndex::new(config).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        for id in 1..=40 {
            let angle = 0.03 * id as f32;
            index.insert(id, &[angle.cos(), angle.sin()]).unwrap();
        }

        let max1 = config.max_degree(1);
        assert_eq!(max1, 4);
        assert!(index.degree(1, 0) <= max1);
        for node in 0..index.len() as NodeIndex {
            assert!(index.degree(1, node) <= max1);
            assert!(index.degree(0, node) <= config.max_degree(0));
        }
    }

    #[test]
    fn graph_edges_respect_node_levels() {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            max_level: 4,
            rng_seed: Some(4),
            ..Config::default()
        })
        .unwrap();
        for (id, vector) in [[1.0, 0.0], [0.0, 1.0], unit_diagonal(), [-1.0, 0.0]]
            .iter()
            .enumerate()
        {
            index.insert(id as u32, vector).unwrap();
        }
        for level in 0..index.layer_count() {
            for node in 0..index.len() as NodeIndex {
                let meta = index.node(node).unwrap();
                if meta.level < level {
                    assert!(index.edges(level, node).is_empty());
                }
                for &neighbor in index.edges(level, node) {
                    assert!(index.node(neighbor).unwrap().level >= level);
                }
            }
        }
    }

    #[test]
    fn search_returns_k_when_larger_than_ef_search() {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            ef_search: 2,
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
        let hits = index.search(&[1.0, 0.0], 4).unwrap();
        assert_eq!(hits.len(), 4);
        assert_eq!(index.search_with_ef(&[1.0, 0.0], 4, 2).unwrap(), hits);
        assert_eq!(index.search_with_ef(&[1.0, 0.0], 1, 2).unwrap().len(), 1);
        assert_eq!(index.config().ef_search, 2);
    }

    #[test]
    fn set_ef_search_rejects_zero() {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            rng_seed: Some(1),
            ..Config::default()
        })
        .unwrap();
        assert!(matches!(
            HnswIndex::new(Config {
                dim: 2,
                ef_construction: 0,
                ..Config::default()
            }),
            Err(Error::InvalidConfig(_))
        ));
        assert!(matches!(
            HnswIndex::new(Config {
                dim: 2,
                ef_search: 0,
                ..Config::default()
            }),
            Err(Error::InvalidConfig(_))
        ));
        assert!(matches!(
            index.set_ef_search(0),
            Err(Error::InvalidConfig(_))
        ));
        assert_eq!(index.config().ef_search, Config::default().ef_search);
    }

    #[test]
    fn set_ef_search_applies_at_query_time() {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            ef_search: 1,
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
        assert_eq!(index.config().ef_search, 1);
        index.set_ef_search(4).unwrap();
        assert_eq!(index.config().ef_search, 4);
        let hits = index.search(&[1.0, 0.0], 4).unwrap();
        assert_eq!(hits.len(), 4);
        assert_eq!(hits[0].id, 0);
        assert_eq!(index.config().dim, 2);
        assert_eq!(index.graph().node_count(), index.len() as NodeIndex);
        assert_eq!(index.degree(0, 0), index.edges(0, 0).len());
    }

    #[test]
    fn dimension_mismatches_return_errors() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        assert!(matches!(
            index.insert(0, &[1.0]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.search(&[1.0], 1),
            Err(Error::DimensionMismatch { .. })
        ));
        index.set_check_vectors(true);
        assert!(matches!(
            index.insert(0, &[f32::NAN]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn check_vectors_rejects_non_finite_and_unnormalized() {
        let mut index = HnswIndex::new(Config {
            dim: 2,
            check_vectors: true,
            rng_seed: Some(1),
            ..Config::default()
        })
        .unwrap();
        assert!(matches!(
            index.insert(0, &[f32::NAN, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            index.insert(0, &[f32::INFINITY, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            index.insert(0, &[2.0, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(index.is_empty());

        index.insert(0, &[1.0, 0.0]).unwrap();
        assert_eq!(index.search(&[1.0, 0.0], 1).unwrap()[0].id, 0);
        index.insert(1, &[1.009, 0.0]).unwrap();
        assert!(matches!(
            index.search(&[f32::NAN, 0.0], 1),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            index.search(&[0.0, 0.0], 1),
            Err(Error::InvalidVector(_))
        ));
        assert_eq!(index.search(&[1.0, 0.0], 2).unwrap().len(), 2);

        index.set_check_vectors(false);
        assert!(!index.config().check_vectors);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "unit-normalized")]
    fn debug_builds_assert_unnormalized_insert() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        let _ = index.insert(0, &[2.0, 0.0]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "unit-normalized")]
    fn debug_builds_assert_non_finite_search() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        let _ = index.search(&[f32::NAN, 0.0], 1);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn release_builds_accept_unnormalized_unless_check_vectors() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.insert(0, &[2.0, 0.0]).unwrap();
        assert_eq!(index.search(&[2.0, 0.0], 1).unwrap()[0].id, 0);
        index.set_check_vectors(true);
        assert!(matches!(
            index.insert(1, &[2.0, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            index.search(&[f32::INFINITY, 0.0], 1),
            Err(Error::InvalidVector(_))
        ));
    }

    #[derive(Debug, PartialEq)]
    struct IndexSnapshot {
        len: usize,
        entry_point: Option<NodeIndex>,
        entry_level: u8,
        ids: Vec<ExternalId>,
        mapping: HashMap<ExternalId, NodeIndex>,
        edges: Vec<(u8, NodeIndex, Vec<NodeIndex>)>,
        vectors: Vec<f32>,
        offsets: Vec<u32>,
    }

    fn snapshot(index: &HnswIndex) -> IndexSnapshot {
        let edges = (0..index.len() as NodeIndex)
            .flat_map(|node| {
                let top = index.node(node).map_or(0, |meta| meta.level);
                (0..=top).map(move |level| (level, node, index.edges(level, node).to_vec()))
            })
            .collect();
        IndexSnapshot {
            len: index.len(),
            entry_point: index.entry_point,
            entry_level: index.entry_level,
            ids: index
                .graph
                .node_data
                .iter()
                .map(|node| node.external_id)
                .collect(),
            mapping: index.external_to_internal.clone(),
            edges,
            vectors: index.vector_data.clone(),
            offsets: index.vector_offsets.clone(),
        }
    }

    fn assert_id_absent(index: &HnswIndex, id: ExternalId) {
        assert!(!index.external_to_internal.contains_key(&id));
        assert!(
            index
                .graph
                .node_data
                .iter()
                .all(|node| node.external_id != id)
        );
        let hits = index.search(&[1.0, 0.0], index.len().max(1)).unwrap();
        assert!(hits.iter().all(|hit| hit.id != id));
    }

    #[test]
    fn failed_first_insert_leaves_empty_index() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.fail_insert_after_append = true;
        assert!(matches!(
            index.insert(1, &[1.0, 0.0]),
            Err(Error::InvalidNode(_))
        ));
        assert!(index.is_empty());
        assert!(index.entry_point.is_none());
        assert!(index.external_to_internal.is_empty());
        index.insert(1, &[1.0, 0.0]).unwrap();
        assert_eq!(index.search(&[1.0, 0.0], 1).unwrap()[0].id, 1);
    }

    #[test]
    fn failed_insert_after_append_can_be_retried() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        index.insert(1, &[0.0, 1.0]).unwrap();
        let before = snapshot(&index);

        index.fail_insert_after_append = true;
        assert!(matches!(
            index.insert(7, &unit_diagonal()),
            Err(Error::InvalidNode(_))
        ));
        assert_eq!(snapshot(&index), before);
        assert_id_absent(&index, 7);

        index.insert(7, &unit_diagonal()).unwrap();
        assert_eq!(index.search(&unit_diagonal(), 1).unwrap()[0].id, 7);
        assert_eq!(index.len(), before.len + 1);
    }

    #[test]
    fn failed_descent_rolls_back_appended_node() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        let before = snapshot(&index);
        index.entry_point = Some(99);
        assert!(matches!(
            index.insert(1, &[0.0, 1.0]),
            Err(Error::InvalidNode(99))
        ));
        assert_eq!(index.len(), before.len);
        assert_eq!(index.vector_data, before.vectors);
        assert_eq!(index.external_to_internal, before.mapping);
        assert!(!index.external_to_internal.contains_key(&1));
        assert!(
            index
                .graph
                .node_data
                .iter()
                .all(|node| node.external_id != 1)
        );
        assert_eq!(index.entry_point, Some(99));

        index.entry_point = before.entry_point;
        index.insert(1, &[0.0, 1.0]).unwrap();
        assert_eq!(index.search(&[0.0, 1.0], 1).unwrap()[0].id, 1);
    }

    #[test]
    fn failed_insert_after_prune_restores_adjacency() {
        let mut index = heuristic_index();
        index.insert(0, &unit_at(0.0)).unwrap();
        index.insert(1, &unit_at(5.0)).unwrap();
        index.insert(2, &unit_at(6.0)).unwrap();
        index.insert(3, &unit_at(7.0)).unwrap();
        index.insert(4, &unit_at(8.0)).unwrap();
        let before = snapshot(&index);

        index.fail_insert_after_first_link = true;
        assert!(matches!(
            index.insert(9, &unit_at(-30.0)),
            Err(Error::InvalidNode(_))
        ));
        assert_eq!(snapshot(&index), before);
        assert_id_absent(&index, 9);

        index.insert(9, &unit_at(-30.0)).unwrap();
        assert!(
            index
                .search(&unit_at(-30.0), 1)
                .unwrap()
                .iter()
                .any(|hit| hit.id == 9)
        );
        assert!(index.degree(0, 0) <= index.config().max_degree(0));
    }

    #[test]
    fn failed_reverse_edge_does_not_leave_one_sided_link() {
        let mut index = HnswIndex::new(config(2)).unwrap();
        index.insert(0, &[1.0, 0.0]).unwrap();
        index.insert(1, &[0.0, 1.0]).unwrap();
        let before = snapshot(&index);

        index.graph.fail_next_add_edge_from(0);
        assert!(matches!(
            index.insert(2, &unit_diagonal()),
            Err(Error::InvalidNode(0))
        ));
        assert_eq!(snapshot(&index), before);
        assert_id_absent(&index, 2);
        assert!(!index.has_edge(0, 0, 2));
        assert!(!index.has_edge(0, 2, 0));

        index.insert(2, &unit_diagonal()).unwrap();
        assert_eq!(index.search(&unit_diagonal(), 1).unwrap()[0].id, 2);
    }

    fn deterministic_unit_vector(seed: usize, dim: usize) -> Vec<f32> {
        let mut vector = (0..dim)
            .map(|dimension| {
                let value = ((seed + 1) * (dimension + 3)) as f32;
                (value * 0.173).sin() + (value * 0.071).cos()
            })
            .collect::<Vec<_>>();
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        for value in &mut vector {
            *value /= norm;
        }
        vector
    }

    fn recall_at_k(config: Config, count: usize, query_count: usize, k: usize) -> f32 {
        let dim = usize::from(config.dim);
        let vectors = (0..count)
            .map(|seed| deterministic_unit_vector(seed, dim))
            .collect::<Vec<_>>();
        let mut index = HnswIndex::new(config).unwrap();
        for (id, vector) in vectors.iter().enumerate() {
            index.insert(id as u32, vector).unwrap();
        }

        let mut recalled = 0;
        for query_index in 0..query_count {
            let query = deterministic_unit_vector(10_000 + query_index, dim);
            let approximate = index
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|hit| hit.id)
                .collect::<Vec<_>>();
            let mut exact = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| (id as u32, cosine_distance_unchecked(vector, &query)))
                .collect::<Vec<_>>();
            exact.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            recalled += exact
                .iter()
                .take(k)
                .filter(|(id, _)| approximate.contains(id))
                .count();
        }
        recalled as f32 / (query_count * k) as f32
    }

    #[test]
    fn deterministic_recall_at_ten_exceeds_baseline() {
        let recall = recall_at_k(
            Config {
                dim: 16,
                rng_seed: Some(7),
                ..Config::default()
            },
            256,
            20,
            10,
        );
        assert!(recall >= 0.95, "recall@10 regressed to {recall:.3}");
    }

    #[cfg(not(miri))]
    #[test]
    fn larger_seeded_graph_keeps_high_recall_at_ten() {
        // Default 256×16-d / M=16 / ef=100 still scores 1.0. This larger
        // graph with M=8 and ef_search=32 scores ~0.96, so the 0.90 floor
        // actually moves when neighbor selection or connectivity regresses.
        let recall = recall_at_k(
            Config {
                dim: 32,
                m: 8,
                ef_construction: 64,
                ef_search: 32,
                level_mult: Config::level_mult_for_m(8),
                rng_seed: Some(7),
                ..Config::default()
            },
            1024,
            32,
            10,
        );
        assert!(recall >= 0.90, "recall@10 regressed to {recall:.3}");
    }
}
