use std::{collections::HashMap, path::Path};

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    Config, Error, ExternalId, Graph, LoadedHnsw, NodeIndex, Result,
    layer::{
        Candidate, SearchGraph, VectorStore, search_knn, search_layer, search_layer_excluding,
        select_closest,
    },
    vector::cosine_distance,
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
/// [`Config::m`] are fixed at [`HnswIndex::new`]; use [`HnswIndex::set_ef_search`]
/// to change the query candidate width. After reverse-link pruning, adjacency
/// may be directed: a dropped outgoing edge is not removed from the peer.
#[derive(Debug)]
pub struct HnswIndex {
    config: Config,
    pub graph: Graph,
    pub(crate) vector_data: Vec<f32>,
    pub(crate) vector_offsets: Vec<u32>,
    external_to_internal: HashMap<ExternalId, NodeIndex>,
    pub(crate) entry_point: Option<NodeIndex>,
    pub(crate) entry_level: u8,
    rng: StdRng,
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
    pub fn set_ef_search(&mut self, ef_search: u16) {
        self.config.ef_search = ef_search;
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
    /// New nodes take the nearest [`Config::new_node_neighbors`] links. Each
    /// reverse link is then pruned on the neighbor only, so a dropped outgoing
    /// edge can remain as an incoming edge on the peer.
    pub fn insert(&mut self, id: ExternalId, vector: &[f32]) -> Result<()> {
        self.check_dimension(vector)?;
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
        self.vector_data.extend_from_slice(vector);
        self.vector_offsets.push(vector_offset);
        self.graph
            .insert_node(node_index, id, level, vector_offset)?;
        self.external_to_internal.insert(id, node_index);

        let Some(mut entry_point) = self.entry_point else {
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
            )?;
            if current_level == 0 {
                break;
            }
            current_level -= 1;
        }

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
    /// returns up to `k` results when the graph contains them.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.check_dimension(query)?;
        let store = self.vector_store();
        let candidates = search_knn(
            &self.graph,
            &store,
            query,
            k,
            self.config.ef_search,
            self.entry_point,
            self.entry_level,
        )?;
        Ok(hits_from_candidates(&self.graph, candidates))
    }

    /// Inserts a batch and assigns dense external IDs starting at zero.
    pub fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        for (index, vector) in vectors.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| Error::CapacityExceeded("external ID"))?;
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Writes a validated `.hnsw` snapshot that can be searched later without
    /// re-inserting vectors.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_file(self, path)
    }

    /// Memory-maps a previously saved `.hnsw` snapshot for query-only search.
    ///
    /// Prefer [`LoadedHnsw::open`] or [`crate::load_file`]; this is the same
    /// mapping constructor and does **not** rebuild a mutable [`HnswIndex`].
    /// Further inserts still require a live builder.
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
        for candidate in candidates.iter().take(max_neighbors) {
            self.graph
                .add_bidirectional_edge(level, node_index, candidate.node_index)?;
            self.prune_neighbors(level, candidate.node_index)?;
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
                distance: cosine_distance(store.get(neighbor), query),
            })
            .collect();
        let kept = select_closest(scored, max_degree)
            .into_iter()
            .map(|candidate| candidate.node_index);
        self.graph.set_edges(level, node_index, kept)
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
    use crate::vector::cosine_distance;

    fn config(dim: u16) -> Config {
        Config {
            dim,
            rng_seed: Some(1234),
            ..Config::default()
        }
    }

    #[test]
    fn insert_and_search() {
        let mut index = HnswIndex::new(config(4)).unwrap();
        index.insert(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 0.0, 1.0, 0.0]).unwrap();
        let results = index.search(&[0.9, 0.1, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn build_batch_and_search() {
        let vectors: [&[f32]; 4] = [
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0],
        ];
        let mut index = HnswIndex::new(config(3)).unwrap();
        index.build(&vectors).unwrap();
        assert_eq!(index.search(&[0.9, 0.9, 0.0], 2).unwrap().len(), 2);
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
        for (id, vector) in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
            .iter()
            .enumerate()
        {
            left.insert(id as u32, vector).unwrap();
            right.insert(id as u32, vector).unwrap();
        }
        assert_eq!(left.graph.node_data, right.graph.node_data);
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
        let hub_neighbors = index.graph.edges(0, 0);
        assert!(
            hub_neighbors.len() <= max0,
            "hub degree {} exceeded Mmax0 {max0}",
            hub_neighbors.len()
        );
        assert!(
            hub_neighbors.len() >= usize::from(config.new_node_neighbors(0)),
            "hub should remain connected after nearby inserts"
        );

        let store = index.vector_store();
        let hub = store.get(0);
        let farthest_kept = hub_neighbors
            .iter()
            .map(|&neighbor| cosine_distance(store.get(neighbor), hub))
            .max_by(|left, right| left.total_cmp(right))
            .unwrap();
        for node in 1..index.graph.node_count() {
            if hub_neighbors.contains(&node) || !index.graph.has_edge(0, node, 0) {
                continue;
            }
            assert!(
                cosine_distance(store.get(node), hub) >= farthest_kept,
                "dropped peer {node} is closer than a kept hub neighbor"
            );
        }

        for node in 0..index.graph.node_count() {
            let meta = index.graph.node(node).unwrap();
            for level in 0..=meta.level {
                let degree = index.graph.edges(level, node).len();
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

        let hub_neighbors = index.graph.edges(0, 0);
        assert!(hub_neighbors.len() <= config.max_degree(0));
        let dropped = (1..index.graph.node_count())
            .find(|&node| !hub_neighbors.contains(&node) && index.graph.has_edge(0, node, 0));
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
        for node in 0..index.graph.node_count() {
            assert!(index.graph.edges(0, node).len() <= config.max_degree(0));
            let upper = index.graph.edges(1, node).len();
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
        index.set_ef_search(8);
        assert_eq!(snapshot.m, 1);
        assert_eq!(index.config().m, 4);
        assert_eq!(index.config().ef_search, 8);

        index.insert(0, &[1.0, 0.0]).unwrap();
        for id in 1..=20 {
            let angle = 0.04 * id as f32;
            index.insert(id, &[angle.cos(), angle.sin()]).unwrap();
        }
        assert!(index.graph.edges(0, 0).len() <= 8);
        assert!(index.graph.edges(0, 0).len() > 2);
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
        assert!(index.graph.edges(1, 0).len() <= max1);
        for node in 0..index.graph.node_count() {
            assert!(index.graph.edges(1, node).len() <= max1);
            assert!(index.graph.edges(0, node).len() <= config.max_degree(0));
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
        for (id, vector) in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
            .iter()
            .enumerate()
        {
            index.insert(id as u32, vector).unwrap();
        }
        for level in 0..index.graph.layer_count() {
            for node in 0..index.graph.node_count() {
                let meta = index.graph.node(node).unwrap();
                if meta.level < level {
                    assert!(index.graph.edges(level, node).is_empty());
                }
                for &neighbor in index.graph.edges(level, node) {
                    assert!(index.graph.node(neighbor).unwrap().level >= level);
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

    #[test]
    fn deterministic_recall_at_ten_exceeds_baseline() {
        const DIM: usize = 16;
        const COUNT: usize = 256;
        const QUERY_COUNT: usize = 20;
        const K: usize = 10;

        let vectors = (0..COUNT)
            .map(|seed| deterministic_unit_vector(seed, DIM))
            .collect::<Vec<_>>();
        let mut index = HnswIndex::new(Config {
            dim: DIM as u16,
            rng_seed: Some(7),
            ..Config::default()
        })
        .unwrap();
        for (id, vector) in vectors.iter().enumerate() {
            index.insert(id as u32, vector).unwrap();
        }

        let mut recalled = 0;
        for query_index in 0..QUERY_COUNT {
            let query = deterministic_unit_vector(10_000 + query_index, DIM);
            let approximate = index
                .search(&query, K)
                .unwrap()
                .into_iter()
                .map(|hit| hit.id)
                .collect::<Vec<_>>();
            let mut exact = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| (id as u32, cosine_distance(vector, &query)))
                .collect::<Vec<_>>();
            exact.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            recalled += exact
                .iter()
                .take(K)
                .filter(|(id, _)| approximate.contains(id))
                .count();
        }

        let recall = recalled as f32 / (QUERY_COUNT * K) as f32;
        assert!(recall >= 0.8, "recall@10 regressed to {recall:.3}");
    }
}
