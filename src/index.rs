use std::{collections::HashMap, path::Path};

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    Config, Error, ExternalId, Graph, NodeIndex, Result,
    layer::{Candidate, VectorStore, search_layer, search_layer_excluding},
    vector::cosine_distance,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    pub id: ExternalId,
    pub distance: f32,
}

/// Mutable HNSW index supporting incremental insertion and nearest-neighbor search.
#[derive(Debug)]
pub struct HnswIndex {
    pub config: Config,
    pub graph: Graph,
    pub(crate) vector_data: Vec<f32>,
    pub(crate) vector_offsets: Vec<u32>,
    external_to_internal: HashMap<ExternalId, NodeIndex>,
    pub(crate) entry_point: Option<NodeIndex>,
    pub(crate) entry_level: u8,
    rng: StdRng,
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

    /// Inserts a normalized vector associated with a caller-facing external ID.
    pub fn insert(&mut self, id: ExternalId, vector: &[f32]) -> Result<()> {
        self.config.validate()?;
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
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.check_dimension(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let Some(mut entry_point) = self.entry_point else {
            return Ok(Vec::new());
        };
        let store = self.vector_store();

        let mut level = self.entry_level;
        while level > 0 {
            entry_point = search_layer(&self.graph, &store, entry_point, level, query, 1)?.nearest;
            level -= 1;
        }

        let result = search_layer(
            &self.graph,
            &store,
            entry_point,
            0,
            query,
            u32::from(self.config.ef_search),
        )?;
        let mut hits = result
            .candidates
            .into_iter()
            .take(k)
            .map(|candidate| {
                let node = self
                    .graph
                    .node(candidate.node_index)
                    .expect("search candidates refer to existing nodes");
                SearchHit {
                    id: node.external_id,
                    distance: cosine_distance(store.get(candidate.node_index), query),
                }
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(hits)
    }

    /// Inserts a batch and assigns dense external IDs starting at zero.
    pub fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        for (index, vector) in vectors.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| Error::CapacityExceeded("external ID"))?;
            self.insert(id, vector)?;
        }
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_file(self, path)
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
        let max_neighbors = if level == 0 {
            self.config.m
        } else {
            (self.config.m / 2).max(1)
        };
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

        for candidate in candidates.iter().take(usize::from(max_neighbors)) {
            self.graph
                .add_bidirectional_edge(level, node_index, candidate.node_index)?;
        }
        Ok(select_entry_point_for_level(
            &self.graph,
            &candidates,
            entry_point,
            required_entry_level,
        ))
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
