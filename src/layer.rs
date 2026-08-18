use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use crate::{Error, Graph, NodeIndex, NodeMeta, Result, vector::cosine_distance_unchecked};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub node_index: NodeIndex,
    pub distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        compare_candidates(self, other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_candidates(self, other)
    }
}

/// Generation-stamped visit marks reused across layers of one search or insert.
///
/// Incrementing the generation is an O(1) clear. Wrapping back to 0 fills
/// the stamp buffer and resumes at 1 so stale marks cannot collide.
#[derive(Default)]
pub(crate) struct VisitedList {
    stamps: Vec<u32>,
    generation: u32,
}

impl std::fmt::Debug for VisitedList {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VisitedList")
            .field("len", &self.stamps.len())
            .field("generation", &self.generation)
            .finish()
    }
}

impl VisitedList {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Grows to `node_count` if needed and starts a new generation.
    pub(crate) fn prepare(&mut self, node_count: usize) {
        if self.stamps.len() < node_count {
            self.stamps.resize(node_count, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
    }

    /// Marks `node` as visited for the current generation.
    ///
    /// Returns `None` when `node` is outside the stamp buffer (callers skip
    /// neighbors `>= visited.len()`, the mmap-safety check from loaded search).
    /// `Some(true)` means the node was already marked this generation.
    pub(crate) fn mark(&mut self, node: NodeIndex) -> Option<bool> {
        let slot = self.stamps.get_mut(node as usize)?;
        let seen = *slot == self.generation;
        *slot = self.generation;
        Some(seen)
    }
}

#[derive(Debug)]
pub(crate) struct SearchResult {
    pub nearest: NodeIndex,
    pub candidates: Vec<Candidate>,
}

/// Graph view used by layer search. Implemented by the mutable construction
/// graph and by the memory-mapped snapshot.
pub(crate) trait SearchGraph {
    fn node_count(&self) -> NodeIndex;
    fn node(&self, node_index: NodeIndex) -> Option<NodeMeta>;
    fn neighbors(&self, level: u8, node_index: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_;
}

/// Vector view used by layer search. Implemented by the owned store and by
/// mmap-backed decoded vectors.
pub(crate) trait SearchVectors {
    fn distance(&self, node_index: NodeIndex, query: &[f32]) -> Result<f32>;
}

pub(crate) struct VectorStore<'a> {
    pub data: &'a [f32],
    pub offsets: &'a [u32],
    pub dim: u16,
}

impl VectorStore<'_> {
    pub(crate) fn get(&self, node_index: NodeIndex) -> &[f32] {
        let start = self.offsets[node_index as usize] as usize;
        &self.data[start..start + usize::from(self.dim)]
    }
}

impl SearchGraph for Graph {
    fn node_count(&self) -> NodeIndex {
        Graph::node_count(self)
    }

    fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
        Graph::node(self, node_index)
    }

    fn neighbors(&self, level: u8, node_index: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        Graph::edges(self, level, node_index).iter().copied()
    }
}

impl SearchVectors for VectorStore<'_> {
    fn distance(&self, node_index: NodeIndex, query: &[f32]) -> Result<f32> {
        Ok(cosine_distance_unchecked(self.get(node_index), query))
    }
}

#[cfg(test)]
pub(crate) fn select_closest(mut candidates: Vec<Candidate>, max: usize) -> Vec<Candidate> {
    candidates.sort_by(compare_candidates);
    candidates.truncate(max);
    candidates
}

/// Malkov & Yashunin Alg. 4 / hnswlib `getNeighborsByHeuristic2`.
///
/// Candidates are considered nearest-first. A candidate is kept only when it
/// is at least as close to the query as to every already chosen neighbor, so
/// the selected set is more spatially diverse than raw nearest-`max`.
pub(crate) fn select_neighbors_heuristic(
    store: &VectorStore<'_>,
    candidates: &[Candidate],
    max: usize,
) -> Vec<Candidate> {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(compare_candidates);
    if max == 0 || ordered.is_empty() {
        return Vec::new();
    }
    if ordered.len() < max {
        return ordered;
    }

    let mut selected: Vec<Candidate> = Vec::with_capacity(max);
    for candidate in ordered {
        if selected.len() >= max {
            break;
        }
        let vector = store.get(candidate.node_index);
        let diverse = selected.iter().all(|chosen| {
            cosine_distance_unchecked(store.get(chosen.node_index), vector) >= candidate.distance
        });
        if diverse {
            selected.push(candidate);
        }
    }
    selected
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.node_index.cmp(&right.node_index))
}

fn distance_to_node<S: SearchVectors>(
    store: &S,
    node_index: NodeIndex,
    query: &[f32],
) -> Result<f32> {
    store.distance(node_index, query)
}

fn insert_bounded(results: &mut Vec<Candidate>, candidate: Candidate, ef: usize) {
    if results.len() >= ef
        && compare_candidates(
            &candidate,
            results.last().expect("non-empty bounded results"),
        ) != Ordering::Less
    {
        return;
    }
    let position = results
        .binary_search_by(|probe| compare_candidates(probe, &candidate))
        .unwrap_or_else(|position| position);
    results.insert(position, candidate);
    results.truncate(ef);
}

#[cfg(test)]
pub(crate) fn search_layer<G, S>(
    graph: &G,
    store: &S,
    entry_point: NodeIndex,
    level: u8,
    query: &[f32],
    ef: u32,
) -> Result<SearchResult>
where
    G: SearchGraph,
    S: SearchVectors,
{
    let mut visited = VisitedList::new();
    search_layer_excluding(
        graph,
        store,
        entry_point,
        level,
        query,
        ef,
        None,
        &mut visited,
    )
}

/// Greedy search at one layer. `visited` is prepared for this hop so a caller
/// can reuse the stamp buffer across layers. The frontier is a min-heap on
/// distance, then node index (the same order as the previous full-sort hop).
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_layer_excluding<G, S>(
    graph: &G,
    store: &S,
    entry_point: NodeIndex,
    level: u8,
    query: &[f32],
    ef: u32,
    excluded: Option<NodeIndex>,
    visited: &mut VisitedList,
) -> Result<SearchResult>
where
    G: SearchGraph,
    S: SearchVectors,
{
    let entry_meta = graph
        .node(entry_point)
        .ok_or(Error::InvalidNode(entry_point))?;
    if entry_meta.level < level {
        return Err(Error::InvalidLayer(level));
    }

    let ef = usize::try_from(ef.max(1)).expect("u32 fits usize on supported targets");
    visited.prepare(graph.node_count() as usize);
    let Some(seen) = visited.mark(entry_point) else {
        return Err(Error::InvalidNode(entry_point));
    };
    debug_assert!(!seen, "fresh generation must not have marked the entry");

    let mut frontier = BinaryHeap::with_capacity(ef);
    let mut results = Vec::new();

    if excluded != Some(entry_point) {
        let entry = Candidate {
            node_index: entry_point,
            distance: distance_to_node(store, entry_point, query)?,
        };
        frontier.push(Reverse(entry));
        insert_bounded(&mut results, entry, ef);
    }

    while let Some(Reverse(current)) = frontier.pop() {
        if results.len() >= ef
            && compare_candidates(
                &current,
                results.last().expect("full result set is non-empty"),
            ) == Ordering::Greater
        {
            break;
        }

        for neighbor in graph.neighbors(level, current.node_index) {
            // Neighbors outside the stamp buffer are skipped (mmap safety).
            let Some(seen) = visited.mark(neighbor) else {
                continue;
            };
            if seen {
                continue;
            }
            if excluded == Some(neighbor) {
                continue;
            }
            let Some(meta) = graph.node(neighbor) else {
                continue;
            };
            if meta.level < level {
                continue;
            }
            let Ok(distance) = distance_to_node(store, neighbor, query) else {
                continue;
            };
            let candidate = Candidate {
                node_index: neighbor,
                distance,
            };
            if results.len() < ef
                || compare_candidates(
                    &candidate,
                    results.last().expect("non-empty bounded results"),
                ) == Ordering::Less
            {
                frontier.push(Reverse(candidate));
                insert_bounded(&mut results, candidate, ef);
            }
        }
    }

    debug_assert!(
        results.windows(2).all(|pair| pair[0] <= pair[1]),
        "insert_bounded keeps results nearest-first with node-index ties"
    );
    let nearest = results
        .first()
        .map_or(entry_point, |candidate| candidate.node_index);
    Ok(SearchResult {
        nearest,
        candidates: results,
    })
}

/// Greedy multi-layer descent followed by a layer-0 candidate search.
pub(crate) fn search_knn<G, S>(
    graph: &G,
    store: &S,
    query: &[f32],
    k: usize,
    ef_search: u16,
    entry_point: Option<NodeIndex>,
    entry_level: u8,
) -> Result<Vec<Candidate>>
where
    G: SearchGraph,
    S: SearchVectors,
{
    if k == 0 {
        return Ok(Vec::new());
    }
    let Some(mut entry_point) = entry_point else {
        return Ok(Vec::new());
    };

    let mut visited = VisitedList::new();
    let mut level = entry_level;
    while level > 0 {
        entry_point = search_layer_excluding(
            graph,
            store,
            entry_point,
            level,
            query,
            1,
            None,
            &mut visited,
        )?
        .nearest;
        level -= 1;
    }

    // HNSW / hnswlib expand the layer-0 candidate list to at least k so a
    // caller asking for more neighbors than `ef_search` still receives them.
    let requested = u32::try_from(k).unwrap_or(u32::MAX);
    let ef = u32::from(ef_search).max(requested);
    let result =
        search_layer_excluding(graph, store, entry_point, 0, query, ef, None, &mut visited)?;
    Ok(result.candidates.into_iter().take(k).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_nodes(levels: &[u8]) -> Graph {
        let mut graph = Graph::new();
        for (index, &level) in levels.iter().enumerate() {
            graph
                .insert_node(index as u32, index as u32 + 10, level, index as u32)
                .unwrap();
        }
        graph
    }

    fn angle_vector(degrees: f32) -> [f32; 2] {
        let radians = degrees.to_radians();
        [radians.cos(), radians.sin()]
    }

    #[test]
    fn select_closest_keeps_nearest_neighbors() {
        let candidates = vec![
            Candidate {
                node_index: 3,
                distance: 0.4,
            },
            Candidate {
                node_index: 1,
                distance: 0.1,
            },
            Candidate {
                node_index: 2,
                distance: 0.1,
            },
            Candidate {
                node_index: 4,
                distance: 0.9,
            },
        ];
        let kept = select_closest(candidates, 2);
        assert_eq!(
            kept.iter()
                .map(|candidate| candidate.node_index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn heuristic_keeps_a_more_diverse_set_than_nearest_m() {
        let query = angle_vector(0.0);
        let vectors = [angle_vector(5.0), angle_vector(6.0), angle_vector(-30.0)];
        let data = vectors.iter().flatten().copied().collect::<Vec<_>>();
        let offsets = [0_u32, 2, 4];
        let store = VectorStore {
            data: &data,
            offsets: &offsets,
            dim: 2,
        };
        let candidates = (0..3)
            .map(|node| Candidate {
                node_index: node,
                distance: cosine_distance_unchecked(store.get(node), &query),
            })
            .collect::<Vec<_>>();

        assert_eq!(
            select_closest(candidates.clone(), 2)
                .iter()
                .map(|candidate| candidate.node_index)
                .collect::<Vec<_>>(),
            vec![0, 1],
            "Alg. 3 keeps the two nearest (A, B)"
        );
        assert_eq!(
            select_neighbors_heuristic(&store, &candidates, 2)
                .iter()
                .map(|candidate| candidate.node_index)
                .collect::<Vec<_>>(),
            vec![0, 2],
            "Alg. 4 keeps the diverse pair (A, C)"
        );
    }

    #[test]
    fn search_stays_in_entry_component() {
        let mut graph = graph_with_nodes(&[0, 0, 0]);
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        let data = [1.0, 0.5, 0.0];
        let store = VectorStore {
            data: &data,
            offsets: &[0, 1, 2],
            dim: 1,
        };
        let result = search_layer(&graph, &store, 0, 0, &[1.0], 3).unwrap();
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.node_index != 2)
        );
    }

    #[test]
    fn greedy_descent_follows_improving_edges() {
        let mut graph = graph_with_nodes(&[0, 0, 0]);
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        graph.add_bidirectional_edge(0, 1, 2).unwrap();
        let data = [-1.0, 0.5, 1.0];
        let store = VectorStore {
            data: &data,
            offsets: &[0, 1, 2],
            dim: 1,
        };
        let result = search_layer(&graph, &store, 0, 0, &[1.0], 1).unwrap();
        assert_eq!(result.nearest, 2);
        assert_eq!(result.candidates.len(), 1);
    }

    #[test]
    fn out_of_range_neighbors_are_skipped() {
        struct OversizedGraph(Graph);

        impl SearchGraph for OversizedGraph {
            fn node_count(&self) -> NodeIndex {
                self.0.node_count()
            }

            fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
                self.0.node(node_index)
            }

            fn neighbors(
                &self,
                level: u8,
                node_index: NodeIndex,
            ) -> impl Iterator<Item = NodeIndex> + '_ {
                self.0
                    .edges(level, node_index)
                    .iter()
                    .copied()
                    .chain(std::iter::once(u32::MAX))
            }
        }

        let mut graph = graph_with_nodes(&[0, 0]);
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        let data = [1.0, 0.0];
        let store = VectorStore {
            data: &data,
            offsets: &[0, 1],
            dim: 1,
        };
        let result = search_layer(&OversizedGraph(graph), &store, 0, 0, &[1.0], 2).unwrap();
        assert_eq!(result.candidates.len(), 2);
    }

    #[test]
    fn search_knn_returns_k_when_larger_than_ef_search() {
        let mut graph = graph_with_nodes(&[0, 0, 0, 0]);
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        graph.add_bidirectional_edge(0, 1, 2).unwrap();
        graph.add_bidirectional_edge(0, 2, 3).unwrap();
        let data = [1.0, 0.5, 0.0, -0.5];
        let store = VectorStore {
            data: &data,
            offsets: &[0, 1, 2, 3],
            dim: 1,
        };
        let candidates = search_knn(&graph, &store, &[1.0], 4, 2, Some(0), 0).unwrap();
        assert_eq!(candidates.len(), 4);
    }

    #[test]
    fn visited_list_marks_once_and_resets_on_wrap() {
        let mut visited = VisitedList::new();
        visited.prepare(2);
        assert_eq!(visited.mark(0), Some(false));
        assert_eq!(visited.mark(0), Some(true));
        assert_eq!(visited.mark(1), Some(false));
        assert_eq!(visited.mark(99), None);

        visited.prepare(2);
        assert_eq!(visited.mark(0), Some(false));
        assert_eq!(visited.mark(1), Some(false));

        visited.generation = u32::MAX;
        visited.prepare(2);
        assert_eq!(visited.generation, 1);
        assert!(visited.stamps.iter().all(|stamp| *stamp == 0));
        assert_eq!(visited.mark(0), Some(false));
        assert_eq!(visited.mark(0), Some(true));
    }

    #[test]
    fn search_breaks_distance_ties_by_node_index() {
        let mut graph = graph_with_nodes(&[0, 0, 0]);
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        graph.add_bidirectional_edge(0, 0, 2).unwrap();
        let data = [0.0, 1.0, 1.0];
        let store = VectorStore {
            data: &data,
            offsets: &[0, 1, 2],
            dim: 1,
        };
        let result = search_layer(&graph, &store, 0, 0, &[1.0], 3).unwrap();
        let ids = result
            .candidates
            .iter()
            .map(|candidate| candidate.node_index)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 2, 0]);
        assert_eq!(result.candidates[0].distance, result.candidates[1].distance);
    }

    #[test]
    fn excluded_node_is_not_returned() {
        let mut graph = graph_with_nodes(&[0, 0, 0]);
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        graph.add_bidirectional_edge(0, 1, 2).unwrap();
        let data = [1.0, 0.5, 0.0];
        let store = VectorStore {
            data: &data,
            offsets: &[0, 1, 2],
            dim: 1,
        };
        let mut visited = VisitedList::new();
        let result =
            search_layer_excluding(&graph, &store, 0, 0, &[1.0], 3, Some(2), &mut visited).unwrap();
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.node_index != 2)
        );
    }
}
