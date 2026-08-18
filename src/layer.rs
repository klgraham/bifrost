use std::cmp::Ordering;

use crate::{Error, Graph, NodeIndex, NodeMeta, Result, vector::cosine_distance};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub node_index: NodeIndex,
    pub distance: f32,
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
        Ok(cosine_distance(self.get(node_index), query))
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
            cosine_distance(store.get(chosen.node_index), vector) >= candidate.distance
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
    search_layer_excluding(graph, store, entry_point, level, query, ef, None)
}

pub(crate) fn search_layer_excluding<G, S>(
    graph: &G,
    store: &S,
    entry_point: NodeIndex,
    level: u8,
    query: &[f32],
    ef: u32,
    excluded: Option<NodeIndex>,
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
    let mut visited = vec![false; graph.node_count() as usize];
    let mut frontier = Vec::new();
    let mut results = Vec::new();
    let Some(entry_slot) = visited.get_mut(entry_point as usize) else {
        return Err(Error::InvalidNode(entry_point));
    };
    *entry_slot = true;

    if excluded != Some(entry_point) {
        let entry = Candidate {
            node_index: entry_point,
            distance: distance_to_node(store, entry_point, query)?,
        };
        frontier.push(entry);
        insert_bounded(&mut results, entry, ef);
    }

    while !frontier.is_empty() {
        frontier.sort_by(|left, right| compare_candidates(right, left));
        let current = frontier.pop().expect("frontier is non-empty");
        if results.len() >= ef
            && compare_candidates(
                &current,
                results.last().expect("full result set is non-empty"),
            ) == Ordering::Greater
        {
            break;
        }

        for neighbor in graph.neighbors(level, current.node_index) {
            let Some(seen) = visited.get_mut(neighbor as usize) else {
                continue;
            };
            if *seen {
                continue;
            }
            *seen = true;
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
                frontier.push(candidate);
                insert_bounded(&mut results, candidate, ef);
            }
        }
    }

    results.sort_by(compare_candidates);
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

    let mut level = entry_level;
    while level > 0 {
        entry_point = search_layer(graph, store, entry_point, level, query, 1)?.nearest;
        level -= 1;
    }

    // HNSW / hnswlib expand the layer-0 candidate list to at least k so a
    // caller asking for more neighbors than `ef_search` still receives them.
    let requested = u32::try_from(k).unwrap_or(u32::MAX);
    let ef = u32::from(ef_search).max(requested);
    let result = search_layer(graph, store, entry_point, 0, query, ef)?;
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
                distance: cosine_distance(store.get(node), &query),
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
        let result = search_layer_excluding(&graph, &store, 0, 0, &[1.0], 3, Some(2)).unwrap();
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.node_index != 2)
        );
    }
}
