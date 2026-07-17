use std::cmp::Ordering;

use crate::{Error, Graph, NodeIndex, Result, vector::cosine_distance};

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

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.node_index.cmp(&right.node_index))
}

fn distance_to_node(store: &VectorStore<'_>, node_index: NodeIndex, query: &[f32]) -> f32 {
    cosine_distance(store.get(node_index), query)
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

pub(crate) fn search_layer(
    graph: &Graph,
    store: &VectorStore<'_>,
    entry_point: NodeIndex,
    level: u8,
    query: &[f32],
    ef: u32,
) -> Result<SearchResult> {
    search_layer_excluding(graph, store, entry_point, level, query, ef, None)
}

pub(crate) fn search_layer_excluding(
    graph: &Graph,
    store: &VectorStore<'_>,
    entry_point: NodeIndex,
    level: u8,
    query: &[f32],
    ef: u32,
    excluded: Option<NodeIndex>,
) -> Result<SearchResult> {
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
    visited[entry_point as usize] = true;

    if excluded != Some(entry_point) {
        let entry = Candidate {
            node_index: entry_point,
            distance: distance_to_node(store, entry_point, query),
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

        for &neighbor in graph.edges(level, current.node_index) {
            if visited[neighbor as usize] {
                continue;
            }
            visited[neighbor as usize] = true;
            if excluded == Some(neighbor) {
                continue;
            }
            let Some(meta) = graph.node(neighbor) else {
                continue;
            };
            if meta.level < level {
                continue;
            }
            let candidate = Candidate {
                node_index: neighbor,
                distance: distance_to_node(store, neighbor, query),
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
