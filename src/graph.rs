use crate::{Error, Result};

/// Caller-facing identifier stored with each vector.
///
/// Chosen by [`crate::HnswIndex::insert`]. [`crate::HnswIndex::build`] assigns
/// dense values `0..n-1`. IDs stay unique for the life of the index; there is
/// no delete API that would free one.
pub type ExternalId = u32;

/// Dense internal node index, assigned in insertion order (`0..n-1`).
///
/// Used by [`Graph`] accessors and [`crate::HnswIndex::entry_point`]. Distinct
/// from [`ExternalId`].
pub type NodeIndex = u32;

/// Per-node metadata stored in the construction graph and on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMeta {
    /// Caller-facing ID for this node.
    pub external_id: ExternalId,
    /// Highest layer this node occupies. The node has adjacency slots on
    /// layers `0..=level`.
    pub level: u8,
    /// Start of the node's vector in the packed `f32` store, as an element
    /// index (not a byte offset). The on-disk layout uses the same unit:
    /// multiply by 4 to reach the vector section.
    pub vector_offset: u32,
}

/// Per-layer graph used while constructing an index.
///
/// Inspect a live index through [`crate::HnswIndex::graph`] or the index's
/// read-only accessors. The graph stored on an index cannot be replaced, and
/// mutation methods here are crate-private so external callers cannot desync
/// adjacency from vectors.
///
/// Adjacency lists are directed. Insertion adds both directed edges atomically
/// (a reverse-edge failure undoes the forward write), then reverse links are
/// pruned on the neighbor only, so a dropped `A → B` edge can leave `B → A`
/// in place.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub(crate) node_data: Vec<NodeMeta>,
    construction_layers: Vec<Vec<Vec<NodeIndex>>>,
    #[cfg(test)]
    fail_add_edge_from: Option<NodeIndex>,
}

#[cfg(test)]
impl Graph {
    pub(crate) fn fail_next_add_edge_from(&mut self, source: NodeIndex) {
        self.fail_add_edge_from = Some(source);
    }
}

impl Graph {
    /// Empty construction graph: no nodes and no layers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Converts a node-list length to [`NodeIndex`] without panicking.
    ///
    /// Insertion rejects a further node when `len` is already [`u32::MAX`], so
    /// a well-formed graph has at most that many nodes (`0..=u32::MAX - 1`)
    /// and [`Graph::node_count`] stays exact. If that cap is bypassed, the
    /// count saturates instead of panicking on the public accessor.
    #[must_use]
    pub(crate) fn node_index_from_len(len: usize) -> NodeIndex {
        u32::try_from(len).unwrap_or(u32::MAX)
    }

    /// `true` when another insert cannot keep the node count in [`u32`].
    ///
    /// `len` is the next [`NodeIndex`]. `len == u32::MAX` already holds
    /// indices `0..=u32::MAX - 1`; that further insert is rejected. Index
    /// [`u32::MAX`] is unused because the resulting count would be `2^32`,
    /// which does not fit in [`Graph::node_count`] or the on-disk header,
    /// and search sizes the visited list from that count so the extra node
    /// would be skipped.
    #[must_use]
    pub(crate) fn node_len_at_u32_cap(len: usize) -> bool {
        len >= u32::MAX as usize
    }

    fn check_insert_node(len: usize, node_index: NodeIndex) -> Result<()> {
        if Self::node_len_at_u32_cap(len) {
            return Err(Error::CapacityExceeded("node count"));
        }
        if node_index != Self::node_index_from_len(len) {
            return Err(Error::InvalidNode(node_index));
        }
        Ok(())
    }

    pub(crate) fn insert_node(
        &mut self,
        node_index: NodeIndex,
        external_id: ExternalId,
        level: u8,
        vector_offset: u32,
    ) -> Result<()> {
        Self::check_insert_node(self.node_data.len(), node_index)?;

        let old_node_count = self.node_data.len();
        while self.construction_layers.len() <= usize::from(level) {
            self.construction_layers
                .push(vec![Vec::new(); old_node_count]);
        }
        for layer in &mut self.construction_layers {
            layer.push(Vec::new());
        }
        self.node_data.push(NodeMeta {
            external_id,
            level,
            vector_offset,
        });
        Ok(())
    }

    /// Outgoing neighbors of `node_index` at `level`.
    ///
    /// Empty when the node is missing or `node.level < level`. The slice is
    /// sorted and unique. Adjacency is directed: reverse-link prune drops an
    /// outgoing edge on one node and leaves the peer's reverse edge in place,
    /// so `edges(level, a)` containing `b` does not imply the reverse.
    #[must_use]
    pub fn edges(&self, level: u8, node_index: NodeIndex) -> &[NodeIndex] {
        let Some(node) = self.node(node_index) else {
            return &[];
        };
        if node.level < level {
            return &[];
        }
        self.construction_layers
            .get(usize::from(level))
            .and_then(|layer| layer.get(node_index as usize))
            .map_or(&[], Vec::as_slice)
    }

    pub(crate) fn add_edge(
        &mut self,
        level: u8,
        source: NodeIndex,
        destination: NodeIndex,
    ) -> Result<bool> {
        let Some(source_meta) = self.node(source) else {
            return Err(Error::InvalidNode(source));
        };
        let Some(destination_meta) = self.node(destination) else {
            return Err(Error::InvalidNode(destination));
        };
        if source_meta.level < level || destination_meta.level < level {
            return Err(Error::InvalidLayer(level));
        }
        let Some(layer) = self.construction_layers.get_mut(usize::from(level)) else {
            return Err(Error::InvalidLayer(level));
        };
        #[cfg(test)]
        if self.fail_add_edge_from == Some(source) {
            self.fail_add_edge_from = None;
            return Err(Error::InvalidNode(source));
        }
        let edges = &mut layer[source as usize];
        Ok(match edges.binary_search(&destination) {
            Ok(_) => false,
            Err(position) => {
                edges.insert(position, destination);
                true
            }
        })
    }

    pub(crate) fn remove_edge(&mut self, level: u8, source: NodeIndex, destination: NodeIndex) {
        let Some(layer) = self.construction_layers.get_mut(usize::from(level)) else {
            return;
        };
        let Some(edges) = layer.get_mut(source as usize) else {
            return;
        };
        if let Ok(position) = edges.binary_search(&destination) {
            edges.remove(position);
        }
    }

    /// Adds both directed edges, or neither if the reverse write fails.
    pub(crate) fn add_bidirectional_edge(
        &mut self,
        level: u8,
        left: NodeIndex,
        right: NodeIndex,
    ) -> Result<()> {
        let added_forward = self.add_edge(level, left, right)?;
        match self.add_edge(level, right, left) {
            Ok(_) => Ok(()),
            Err(error) => {
                if added_forward {
                    self.remove_edge(level, left, right);
                }
                Err(error)
            }
        }
    }

    /// Replaces an adjacency list without re-validating neighbors.
    ///
    /// Used to restore a snapshot taken before insert mutations. `edges` must
    /// already be sorted and unique.
    pub(crate) fn replace_edges(&mut self, level: u8, source: NodeIndex, edges: Vec<NodeIndex>) {
        let Some(layer) = self.construction_layers.get_mut(usize::from(level)) else {
            return;
        };
        let Some(slot) = layer.get_mut(source as usize) else {
            return;
        };
        *slot = edges;
    }

    /// Drops the most recently inserted node and any layers allocated for it.
    pub(crate) fn pop_last_node(&mut self, previous_layer_count: u8) {
        let _ = self.node_data.pop();
        for layer in &mut self.construction_layers {
            let _ = layer.pop();
        }
        self.construction_layers
            .truncate(usize::from(previous_layer_count));
    }

    pub(crate) fn set_edges(
        &mut self,
        level: u8,
        source: NodeIndex,
        neighbors: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<()> {
        let Some(source_meta) = self.node(source) else {
            return Err(Error::InvalidNode(source));
        };
        if source_meta.level < level {
            return Err(Error::InvalidLayer(level));
        }
        if usize::from(level) >= self.construction_layers.len() {
            return Err(Error::InvalidLayer(level));
        }

        let mut edges = neighbors.into_iter().collect::<Vec<_>>();
        edges.sort_unstable();
        edges.dedup();
        for &destination in &edges {
            let Some(destination_meta) = self.node(destination) else {
                return Err(Error::InvalidNode(destination));
            };
            if destination_meta.level < level {
                return Err(Error::InvalidLayer(level));
            }
        }
        self.construction_layers[usize::from(level)][source as usize] = edges;
        Ok(())
    }

    /// Number of nodes in the construction graph.
    ///
    /// After a successful insert the value is exact and at most [`u32::MAX`]
    /// (indices `0..=u32::MAX - 1`). If the `u32` cap is bypassed, this
    /// saturates at [`u32::MAX`] instead of panicking.
    #[must_use]
    pub fn node_count(&self) -> NodeIndex {
        Self::node_index_from_len(self.node_data.len())
    }

    /// Number of allocated layers (highest assigned node level plus one).
    ///
    /// `0` on an empty graph.
    #[must_use]
    pub fn layer_count(&self) -> u8 {
        u8::try_from(self.construction_layers.len())
            .expect("layer count cannot exceed the u8 node level range")
    }

    /// Metadata for a dense internal node, if `node_index` exists.
    #[must_use]
    pub fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
        self.node_data.get(node_index as usize).copied()
    }

    /// Whether a directed edge exists from `source` to `destination` at `level`.
    ///
    /// This is a membership test on [`Graph::edges`]. A dropped reverse link
    /// can make the relation one-sided.
    #[must_use]
    pub fn has_edge(&self, level: u8, source: NodeIndex, destination: NodeIndex) -> bool {
        self.edges(level, source)
            .binary_search(&destination)
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_dense_nodes_and_allocates_layers() {
        let mut graph = Graph::new();
        graph.insert_node(0, 100, 3, 0).unwrap();
        graph.insert_node(1, 200, 2, 100).unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.layer_count(), 4);
        assert_eq!(graph.node(0).unwrap().external_id, 100);
    }

    #[test]
    fn adjacency_is_sorted_duplicate_free_and_per_layer() {
        let mut graph = Graph::new();
        for node in 0..3 {
            graph.insert_node(node, node + 10, 1, node).unwrap();
        }
        graph.add_edge(0, 0, 2).unwrap();
        graph.add_edge(0, 0, 1).unwrap();
        graph.add_edge(0, 0, 2).unwrap();
        graph.add_edge(1, 0, 2).unwrap();
        assert_eq!(graph.edges(0, 0), &[1, 2]);
        assert_eq!(graph.edges(1, 0), &[2]);
    }

    #[test]
    fn rejects_edges_above_either_node_level() {
        let mut graph = Graph::new();
        graph.insert_node(0, 10, 1, 0).unwrap();
        graph.insert_node(1, 11, 0, 1).unwrap();
        assert!(matches!(
            graph.add_edge(1, 0, 1),
            Err(Error::InvalidLayer(1))
        ));
        assert!(graph.edges(1, 1).is_empty());
    }

    #[test]
    fn bidirectional_edges_are_visible_from_both_nodes() {
        let mut graph = Graph::new();
        graph.insert_node(0, 10, 0, 0).unwrap();
        graph.insert_node(1, 11, 0, 1).unwrap();
        graph.add_bidirectional_edge(0, 0, 1).unwrap();
        assert!(graph.has_edge(0, 0, 1));
        assert!(graph.has_edge(0, 1, 0));
    }

    #[test]
    fn set_edges_replaces_with_sorted_unique_neighbors() {
        let mut graph = Graph::new();
        for node in 0..4 {
            graph.insert_node(node, node + 10, 0, node).unwrap();
        }
        graph.add_edge(0, 0, 1).unwrap();
        graph.add_edge(0, 0, 2).unwrap();
        graph.add_edge(0, 0, 3).unwrap();
        graph.set_edges(0, 0, [3, 1, 3]).unwrap();
        assert_eq!(graph.edges(0, 0), &[1, 3]);
        assert!(matches!(
            graph.set_edges(0, 0, [5]),
            Err(Error::InvalidNode(5))
        ));
    }

    #[test]
    fn bidirectional_edge_is_atomic_when_reverse_fails() {
        let mut graph = Graph::new();
        graph.insert_node(0, 10, 0, 0).unwrap();
        graph.insert_node(1, 11, 0, 1).unwrap();
        graph.fail_add_edge_from = Some(1);
        assert!(matches!(
            graph.add_bidirectional_edge(0, 0, 1),
            Err(Error::InvalidNode(1))
        ));
        assert!(!graph.has_edge(0, 0, 1));
        assert!(!graph.has_edge(0, 1, 0));
        assert_eq!(graph.fail_add_edge_from, None);

        graph.add_edge(0, 0, 1).unwrap();
        graph.fail_add_edge_from = Some(1);
        assert!(matches!(
            graph.add_bidirectional_edge(0, 0, 1),
            Err(Error::InvalidNode(1))
        ));
        assert!(
            graph.has_edge(0, 0, 1),
            "pre-existing forward edge must survive undo"
        );
        assert!(!graph.has_edge(0, 1, 0));
    }

    #[test]
    fn node_index_from_len_saturates_above_u32_max() {
        assert_eq!(Graph::node_index_from_len(0), 0);
        assert_eq!(Graph::node_index_from_len(1), 1);
        assert_eq!(Graph::node_index_from_len(u32::MAX as usize), u32::MAX);
        if let Some(over) = (u32::MAX as usize).checked_add(1) {
            assert_eq!(Graph::node_index_from_len(over), u32::MAX);
        }
    }

    #[test]
    fn insert_node_rejects_u32_capacity_without_allocating() {
        assert!(!Graph::node_len_at_u32_cap(0));
        assert!(!Graph::node_len_at_u32_cap(u32::MAX as usize - 1));
        assert!(Graph::node_len_at_u32_cap(u32::MAX as usize));
        if let Some(over) = (u32::MAX as usize).checked_add(1) {
            assert!(Graph::node_len_at_u32_cap(over));
        }

        // Last legal NodeIndex (`u32::MAX - 1`) is usable: `len == u32::MAX`
        // is a full graph (`0..=u32::MAX - 1`). Index `u32::MAX` is unused
        // so the u32 count stays exact (see `node_len_at_u32_cap`).
        assert_eq!(
            Graph::node_index_from_len(u32::MAX as usize - 1),
            u32::MAX - 1
        );
        Graph::check_insert_node(u32::MAX as usize - 1, u32::MAX - 1).unwrap();
        assert!(matches!(
            Graph::check_insert_node(u32::MAX as usize, u32::MAX),
            Err(Error::CapacityExceeded("node count"))
        ));
        assert!(matches!(
            Graph::check_insert_node(u32::MAX as usize, 0),
            Err(Error::CapacityExceeded("node count"))
        ));
        if let Some(over) = (u32::MAX as usize).checked_add(1) {
            assert!(matches!(
                Graph::check_insert_node(over, u32::MAX),
                Err(Error::CapacityExceeded("node count"))
            ));
        }
        assert!(matches!(
            Graph::check_insert_node(2, 3),
            Err(Error::InvalidNode(3))
        ));
        Graph::check_insert_node(2, 2).unwrap();
    }

    #[test]
    fn pop_last_node_restores_layer_count() {
        let mut graph = Graph::new();
        graph.insert_node(0, 10, 0, 0).unwrap();
        graph.insert_node(1, 11, 3, 1).unwrap();
        assert_eq!(graph.layer_count(), 4);
        graph.pop_last_node(1);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.layer_count(), 1);
        assert_eq!(graph.node(0).unwrap().external_id, 10);
        assert!(graph.node(1).is_none());
    }
}
