use crate::{Error, Result};

pub type ExternalId = u32;
pub type NodeIndex = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMeta {
    pub external_id: ExternalId,
    pub level: u8,
    pub vector_offset: u32,
}

/// Mutable per-layer graph used while constructing an index.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub(crate) node_data: Vec<NodeMeta>,
    construction_layers: Vec<Vec<Vec<NodeIndex>>>,
}

impl Graph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert_node(
        &mut self,
        node_index: NodeIndex,
        external_id: ExternalId,
        level: u8,
        vector_offset: u32,
    ) -> Result<()> {
        if node_index != self.node_count() {
            return Err(Error::InvalidNode(node_index));
        }

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
    ) -> Result<()> {
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
        let edges = &mut layer[source as usize];
        match edges.binary_search(&destination) {
            Ok(_) => {}
            Err(position) => edges.insert(position, destination),
        }
        Ok(())
    }

    pub(crate) fn add_bidirectional_edge(
        &mut self,
        level: u8,
        left: NodeIndex,
        right: NodeIndex,
    ) -> Result<()> {
        self.add_edge(level, left, right)?;
        self.add_edge(level, right, left)
    }

    #[must_use]
    pub fn node_count(&self) -> NodeIndex {
        u32::try_from(self.node_data.len()).expect("graph node count was checked before insertion")
    }

    #[must_use]
    pub fn layer_count(&self) -> u8 {
        u8::try_from(self.construction_layers.len())
            .expect("layer count cannot exceed the u8 node level range")
    }

    #[must_use]
    pub fn node(&self, node_index: NodeIndex) -> Option<NodeMeta> {
        self.node_data.get(node_index as usize).copied()
    }

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
}
