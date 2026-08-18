use crate::{Error, Result};

/// Construction and search parameters for an HNSW index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Vector dimensionality.
    pub dim: u16,
    /// Maximum neighbors selected for a newly inserted node in layer zero.
    ///
    /// Must be greater than zero. Upper layers select at most
    /// [`Config::new_node_neighbors`] neighbors (`max(m / 2, 1)`). New-node
    /// links and reverse-link pruning both use the paper / hnswlib diversity
    /// heuristic, then cap outgoing degree at [`Config::max_degree`]: `2 * m`
    /// at layer 0 (`Mmax0`) and `m` at upper layers (`Mmax`). Pruning is
    /// one-sided: a dropped outgoing edge is not removed from the peer, so
    /// adjacency may become directed.
    pub m: u8,
    /// Candidate search width during insertion.
    pub ef_construction: u16,
    /// Candidate search width during queries.
    ///
    /// `search` uses `max(ef_search, k)` so a request larger than this value
    /// still returns up to `k` hits.
    pub ef_search: u16,
    /// Highest level that may be assigned to a node.
    pub max_level: u8,
    /// Probability of stopping at the current randomly selected level.
    pub level_mult: f64,
    /// Optional seed for repeatable construction with the pinned `rand` version.
    /// Operating-system entropy is used when absent.
    pub rng_seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 384,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_level: 16,
            level_mult: 0.5,
            rng_seed: None,
        }
    }
}

impl Config {
    pub(crate) fn validate(self) -> Result<Self> {
        if self.dim == 0 {
            return Err(Error::InvalidConfig("dim must be greater than zero"));
        }
        if self.m == 0 {
            return Err(Error::InvalidConfig("m must be greater than zero"));
        }
        if self.max_level == 0 || self.max_level == u8::MAX {
            return Err(Error::InvalidConfig("max_level must be between 1 and 254"));
        }
        if !self.level_mult.is_finite() || !(0.0..=1.0).contains(&self.level_mult) {
            return Err(Error::InvalidConfig(
                "level_mult must be finite and between zero and one",
            ));
        }
        Ok(self)
    }

    /// Neighbors selected for a newly inserted node at `level`.
    ///
    /// Layer 0 uses [`Config::m`]; upper layers use `max(m / 2, 1)`.
    #[must_use]
    pub fn new_node_neighbors(&self, level: u8) -> u8 {
        if level == 0 {
            self.m
        } else {
            (self.m / 2).max(1)
        }
    }

    /// Maximum outgoing degree after reverse-link pruning (HNSW `Mmax` / `Mmax0`).
    ///
    /// Layer 0 allows `2 * m` neighbors; upper layers allow `m`.
    #[must_use]
    pub fn max_degree(&self, level: u8) -> usize {
        if level == 0 {
            2 * usize::from(self.m)
        } else {
            usize::from(self.m)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_stable() {
        assert_eq!(
            Config::default(),
            Config {
                dim: 384,
                m: 16,
                ef_construction: 200,
                ef_search: 100,
                max_level: 16,
                level_mult: 0.5,
                rng_seed: None,
            }
        );
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(matches!(
            Config {
                dim: 0,
                ..Config::default()
            }
            .validate(),
            Err(Error::InvalidConfig(_))
        ));
        assert!(matches!(
            Config {
                level_mult: f64::NAN,
                ..Config::default()
            }
            .validate(),
            Err(Error::InvalidConfig(_))
        ));
        assert!(matches!(
            Config {
                max_level: u8::MAX,
                ..Config::default()
            }
            .validate(),
            Err(Error::InvalidConfig(_))
        ));
        assert!(matches!(
            Config {
                m: 0,
                ..Config::default()
            }
            .validate(),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn neighbor_caps_follow_hnsw_mmax() {
        let config = Config {
            m: 16,
            ..Config::default()
        };
        assert_eq!(config.new_node_neighbors(0), 16);
        assert_eq!(config.new_node_neighbors(1), 8);
        assert_eq!(config.max_degree(0), 32);
        assert_eq!(config.max_degree(1), 16);

        let tiny = Config {
            m: 1,
            ..Config::default()
        };
        assert_eq!(tiny.new_node_neighbors(3), 1);
        assert_eq!(tiny.max_degree(0), 2);
        assert_eq!(tiny.max_degree(2), 1);
    }
}
