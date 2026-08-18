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
    ///
    /// Must be greater than zero. A stored value of `0` is rejected rather
    /// than silently clamped during construction search.
    pub ef_construction: u16,
    /// Candidate search width during queries.
    ///
    /// Must be greater than zero. `search` uses `max(ef_search, k)` so a
    /// request larger than this value still returns up to `k` hits.
    pub ef_search: u16,
    /// Highest level that may be assigned to a node.
    pub max_level: u8,
    /// Probability of stopping at the current randomly selected level.
    ///
    /// Insertion stops at the current level with this probability, so
    /// `P(level >= L) = (1 - level_mult)^L`. [`Config::default`] uses
    /// [`Config::level_mult_for_m`] (`1.0 - 1.0 / m`) so
    /// `P(level >= L) = m^{-L}`, matching the HNSW paper and hnswlib.
    /// `1.0` always assigns level zero; `0.0` always assigns
    /// [`Config::max_level`]. Stored snapshots may keep any finite value in
    /// `[0, 1]`, including the former `0.5` default.
    pub level_mult: f64,
    /// Optional seed for repeatable construction with the pinned `rand` version.
    /// Operating-system entropy is used when absent.
    pub rng_seed: Option<u64>,
    /// When `true`, insert and search return [`crate::Error::InvalidVector`]
    /// for non-finite coordinates or an L2 norm more than
    /// [`crate::vector::UNIT_NORM_TOLERANCE`] from `1`.
    ///
    /// Default is `false`: release builds keep the previous unchecked
    /// contract. Debug builds always `debug_assert` the same conditions so
    /// NaN/Inf and clearly unnormalized inputs fail in development. The flag
    /// is not stored in `.hnsw` snapshots; call
    /// [`crate::LoadedHnsw::set_check_vectors`] after mapping.
    pub check_vectors: bool,
}

impl Default for Config {
    fn default() -> Self {
        let m = 16;
        Self {
            dim: 384,
            m,
            ef_construction: 200,
            ef_search: 100,
            max_level: 16,
            level_mult: Self::level_mult_for_m(m),
            rng_seed: None,
            check_vectors: false,
        }
    }
}

impl Config {
    /// Stop probability that yields `P(level >= L) = m^{-L}`.
    ///
    /// This is the HNSW paper / hnswlib formula: `1.0 - 1.0 / m`. `m` must
    /// be greater than zero. `level_mult` is stored independently of `m`, so
    /// a struct update that changes only `m` should call this if the paper
    /// distribution is wanted.
    #[must_use]
    pub fn level_mult_for_m(m: u8) -> f64 {
        1.0 - 1.0 / f64::from(m)
    }

    pub(crate) fn validate(self) -> Result<Self> {
        if self.dim == 0 {
            return Err(Error::InvalidConfig("dim must be greater than zero"));
        }
        if self.m == 0 {
            return Err(Error::InvalidConfig("m must be greater than zero"));
        }
        if self.ef_construction == 0 {
            return Err(Error::InvalidConfig(
                "ef_construction must be greater than zero",
            ));
        }
        if self.ef_search == 0 {
            return Err(Error::InvalidConfig("ef_search must be greater than zero"));
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
                level_mult: 0.9375,
                rng_seed: None,
                check_vectors: false,
            }
        );
        assert_eq!(Config::default().level_mult, Config::level_mult_for_m(16));
        assert_eq!(Config::level_mult_for_m(16), 1.0 - 1.0 / 16.0);
        assert_eq!(Config::level_mult_for_m(8), 1.0 - 1.0 / 8.0);
    }

    #[test]
    fn historical_level_mult_half_still_validates() {
        assert!(
            Config {
                level_mult: 0.5,
                ..Config::default()
            }
            .validate()
            .is_ok()
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
        assert!(matches!(
            Config {
                ef_construction: 0,
                ..Config::default()
            }
            .validate(),
            Err(Error::InvalidConfig(_))
        ));
        assert!(matches!(
            Config {
                ef_search: 0,
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
