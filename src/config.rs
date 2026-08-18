use crate::{Error, Result};

/// Construction and search parameters for an HNSW index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Vector dimensionality.
    pub dim: u16,
    /// Maximum neighbors selected for a new node in layer zero.
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
    }
}
