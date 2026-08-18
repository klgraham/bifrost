use std::{fs, path::PathBuf, time::SystemTime};

use hnsw_rs::{Config, HnswIndex, LoadedHnsw};

fn temporary_file(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hnsw-rs-search-{label}-{}-{:?}.hnsw",
        std::process::id(),
        SystemTime::now()
    ))
}

fn four_node_index(ef_search: u16) -> HnswIndex {
    let mut index = HnswIndex::new(Config {
        dim: 2,
        ef_search,
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
    index
}

#[test]
fn search_returns_k_when_larger_than_ef_search() {
    let index = four_node_index(2);
    let query = [1.0_f32, 0.0];
    let live = index.search(&query, 4).unwrap();
    assert_eq!(live.len(), 4);
    assert_eq!(index.search_with_ef(&query, 4, 2).unwrap(), live);
    assert_eq!(index.config().ef_search, 2);

    let path = temporary_file("k-gt-ef");
    index.save(&path).unwrap();
    let loaded = LoadedHnsw::open(&path).unwrap();
    assert_eq!(loaded.search(&query, 4).unwrap(), live);
    assert_eq!(loaded.search_with_ef(&query, 4, 2).unwrap(), live);
    drop(loaded);
    fs::remove_file(path).unwrap();
}
