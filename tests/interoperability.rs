use std::{fs, path::PathBuf, time::SystemTime};

use hnsw_rs::{Config, HnswIndex, VERSION, load_file};

fn decode_hex(input: &str) -> Vec<u8> {
    let compact = input.trim();
    assert_eq!(compact.len() % 2, 0);
    compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(text, 16).unwrap()
        })
        .collect()
}

fn v3_fixture() -> Vec<u8> {
    decode_hex(include_str!("fixtures/v3.hex"))
}

fn temporary_file(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hnsw-rs-interop-{label}-{}-{:?}.hnsw",
        std::process::id(),
        SystemTime::now()
    ))
}

fn fixture_index() -> HnswIndex {
    let mut index = HnswIndex::new(Config {
        dim: 2,
        max_level: 2,
        level_mult: 0.0,
        rng_seed: Some(1),
        ..Config::default()
    })
    .unwrap();
    index.insert(100, &[1.0, 0.0]).unwrap();
    index.config.level_mult = 1.0;
    index.insert(200, &[0.0, 1.0]).unwrap();
    index.insert(300, &[-1.0, 0.0]).unwrap();
    index
}

#[test]
fn loads_v3_fixture() {
    let path = temporary_file("load-v3");
    fs::write(&path, v3_fixture()).unwrap();
    let loaded = load_file(&path).unwrap();
    assert_eq!(loaded.header().version, VERSION);
    assert_eq!(loaded.header().node_count, 3);
    assert_eq!(loaded.node(0).unwrap().external_id, 100);
    assert_eq!(loaded.node(1).unwrap().external_id, 200);
    assert_eq!(loaded.node(2).unwrap().external_id, 300);
    assert_eq!(
        loaded.vector(2).unwrap().iter().collect::<Vec<_>>(),
        [-1.0, 0.0]
    );
    drop(loaded);
    fs::remove_file(path).unwrap();
}

#[test]
fn loaded_fixture_search_matches_live_index() {
    let path = temporary_file("search-v3");
    fs::write(&path, v3_fixture()).unwrap();
    let loaded = load_file(&path).unwrap();
    let live = fixture_index();
    let query = [0.9_f32, 0.1];
    assert_eq!(loaded.search(&query, 3).unwrap(), live.search(&query, 3).unwrap());
    assert_eq!(loaded.search(&[0.0, 1.0], 1).unwrap()[0].id, 200);
    drop(loaded);
    fs::remove_file(path).unwrap();
}

#[test]
fn writer_matches_v3_fixture() {
    let path = temporary_file("write-v3");
    fixture_index().save(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), v3_fixture());
    fs::remove_file(path).unwrap();
}
