use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

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
        "hnsw-rs-interop-{label}-{}-{}.hnsw",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
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
    index.set_level_mult(1.0).unwrap();
    index.insert(200, &[0.0, 1.0]).unwrap();
    index.insert(300, &[-1.0, 0.0]).unwrap();
    index
}

fn fixture_vector(node: u32) -> [f32; 2] {
    match node {
        0 => [1.0, 0.0],
        1 => [0.0, 1.0],
        2 => [-1.0, 0.0],
        _ => panic!("v3 fixture has three nodes"),
    }
}

#[test]
fn loads_v3_fixture() {
    let path = temporary_file("load-v3");
    fs::write(&path, v3_fixture()).unwrap();
    let loaded = load_file(&path).unwrap();
    let live = fixture_index();
    let header = loaded.header();
    assert_eq!(header.version, VERSION);
    assert_eq!(header.node_count, live.len() as u32);
    assert_eq!(header.dim, live.config().dim);
    assert_eq!(header.m, live.config().m);
    assert_eq!(header.ef_construction, live.config().ef_construction);
    assert_eq!(header.ef_search, live.config().ef_search);
    assert_eq!(header.max_level, live.config().max_level);
    assert_eq!(header.level_mult, live.config().level_mult);
    assert_eq!(header.layer_count, live.layer_count());
    assert_eq!(header.entry_point, live.entry_point().unwrap());
    assert_eq!(header.entry_level, live.entry_level());
    assert_eq!(loaded.node(3), None);
    for node in 0..live.len() as u32 {
        assert_eq!(loaded.node(node), live.node(node));
        assert_eq!(
            loaded.vector(node).unwrap().iter().collect::<Vec<_>>(),
            fixture_vector(node)
        );
        for level in 0..live.layer_count() {
            assert_eq!(
                loaded.edges(level, node).iter().collect::<Vec<_>>(),
                live.edges(level, node)
            );
        }
    }
    drop(loaded);
    fs::remove_file(path).unwrap();
}

#[test]
fn loaded_fixture_search_matches_live_index() {
    let path = temporary_file("search-v3");
    fs::write(&path, v3_fixture()).unwrap();
    let loaded = load_file(&path).unwrap();
    let live = fixture_index();
    let query = [0.9998_f32, 0.02];
    assert_eq!(
        loaded.search(&query, 3).unwrap(),
        live.search(&query, 3).unwrap()
    );
    for (vector, id) in [([1.0_f32, 0.0], 100), ([0.0, 1.0], 200), ([-1.0, 0.0], 300)] {
        assert_eq!(loaded.search(&vector, 1).unwrap()[0].id, id);
        assert_eq!(live.search(&vector, 1).unwrap()[0].id, id);
    }
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
