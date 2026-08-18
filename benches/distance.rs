use std::{hint::black_box, time::Instant};

use hnsw_rs::vector::{cosine_distance, dot};

fn run(name: &str, iterations: usize, mut operation: impl FnMut()) {
    for _ in 0..1_000 {
        operation();
    }
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    let elapsed = started.elapsed();
    println!(
        "{name}: {:.2} ns/iteration ({iterations} iterations)",
        elapsed.as_nanos() as f64 / iterations as f64
    );
}

fn main() {
    let iterations = std::env::var("HNSW_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000_000);
    let left = (0..384)
        .map(|index| (index as f32 / 384.0).sin())
        .collect::<Vec<_>>();
    let right = (0..384)
        .map(|index| (index as f32 / 384.0).cos())
        .collect::<Vec<_>>();

    run("dot/f32x384", iterations, || {
        black_box(dot(black_box(&left), black_box(&right)).expect("equal lengths"));
    });
    run("cosine_distance/f32x384", iterations, || {
        black_box(cosine_distance(black_box(&left), black_box(&right)).expect("equal lengths"));
    });
}
