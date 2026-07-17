use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use hnsw_rs::{Config, HnswIndex, vector::dot};
use hnsw_rs_upstream::prelude::{DistDot, Hnsw};
use usearch::{Index as UsearchIndex, IndexOptions, MetricKind, ScalarKind};

const UPSTREAM_VERSION: &str = "0.3.4";
const USEARCH_VERSION: &str = "2.26.0";

#[derive(Clone, Copy)]
struct Parameters {
    vectors: usize,
    dimensions: usize,
    queries: usize,
    repetitions: usize,
    k: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
}

impl Parameters {
    fn from_env() -> Self {
        Self {
            vectors: env_usize("HNSW_BENCH_VECTORS", 10_000),
            dimensions: env_usize("HNSW_BENCH_DIMENSIONS", 384),
            queries: env_usize("HNSW_BENCH_QUERIES", 100),
            repetitions: env_usize("HNSW_BENCH_REPETITIONS", 5),
            k: env_usize("HNSW_BENCH_K", 10),
            m: env_usize("HNSW_BENCH_M", 16),
            ef_construction: env_usize("HNSW_BENCH_EF_CONSTRUCTION", 200),
            ef_search: env_usize("HNSW_BENCH_EF_SEARCH", 100),
            seed: env_u64("HNSW_BENCH_SEED", 42),
        }
    }

    fn validate(self, ef_searches: &[usize]) {
        assert!(
            (1..=u32::MAX as usize).contains(&self.vectors),
            "HNSW_BENCH_VECTORS must be positive and fit in u32"
        );
        assert!(
            (1..=u16::MAX as usize).contains(&self.dimensions),
            "HNSW_BENCH_DIMENSIONS must fit in u16"
        );
        assert!(self.queries > 0, "HNSW_BENCH_QUERIES must be positive");
        assert!(
            self.repetitions > 0,
            "HNSW_BENCH_REPETITIONS must be positive"
        );
        assert!(
            (1..=self.vectors).contains(&self.k),
            "HNSW_BENCH_K must be between 1 and the vector count"
        );
        assert!(
            (2..=u8::MAX as usize).contains(&self.m),
            "HNSW_BENCH_M must be between 2 and 255"
        );
        assert!(!ef_searches.is_empty(), "ef_search sweep must not be empty");
        for ef_search in ef_searches {
            assert!(
                (self.k..=u16::MAX as usize).contains(ef_search),
                "every ef_search value must be at least k and fit in u16"
            );
        }
        assert!(
            (1..=u16::MAX as usize).contains(&self.ef_construction),
            "HNSW_BENCH_EF_CONSTRUCTION must fit in u16"
        );
        assert!(
            self.queries
                .checked_mul(self.repetitions)
                .is_some_and(|samples| u32::try_from(samples).is_ok()),
            "the timed query sample count must fit in u32"
        );
    }
}

struct SearchMeasurement {
    ids: Vec<Vec<u32>>,
    mean: Duration,
    p50: Duration,
    p95: Duration,
}

struct SearchRow<'a> {
    name: &'a str,
    search: SearchMeasurement,
    recall: f64,
}

fn main() {
    let parameters = Parameters::from_env();
    let ef_searches =
        env_usize_list("HNSW_BENCH_EF_SEARCHES").unwrap_or_else(|| vec![parameters.ef_search]);
    parameters.validate(&ef_searches);

    println!(
        "Generating {} indexed vectors and {} queries ({} dimensions)...",
        parameters.vectors, parameters.queries, parameters.dimensions
    );
    let vectors =
        generate_normalized_vectors(parameters.vectors, parameters.dimensions, parameters.seed);
    let queries = generate_normalized_vectors(
        parameters.queries,
        parameters.dimensions,
        parameters.seed ^ 0xa076_1d64_78bd_642f,
    );

    let truth_started = Instant::now();
    let truth = exact_top_k(&vectors, &queries, parameters.k);
    let truth_elapsed = truth_started.elapsed();

    println!("Building hnsw-rs (local checkout)...");
    let build_started = Instant::now();
    let mut ours = build_ours(&vectors, parameters);
    let ours_build = build_started.elapsed();

    println!("Building upstream hnsw_rs {UPSTREAM_VERSION}...");
    let build_started = Instant::now();
    let upstream = build_upstream(&vectors, parameters);
    let upstream_build = build_started.elapsed();

    println!("Building USearch {USEARCH_VERSION}...");
    let build_started = Instant::now();
    let usearch = build_usearch(&vectors, parameters);
    let usearch_build = build_started.elapsed();
    let usearch_acceleration = usearch.hardware_acceleration();

    println!();
    println!("# HNSW competitor benchmark");
    println!();
    println!(
        "- Dataset: {} generated unit vectors, {} dimensions, seed {}",
        parameters.vectors, parameters.dimensions, parameters.seed
    );
    println!(
        "- Search: {} generated queries, k={}, {} timed repetitions",
        parameters.queries, parameters.k, parameters.repetitions
    );
    println!(
        "- HNSW parameters: M={}, ef_construction={}, ef_search={ef_searches:?}",
        parameters.m, parameters.ef_construction
    );
    println!(
        "- Platform: {}/{}; USearch acceleration: {}",
        env::consts::OS,
        env::consts::ARCH,
        usearch_acceleration
    );
    println!("- Metric: exact inner product over pre-normalized f32 vectors");
    println!(
        "- Exact ground truth time: {:.3} s",
        truth_elapsed.as_secs_f64()
    );
    println!();
    println!("| implementation | build (s) | build vectors/s |");
    println!("|---|---:|---:|");
    for (name, build) in [
        ("hnsw-rs (this crate)", ours_build),
        ("hnsw_rs (upstream)", upstream_build),
        ("USearch", usearch_build),
    ] {
        println!(
            "| {} | {:.3} | {:.0} |",
            name,
            build.as_secs_f64(),
            parameters.vectors as f64 / build.as_secs_f64(),
        );
    }

    for ef_search in ef_searches {
        ours.config.ef_search = ef_search as u16;
        usearch.change_expansion_search(ef_search);

        let ours_search = measure_search(&queries, parameters.repetitions, |query| {
            ours.search(query, parameters.k)
                .expect("hnsw-rs search failed")
                .into_iter()
                .map(|hit| hit.id)
                .collect()
        });
        let upstream_search = measure_search(&queries, parameters.repetitions, |query| {
            upstream
                .search(query, parameters.k, ef_search)
                .into_iter()
                .map(|neighbour| {
                    u32::try_from(neighbour.d_id).expect("benchmark IDs must fit in u32")
                })
                .collect()
        });
        let usearch_search = measure_search(&queries, parameters.repetitions, |query| {
            usearch
                .search(query, parameters.k)
                .expect("USearch search failed")
                .keys
                .into_iter()
                .map(|id| u32::try_from(id).expect("benchmark IDs must fit in u32"))
                .collect()
        });
        let rows = [
            SearchRow {
                name: "hnsw-rs (this crate)",
                recall: recall_at_k(&truth, &ours_search.ids),
                search: ours_search,
            },
            SearchRow {
                name: "hnsw_rs (upstream)",
                recall: recall_at_k(&truth, &upstream_search.ids),
                search: upstream_search,
            },
            SearchRow {
                name: "USearch",
                recall: recall_at_k(&truth, &usearch_search.ids),
                search: usearch_search,
            },
        ];

        println!();
        println!("## ef_search={ef_search}");
        println!();
        println!(
            "| implementation | query mean (us) | p50 (us) | p95 (us) | queries/s | recall@{} |",
            parameters.k
        );
        println!("|---|---:|---:|---:|---:|---:|");
        for row in rows {
            println!(
                "| {} | {:.2} | {:.2} | {:.2} | {:.0} | {:.4} |",
                row.name,
                micros(row.search.mean),
                micros(row.search.p50),
                micros(row.search.p95),
                1.0 / row.search.mean.as_secs_f64(),
                row.recall,
            );
        }
    }

    println!();
    println!(
        "All builds and searches are single-caller and in-memory; dependency setup and data generation are excluded."
    );
    println!("USearch uses f32 storage. Timings include each crate's public Rust API boundary.");
}

fn build_ours(vectors: &[Vec<f32>], parameters: Parameters) -> HnswIndex {
    let mut index = HnswIndex::new(Config {
        dim: parameters.dimensions as u16,
        m: parameters.m as u8,
        ef_construction: parameters.ef_construction as u16,
        ef_search: parameters.ef_search as u16,
        max_level: 16,
        // Standard HNSW level sampling reaches level L with probability M^-L.
        level_mult: 1.0 - 1.0 / parameters.m as f64,
        rng_seed: Some(parameters.seed),
    })
    .expect("valid hnsw-rs benchmark configuration");
    for (id, vector) in vectors.iter().enumerate() {
        index
            .insert(id as u32, vector)
            .expect("hnsw-rs insertion failed");
    }
    index
}

fn build_upstream(vectors: &[Vec<f32>], parameters: Parameters) -> Hnsw<'static, f32, DistDot> {
    let index = Hnsw::<f32, DistDot>::new(
        parameters.m,
        parameters.vectors,
        16,
        parameters.ef_construction,
        DistDot {},
    );
    for (id, vector) in vectors.iter().enumerate() {
        index.insert((vector, id));
    }
    index
}

fn build_usearch(vectors: &[Vec<f32>], parameters: Parameters) -> UsearchIndex {
    let options = IndexOptions {
        dimensions: parameters.dimensions,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: parameters.m,
        expansion_add: parameters.ef_construction,
        expansion_search: parameters.ef_search,
        multi: false,
    };
    let index = UsearchIndex::new(&options).expect("valid USearch benchmark configuration");
    index
        .reserve(parameters.vectors)
        .expect("USearch reservation failed");
    for (id, vector) in vectors.iter().enumerate() {
        index
            .add(id as u64, vector)
            .expect("USearch insertion failed");
    }
    index
}

fn measure_search(
    queries: &[Vec<f32>],
    repetitions: usize,
    mut search: impl FnMut(&[f32]) -> Vec<u32>,
) -> SearchMeasurement {
    for query in queries {
        black_box(search(black_box(query)));
    }

    let mut ids = Vec::with_capacity(queries.len());
    let mut samples = Vec::with_capacity(queries.len() * repetitions);
    for repetition in 0..repetitions {
        for query in queries {
            let started = Instant::now();
            let result = black_box(search(black_box(query)));
            samples.push(started.elapsed());
            if repetition == 0 {
                ids.push(result);
            } else {
                black_box(result);
            }
        }
    }

    samples.sort_unstable();
    let total = samples.iter().sum::<Duration>();
    SearchMeasurement {
        ids,
        mean: total / u32::try_from(samples.len()).expect("sample count must fit in u32"),
        p50: percentile(&samples, 0.50),
        p95: percentile(&samples, 0.95),
    }
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index]
}

fn exact_top_k(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut scores = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| (dot(vector, query), id as u32))
                .collect::<Vec<_>>();
            scores.select_nth_unstable_by(k - 1, |left, right| {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            scores.truncate(k);
            scores.sort_unstable_by(|left, right| {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            scores.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

fn recall_at_k(truth: &[Vec<u32>], approximate: &[Vec<u32>]) -> f64 {
    assert_eq!(truth.len(), approximate.len());
    let recalled = truth
        .iter()
        .zip(approximate)
        .map(|(expected, actual)| actual.iter().filter(|id| expected.contains(id)).count())
        .sum::<usize>();
    recalled as f64 / truth.iter().map(Vec::len).sum::<usize>() as f64
}

fn generate_normalized_vectors(count: usize, dimensions: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64(seed);
    (0..count)
        .map(|_| {
            let mut vector = (0..dimensions)
                .map(|_| rng.next_f32() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            for value in &mut vector {
                *value /= norm;
            }
            vector
        })
        .collect()
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 40) as f32 / (1_u32 << 24) as f32
    }
}

fn env_usize_list(name: &str) -> Option<Vec<usize>> {
    env::var(name).ok().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .map(|item| {
                item.parse()
                    .unwrap_or_else(|_| panic!("{name} must be a comma-separated integer list"))
            })
            .collect()
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer"))
        })
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer"))
        })
        .unwrap_or(default)
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}
