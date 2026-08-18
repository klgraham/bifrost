use std::{
    collections::HashMap,
    env,
    error::Error,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use hnsw_rs::{Config, HnswIndex, vector::dot};
use hnsw_rs_upstream::prelude::{Distance, Hnsw};
use usearch::{Index as UsearchIndex, IndexOptions, MetricKind, ScalarKind};

const UPSTREAM_VERSION: &str = "0.3.4";
const USEARCH_VERSION: &str = "2.26.0";
const FIXTURE_FORMAT: &str = "hnsw-rs-embedding-fixture-v1";

type Result<T> = std::result::Result<T, Box<dyn Error>>;

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
    semantic: Option<SemanticMetrics>,
}

struct Dataset {
    label: String,
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    relevance: Option<Relevance>,
    fixture_directory: Option<PathBuf>,
}

struct Relevance {
    by_query: Vec<Vec<(u32, u32)>>,
    evaluated_queries: usize,
}

#[derive(Clone, Copy)]
struct SemanticMetrics {
    ndcg: f64,
    recall: f64,
}

#[derive(Clone, Copy)]
struct UpstreamInnerProduct;

impl Distance<f32> for UpstreamInnerProduct {
    fn eval(&self, left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        (1.0 - left
            .iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum::<f32>())
        .max(0.0)
    }
}

fn main() -> Result<()> {
    let mut parameters = Parameters::from_env();
    let dataset = if let Some(path) = env::var_os("HNSW_BENCH_FIXTURE") {
        let fixture = load_fixture(
            &PathBuf::from(path),
            env_usize_optional("HNSW_BENCH_VECTORS"),
            env_usize_optional("HNSW_BENCH_QUERIES"),
        )?;
        if let Some(dimensions) = env_usize_optional("HNSW_BENCH_DIMENSIONS") {
            assert_eq!(
                dimensions,
                fixture.vectors[0].len(),
                "HNSW_BENCH_DIMENSIONS does not match the fixture"
            );
        }
        parameters.vectors = fixture.vectors.len();
        parameters.dimensions = fixture.vectors[0].len();
        parameters.queries = fixture.queries.len();
        fixture
    } else {
        println!(
            "Generating {} indexed vectors and {} queries ({} dimensions)...",
            parameters.vectors, parameters.queries, parameters.dimensions
        );
        Dataset {
            label: format!(
                "{} generated unit vectors, {} dimensions, seed {}",
                parameters.vectors, parameters.dimensions, parameters.seed
            ),
            vectors: generate_normalized_vectors(
                parameters.vectors,
                parameters.dimensions,
                parameters.seed,
            ),
            queries: generate_normalized_vectors(
                parameters.queries,
                parameters.dimensions,
                parameters.seed ^ 0xa076_1d64_78bd_642f,
            ),
            relevance: None,
            fixture_directory: None,
        }
    };
    let ef_searches =
        env_usize_list("HNSW_BENCH_EF_SEARCHES").unwrap_or_else(|| vec![parameters.ef_search]);
    parameters.validate(&ef_searches);

    let vectors = &dataset.vectors;
    let queries = &dataset.queries;

    let truth_started = Instant::now();
    let truth = exact_top_k(vectors, queries, parameters.k);
    let truth_elapsed = truth_started.elapsed();

    println!("Building hnsw-rs (local checkout)...");
    let build_started = Instant::now();
    let mut ours = build_ours(vectors, parameters);
    let ours_build = build_started.elapsed();

    println!("Building upstream hnsw_rs {UPSTREAM_VERSION}...");
    let build_started = Instant::now();
    let upstream = build_upstream(vectors, parameters);
    let upstream_build = build_started.elapsed();

    println!("Building USearch {USEARCH_VERSION}...");
    let build_started = Instant::now();
    let usearch = build_usearch(vectors, parameters);
    let usearch_build = build_started.elapsed();
    let usearch_acceleration = usearch.hardware_acceleration();

    println!();
    println!("# HNSW competitor benchmark");
    println!();
    println!("- Dataset: {}", dataset.label);
    println!(
        "- Search: {} queries, k={}, {} timed repetitions",
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
    if let Some(relevance) = &dataset.relevance {
        println!(
            "- Semantic evaluation: BEIR test qrels for {} queries with relevant indexed documents",
            relevance.evaluated_queries
        );
    }
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
        ours.set_ef_search(ef_search as u16)
            .expect("benchmark ef_search values are validated before the run");
        usearch.change_expansion_search(ef_search);

        let ours_search = measure_search(queries, parameters.repetitions, |query| {
            ours.search(query, parameters.k)
                .expect("hnsw-rs search failed")
                .into_iter()
                .map(|hit| hit.id)
                .collect()
        });
        let upstream_search = measure_search(queries, parameters.repetitions, |query| {
            upstream
                .search(query, parameters.k, ef_search)
                .into_iter()
                .map(|neighbour| {
                    u32::try_from(neighbour.d_id).expect("benchmark IDs must fit in u32")
                })
                .collect()
        });
        let usearch_search = measure_search(queries, parameters.repetitions, |query| {
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
                semantic: dataset
                    .relevance
                    .as_ref()
                    .map(|relevance| semantic_metrics(relevance, &ours_search.ids, parameters.k)),
                search: ours_search,
            },
            SearchRow {
                name: "hnsw_rs (upstream)",
                recall: recall_at_k(&truth, &upstream_search.ids),
                semantic: dataset.relevance.as_ref().map(|relevance| {
                    semantic_metrics(relevance, &upstream_search.ids, parameters.k)
                }),
                search: upstream_search,
            },
            SearchRow {
                name: "USearch",
                recall: recall_at_k(&truth, &usearch_search.ids),
                semantic: dataset.relevance.as_ref().map(|relevance| {
                    semantic_metrics(relevance, &usearch_search.ids, parameters.k)
                }),
                search: usearch_search,
            },
        ];

        println!();
        println!("## ef_search={ef_search}");
        println!();
        if dataset.relevance.is_some() {
            println!(
                "| implementation | query mean (us) | p50 (us) | p95 (us) | queries/s | exact recall@{} | nDCG@{} | qrels recall@{} |",
                parameters.k, parameters.k, parameters.k
            );
            println!("|---|---:|---:|---:|---:|---:|---:|---:|");
        } else {
            println!(
                "| implementation | query mean (us) | p50 (us) | p95 (us) | queries/s | exact recall@{} |",
                parameters.k
            );
            println!("|---|---:|---:|---:|---:|---:|");
        }
        for row in rows {
            if let Some(semantic) = row.semantic {
                println!(
                    "| {} | {:.2} | {:.2} | {:.2} | {:.0} | {:.4} | {:.4} | {:.4} |",
                    row.name,
                    micros(row.search.mean),
                    micros(row.search.p50),
                    micros(row.search.p95),
                    1.0 / row.search.mean.as_secs_f64(),
                    row.recall,
                    semantic.ndcg,
                    semantic.recall,
                );
            } else {
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
    }

    println!();
    println!(
        "All builds and searches are single-caller and in-memory; dependency setup and data generation are excluded."
    );
    println!("USearch uses f32 storage. Timings include each crate's public Rust API boundary.");
    if let Some(directory) = &dataset.fixture_directory {
        let indexes = directory.join("indexes");
        fs::create_dir_all(&indexes)?;
        let path = indexes.join(format!(
            "hnsw-rs-m{}-efc{}-seed{}.hnsw",
            parameters.m, parameters.ef_construction, parameters.seed
        ));
        ours.save(&path)?;
        println!("Saved the hnsw-rs index to {}.", path.display());
    }
    Ok(())
}

fn load_fixture(
    directory: &Path,
    vector_limit: Option<usize>,
    query_limit: Option<usize>,
) -> Result<Dataset> {
    let manifest = load_manifest(&directory.join("manifest.txt"))?;
    if manifest.get("format").map(String::as_str) != Some(FIXTURE_FORMAT) {
        return Err(format!(
            "{} is not a supported embedding fixture",
            directory.display()
        )
        .into());
    }
    let dimensions = manifest_usize(&manifest, "dimensions")?;
    let corpus_count = manifest_usize(&manifest, "corpus_count")?;
    let query_count = manifest_usize(&manifest, "query_count")?;
    let vectors_to_load = vector_limit.unwrap_or(corpus_count).min(corpus_count);
    let queries_to_load = query_limit.unwrap_or(query_count).min(query_count);
    if vectors_to_load == 0 || queries_to_load == 0 {
        return Err("fixture vector and query counts must be positive".into());
    }

    let vectors = read_f32_vectors(
        &directory.join("corpus.f32"),
        corpus_count,
        dimensions,
        vectors_to_load,
    )?;
    let queries = read_f32_vectors(
        &directory.join("queries.f32"),
        query_count,
        dimensions,
        queries_to_load,
    )?;
    let corpus_ids = read_ids(
        &directory.join("corpus-ids.txt"),
        corpus_count,
        vectors_to_load,
    )?;
    let query_ids = read_ids(
        &directory.join("query-ids.txt"),
        query_count,
        queries_to_load,
    )?;
    let qrels_path = directory.join("qrels-test.tsv");
    let relevance = if qrels_path.exists() {
        Some(load_qrels(&qrels_path, &corpus_ids, &query_ids)?)
    } else {
        None
    };

    let dataset = manifest
        .get("dataset")
        .map(String::as_str)
        .unwrap_or("embedding fixture");
    let model = manifest
        .get("model")
        .map(String::as_str)
        .unwrap_or("unknown model");
    println!(
        "Loaded {} indexed vectors and {} queries from {}...",
        vectors.len(),
        queries.len(),
        directory.display()
    );
    Ok(Dataset {
        label: format!(
            "{dataset}, {model}, {dimensions} dimensions ({} corpus vectors, {} queries)",
            vectors.len(),
            queries.len()
        ),
        vectors,
        queries,
        relevance,
        fixture_directory: Some(directory.to_owned()),
    })
}

fn load_manifest(path: &Path) -> Result<HashMap<String, String>> {
    let text = fs::read_to_string(path)?;
    let mut manifest = HashMap::new();
    for (line_number, line) in text.lines().enumerate() {
        let (key, value) = line.split_once('=').ok_or_else(|| {
            format!(
                "{}:{}: expected a key=value entry",
                path.display(),
                line_number + 1
            )
        })?;
        if manifest.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("{}: duplicate manifest key {key}", path.display()).into());
        }
    }
    Ok(manifest)
}

fn manifest_usize(manifest: &HashMap<String, String>, key: &str) -> Result<usize> {
    manifest
        .get(key)
        .ok_or_else(|| format!("fixture manifest is missing {key}"))?
        .parse()
        .map_err(|_| format!("fixture manifest {key} must be an integer").into())
}

fn read_f32_vectors(
    path: &Path,
    manifest_count: usize,
    dimensions: usize,
    load_count: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected_bytes = manifest_count
        .checked_mul(dimensions)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .ok_or("fixture vector file size overflow")?;
    let bytes = fs::read(path)?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{} has {} bytes; expected {expected_bytes}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let values_to_load = load_count
        .checked_mul(dimensions)
        .ok_or("fixture vector count overflow")?;
    let values = bytes[..values_to_load * size_of::<f32>()]
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect::<Vec<_>>();
    let mut vectors = Vec::with_capacity(load_count);
    for (index, values) in values.chunks_exact(dimensions).enumerate() {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 0.01 {
            return Err(format!(
                "{} vector {index} has invalid L2 norm {norm}",
                path.display()
            )
            .into());
        }
        vectors.push(values.to_vec());
    }
    Ok(vectors)
}

fn read_ids(path: &Path, manifest_count: usize, load_count: usize) -> Result<Vec<String>> {
    let ids = fs::read_to_string(path)?
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if ids.len() != manifest_count {
        return Err(format!(
            "{} has {} IDs; expected {manifest_count}",
            path.display(),
            ids.len()
        )
        .into());
    }
    if ids.iter().any(String::is_empty) {
        return Err(format!("{} contains an empty ID", path.display()).into());
    }
    let unique = ids.iter().collect::<std::collections::HashSet<_>>();
    if unique.len() != ids.len() {
        return Err(format!("{} contains duplicate IDs", path.display()).into());
    }
    Ok(ids.into_iter().take(load_count).collect())
}

fn load_qrels(path: &Path, corpus_ids: &[String], query_ids: &[String]) -> Result<Relevance> {
    let corpus_by_id = corpus_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index as u32))
        .collect::<HashMap<_, _>>();
    let query_by_id = query_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let text = fs::read_to_string(path)?;
    let mut relevance = vec![Vec::new(); query_ids.len()];
    for (line_number, line) in text.lines().enumerate() {
        if line_number == 0 && line.to_ascii_lowercase().contains("query") {
            continue;
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != 3 {
            return Err(format!(
                "{}:{}: expected query-id, corpus-id, score",
                path.display(),
                line_number + 1
            )
            .into());
        }
        let Some(&query_index) = query_by_id.get(columns[0]) else {
            continue;
        };
        let Some(&corpus_index) = corpus_by_id.get(columns[1]) else {
            continue;
        };
        let score = columns[2].parse::<u32>().map_err(|_| {
            format!(
                "{}:{}: relevance score must be a non-negative integer",
                path.display(),
                line_number + 1
            )
        })?;
        if score > 0 {
            relevance[query_index].push((corpus_index, score));
        }
    }
    for entries in &mut relevance {
        entries.sort_unstable_by_key(|&(id, _)| id);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(
                format!("{} contains duplicate query/document pairs", path.display()).into(),
            );
        }
    }
    let evaluated_queries = relevance
        .iter()
        .filter(|entries| !entries.is_empty())
        .count();
    if evaluated_queries == 0 {
        return Err(format!(
            "{} has no qrels for the loaded fixture subset",
            path.display()
        )
        .into());
    }
    Ok(Relevance {
        by_query: relevance,
        evaluated_queries,
    })
}

fn semantic_metrics(relevance: &Relevance, results: &[Vec<u32>], k: usize) -> SemanticMetrics {
    assert_eq!(relevance.by_query.len(), results.len());
    let mut ndcg = 0.0;
    let mut recall = 0.0;
    for (expected, actual) in relevance.by_query.iter().zip(results) {
        if expected.is_empty() {
            continue;
        }
        let scores = expected.iter().copied().collect::<HashMap<_, _>>();
        let dcg = actual
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, id)| {
                let relevance = f64::from(scores.get(id).copied().unwrap_or(0));
                (2.0_f64.powf(relevance) - 1.0) / (rank as f64 + 2.0).log2()
            })
            .sum::<f64>();
        let mut ideal = expected.iter().map(|&(_, score)| score).collect::<Vec<_>>();
        ideal.sort_unstable_by(|left, right| right.cmp(left));
        let idcg = ideal
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, &relevance)| {
                (2.0_f64.powf(f64::from(relevance)) - 1.0) / (rank as f64 + 2.0).log2()
            })
            .sum::<f64>();
        ndcg += dcg / idcg;
        let recalled = actual
            .iter()
            .take(k)
            .filter(|id| scores.contains_key(id))
            .count();
        recall += recalled as f64 / expected.len() as f64;
    }
    SemanticMetrics {
        ndcg: ndcg / relevance.evaluated_queries as f64,
        recall: recall / relevance.evaluated_queries as f64,
    }
}

fn build_ours(vectors: &[Vec<f32>], parameters: Parameters) -> HnswIndex {
    let mut index = HnswIndex::new(Config {
        dim: parameters.dimensions as u16,
        m: parameters.m as u8,
        ef_construction: parameters.ef_construction as u16,
        ef_search: parameters.ef_search as u16,
        max_level: 16,
        // Same paper / hnswlib default as Config::default() for M=16:
        // P(level >= L) = M^{-L}.
        level_mult: Config::level_mult_for_m(parameters.m as u8),
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

fn build_upstream(
    vectors: &[Vec<f32>],
    parameters: Parameters,
) -> Hnsw<'static, f32, UpstreamInnerProduct> {
    let index = Hnsw::<f32, UpstreamInnerProduct>::new(
        parameters.m,
        parameters.vectors,
        16,
        parameters.ef_construction,
        UpstreamInnerProduct,
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
                .map(|(id, vector)| (dot(vector, query).expect("equal lengths"), id as u32))
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
    env_usize_optional(name).unwrap_or(default)
}

fn env_usize_optional(name: &str) -> Option<usize> {
    env::var(name).ok().map(|value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
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

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn loads_fixture_and_scores_qrels() -> Result<()> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = env::temp_dir().join(format!(
            "hnsw-rs-fixture-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory)?;
        fs::write(
            directory.join("manifest.txt"),
            "format=hnsw-rs-embedding-fixture-v1\n\
             dataset=test\n\
             model=test-model\n\
             dimensions=2\n\
             corpus_count=3\n\
             query_count=2\n",
        )?;
        write_vectors(
            &directory.join("corpus.f32"),
            &[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        )?;
        write_vectors(&directory.join("queries.f32"), &[[1.0, 0.0], [-1.0, 0.0]])?;
        fs::write(directory.join("corpus-ids.txt"), "d1\nd2\nd3\n")?;
        fs::write(directory.join("query-ids.txt"), "q1\nq2\n")?;
        fs::write(
            directory.join("qrels-test.tsv"),
            "query-id\tcorpus-id\tscore\nq1\td1\t2\nq2\td3\t1\n",
        )?;

        let dataset = load_fixture(&directory, None, None)?;
        assert_eq!(dataset.vectors.len(), 3);
        assert_eq!(dataset.queries.len(), 2);
        let semantic = semantic_metrics(
            dataset.relevance.as_ref().expect("test qrels"),
            &[vec![0], vec![2]],
            1,
        );
        assert_eq!(semantic.ndcg, 1.0);
        assert_eq!(semantic.recall, 1.0);

        let parameters = Parameters {
            vectors: dataset.vectors.len(),
            dimensions: 2,
            queries: dataset.queries.len(),
            repetitions: 1,
            k: 1,
            m: 2,
            ef_construction: 10,
            ef_search: 10,
            seed: 42,
        };
        let ours = build_ours(&dataset.vectors, parameters);
        let upstream = build_upstream(&dataset.vectors, parameters);
        let usearch = build_usearch(&dataset.vectors, parameters);
        for query in &dataset.queries {
            assert_eq!(ours.search(query, 1)?.len(), 1);
            assert_eq!(upstream.search(query, 1, 10).len(), 1);
            assert_eq!(usearch.search(query, 1)?.keys.len(), 1);
        }

        fs::remove_dir_all(directory)?;
        Ok(())
    }

    fn write_vectors<const DIMENSIONS: usize>(
        path: &Path,
        vectors: &[[f32; DIMENSIONS]],
    ) -> Result<()> {
        let mut file = File::create(path)?;
        for vector in vectors {
            for value in vector {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }
}
