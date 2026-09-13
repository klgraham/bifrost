use std::{
    collections::HashMap,
    env,
    error::Error,
    fs,
    hint::black_box,
    io,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

use bifrost::{Config, HnswIndex, vector::dot};
use bifrost_benchmark_fixture::{Artifact, ArtifactResolution, MANIFEST_FILE, VerifiedFixture};
use hnsw_rs_upstream::prelude::{Distance, Hnsw};
use usearch::{Index as UsearchIndex, IndexOptions, MetricKind, ScalarKind};

const UPSTREAM_VERSION: &str = "0.3.4";
const USEARCH_VERSION: &str = "2.26.0";
const LEGACY_FIXTURE_FORMAT: &str = "hnsw-rs-embedding-fixture-v1";

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
    artifact: Option<ArtifactIdentity>,
}

#[derive(Debug, PartialEq, Eq)]
enum ArtifactIdentity {
    Resolved(ArtifactResolution),
    Local {
        artifact_id: String,
        revision: String,
    },
}

impl ArtifactIdentity {
    fn report_lines(&self) -> Vec<String> {
        match self {
            Self::Resolved(resolution) => vec![
                format!("- Artifact: {}", resolution.artifact_id),
                format!("- Artifact repository: {}", resolution.repository),
                format!("- Artifact revision: {}", resolution.revision),
                format!(
                    "- Artifact manifest SHA-256: {}",
                    resolution.manifest_sha256
                ),
            ],
            Self::Local {
                artifact_id,
                revision,
            } => vec![
                format!("- Artifact: {artifact_id}"),
                format!("- Artifact revision: {revision}"),
            ],
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Relevance {
    by_query: Vec<Vec<(u32, u32)>>,
    evaluated_queries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
            artifact: None,
        }
    };
    let output_directory = dataset
        .fixture_directory
        .as_deref()
        .map(output_directory_from_env)
        .transpose()?;
    let ef_searches =
        env_usize_list("HNSW_BENCH_EF_SEARCHES").unwrap_or_else(|| vec![parameters.ef_search]);
    parameters.validate(&ef_searches);

    let vectors = &dataset.vectors;
    let queries = &dataset.queries;

    let truth_started = Instant::now();
    let truth = exact_top_k(vectors, queries, parameters.k);
    let truth_elapsed = truth_started.elapsed();

    println!("Building Bifrost (local checkout)...");
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
    if let Some(artifact) = &dataset.artifact {
        for line in artifact.report_lines() {
            println!("{line}");
        }
    }
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
        ("Bifrost (this crate)", ours_build),
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
                .expect("Bifrost search failed")
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
                name: "Bifrost (this crate)",
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
    if let Some(directory) = output_directory {
        let path = save_bifrost_index(&ours, &directory, parameters)?;
        println!("Saved the Bifrost index to {}.", path.display());
    }
    Ok(())
}

fn save_bifrost_index(
    index: &HnswIndex,
    output_directory: &Path,
    parameters: Parameters,
) -> Result<PathBuf> {
    let path = output_directory.join(format!(
        "bifrost-m{}-efc{}-seed{}.hnsw",
        parameters.m, parameters.ef_construction, parameters.seed
    ));
    index.save(&path)?;
    Ok(path)
}

fn load_fixture(
    directory: &Path,
    vector_limit: Option<usize>,
    query_limit: Option<usize>,
) -> Result<Dataset> {
    load_fixture_with_expected(directory, vector_limit, query_limit, None)
}

fn load_fixture_with_expected(
    directory: &Path,
    vector_limit: Option<usize>,
    query_limit: Option<usize>,
    expected: Option<&Artifact>,
) -> Result<Dataset> {
    if directory.join(MANIFEST_FILE).exists() || expected.is_some() {
        load_v2_fixture(directory, vector_limit, query_limit, expected)
    } else {
        load_legacy_fixture(directory, vector_limit, query_limit)
    }
}

fn load_v2_fixture(
    directory: &Path,
    vector_limit: Option<usize>,
    query_limit: Option<usize>,
    expected: Option<&Artifact>,
) -> Result<Dataset> {
    let fixture = VerifiedFixture::load(directory, expected)?;
    let corpus_count = fixture.corpus_vectors.len();
    let query_count = fixture.query_vectors.len();
    let vectors_to_load = vector_limit.unwrap_or(corpus_count).min(corpus_count);
    let queries_to_load = query_limit.unwrap_or(query_count).min(query_count);
    if vectors_to_load == 0 || queries_to_load == 0 {
        return Err("fixture vector and query counts must be positive".into());
    }

    let vectors = fixture
        .corpus_vectors
        .into_iter()
        .take(vectors_to_load)
        .collect::<Vec<_>>();
    let queries = fixture
        .query_vectors
        .into_iter()
        .take(queries_to_load)
        .collect::<Vec<_>>();
    let corpus_ids = fixture
        .corpus_ids
        .into_iter()
        .take(vectors_to_load)
        .collect::<Vec<_>>();
    let query_ids = fixture
        .query_ids
        .into_iter()
        .take(queries_to_load)
        .collect::<Vec<_>>();
    let relevance = fixture
        .qrels_path
        .as_deref()
        .map(|path| load_qrels(path, &corpus_ids, &query_ids))
        .transpose()?;
    let artifact = match fixture.resolution {
        Some(resolution) => ArtifactIdentity::Resolved(resolution),
        None => ArtifactIdentity::Local {
            artifact_id: fixture.metadata.artifact_id.clone(),
            revision: "local-unpublished".to_owned(),
        },
    };
    let metadata = fixture.metadata;

    println!(
        "Loaded {} indexed vectors and {} queries from {}...",
        vectors.len(),
        queries.len(),
        directory.display()
    );
    Ok(Dataset {
        label: format!(
            "{}, {}, {} dimensions ({} corpus vectors, {} queries)",
            metadata.dataset,
            metadata.requested_model,
            metadata.dimensions,
            vectors.len(),
            queries.len()
        ),
        vectors,
        queries,
        relevance,
        fixture_directory: Some(directory.to_owned()),
        artifact: Some(artifact),
    })
}

fn load_legacy_fixture(
    directory: &Path,
    vector_limit: Option<usize>,
    query_limit: Option<usize>,
) -> Result<Dataset> {
    let manifest = load_manifest(&directory.join("manifest.txt"))?;
    if manifest.get("format").map(String::as_str) != Some(LEGACY_FIXTURE_FORMAT) {
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
        artifact: Some(ArtifactIdentity::Local {
            artifact_id: "legacy-local-fixture".to_owned(),
            revision: env::var("HNSW_BENCH_ARTIFACT_REVISION")
                .unwrap_or_else(|_| "local-unversioned".to_owned()),
        }),
    })
}

fn output_directory_from_env(fixture_directory: &Path) -> Result<PathBuf> {
    let output = env::var_os("HNSW_BENCH_OUTPUT_DIR")
        .ok_or("HNSW_BENCH_OUTPUT_DIR must name a writable directory that does not overlap HNSW_BENCH_FIXTURE")?;
    if output.is_empty() {
        return Err("HNSW_BENCH_OUTPUT_DIR must not be empty".into());
    }
    prepare_output_directory(fixture_directory, Path::new(&output))
}

fn prepare_output_directory(fixture_directory: &Path, output_directory: &Path) -> Result<PathBuf> {
    let fixture_directory = fs::canonicalize(fixture_directory)?;
    let output_directory = comparable_path(output_directory)?;
    reject_fixture_output(&fixture_directory, &output_directory)?;
    fs::create_dir_all(&output_directory)?;
    let output_directory = fs::canonicalize(output_directory)?;
    reject_fixture_output(&fixture_directory, &output_directory)?;
    Ok(output_directory)
}

fn reject_fixture_output(fixture_directory: &Path, output_directory: &Path) -> Result<()> {
    if output_directory.starts_with(fixture_directory)
        || fixture_directory.starts_with(output_directory)
    {
        return Err(format!(
            "HNSW_BENCH_OUTPUT_DIR ({}) must not overlap HNSW_BENCH_FIXTURE ({})",
            output_directory.display(),
            fixture_directory.display()
        )
        .into());
    }
    Ok(())
}

fn comparable_path(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        env::current_dir()?.join(path)
    };
    let mut missing = Vec::new();
    let mut existing = absolute.as_path();
    while !existing.exists() {
        let name = existing.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no existing ancestor for {}", path.display()),
            )
        })?;
        missing.push(name.to_owned());
        existing = existing.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no existing ancestor for {}", path.display()),
            )
        })?;
    }
    let mut resolved = fs::canonicalize(existing)?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Ok(normalize_path(&resolved))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
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
        ..Config::default()
    })
    .expect("valid Bifrost benchmark configuration");
    for (id, vector) in vectors.iter().enumerate() {
        index
            .insert(id as u32, vector)
            .expect("Bifrost insertion failed");
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
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use bifrost_benchmark_fixture::{
        FIXTURE_SCHEMA, FixtureCounts, FixtureFile, FixtureFileKind, FixtureManifest, Publication,
        RESOLUTION_FILE, SourceMetadata, VectorRepresentation, sha256_hex,
    };

    #[test]
    fn v2_local_and_resolved_fixtures_have_identical_inputs_and_results() -> Result<()> {
        let local_directory = TestDirectory::new("v2-local-fixture")?;
        let resolved_directory = TestDirectory::new("v2-resolved-fixture")?;
        write_v2_fixture(local_directory.path())?;
        write_v2_fixture_with_resolution(resolved_directory.path())?;
        let manifest_sha256 = sha256_hex(&fs::read(resolved_directory.path().join(MANIFEST_FILE))?);

        let local = load_fixture(local_directory.path(), None, None)?;
        let dataset = load_fixture(resolved_directory.path(), None, None)?;
        for relative in [
            MANIFEST_FILE,
            "corpus.f32",
            "queries.f32",
            "corpus-ids.txt",
            "query-ids.txt",
            "qrels-test.tsv",
        ] {
            assert_eq!(
                fs::read(local_directory.path().join(relative))?,
                fs::read(resolved_directory.path().join(relative))?,
                "local and resolved fixtures differ at {relative}"
            );
        }
        assert_eq!(dataset.vectors, local.vectors);
        assert_eq!(dataset.queries, local.queries);
        assert_eq!(dataset.relevance, local.relevance);
        assert_eq!(
            dataset.artifact,
            Some(ArtifactIdentity::Resolved(ArtifactResolution {
                artifact_id: "test-v2-2d".to_owned(),
                repository: "test/bifrost-benchmarks".to_owned(),
                revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                manifest_sha256: manifest_sha256.clone(),
            }))
        );
        assert_eq!(
            dataset
                .artifact
                .as_ref()
                .expect("resolved fixture artifact")
                .report_lines(),
            vec![
                "- Artifact: test-v2-2d".to_owned(),
                "- Artifact repository: test/bifrost-benchmarks".to_owned(),
                "- Artifact revision: 0123456789abcdef0123456789abcdef01234567".to_owned(),
                format!("- Artifact manifest SHA-256: {manifest_sha256}"),
            ]
        );
        assert_eq!(dataset.vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(dataset.queries, vec![vec![1.0, 0.0]]);

        let parameters = test_parameters(&dataset);
        let truth = exact_top_k(&dataset.vectors, &dataset.queries, parameters.k);
        assert_eq!(
            truth,
            exact_top_k(&local.vectors, &local.queries, parameters.k)
        );
        let ours = build_ours(&dataset.vectors, parameters);
        let upstream = build_upstream(&dataset.vectors, parameters);
        let usearch = build_usearch(&dataset.vectors, parameters);
        let ours_results = measure_search(&dataset.queries, 1, |query| {
            ours.search(query, parameters.k)
                .expect("Bifrost test search")
                .into_iter()
                .map(|hit| hit.id)
                .collect()
        })
        .ids;
        let upstream_results = measure_search(&dataset.queries, 1, |query| {
            upstream
                .search(query, parameters.k, parameters.ef_search)
                .into_iter()
                .map(|neighbour| u32::try_from(neighbour.d_id).expect("test IDs must fit in u32"))
                .collect()
        })
        .ids;
        let usearch_results = measure_search(&dataset.queries, 1, |query| {
            usearch
                .search(query, parameters.k)
                .expect("USearch test search")
                .keys
                .into_iter()
                .map(|id| u32::try_from(id).expect("test IDs must fit in u32"))
                .collect()
        })
        .ids;
        assert_eq!(ours_results, truth);
        assert_eq!(upstream_results, truth);
        assert_eq!(usearch_results, truth);
        let relevance = dataset.relevance.as_ref().expect("v2 test qrels");
        let expected_semantic = semantic_metrics(relevance, &truth, parameters.k);
        assert_eq!(
            semantic_metrics(relevance, &ours_results, parameters.k),
            expected_semantic
        );
        assert_eq!(
            semantic_metrics(relevance, &upstream_results, parameters.k),
            expected_semantic
        );
        assert_eq!(
            semantic_metrics(relevance, &usearch_results, parameters.k),
            expected_semantic
        );
        Ok(())
    }

    #[test]
    fn propagates_catalog_mismatch_from_shared_validation() -> Result<()> {
        let directory = TestDirectory::new("v2-catalog-mismatch")?;
        write_v2_fixture(directory.path())?;
        let expected = Artifact {
            id: "different-artifact".to_owned(),
            dataset: "test".to_owned(),
            model: "test-model".to_owned(),
            dimensions: 2,
            publication: Publication::Planned {
                repository: "test/bifrost-benchmarks".to_owned(),
                prefix: "fixtures/{manifest_sha256}".to_owned(),
                blocked_by: "test fixture is not published".to_owned(),
            },
        };

        let shared_error = match VerifiedFixture::load(directory.path(), Some(&expected)) {
            Ok(_) => panic!("catalog mismatch must fail"),
            Err(error) => error.to_string(),
        };
        let benchmark_error = error_message(load_fixture_with_expected(
            directory.path(),
            None,
            None,
            Some(&expected),
        ));
        assert_eq!(benchmark_error, shared_error);
        Ok(())
    }

    #[test]
    fn loads_legacy_v1_fixture_and_scores_qrels() -> Result<()> {
        let directory = TestDirectory::new("legacy-fixture")?;
        write_legacy_fixture(directory.path())?;

        let dataset = load_fixture(directory.path(), None, None)?;
        assert_eq!(dataset.vectors.len(), 3);
        assert_eq!(dataset.queries.len(), 2);
        let Some(ArtifactIdentity::Local { artifact_id, .. }) = dataset.artifact.as_ref() else {
            panic!("legacy fixture must have local artifact identity");
        };
        assert_eq!(artifact_id, "legacy-local-fixture");
        let semantic = semantic_metrics(
            dataset.relevance.as_ref().expect("test qrels"),
            &[vec![0], vec![2]],
            1,
        );
        assert_eq!(semantic.ndcg, 1.0);
        assert_eq!(semantic.recall, 1.0);

        let parameters = test_parameters(&dataset);
        let ours = build_ours(&dataset.vectors, parameters);
        let upstream = build_upstream(&dataset.vectors, parameters);
        let usearch = build_usearch(&dataset.vectors, parameters);
        for query in &dataset.queries {
            assert_eq!(ours.search(query, 1)?.len(), 1);
            assert_eq!(upstream.search(query, 1, 10).len(), 1);
            assert_eq!(usearch.search(query, 1)?.keys.len(), 1);
        }

        Ok(())
    }

    #[test]
    fn read_only_fixture_benchmark_writes_only_to_separate_output() -> Result<()> {
        let fixture = TestDirectory::new("read-only-fixture")?;
        let output = TestDirectory::new("benchmark-output")?;
        write_v2_fixture(fixture.path())?;
        make_fixture_read_only(fixture.path())?;
        let fixture_before = snapshot_files(fixture.path())?;

        let dataset = load_fixture(fixture.path(), None, None)?;
        let parameters = test_parameters(&dataset);
        let index = build_ours(&dataset.vectors, parameters);
        let output = prepare_output_directory(fixture.path(), output.path())?;
        let index_path = save_bifrost_index(&index, &output, parameters)?;

        assert!(index_path.starts_with(&output));
        assert!(index_path.is_file());
        assert_eq!(snapshot_files(fixture.path())?, fixture_before);
        assert!(!fixture.path().join("indexes").exists());
        Ok(())
    }

    #[test]
    fn rejects_output_paths_that_overlap_fixture() -> Result<()> {
        let cache = TestDirectory::new("fixture-output-guard")?;
        let fixture = cache.path().join("cache").join("artifact");
        fs::create_dir_all(&fixture)?;
        write_legacy_fixture(&fixture)?;

        let equal = error_message(prepare_output_directory(&fixture, &fixture));
        assert!(equal.contains("must not overlap HNSW_BENCH_FIXTURE"));

        let nested = fixture.join("indexes").join("run");
        let nested_error = error_message(prepare_output_directory(&fixture, &nested));
        assert!(nested_error.contains("must not overlap HNSW_BENCH_FIXTURE"));
        assert!(!nested.exists());

        let ancestor = cache.path().join("cache");
        let ancestor_error = error_message(prepare_output_directory(&fixture, &ancestor));
        assert!(ancestor_error.contains("must not overlap HNSW_BENCH_FIXTURE"));
        Ok(())
    }

    fn write_legacy_fixture(directory: &Path) -> Result<()> {
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
        Ok(())
    }

    fn write_v2_fixture(directory: &Path) -> Result<()> {
        let corpus_vectors = vector_bytes(&[[1.0, 0.0], [0.0, 1.0]]);
        let query_vectors = vector_bytes(&[[1.0, 0.0]]);
        let corpus_ids = b"d1\nd2\n";
        let query_ids = b"q1\n";
        let qrels = b"query-id\tcorpus-id\tscore\nq1\td1\t2\n";
        fs::write(directory.join("corpus.f32"), &corpus_vectors)?;
        fs::write(directory.join("queries.f32"), &query_vectors)?;
        fs::write(directory.join("corpus-ids.txt"), corpus_ids)?;
        fs::write(directory.join("query-ids.txt"), query_ids)?;
        fs::write(directory.join("qrels-test.tsv"), qrels)?;

        FixtureManifest {
            schema: FIXTURE_SCHEMA.to_owned(),
            artifact_id: "test-v2-2d".to_owned(),
            dataset: "test".to_owned(),
            source: SourceMetadata {
                url: "https://example.com/test".to_owned(),
                revision: "test-source-v1".to_owned(),
                sha256: sha256_hex(b"test source"),
                license: "CC0-1.0".to_owned(),
            },
            selection: "all test fixture records".to_owned(),
            preprocessing: "title, then one space, then body".to_owned(),
            corpus_input_sha256: sha256_hex(b"d1\0first\nd2\0second\n"),
            query_input_sha256: sha256_hex(b"q1\0first query\n"),
            requested_model: "test-model".to_owned(),
            returned_model: "test-model".to_owned(),
            dimensions: 2,
            normalization: "l2".to_owned(),
            derivation: None,
            generator_revision: "1".repeat(40),
            representation: VectorRepresentation {
                scalar: "f32".to_owned(),
                endianness: "little".to_owned(),
            },
            counts: FixtureCounts {
                corpus: 2,
                queries: 1,
            },
            files: vec![
                fixture_file(
                    FixtureFileKind::CorpusVectors,
                    "corpus.f32",
                    &corpus_vectors,
                ),
                fixture_file(FixtureFileKind::QueryVectors, "queries.f32", &query_vectors),
                fixture_file(FixtureFileKind::CorpusIds, "corpus-ids.txt", corpus_ids),
                fixture_file(FixtureFileKind::QueryIds, "query-ids.txt", query_ids),
                fixture_file(FixtureFileKind::Qrels, "qrels-test.tsv", qrels),
            ],
        }
        .write(directory.join(MANIFEST_FILE))?;
        Ok(())
    }

    fn write_v2_fixture_with_resolution(directory: &Path) -> Result<()> {
        write_v2_fixture(directory)?;
        let manifest = fs::read(directory.join(MANIFEST_FILE))?;
        ArtifactResolution {
            artifact_id: "test-v2-2d".to_owned(),
            repository: "test/bifrost-benchmarks".to_owned(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            manifest_sha256: sha256_hex(&manifest),
        }
        .write(directory.join(RESOLUTION_FILE))?;
        Ok(())
    }

    fn fixture_file(kind: FixtureFileKind, path: &str, bytes: &[u8]) -> FixtureFile {
        FixtureFile {
            kind,
            path: path.to_owned(),
            byte_count: bytes.len() as u64,
            sha256: sha256_hex(bytes),
        }
    }

    fn vector_bytes<const DIMENSIONS: usize>(vectors: &[[f32; DIMENSIONS]]) -> Vec<u8> {
        vectors
            .iter()
            .flat_map(|vector| vector.iter())
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[cfg(unix)]
    fn make_fixture_read_only(directory: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o444))?;
        }
        fs::set_permissions(directory, fs::Permissions::from_mode(0o555))?;
        Ok(())
    }

    #[cfg(not(unix))]
    fn make_fixture_read_only(_directory: &Path) -> Result<()> {
        Ok(())
    }

    fn test_parameters(dataset: &Dataset) -> Parameters {
        Parameters {
            vectors: dataset.vectors.len(),
            dimensions: 2,
            queries: dataset.queries.len(),
            repetitions: 1,
            k: 1,
            m: 2,
            ef_construction: 10,
            ef_search: 10,
            seed: 42,
        }
    }

    fn snapshot_files(directory: &Path) -> Result<Vec<(PathBuf, Vec<u8>)>> {
        let mut snapshot = fs::read_dir(directory)?
            .map(|entry| {
                let path = entry?.path();
                let bytes = fs::read(&path)?;
                let name = PathBuf::from(path.file_name().expect("fixture entry name"));
                Ok::<_, io::Error>((name, bytes))
            })
            .collect::<io::Result<Vec<_>>>()?;
        snapshot.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        Ok(snapshot)
    }

    fn error_message<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(error) => error.to_string(),
        }
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

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Result<Self> {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let path =
                env::temp_dir().join(format!("bifrost-{label}-{}-{nonce}", std::process::id()));
            fs::create_dir_all(&path)?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            #[cfg(unix)]
            restore_writable(&self.0);
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    fn restore_writable(path: &Path) {
        use std::os::unix::fs::PermissionsExt;

        let Ok(metadata) = fs::symlink_metadata(path) else {
            return;
        };
        if metadata.file_type().is_symlink() {
            return;
        }
        if metadata.is_dir() {
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o755));
        } else {
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o644));
            return;
        }
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                restore_writable(&entry.path());
            }
        }
    }
}
