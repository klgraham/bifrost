use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use bifrost_benchmark_fixture::{
    Artifact, FIXTURE_SCHEMA, FixtureCounts, FixtureFile, FixtureFileKind, FixtureManifest,
    SourceMetadata, VectorRepresentation, VerifiedFixture, input_sha256,
};
use reqwest::{StatusCode, blocking::Client};
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

use crate::{
    ToolResult, context, error,
    util::{
        ArtifactLock, atomic_write, atomic_write_under, clean_stale_stages, create_dir_all_under,
        ensure_safe_descendant, install_directory, is_commit_sha, is_sha256, make_tree_read_only,
        prepare_mutation_root, reject_overlapping_paths, remove_tree, sha256_bytes, sha256_file,
        unique_stage,
    },
};

pub const FIQA_URL: &str =
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip";
pub const FIQA_SHA256: &str = "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02";
pub const FIQA_REVISION: &str = "BEIR-fiqa.zip-sha256-32c7df99";
pub const FIQA_LICENSE: &str = "CC BY-SA 4.0";
pub const PREPROCESSING: &str = "sort IDs ascending; corpus trims title/text and joins both with two newlines; queries trim text and include FiQA test-qrel IDs";
pub const SELECTION: &str = "all nonempty corpus records; test-qrel queries with an indexed corpus document; test qrels filtered to loaded query and document IDs";

const ARCHIVE_NAME: &str = "fiqa.zip";
const MANIFEST_NAME: &str = "manifest.json";
const CORPUS_VECTORS: &str = "corpus.f32";
const QUERY_VECTORS: &str = "queries.f32";
const CORPUS_IDS: &str = "corpus-ids.txt";
const QUERY_IDS: &str = "query-ids.txt";
const QRELS: &str = "qrels-test.tsv";
const MAX_BATCH_INPUTS: usize = 2048;

#[derive(Debug, Clone)]
pub struct FiqaSourceSpec {
    pub url: String,
    pub revision: String,
    pub sha256: String,
    pub license: String,
}

impl Default for FiqaSourceSpec {
    fn default() -> Self {
        Self {
            url: FIQA_URL.to_owned(),
            revision: FIQA_REVISION.to_owned(),
            sha256: FIQA_SHA256.to_owned(),
            license: FIQA_LICENSE.to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FiqaOptions {
    pub artifact: Artifact,
    pub workspace: PathBuf,
    pub output: PathBuf,
    pub batch_size: usize,
    pub repair_batches: bool,
    pub generator_revision: String,
    pub source: FiqaSourceSpec,
}

impl FiqaOptions {
    fn validate(&self) -> ToolResult<()> {
        if !(1..=MAX_BATCH_INPUTS).contains(&self.batch_size) {
            return error(format!(
                "batch size must be between 1 and {MAX_BATCH_INPUTS}"
            ));
        }
        if !is_commit_sha(&self.generator_revision) {
            return error("generator revision must be a 40-character lowercase Git commit");
        }
        if self.source.url.trim().is_empty()
            || self.source.revision.trim().is_empty()
            || self.source.license.trim().is_empty()
            || !is_sha256(&self.source.sha256)
        {
            return error("FiQA source metadata is incomplete or invalid");
        }
        Ok(())
    }
}

pub trait SourceArchive: Send + Sync {
    fn download(&self, url: &str) -> ToolResult<Vec<u8>>;
}

pub struct HttpFiqaSource {
    client: Client,
}

impl HttpFiqaSource {
    pub fn new() -> ToolResult<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .user_agent("bifrost-benchmark-artifacts/1")
                .build()?,
        })
    }
}

impl SourceArchive for HttpFiqaSource {
    fn download(&self, url: &str) -> ToolResult<Vec<u8>> {
        eprintln!("Downloading FiQA source archive from {url}");
        let bytes = self.client.get(url).send()?.error_for_status()?.bytes()?;
        Ok(bytes.to_vec())
    }
}

pub trait Embedder: Send + Sync {
    fn embed(
        &self,
        model: &str,
        dimensions: usize,
        inputs: &[String],
    ) -> ToolResult<EmbeddingBatch>;
}

pub struct OpenAiEmbedder {
    client: Client,
    endpoint: String,
    api_key_environment: String,
}

impl OpenAiEmbedder {
    pub fn new(
        endpoint: impl Into<String>,
        api_key_environment: impl Into<String>,
    ) -> ToolResult<Self> {
        let endpoint = endpoint.into();
        if endpoint.trim().is_empty() {
            return error("OpenAI endpoint must not be empty");
        }
        let api_key_environment = api_key_environment.into();
        if api_key_environment.trim().is_empty() {
            return error("OpenAI API-key environment variable must not be empty");
        }
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(180))
                .user_agent("bifrost-benchmark-artifacts/1")
                .build()?,
            endpoint,
            api_key_environment,
        })
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    dimensions: usize,
    encoding_format: &'static str,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct EmbeddingBatch {
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    #[serde(default)]
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct EmbeddingItem {
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct EmbeddingUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

impl Embedder for OpenAiEmbedder {
    fn embed(
        &self,
        model: &str,
        dimensions: usize,
        inputs: &[String],
    ) -> ToolResult<EmbeddingBatch> {
        let api_key = std::env::var(&self.api_key_environment).map_err(|_| {
            crate::ToolError::new(format!(
                "{} must be set to generate embeddings",
                self.api_key_environment
            ))
        })?;
        if api_key.trim().is_empty() {
            return error(format!("{} is empty", self.api_key_environment));
        }
        let request = EmbeddingRequest {
            model,
            input: inputs,
            dimensions,
            encoding_format: "float",
        };
        let mut last_error = String::new();
        for attempt in 0..6 {
            let response = self
                .client
                .post(&self.endpoint)
                .bearer_auth(&api_key)
                .json(&request)
                .send();
            match response {
                Ok(response) if response.status().is_success() => return Ok(response.json()?),
                Ok(response) => {
                    let status = response.status();
                    let retryable =
                        status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                    last_error = format!(
                        "OpenAI embeddings request failed with {status}: {}",
                        response.text()?
                    );
                    if !retryable {
                        return error(last_error);
                    }
                }
                Err(request_error) => {
                    last_error = format!("OpenAI embeddings request failed: {request_error}");
                }
            }
            if attempt < 5 {
                thread::sleep(Duration::from_secs(1_u64 << attempt));
            }
        }
        error(last_error)
    }
}

#[derive(Debug, Deserialize)]
struct BeirRecord {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(Debug, Clone)]
struct TextRecord {
    id: String,
    text: String,
}

#[derive(Debug, Clone)]
struct Qrel {
    query_id: String,
    corpus_id: String,
    score: u32,
}

#[derive(Debug, Deserialize, Serialize)]
struct CachedBatch {
    schema: String,
    role: String,
    requested_model: String,
    dimensions: usize,
    preprocessing: String,
    generator_revision: String,
    ids: Vec<String>,
    input_sha256: String,
    response: EmbeddingBatch,
}

pub fn prepare_fiqa(
    options: &FiqaOptions,
    source: &dyn SourceArchive,
    embedder: &dyn Embedder,
) -> ToolResult<PathBuf> {
    options.validate()?;
    reject_overlapping_paths(
        &options.workspace,
        "prepare workspace",
        &options.output,
        "fixture output",
    )?;
    let output_parent = options.output.parent().ok_or_else(|| {
        crate::ToolError::new(format!(
            "{} has no parent directory",
            options.output.display()
        ))
    })?;
    prepare_mutation_root(output_parent, "fixture output parent")?;
    prepare_mutation_root(&options.workspace, "prepare workspace")?;
    ensure_safe_descendant(output_parent, &options.output, "fixture output")?;
    if fixture_is_complete(&options.output, options) {
        make_tree_read_only(&options.output)?;
        return Ok(fs::canonicalize(&options.output)?);
    }

    let _lock = ArtifactLock::acquire(&options.workspace, &options.artifact.id)?;
    ensure_safe_descendant(output_parent, &options.output, "fixture output")?;
    if fixture_is_complete(&options.output, options) {
        make_tree_read_only(&options.output)?;
        return Ok(fs::canonicalize(&options.output)?);
    }

    let source_directory = options
        .workspace
        .join("source")
        .join(&options.source.sha256);
    create_dir_all_under(
        &options.workspace,
        &source_directory,
        "FiQA source cache directory",
    )?;
    let archive_path = source_directory.join(ARCHIVE_NAME);
    ensure_safe_descendant(&options.workspace, &archive_path, "FiQA source archive")?;
    ensure_archive(&options.workspace, &archive_path, &options.source, source)?;
    let extracted = extract_source_files(&archive_path, &source_directory)?;
    let corpus = load_corpus(&extracted.corpus)?;
    let qrels = load_qrels(&extracted.qrels)?;
    let corpus_ids = corpus
        .iter()
        .map(|record| record.id.as_str())
        .collect::<HashSet<_>>();
    let queries = load_queries(&extracted.queries, &qrels, &corpus_ids)?;
    let query_ids = queries
        .iter()
        .map(|record| record.id.as_str())
        .collect::<HashSet<_>>();
    let qrels = qrels
        .into_iter()
        .filter(|qrel| {
            query_ids.contains(qrel.query_id.as_str())
                && corpus_ids.contains(qrel.corpus_id.as_str())
        })
        .collect::<Vec<_>>();

    let stage_root = output_parent.join(".staging");
    create_dir_all_under(output_parent, &stage_root, "fixture staging root")?;
    clean_stale_stages(&stage_root, &options.artifact.id)?;
    let stage = unique_stage(&stage_root, &options.artifact.id);
    create_dir_all_under(&stage_root, &stage, "fixture staging directory")?;
    let result = (|| -> ToolResult<()> {
        context(
            build_fixture(options, embedder, &corpus, &queries, &qrels, &stage),
            "building staged fixture",
        )?;
        context(
            VerifiedFixture::load(&stage, Some(&options.artifact)),
            "verifying staged fixture",
        )?;
        context(
            make_tree_read_only(&stage),
            "making staged fixture read-only",
        )?;
        context(
            install_directory(output_parent, &stage, &options.output),
            "installing staged fixture",
        )?;
        Ok(())
    })();
    if result.is_err() && crate::util::path_present(&stage) {
        let _ = remove_tree(&stage);
    }
    result?;
    Ok(fs::canonicalize(&options.output)?)
}

fn fixture_is_complete(path: &Path, options: &FiqaOptions) -> bool {
    let Ok(fixture) = VerifiedFixture::load(path, Some(&options.artifact)) else {
        return false;
    };
    let metadata = &fixture.metadata;
    metadata.source.url == options.source.url
        && metadata.source.revision == options.source.revision
        && metadata.source.sha256 == options.source.sha256
        && metadata.source.license == options.source.license
        && metadata.preprocessing == PREPROCESSING
        && metadata.selection == SELECTION
        && metadata.requested_model == options.artifact.model
        && metadata.returned_model == options.artifact.model
        && metadata.dimensions == options.artifact.dimensions
        && metadata.normalization == "l2"
        && metadata.representation.scalar == "f32"
        && metadata.representation.endianness == "little"
        && metadata.generator_revision == options.generator_revision
        && metadata.derivation.as_deref()
            == Some(expected_derivation(options.artifact.dimensions).as_str())
}

fn ensure_archive(
    workspace: &Path,
    path: &Path,
    spec: &FiqaSourceSpec,
    source: &dyn SourceArchive,
) -> ToolResult<()> {
    if path.exists() {
        let actual = sha256_file(path)?;
        if actual == spec.sha256 {
            return Ok(());
        }
        return error(format!(
            "cached FiQA archive {} has SHA-256 {actual}, expected {}; remove it explicitly to redownload",
            path.display(),
            spec.sha256
        ));
    }
    let bytes = source.download(&spec.url)?;
    let actual = sha256_bytes(&bytes);
    if actual != spec.sha256 {
        return error(format!(
            "downloaded FiQA archive has SHA-256 {actual}, expected {}",
            spec.sha256
        ));
    }
    atomic_write_under(workspace, path, &bytes)
}

struct ExtractedSource {
    corpus: PathBuf,
    queries: PathBuf,
    qrels: PathBuf,
}

fn extract_source_files(archive_path: &Path, output: &Path) -> ToolResult<ExtractedSource> {
    let extracted = ExtractedSource {
        corpus: output.join("corpus.jsonl"),
        queries: output.join("queries.jsonl"),
        qrels: output.join(QRELS),
    };
    for (suffix, destination) in [
        ("fiqa/corpus.jsonl", &extracted.corpus),
        ("fiqa/queries.jsonl", &extracted.queries),
        ("fiqa/qrels/test.tsv", &extracted.qrels),
    ] {
        ensure_safe_descendant(output, destination, "extracted FiQA source file")?;
        let file = File::open(archive_path)?;
        let mut archive = ZipArchive::new(file)?;
        let index = (0..archive.len())
            .find(|index| {
                archive
                    .by_index(*index)
                    .is_ok_and(|entry| entry.name().ends_with(suffix))
            })
            .ok_or_else(|| {
                crate::ToolError::new(format!("{suffix} is missing from the FiQA archive"))
            })?;
        let mut entry = archive.by_index(index)?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        if fs::read(destination).map_or(true, |current| current != bytes) {
            atomic_write_under(output, destination, &bytes)?;
        }
    }
    Ok(extracted)
}

fn load_corpus(path: &Path) -> ToolResult<Vec<TextRecord>> {
    let mut records = load_records(path)?
        .into_iter()
        .filter_map(|record| {
            let title = record.title.trim();
            let text = record.text.trim();
            let text = match (title, text) {
                ("", "") => return None,
                ("", text) => text.to_owned(),
                (title, "") => title.to_owned(),
                (title, text) => format!("{title}\n\n{text}"),
            };
            Some(TextRecord {
                id: record.id,
                text,
            })
        })
        .collect::<Vec<_>>();
    records.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    require_unique_records(&records, "FiQA corpus")?;
    if records.is_empty() {
        return error("FiQA corpus contains no nonempty records");
    }
    Ok(records)
}

fn load_queries(
    path: &Path,
    qrels: &[Qrel],
    corpus_ids: &HashSet<&str>,
) -> ToolResult<Vec<TextRecord>> {
    let relevant_query_ids = qrels
        .iter()
        .filter(|qrel| corpus_ids.contains(qrel.corpus_id.as_str()))
        .map(|qrel| qrel.query_id.as_str())
        .collect::<HashSet<_>>();
    let mut queries = load_records(path)?
        .into_iter()
        .filter(|record| relevant_query_ids.contains(record.id.as_str()))
        .map(|record| TextRecord {
            id: record.id,
            text: record.text.trim().to_owned(),
        })
        .collect::<Vec<_>>();
    queries.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    require_unique_records(&queries, "FiQA test queries")?;
    if queries.iter().any(|record| record.text.is_empty()) {
        return error("FiQA test queries contain an empty query");
    }
    let loaded = queries
        .iter()
        .map(|record| record.id.as_str())
        .collect::<HashSet<_>>();
    if loaded != relevant_query_ids {
        return error(
            "FiQA queries do not contain every test-qrel query with an indexed corpus record",
        );
    }
    if queries.is_empty() {
        return error("FiQA test split contains no usable queries");
    }
    Ok(queries)
}

fn load_records(path: &Path) -> ToolResult<Vec<BeirRecord>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line = line?;
            serde_json::from_str(&line).map_err(|parse_error| {
                Box::new(crate::ToolError::new(format!(
                    "{}:{}: invalid JSON: {parse_error}",
                    path.display(),
                    index + 1
                ))) as Box<dyn std::error::Error + Send + Sync>
            })
        })
        .collect()
}

fn require_unique_records(records: &[TextRecord], label: &str) -> ToolResult<()> {
    for pair in records.windows(2) {
        if pair[0].id == pair[1].id {
            return error(format!("{label} repeats ID {:?}", pair[0].id));
        }
    }
    Ok(())
}

fn load_qrels(path: &Path) -> ToolResult<Vec<Qrel>> {
    let mut qrels = Vec::new();
    let mut pairs = HashSet::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if index == 0 {
            if line != "query-id\tcorpus-id\tscore" {
                return error(format!(
                    "{}:1: expected exact query-id, corpus-id, score header",
                    path.display()
                ));
            }
            continue;
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != 3 {
            return error(format!(
                "{}:{}: expected query-id, corpus-id, score",
                path.display(),
                index + 1
            ));
        }
        let score = columns[2].parse::<u32>().map_err(|parse_error| {
            crate::ToolError::new(format!(
                "{}:{}: invalid relevance score: {parse_error}",
                path.display(),
                index + 1
            ))
        })?;
        if columns[0].is_empty() || columns[1].is_empty() {
            return error(format!(
                "{}:{}: query and corpus IDs must be nonempty",
                path.display(),
                index + 1
            ));
        }
        if score == 0 {
            return error(format!(
                "{}:{}: relevance score must be positive",
                path.display(),
                index + 1
            ));
        }
        if !pairs.insert((columns[0].to_owned(), columns[1].to_owned())) {
            return error(format!(
                "{}:{}: duplicate qrel pair {:?}, {:?}",
                path.display(),
                index + 1,
                columns[0],
                columns[1]
            ));
        }
        qrels.push(Qrel {
            query_id: columns[0].to_owned(),
            corpus_id: columns[1].to_owned(),
            score,
        });
    }
    qrels.sort_unstable_by(|left, right| {
        (&left.query_id, &left.corpus_id, left.score).cmp(&(
            &right.query_id,
            &right.corpus_id,
            right.score,
        ))
    });
    if qrels.is_empty() {
        return error("FiQA test qrels are empty");
    }
    Ok(qrels)
}

fn build_fixture(
    options: &FiqaOptions,
    embedder: &dyn Embedder,
    corpus: &[TextRecord],
    queries: &[TextRecord],
    qrels: &[Qrel],
    stage: &Path,
) -> ToolResult<()> {
    let corpus_embeddings = embed_records(options, embedder, "corpus", corpus)?;
    let query_embeddings = embed_records(options, embedder, "queries", queries)?;
    if corpus_embeddings.returned_model != query_embeddings.returned_model {
        return error(format!(
            "embedding response model changed between corpus ({}) and queries ({})",
            corpus_embeddings.returned_model, query_embeddings.returned_model
        ));
    }

    write_vectors(&stage.join(CORPUS_VECTORS), &corpus_embeddings.vectors)?;
    write_vectors(&stage.join(QUERY_VECTORS), &query_embeddings.vectors)?;
    write_ids(&stage.join(CORPUS_IDS), corpus)?;
    write_ids(&stage.join(QUERY_IDS), queries)?;
    write_qrels(&stage.join(QRELS), qrels)?;

    let files = [
        (FixtureFileKind::CorpusVectors, CORPUS_VECTORS),
        (FixtureFileKind::QueryVectors, QUERY_VECTORS),
        (FixtureFileKind::CorpusIds, CORPUS_IDS),
        (FixtureFileKind::QueryIds, QUERY_IDS),
        (FixtureFileKind::Qrels, QRELS),
    ]
    .into_iter()
    .map(|(kind, relative)| fixture_file(stage, kind, relative))
    .collect::<ToolResult<Vec<_>>>()?;
    let manifest = FixtureManifest {
        schema: FIXTURE_SCHEMA.to_owned(),
        artifact_id: options.artifact.id.clone(),
        dataset: options.artifact.dataset.clone(),
        source: SourceMetadata {
            url: options.source.url.clone(),
            revision: options.source.revision.clone(),
            sha256: options.source.sha256.clone(),
            license: options.source.license.clone(),
        },
        preprocessing: PREPROCESSING.to_owned(),
        selection: SELECTION.to_owned(),
        corpus_input_sha256: input_sha256(
            corpus
                .iter()
                .map(|record| (record.id.as_str(), record.text.as_str())),
        ),
        query_input_sha256: input_sha256(
            queries
                .iter()
                .map(|record| (record.id.as_str(), record.text.as_str())),
        ),
        requested_model: options.artifact.model.clone(),
        returned_model: corpus_embeddings.returned_model,
        dimensions: options.artifact.dimensions,
        normalization: "l2".to_owned(),
        derivation: Some(expected_derivation(options.artifact.dimensions)),
        generator_revision: options.generator_revision.clone(),
        representation: VectorRepresentation {
            scalar: "f32".to_owned(),
            endianness: "little".to_owned(),
        },
        counts: FixtureCounts {
            corpus: corpus.len(),
            queries: queries.len(),
        },
        files,
    };
    manifest.write(stage.join(MANIFEST_NAME))?;
    Ok(())
}

fn expected_derivation(dimensions: usize) -> String {
    format!(
        "independent embedding requests at {dimensions} dimensions; not truncated or projected from another fixture"
    )
}

struct PackedEmbeddings {
    returned_model: String,
    vectors: Vec<Vec<f32>>,
}

fn embed_records(
    options: &FiqaOptions,
    embedder: &dyn Embedder,
    role: &str,
    records: &[TextRecord],
) -> ToolResult<PackedEmbeddings> {
    let cache_directory = options
        .workspace
        .join("batches")
        .join(&options.artifact.id)
        .join(role);
    create_dir_all_under(
        &options.workspace,
        &cache_directory,
        "embedding batch cache directory",
    )?;
    let mut vectors = Vec::with_capacity(records.len());
    let mut returned_model: Option<String> = None;
    for (batch_index, batch) in records.chunks(options.batch_size).enumerate() {
        let identity = batch_identity(
            role,
            batch,
            &options.artifact.model,
            options.artifact.dimensions,
            &options.generator_revision,
        );
        let cache_path = cache_directory.join(format!("{identity}.json"));
        ensure_safe_descendant(
            &options.workspace,
            &cache_path,
            "embedding batch cache file",
        )?;
        let cached = match load_cached_batch(&cache_path) {
            Ok(Some(cached)) => match validate_cached_batch(&cached, role, batch, options) {
                Ok(()) => cached,
                Err(_cache_error) if options.repair_batches => {
                    quarantine_cached_batch(&options.workspace, &cache_path)?;
                    request_and_cache_batch(
                        embedder,
                        role,
                        batch,
                        options,
                        &cache_path,
                        batch_index,
                    )?
                }
                Err(cache_error) => {
                    return error(format!(
                        "cached batch {} is invalid: {cache_error}; pass --repair-batches to replace it",
                        cache_path.display()
                    ));
                }
            },
            Err(_cache_error) if options.repair_batches => {
                quarantine_cached_batch(&options.workspace, &cache_path)?;
                request_and_cache_batch(embedder, role, batch, options, &cache_path, batch_index)?
            }
            Err(cache_error) => {
                return error(format!(
                    "cached batch {} cannot be read: {cache_error}; pass --repair-batches to replace it",
                    cache_path.display()
                ));
            }
            Ok(None) => {
                request_and_cache_batch(embedder, role, batch, options, &cache_path, batch_index)?
            }
        };
        validate_cached_batch(&cached, role, batch, options)?;
        match &returned_model {
            Some(model) if model != &cached.response.model => {
                return error(format!(
                    "embedding response model changed from {model} to {}",
                    cached.response.model
                ));
            }
            None => returned_model = Some(cached.response.model.clone()),
            Some(_) => {}
        }
        let mut items = cached.response.data;
        items.sort_unstable_by_key(|item| item.index);
        vectors.extend(items.into_iter().map(|item| item.embedding));
    }
    Ok(PackedEmbeddings {
        returned_model: returned_model
            .ok_or_else(|| crate::ToolError::new("no embeddings were produced"))?,
        vectors,
    })
}

fn quarantine_cached_batch(root: &Path, path: &Path) -> ToolResult<()> {
    let quarantine = path.with_extension(format!("invalid-{}", std::process::id()));
    ensure_safe_descendant(root, path, "embedding batch cache file")?;
    ensure_safe_descendant(root, &quarantine, "embedding batch quarantine")?;
    if quarantine.exists() {
        fs::remove_file(&quarantine)?;
    }
    fs::rename(path, quarantine)?;
    Ok(())
}

fn request_and_cache_batch(
    embedder: &dyn Embedder,
    role: &str,
    batch: &[TextRecord],
    options: &FiqaOptions,
    cache_path: &Path,
    batch_index: usize,
) -> ToolResult<CachedBatch> {
    eprintln!(
        "Embedding {role} batch {batch_index} ({} inputs, {} dimensions)",
        batch.len(),
        options.artifact.dimensions
    );
    let inputs = batch
        .iter()
        .map(|record| record.text.clone())
        .collect::<Vec<_>>();
    let response = embedder.embed(
        &options.artifact.model,
        options.artifact.dimensions,
        &inputs,
    )?;
    let cached = CachedBatch {
        schema: "bifrost-embedding-batch-v2".to_owned(),
        role: role.to_owned(),
        requested_model: options.artifact.model.clone(),
        dimensions: options.artifact.dimensions,
        preprocessing: PREPROCESSING.to_owned(),
        generator_revision: options.generator_revision.clone(),
        ids: batch.iter().map(|record| record.id.clone()).collect(),
        input_sha256: input_sha256(
            batch
                .iter()
                .map(|record| (record.id.as_str(), record.text.as_str())),
        ),
        response,
    };
    validate_cached_batch(&cached, role, batch, options)?;
    let mut bytes = serde_json::to_vec_pretty(&cached)?;
    bytes.push(b'\n');
    atomic_write_under(&options.workspace, cache_path, &bytes)?;
    Ok(cached)
}

fn load_cached_batch(path: &Path) -> ToolResult<Option<CachedBatch>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(read_error) if read_error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(read_error) => Err(read_error.into()),
    }
}

fn validate_cached_batch(
    cached: &CachedBatch,
    role: &str,
    batch: &[TextRecord],
    options: &FiqaOptions,
) -> ToolResult<()> {
    let expected_ids = batch
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let expected_input = input_sha256(
        batch
            .iter()
            .map(|record| (record.id.as_str(), record.text.as_str())),
    );
    if cached.schema != "bifrost-embedding-batch-v2"
        || cached.role != role
        || cached.requested_model != options.artifact.model
        || cached.dimensions != options.artifact.dimensions
        || cached.preprocessing != PREPROCESSING
        || cached.generator_revision != options.generator_revision
        || cached.ids.iter().map(String::as_str).collect::<Vec<_>>() != expected_ids
        || cached.input_sha256 != expected_input
    {
        return error("cached embedding batch does not match its content identity");
    }
    if cached.response.model != options.artifact.model {
        return error(format!(
            "embedding response model {:?} does not match requested model {:?}",
            cached.response.model, options.artifact.model
        ));
    }
    if cached.response.data.len() != batch.len() {
        return error(format!(
            "embedding response contains {} rows for {} inputs",
            cached.response.data.len(),
            batch.len()
        ));
    }
    let mut indexes = cached
        .response
        .data
        .iter()
        .map(|item| item.index)
        .collect::<Vec<_>>();
    indexes.sort_unstable();
    if indexes != (0..batch.len()).collect::<Vec<_>>() {
        return error("embedding response indexes are not exactly 0..input_count");
    }
    for item in &cached.response.data {
        if item.embedding.len() != options.artifact.dimensions {
            return error(format!(
                "embedding {} has {} dimensions; expected {}",
                item.index,
                item.embedding.len(),
                options.artifact.dimensions
            ));
        }
        let norm = item
            .embedding
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 0.01 {
            return error(format!(
                "embedding {} has invalid L2 norm {norm}",
                item.index
            ));
        }
    }
    Ok(())
}

fn batch_identity(
    role: &str,
    batch: &[TextRecord],
    model: &str,
    dimensions: usize,
    generator_revision: &str,
) -> String {
    let input = input_sha256(
        batch
            .iter()
            .map(|record| (record.id.as_str(), record.text.as_str())),
    );
    sha256_bytes(
        format!("bifrost-embedding-batch-identity-v2\0{role}\0{model}\0{dimensions}\0{PREPROCESSING}\0{generator_revision}\0{input}")
            .as_bytes(),
    )
}

fn write_vectors(path: &Path, vectors: &[Vec<f32>]) -> ToolResult<()> {
    let value_count = vectors
        .iter()
        .try_fold(0_usize, |count, vector| count.checked_add(vector.len()))
        .ok_or_else(|| crate::ToolError::new("embedding byte count overflow"))?;
    let mut bytes = Vec::with_capacity(value_count.saturating_mul(size_of::<f32>()));
    for vector in vectors {
        for value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    atomic_write(path, &bytes)
}

fn write_ids(path: &Path, records: &[TextRecord]) -> ToolResult<()> {
    let mut bytes = Vec::new();
    for record in records {
        writeln!(bytes, "{}", record.id)?;
    }
    atomic_write(path, &bytes)
}

fn write_qrels(path: &Path, qrels: &[Qrel]) -> ToolResult<()> {
    let mut bytes = b"query-id\tcorpus-id\tscore\n".to_vec();
    for qrel in qrels {
        writeln!(
            bytes,
            "{}\t{}\t{}",
            qrel.query_id, qrel.corpus_id, qrel.score
        )?;
    }
    atomic_write(path, &bytes)
}

fn fixture_file(root: &Path, kind: FixtureFileKind, relative: &str) -> ToolResult<FixtureFile> {
    let path = root.join(relative);
    Ok(FixtureFile {
        kind,
        path: relative.to_owned(),
        byte_count: fs::metadata(&path)?.len(),
        sha256: sha256_file(&path)?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use bifrost_benchmark_fixture::Publication;

    struct MemorySource {
        bytes: Vec<u8>,
        calls: AtomicUsize,
    }

    impl SourceArchive for MemorySource {
        fn download(&self, _url: &str) -> ToolResult<Vec<u8>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.bytes.clone())
        }
    }

    struct RecordingEmbedder {
        calls: AtomicUsize,
        batches: Mutex<Vec<Vec<String>>>,
    }

    impl Embedder for RecordingEmbedder {
        fn embed(
            &self,
            model: &str,
            dimensions: usize,
            inputs: &[String],
        ) -> ToolResult<EmbeddingBatch> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.batches.lock().unwrap().push(inputs.to_vec());
            let value = 1.0 / (dimensions as f32).sqrt();
            Ok(EmbeddingBatch {
                data: inputs
                    .iter()
                    .enumerate()
                    .map(|(index, _)| EmbeddingItem {
                        embedding: vec![value; dimensions],
                        index,
                    })
                    .collect(),
                model: model.to_owned(),
                usage: EmbeddingUsage::default(),
            })
        }
    }

    fn artifact(dimensions: usize) -> Artifact {
        Artifact {
            id: format!("fiqa-test-{dimensions}"),
            dataset: "BEIR FiQA-2018".to_owned(),
            model: "text-embedding-3-small".to_owned(),
            dimensions,
            publication: Publication::Planned {
                repository: "owner/repository".to_owned(),
                prefix: format!("fixtures/{dimensions}/{{manifest_sha256}}"),
                blocked_by: "test fixture is not for publication".to_owned(),
            },
        }
    }

    fn tiny_archive() -> Vec<u8> {
        let cursor = std::io::Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("fiqa/corpus.jsonl", options).unwrap();
        writer
            .write_all(b"{\"_id\":\"d2\",\"title\":\"Two\",\"text\":\"Body\"}\n{\"_id\":\"d1\",\"title\":\"\",\"text\":\"One\"}\n")
            .unwrap();
        writer.start_file("fiqa/queries.jsonl", options).unwrap();
        writer
            .write_all(b"{\"_id\":\"q1\",\"text\":\"Question\"}\n")
            .unwrap();
        writer.start_file("fiqa/qrels/test.tsv", options).unwrap();
        writer
            .write_all(b"query-id\tcorpus-id\tscore\r\nq1\td1\t2\r\n")
            .unwrap();
        writer.finish().unwrap().into_inner()
    }

    fn options(temp: &tempfile::TempDir, source_bytes: &[u8]) -> FiqaOptions {
        FiqaOptions {
            artifact: artifact(4),
            workspace: temp.path().join("workspace"),
            output: temp.path().join("fixture"),
            batch_size: 1,
            repair_batches: false,
            generator_revision: "1".repeat(40),
            source: FiqaSourceSpec {
                url: "https://example.invalid/fiqa.zip".to_owned(),
                revision: "test-source".to_owned(),
                sha256: sha256_bytes(source_bytes),
                license: FIQA_LICENSE.to_owned(),
            },
        }
    }

    #[test]
    fn completed_fixture_makes_no_source_or_embedding_calls() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let options = options(&temp, &archive);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        let source_calls = source.calls.load(Ordering::SeqCst);
        let embedding_calls = embedder.calls.load(Ordering::SeqCst);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        assert_eq!(source.calls.load(Ordering::SeqCst), source_calls);
        assert_eq!(embedder.calls.load(Ordering::SeqCst), embedding_calls);
    }

    #[test]
    fn changed_source_metadata_rebuilds_the_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let mut options = options(&temp, &archive);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        options.source.revision = "corrected-source-revision".to_owned();
        prepare_fiqa(&options, &source, &embedder).unwrap();
        let manifest = FixtureManifest::load(options.output.join(MANIFEST_NAME)).unwrap();
        assert_eq!(manifest.source.revision, "corrected-source-revision");
    }

    #[test]
    fn changed_generator_revision_rebuilds_the_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let mut options = options(&temp, &archive);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        let calls = embedder.calls.load(Ordering::SeqCst);
        options.generator_revision = "2".repeat(40);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        let manifest = FixtureManifest::load(options.output.join(MANIFEST_NAME)).unwrap();
        assert_eq!(manifest.generator_revision, "2".repeat(40));
        assert!(embedder.calls.load(Ordering::SeqCst) > calls);
    }

    #[test]
    fn generator_revision_rejects_a_moving_label() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let mut options = options(&temp, &archive);
        options.generator_revision = "main".to_owned();

        let failure = options.validate().unwrap_err();

        assert!(failure.to_string().contains("40-character lowercase"));
    }

    #[test]
    fn qrels_reject_duplicate_query_document_pairs() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("qrels.tsv");
        fs::write(&path, b"query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\n").unwrap();

        let failure = load_qrels(&path).unwrap_err();

        assert!(failure.to_string().contains("duplicate qrel pair"));
    }

    #[test]
    fn qrels_reject_zero_relevance() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("qrels.tsv");
        fs::write(&path, b"query-id\tcorpus-id\tscore\nq1\td1\t0\n").unwrap();

        let failure = load_qrels(&path).unwrap_err();

        assert!(failure.to_string().contains("must be positive"));
    }

    #[test]
    fn qrels_accept_the_official_crlf_encoding() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("qrels.tsv");
        fs::write(&path, b"query-id\tcorpus-id\tscore\r\nq1\td1\t1\r\n").unwrap();

        let qrels = load_qrels(&path).unwrap();

        assert_eq!(qrels.len(), 1);
        assert_eq!(qrels[0].query_id, "q1");
        assert_eq!(qrels[0].corpus_id, "d1");
        assert_eq!(qrels[0].score, 1);
    }

    #[test]
    fn workspace_and_output_must_not_overlap() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let mut options = options(&temp, &archive);
        options.output = options.workspace.join("fixture");
        let failure = prepare_fiqa(&options, &source, &embedder).unwrap_err();
        assert!(failure.to_string().contains("non-nested"));
        assert_eq!(source.calls.load(Ordering::SeqCst), 0);
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[test]
    fn source_cache_symlink_cannot_escape_the_prepare_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let options = options(&temp, &archive);
        fs::create_dir_all(&options.workspace).unwrap();
        let outside = temp.path().join("outside-source-cache");
        fs::create_dir(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        fs::write(&sentinel, b"unchanged").unwrap();
        std::os::unix::fs::symlink(&outside, options.workspace.join("source")).unwrap();

        let failure = prepare_fiqa(&options, &source, &embedder).unwrap_err();

        assert!(failure.to_string().contains("refusing symlink"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
        assert_eq!(source.calls.load(Ordering::SeqCst), 0);
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cached_batches_are_reused_after_an_incomplete_fixture_is_removed() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let options = options(&temp, &archive);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        let calls = embedder.calls.load(Ordering::SeqCst);
        crate::util::remove_tree(&options.output).unwrap();
        prepare_fiqa(&options, &source, &embedder).unwrap();
        assert_eq!(embedder.calls.load(Ordering::SeqCst), calls);
    }

    #[test]
    fn invalid_cached_batch_requires_explicit_repair() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let mut options = options(&temp, &archive);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        crate::util::remove_tree(&options.output).unwrap();
        let cache_file = fs::read_dir(
            options
                .workspace
                .join("batches")
                .join(&options.artifact.id)
                .join("corpus"),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
        fs::write(&cache_file, b"not JSON").unwrap();
        let calls = embedder.calls.load(Ordering::SeqCst);
        let failure = prepare_fiqa(&options, &source, &embedder).unwrap_err();
        assert!(failure.to_string().contains("--repair-batches"));
        assert_eq!(embedder.calls.load(Ordering::SeqCst), calls);

        options.repair_batches = true;
        prepare_fiqa(&options, &source, &embedder).unwrap();
        assert!(embedder.calls.load(Ordering::SeqCst) > calls);
    }

    #[test]
    fn extracted_source_tampering_is_replaced_from_the_verified_archive() {
        let temp = tempfile::tempdir().unwrap();
        let archive = tiny_archive();
        let source = MemorySource {
            bytes: archive.clone(),
            calls: AtomicUsize::new(0),
        };
        let embedder = RecordingEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        };
        let options = options(&temp, &archive);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        crate::util::remove_tree(&options.output).unwrap();
        let corpus = options
            .workspace
            .join("source")
            .join(&options.source.sha256)
            .join("corpus.jsonl");
        fs::write(&corpus, b"tampered\n").unwrap();
        let calls = embedder.calls.load(Ordering::SeqCst);
        prepare_fiqa(&options, &source, &embedder).unwrap();
        assert_eq!(embedder.calls.load(Ordering::SeqCst), calls);
        assert!(!fs::read_to_string(corpus).unwrap().contains("tampered"));
    }

    #[test]
    fn dimensions_change_the_batch_content_identity() {
        let records = vec![TextRecord {
            id: "id".to_owned(),
            text: "text".to_owned(),
        }];
        assert_ne!(
            batch_identity("corpus", &records, "model", 384, &"1".repeat(40)),
            batch_identity("corpus", &records, "model", 1536, &"1".repeat(40))
        );
    }
}
