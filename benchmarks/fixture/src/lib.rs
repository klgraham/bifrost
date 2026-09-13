use std::{
    collections::{HashMap, HashSet},
    error::Error as StdError,
    ffi::OsStr,
    fmt,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CATALOG_SCHEMA: &str = "bifrost-benchmark-catalog-v1";
pub const FIXTURE_SCHEMA: &str = "bifrost-benchmark-fixture-v2";
pub const MANIFEST_FILE: &str = "manifest.json";
pub const RESOLUTION_FILE: &str = ".bifrost-resolution.json";
const F32_BYTES: usize = size_of::<f32>();
const UNIT_NORM_TOLERANCE: f64 = 0.01;
static TEMPORARY_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub type Result<T> = std::result::Result<T, FixtureError>;

#[derive(Debug)]
pub enum FixtureError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidCatalog(String),
    InvalidManifest(String),
    InvalidResolution(String),
    Verification(String),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Json { path, source } => {
                write!(formatter, "{}: invalid JSON: {source}", path.display())
            }
            Self::InvalidCatalog(message) => write!(formatter, "invalid catalog: {message}"),
            Self::InvalidManifest(message) => {
                write!(formatter, "invalid fixture manifest: {message}")
            }
            Self::InvalidResolution(message) => {
                write!(formatter, "invalid artifact resolution: {message}")
            }
            Self::Verification(message) => {
                write!(formatter, "fixture verification failed: {message}")
            }
        }
    }
}

impl StdError for FixtureError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::InvalidCatalog(_)
            | Self::InvalidManifest(_)
            | Self::InvalidResolution(_)
            | Self::Verification(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema: String,
    pub artifacts: Vec<Artifact>,
}

impl Catalog {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let catalog: Self = read_json(path.as_ref())?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != CATALOG_SCHEMA {
            return Err(FixtureError::InvalidCatalog(format!(
                "schema must be {CATALOG_SCHEMA}, got {}",
                self.schema
            )));
        }
        if self.artifacts.is_empty() {
            return Err(FixtureError::InvalidCatalog(
                "artifacts must not be empty".to_owned(),
            ));
        }

        let mut ids = HashSet::with_capacity(self.artifacts.len());
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !ids.insert(artifact.id.as_str()) {
                return Err(FixtureError::InvalidCatalog(format!(
                    "duplicate artifact id {}",
                    artifact.id
                )));
            }
        }
        Ok(())
    }

    pub fn artifact(&self, id: &str) -> Option<&Artifact> {
        self.artifacts.iter().find(|artifact| artifact.id == id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub id: String,
    pub dataset: String,
    pub model: String,
    pub dimensions: usize,
    pub publication: Publication,
}

impl Artifact {
    fn validate(&self) -> Result<()> {
        validate_identifier(&self.id, "artifact id").map_err(FixtureError::InvalidCatalog)?;
        validate_nonempty(&self.dataset, "artifact dataset")
            .map_err(FixtureError::InvalidCatalog)?;
        validate_nonempty(&self.model, "artifact model").map_err(FixtureError::InvalidCatalog)?;
        validate_dimensions(self.dimensions).map_err(FixtureError::InvalidCatalog)?;

        match &self.publication {
            Publication::Planned {
                repository,
                prefix,
                blocked_by,
            } => {
                validate_repository(repository).map_err(FixtureError::InvalidCatalog)?;
                validate_relative_path(prefix, "planned prefix", true)
                    .map_err(FixtureError::InvalidCatalog)?;
                if prefix.matches("{manifest_sha256}").count() != 1
                    || Path::new(prefix).file_name().and_then(OsStr::to_str)
                        != Some("{manifest_sha256}")
                {
                    return Err(FixtureError::InvalidCatalog(
                        "planned prefix must contain exactly one {manifest_sha256}, as its final segment"
                            .to_owned(),
                    ));
                }
                validate_short_text(blocked_by, "planned blocked_by")
                    .map_err(FixtureError::InvalidCatalog)?;
            }
            Publication::Published {
                repository,
                revision,
                prefix,
                manifest_sha256,
            } => {
                validate_repository(repository).map_err(FixtureError::InvalidCatalog)?;
                validate_hex(revision, 40, "published revision")
                    .map_err(FixtureError::InvalidCatalog)?;
                validate_relative_path(prefix, "published prefix", false)
                    .map_err(FixtureError::InvalidCatalog)?;
                validate_sha256(manifest_sha256, "published manifest_sha256")
                    .map_err(FixtureError::InvalidCatalog)?;
                if Path::new(prefix).file_name().and_then(OsStr::to_str) != Some(manifest_sha256) {
                    return Err(FixtureError::InvalidCatalog(
                        "published prefix must end with published manifest_sha256".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Publication {
    Planned {
        repository: String,
        prefix: String,
        blocked_by: String,
    },
    Published {
        repository: String,
        revision: String,
        prefix: String,
        manifest_sha256: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactResolution {
    pub artifact_id: String,
    pub repository: String,
    pub revision: String,
    pub manifest_sha256: String,
}

impl ArtifactResolution {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let resolution: Self = read_json(path.as_ref())?;
        resolution.validate()?;
        Ok(resolution)
    }

    pub fn validate(&self) -> Result<()> {
        validate_identifier(&self.artifact_id, "artifact_id")
            .map_err(FixtureError::InvalidResolution)?;
        validate_repository(&self.repository).map_err(FixtureError::InvalidResolution)?;
        validate_hex(&self.revision, 40, "revision").map_err(FixtureError::InvalidResolution)?;
        validate_sha256(&self.manifest_sha256, "manifest_sha256")
            .map_err(FixtureError::InvalidResolution)
    }

    pub fn from_artifact(artifact: &Artifact) -> Option<Self> {
        match &artifact.publication {
            Publication::Planned { .. } => None,
            Publication::Published {
                repository,
                revision,
                manifest_sha256,
                ..
            } => Some(Self {
                artifact_id: artifact.id.clone(),
                repository: repository.clone(),
                revision: revision.clone(),
                manifest_sha256: manifest_sha256.clone(),
            }),
        }
    }

    pub fn to_json_pretty(&self) -> Result<Vec<u8>> {
        self.validate()?;
        pretty_json(self, RESOLUTION_FILE)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        atomic_write(path, &self.to_json_pretty()?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureManifest {
    pub schema: String,
    pub artifact_id: String,
    pub dataset: String,
    pub source: SourceMetadata,
    pub selection: String,
    pub preprocessing: String,
    pub corpus_input_sha256: String,
    pub query_input_sha256: String,
    pub requested_model: String,
    pub returned_model: String,
    pub dimensions: usize,
    pub normalization: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<String>,
    pub generator_revision: String,
    pub representation: VectorRepresentation,
    pub counts: FixtureCounts,
    pub files: Vec<FixtureFile>,
}

impl FixtureManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let manifest: Self = read_json(path.as_ref())?;
        manifest.validate_metadata()?;
        Ok(manifest)
    }

    pub fn validate_metadata(&self) -> Result<()> {
        if self.schema != FIXTURE_SCHEMA {
            return Err(FixtureError::InvalidManifest(format!(
                "schema must be {FIXTURE_SCHEMA}, got {}",
                self.schema
            )));
        }
        validate_identifier(&self.artifact_id, "artifact_id")
            .map_err(FixtureError::InvalidManifest)?;
        validate_nonempty(&self.dataset, "dataset").map_err(FixtureError::InvalidManifest)?;
        self.source.validate()?;
        validate_nonempty(&self.selection, "selection").map_err(FixtureError::InvalidManifest)?;
        validate_nonempty(&self.preprocessing, "preprocessing")
            .map_err(FixtureError::InvalidManifest)?;
        validate_sha256(&self.corpus_input_sha256, "corpus_input_sha256")
            .map_err(FixtureError::InvalidManifest)?;
        validate_sha256(&self.query_input_sha256, "query_input_sha256")
            .map_err(FixtureError::InvalidManifest)?;
        validate_nonempty(&self.requested_model, "requested_model")
            .map_err(FixtureError::InvalidManifest)?;
        validate_nonempty(&self.returned_model, "returned_model")
            .map_err(FixtureError::InvalidManifest)?;
        validate_dimensions(self.dimensions).map_err(FixtureError::InvalidManifest)?;
        if self.normalization != "l2" {
            return Err(FixtureError::InvalidManifest(format!(
                "normalization must be l2, got {}",
                self.normalization
            )));
        }
        if self.derivation.as_deref().is_some_and(str::is_empty) {
            return Err(FixtureError::InvalidManifest(
                "derivation must not be empty when present".to_owned(),
            ));
        }
        validate_hex(&self.generator_revision, 40, "generator_revision")
            .map_err(FixtureError::InvalidManifest)?;
        self.representation.validate()?;
        if self.counts.corpus == 0 || self.counts.queries == 0 {
            return Err(FixtureError::InvalidManifest(
                "corpus and query counts must be positive".to_owned(),
            ));
        }

        let expected_corpus_bytes = vector_byte_count(self.counts.corpus, self.dimensions)?;
        let expected_query_bytes = vector_byte_count(self.counts.queries, self.dimensions)?;
        let mut paths = HashSet::with_capacity(self.files.len());
        let mut kinds = HashSet::with_capacity(self.files.len());
        for file in &self.files {
            validate_relative_path(&file.path, "fixture file path", false)
                .map_err(FixtureError::InvalidManifest)?;
            if file.path == MANIFEST_FILE {
                return Err(FixtureError::InvalidManifest(format!(
                    "{} is reserved and must not appear in files",
                    MANIFEST_FILE
                )));
            }
            if file.path == RESOLUTION_FILE {
                return Err(FixtureError::InvalidManifest(format!(
                    "{} is reserved and must not appear in files",
                    RESOLUTION_FILE
                )));
            }
            if file.byte_count == 0 {
                return Err(FixtureError::InvalidManifest(format!(
                    "{} has a zero byte count",
                    file.path
                )));
            }
            validate_sha256(&file.sha256, &format!("{} sha256", file.path))
                .map_err(FixtureError::InvalidManifest)?;
            if !paths.insert(file.path.as_str()) {
                return Err(FixtureError::InvalidManifest(format!(
                    "duplicate fixture file path {}",
                    file.path
                )));
            }
            if !kinds.insert(file.kind) {
                return Err(FixtureError::InvalidManifest(format!(
                    "duplicate fixture file kind {}",
                    file.kind.as_str()
                )));
            }
        }

        for required in FixtureFileKind::REQUIRED {
            if !kinds.contains(&required) {
                return Err(FixtureError::InvalidManifest(format!(
                    "missing fixture file kind {}",
                    required.as_str()
                )));
            }
        }
        let corpus = self
            .file(FixtureFileKind::CorpusVectors)
            .expect("required corpus vector file checked");
        if corpus.byte_count != expected_corpus_bytes {
            return Err(FixtureError::InvalidManifest(format!(
                "{} declares {} bytes; corpus shape requires {expected_corpus_bytes}",
                corpus.path, corpus.byte_count
            )));
        }
        let queries = self
            .file(FixtureFileKind::QueryVectors)
            .expect("required query vector file checked");
        if queries.byte_count != expected_query_bytes {
            return Err(FixtureError::InvalidManifest(format!(
                "{} declares {} bytes; query shape requires {expected_query_bytes}",
                queries.path, queries.byte_count
            )));
        }
        Ok(())
    }

    pub fn file(&self, kind: FixtureFileKind) -> Option<&FixtureFile> {
        self.files.iter().find(|file| file.kind == kind)
    }

    pub fn identity_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"bifrost-benchmark-fixture-identity-v1\0");
        for value in [
            self.artifact_id.as_str(),
            self.dataset.as_str(),
            self.source.url.as_str(),
            self.source.revision.as_str(),
            self.source.sha256.as_str(),
            self.source.license.as_str(),
            self.selection.as_str(),
            self.preprocessing.as_str(),
            self.corpus_input_sha256.as_str(),
            self.query_input_sha256.as_str(),
            self.requested_model.as_str(),
            self.returned_model.as_str(),
            self.normalization.as_str(),
            self.generator_revision.as_str(),
            self.representation.scalar.as_str(),
            self.representation.endianness.as_str(),
        ] {
            update_length_prefixed(&mut digest, value.as_bytes());
        }
        digest.update(
            u64::try_from(self.dimensions)
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        digest.update(
            u64::try_from(self.counts.corpus)
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        digest.update(
            u64::try_from(self.counts.queries)
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        match &self.derivation {
            Some(derivation) => {
                digest.update([1]);
                update_length_prefixed(&mut digest, derivation.as_bytes());
            }
            None => digest.update([0]),
        }
        format!("{:x}", digest.finalize())
    }

    pub fn to_json_pretty(&self) -> Result<Vec<u8>> {
        self.validate_metadata()?;
        pretty_json(self, MANIFEST_FILE)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.to_json_pretty()?;
        atomic_write(path, &bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceMetadata {
    pub url: String,
    pub revision: String,
    pub sha256: String,
    pub license: String,
}

impl SourceMetadata {
    fn validate(&self) -> Result<()> {
        validate_nonempty(&self.url, "source.url").map_err(FixtureError::InvalidManifest)?;
        validate_nonempty(&self.revision, "source.revision")
            .map_err(FixtureError::InvalidManifest)?;
        validate_sha256(&self.sha256, "source.sha256").map_err(FixtureError::InvalidManifest)?;
        validate_nonempty(&self.license, "source.license")
            .map_err(FixtureError::InvalidManifest)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VectorRepresentation {
    pub scalar: String,
    pub endianness: String,
}

impl VectorRepresentation {
    fn validate(&self) -> Result<()> {
        if self.scalar != "f32" {
            return Err(FixtureError::InvalidManifest(format!(
                "representation.scalar must be f32, got {}",
                self.scalar
            )));
        }
        if self.endianness != "little" {
            return Err(FixtureError::InvalidManifest(format!(
                "representation.endianness must be little, got {}",
                self.endianness
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureCounts {
    pub corpus: usize,
    pub queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureFile {
    pub kind: FixtureFileKind,
    pub path: String,
    #[serde(rename = "bytes")]
    pub byte_count: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FixtureFileKind {
    CorpusVectors,
    QueryVectors,
    CorpusIds,
    QueryIds,
    Qrels,
}

impl FixtureFileKind {
    const REQUIRED: [Self; 5] = [
        Self::CorpusVectors,
        Self::QueryVectors,
        Self::CorpusIds,
        Self::QueryIds,
        Self::Qrels,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CorpusVectors => "corpus_vectors",
            Self::QueryVectors => "query_vectors",
            Self::CorpusIds => "corpus_ids",
            Self::QueryIds => "query_ids",
            Self::Qrels => "qrels",
        }
    }
}

#[derive(Debug)]
pub struct VerifiedFixture {
    pub corpus_vectors: Vec<Vec<f32>>,
    pub query_vectors: Vec<Vec<f32>>,
    pub corpus_ids: Vec<String>,
    pub query_ids: Vec<String>,
    pub qrels_path: Option<PathBuf>,
    pub metadata: FixtureManifest,
    pub resolution: Option<ArtifactResolution>,
}

impl VerifiedFixture {
    pub fn load(directory: impl AsRef<Path>, expected: Option<&Artifact>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest_path = directory.join(MANIFEST_FILE);
        require_regular_file(&manifest_path)?;
        let manifest_bytes = read_file(&manifest_path)?;
        let resolution_path = directory.join(RESOLUTION_FILE);
        let resolution = match fs::symlink_metadata(&resolution_path) {
            Ok(_) => {
                require_regular_file(&resolution_path)?;
                let resolution = ArtifactResolution::load(&resolution_path)?;
                Some(resolution)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(source) => {
                return Err(FixtureError::Io {
                    path: resolution_path,
                    source,
                });
            }
        };

        if let Some(artifact) = expected {
            artifact.validate().map_err(|error| match error {
                FixtureError::InvalidCatalog(message) => FixtureError::Verification(format!(
                    "expected artifact metadata is invalid: {message}"
                )),
                other => other,
            })?;
            if let Publication::Published {
                manifest_sha256, ..
            } = &artifact.publication
            {
                verify_sha256(&manifest_path, &manifest_bytes, manifest_sha256)?;
            }
        }

        let manifest: FixtureManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|source| FixtureError::Json {
                path: manifest_path.clone(),
                source,
            })?;
        manifest.validate_metadata()?;
        if let Some(resolution) = &resolution {
            verify_resolution(
                &manifest,
                &manifest_path,
                &manifest_bytes,
                resolution,
                expected,
            )?;
        }
        if let Some(artifact) = expected {
            verify_expected_artifact(&manifest, artifact)?;
        }

        verify_directory_allowlist(directory, &manifest)?;
        let mut verified_files = HashMap::with_capacity(manifest.files.len());
        for file in &manifest.files {
            let path = directory.join(&file.path);
            require_regular_file(&path)?;
            let bytes = read_file(&path)?;
            if u64::try_from(bytes.len()).ok() != Some(file.byte_count) {
                return Err(FixtureError::Verification(format!(
                    "{} has {} bytes; manifest declares {}",
                    path.display(),
                    bytes.len(),
                    file.byte_count
                )));
            }
            verify_sha256(&path, &bytes, &file.sha256)?;
            verified_files.insert(file.kind, bytes);
        }

        let corpus_vectors = read_vectors(
            verified_files
                .get(&FixtureFileKind::CorpusVectors)
                .expect("required corpus vectors checked"),
            manifest.counts.corpus,
            manifest.dimensions,
            "corpus vectors",
        )?;
        let query_vectors = read_vectors(
            verified_files
                .get(&FixtureFileKind::QueryVectors)
                .expect("required query vectors checked"),
            manifest.counts.queries,
            manifest.dimensions,
            "query vectors",
        )?;
        let corpus_ids = read_ids(
            verified_files
                .get(&FixtureFileKind::CorpusIds)
                .expect("required corpus IDs checked"),
            manifest.counts.corpus,
            "corpus IDs",
        )?;
        let query_ids = read_ids(
            verified_files
                .get(&FixtureFileKind::QueryIds)
                .expect("required query IDs checked"),
            manifest.counts.queries,
            "query IDs",
        )?;
        validate_qrels(
            verified_files
                .get(&FixtureFileKind::Qrels)
                .expect("required qrels checked"),
            &corpus_ids,
            &query_ids,
        )?;
        let qrels_path = Some(
            directory.join(
                &manifest
                    .file(FixtureFileKind::Qrels)
                    .expect("required qrels metadata checked")
                    .path,
            ),
        );

        Ok(Self {
            corpus_vectors,
            query_vectors,
            corpus_ids,
            query_ids,
            qrels_path,
            metadata: manifest,
            resolution,
        })
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn input_sha256<'a>(records: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"bifrost-benchmark-text-input-v1\0");
    for (id, text) in records {
        update_length_prefixed(&mut digest, id.as_bytes());
        update_length_prefixed(&mut digest, text.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = read_file(path)?;
    serde_json::from_slice(&bytes).map_err(|source| FixtureError::Json {
        path: path.to_owned(),
        source,
    })
}

fn pretty_json<T: Serialize>(value: &T, display_path: &str) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|source| FixtureError::Json {
        path: PathBuf::from(display_path),
        source,
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| FixtureError::Io {
        path: path.to_owned(),
        source,
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| FixtureError::Io {
        path: parent.to_owned(),
        source,
    })?;
    let counter = TEMPORARY_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or(MANIFEST_FILE);
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{counter}", std::process::id()));
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| FixtureError::Io {
                path: temporary.clone(),
                source,
            })?;
        file.write_all(bytes).map_err(|source| FixtureError::Io {
            path: temporary.clone(),
            source,
        })?;
        file.sync_all().map_err(|source| FixtureError::Io {
            path: temporary.clone(),
            source,
        })?;
        fs::rename(&temporary, path).map_err(|source| FixtureError::Io {
            path: path.to_owned(),
            source,
        })?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

fn verify_expected_artifact(manifest: &FixtureManifest, artifact: &Artifact) -> Result<()> {
    if manifest.artifact_id != artifact.id {
        return Err(FixtureError::Verification(format!(
            "artifact id {} does not match catalog id {}",
            manifest.artifact_id, artifact.id
        )));
    }
    if manifest.dataset != artifact.dataset {
        return Err(FixtureError::Verification(format!(
            "dataset {} does not match catalog dataset {}",
            manifest.dataset, artifact.dataset
        )));
    }
    if manifest.requested_model != artifact.model {
        return Err(FixtureError::Verification(format!(
            "requested model {} does not match catalog model {}",
            manifest.requested_model, artifact.model
        )));
    }
    if manifest.dimensions != artifact.dimensions {
        return Err(FixtureError::Verification(format!(
            "{} dimensions do not match catalog dimensions {}",
            manifest.dimensions, artifact.dimensions
        )));
    }
    Ok(())
}

fn verify_resolution(
    manifest: &FixtureManifest,
    manifest_path: &Path,
    manifest_bytes: &[u8],
    resolution: &ArtifactResolution,
    expected: Option<&Artifact>,
) -> Result<()> {
    if resolution.artifact_id != manifest.artifact_id {
        return Err(FixtureError::Verification(format!(
            "resolution artifact id {} does not match manifest artifact id {}",
            resolution.artifact_id, manifest.artifact_id
        )));
    }
    verify_sha256(manifest_path, manifest_bytes, &resolution.manifest_sha256)?;

    if let Some(Artifact {
        publication:
            Publication::Published {
                repository,
                revision,
                manifest_sha256,
                ..
            },
        ..
    }) = expected
    {
        if resolution.repository != *repository {
            return Err(FixtureError::Verification(format!(
                "resolution repository {} does not match catalog repository {repository}",
                resolution.repository
            )));
        }
        if resolution.revision != *revision {
            return Err(FixtureError::Verification(format!(
                "resolution revision {} does not match catalog revision {revision}",
                resolution.revision
            )));
        }
        if !resolution
            .manifest_sha256
            .eq_ignore_ascii_case(manifest_sha256)
        {
            return Err(FixtureError::Verification(
                "resolution manifest_sha256 does not match catalog manifest_sha256".to_owned(),
            ));
        }
    }
    Ok(())
}

fn verify_directory_allowlist(directory: &Path, manifest: &FixtureManifest) -> Result<()> {
    let mut allowed_files = manifest
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect::<HashSet<_>>();
    allowed_files.insert(MANIFEST_FILE.to_owned());
    allowed_files.insert(RESOLUTION_FILE.to_owned());

    let mut allowed_directories = HashSet::new();
    for file in &allowed_files {
        let mut parent = Path::new(file).parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() {
                break;
            }
            allowed_directories.insert(path_to_manifest_string(path)?);
            parent = path.parent();
        }
    }

    inspect_directory(directory, directory, &allowed_files, &allowed_directories)
}

fn inspect_directory(
    root: &Path,
    directory: &Path,
    allowed_files: &HashSet<String>,
    allowed_directories: &HashSet<String>,
) -> Result<()> {
    let entries = fs::read_dir(directory).map_err(|source| FixtureError::Io {
        path: directory.to_owned(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| FixtureError::Io {
            path: directory.to_owned(),
            source,
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| FixtureError::Io {
            path: path.clone(),
            source,
        })?;
        let relative = path.strip_prefix(root).map_err(|_| {
            FixtureError::Verification(format!("{} escapes fixture root", path.display()))
        })?;
        let relative = path_to_manifest_string(relative)?;
        if metadata.file_type().is_symlink() {
            return Err(FixtureError::Verification(format!(
                "{} is a symlink; fixture entries must be regular files",
                path.display()
            )));
        }
        if metadata.is_dir() {
            if !allowed_directories.contains(&relative) {
                return Err(FixtureError::Verification(format!(
                    "unlisted directory {}",
                    path.display()
                )));
            }
            inspect_directory(root, &path, allowed_files, allowed_directories)?;
        } else if metadata.is_file() {
            if !allowed_files.contains(&relative) {
                return Err(FixtureError::Verification(format!(
                    "unlisted file {}",
                    path.display()
                )));
            }
        } else {
            return Err(FixtureError::Verification(format!(
                "{} is not a regular file",
                path.display()
            )));
        }
    }
    Ok(())
}

fn require_regular_file(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|source| FixtureError::Io {
        path: path.to_owned(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(FixtureError::Verification(format!(
            "{} must be a regular, non-symlink file",
            path.display()
        )));
    }
    Ok(())
}

fn verify_sha256(path: &Path, bytes: &[u8], expected: &str) -> Result<()> {
    let actual = sha256_hex(bytes);
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(FixtureError::Verification(format!(
            "{} SHA-256 is {actual}; expected {expected}",
            path.display()
        )));
    }
    Ok(())
}

fn read_vectors(
    bytes: &[u8],
    count: usize,
    dimensions: usize,
    label: &str,
) -> Result<Vec<Vec<f32>>> {
    let expected_bytes = count
        .checked_mul(dimensions)
        .and_then(|values| values.checked_mul(F32_BYTES))
        .ok_or_else(|| FixtureError::Verification(format!("{label} shape overflows usize")))?;
    if bytes.len() != expected_bytes {
        return Err(FixtureError::Verification(format!(
            "{label} has {} bytes; expected {expected_bytes}",
            bytes.len()
        )));
    }

    let mut vectors = Vec::with_capacity(count);
    for (index, row) in bytes.chunks_exact(dimensions * F32_BYTES).enumerate() {
        let values = row
            .chunks_exact(F32_BYTES)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte f32 chunk")))
            .collect::<Vec<_>>();
        let norm = values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
            return Err(FixtureError::Verification(format!(
                "{label} row {index} has invalid L2 norm {norm}"
            )));
        }
        vectors.push(values);
    }
    Ok(vectors)
}

fn read_ids(bytes: &[u8], count: usize, label: &str) -> Result<Vec<String>> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| FixtureError::Verification(format!("{label} is not UTF-8: {error}")))?;
    let ids = text.lines().map(str::to_owned).collect::<Vec<_>>();
    if ids.len() != count {
        return Err(FixtureError::Verification(format!(
            "{label} contains {} IDs; expected {count}",
            ids.len()
        )));
    }
    if ids.iter().any(String::is_empty) {
        return Err(FixtureError::Verification(format!(
            "{label} contains an empty ID"
        )));
    }
    let unique = ids.iter().collect::<HashSet<_>>();
    if unique.len() != ids.len() {
        return Err(FixtureError::Verification(format!(
            "{label} contains duplicate IDs"
        )));
    }
    Ok(ids)
}

fn validate_qrels(bytes: &[u8], corpus_ids: &[String], query_ids: &[String]) -> Result<()> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| FixtureError::Verification(format!("qrels are not UTF-8: {error}")))?;
    let mut lines = text.lines();
    if lines.next() != Some("query-id\tcorpus-id\tscore") {
        return Err(FixtureError::Verification(
            "qrels must start with the header query-id\\tcorpus-id\\tscore".to_owned(),
        ));
    }

    let corpus_ids = corpus_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let query_ids = query_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut pairs = HashSet::new();
    let mut queries_with_relevance = HashSet::new();
    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != 3 {
            return Err(FixtureError::Verification(format!(
                "qrels line {line_number} must contain query-id, corpus-id, and score"
            )));
        }
        let query_id = columns[0];
        let corpus_id = columns[1];
        if !query_ids.contains(query_id) {
            return Err(FixtureError::Verification(format!(
                "qrels line {line_number} references unknown query ID {query_id:?}"
            )));
        }
        if !corpus_ids.contains(corpus_id) {
            return Err(FixtureError::Verification(format!(
                "qrels line {line_number} references unknown corpus ID {corpus_id:?}"
            )));
        }
        let score = columns[2].parse::<u32>().map_err(|_| {
            FixtureError::Verification(format!(
                "qrels line {line_number} score must be a positive integer"
            ))
        })?;
        if score == 0 {
            return Err(FixtureError::Verification(format!(
                "qrels line {line_number} score must be a positive integer"
            )));
        }
        if !pairs.insert((query_id, corpus_id)) {
            return Err(FixtureError::Verification(format!(
                "qrels line {line_number} repeats query/document pair {query_id:?}/{corpus_id:?}"
            )));
        }
        queries_with_relevance.insert(query_id);
    }

    if pairs.is_empty() {
        return Err(FixtureError::Verification(
            "qrels contain no relevance judgments".to_owned(),
        ));
    }
    if let Some(query_id) = query_ids
        .iter()
        .find(|query_id| !queries_with_relevance.contains(**query_id))
    {
        return Err(FixtureError::Verification(format!(
            "qrels contain no relevance judgment for query ID {query_id:?}"
        )));
    }
    Ok(())
}

fn vector_byte_count(count: usize, dimensions: usize) -> Result<u64> {
    count
        .checked_mul(dimensions)
        .and_then(|values| values.checked_mul(F32_BYTES))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| FixtureError::InvalidManifest("vector byte count overflow".to_owned()))
}

fn validate_nonempty(value: &str, name: &str) -> std::result::Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must not be empty"))
    } else {
        Ok(())
    }
}

fn validate_dimensions(dimensions: usize) -> std::result::Result<(), String> {
    if (1..=usize::from(u16::MAX)).contains(&dimensions) {
        Ok(())
    } else {
        Err("dimensions must be between 1 and 65535".to_owned())
    }
}

fn validate_identifier(value: &str, name: &str) -> std::result::Result<(), String> {
    validate_nonempty(value, name)?;
    if value == "."
        || value == ".."
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!(
            "{name} must contain only ASCII letters, digits, '.', '-', or '_'"
        ));
    }
    Ok(())
}

fn validate_repository(repository: &str) -> std::result::Result<(), String> {
    let mut segments = repository.split('/');
    let owner = segments.next().unwrap_or_default();
    let name = segments.next().unwrap_or_default();
    if segments.next().is_some() || owner.is_empty() || name.is_empty() {
        return Err("Hugging Face repository must have the form owner/name".to_owned());
    }
    validate_identifier(owner, "Hugging Face repository owner")?;
    validate_identifier(name, "Hugging Face repository name")
}

fn validate_relative_path(
    value: &str,
    name: &str,
    allow_manifest_placeholder: bool,
) -> std::result::Result<(), String> {
    validate_nonempty(value, name)?;
    if value.starts_with('/') || Path::new(value).is_absolute() {
        return Err(format!("{name} must be relative"));
    }
    if value.contains('\\') {
        return Err(format!("{name} must use '/' separators"));
    }
    for segment in value.split('/') {
        if segment.is_empty() || matches!(segment, "." | "..") {
            return Err(format!(
                "{name} contains path traversal or an empty component"
            ));
        }
        if segment == "{manifest_sha256}" {
            if !allow_manifest_placeholder {
                return Err(format!(
                    "{name} contains an unresolved manifest placeholder"
                ));
            }
        } else {
            validate_identifier(segment, name)?;
        }
    }
    Ok(())
}

fn validate_short_text(value: &str, name: &str) -> std::result::Result<(), String> {
    validate_nonempty(value, name)?;
    if value.chars().count() > 256 {
        return Err(format!("{name} must be at most 256 characters"));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{name} must not contain control characters"));
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> std::result::Result<(), String> {
    validate_hex(value, 64, name)
}

fn validate_hex(value: &str, length: usize, name: &str) -> std::result::Result<(), String> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(format!(
            "{name} must be exactly {length} lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn path_to_manifest_string(path: &Path) -> Result<String> {
    let mut segments = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(segment) => segments.push(
                segment
                    .to_str()
                    .ok_or_else(|| {
                        FixtureError::Verification(format!(
                            "{} contains a non-UTF-8 path component",
                            path.display()
                        ))
                    })?
                    .to_owned(),
            ),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(FixtureError::Verification(format!(
                    "{} is not a safe relative path",
                    path.display()
                )));
            }
        }
    }
    Ok(segments.join("/"))
}

fn update_length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_be_bytes());
    digest.update(bytes);
}
