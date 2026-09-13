mod fetch;
mod prepare;
mod publish;
mod util;

pub use fetch::{FetchOptions, HfDownloader, RemoteFileSource, fetch_artifact};
pub use prepare::{
    Embedder, EmbeddingBatch, EmbeddingItem, FiqaOptions, FiqaSourceSpec, HttpFiqaSource,
    OpenAiEmbedder, SourceArchive, prepare_fiqa,
};
pub use publish::{
    HfPublisher, PublishOptions, PublishReceipt, Publisher, RemoteEntry, UploadFile,
    publish_artifact,
};

use std::{error::Error, fmt};

pub type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug)]
pub struct ToolError(String);

impl ToolError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ToolError {}

pub(crate) fn error<T>(message: impl Into<String>) -> ToolResult<T> {
    Err(Box::new(ToolError::new(message)))
}

pub(crate) fn context<T, E: fmt::Display>(result: Result<T, E>, action: &str) -> ToolResult<T> {
    result.map_err(|source| Box::new(ToolError::new(format!("{action}: {source}"))) as _)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::{fs, io::Write, path::Path};

    use bifrost_benchmark_fixture::{
        Artifact, FIXTURE_SCHEMA, FixtureCounts, FixtureFile, FixtureFileKind, FixtureManifest,
        Publication, SourceMetadata, VectorRepresentation, input_sha256,
    };

    use crate::{
        ToolResult,
        util::{atomic_write, sha256_bytes, sha256_file},
    };

    pub(crate) fn planned_artifact(dimensions: usize) -> Artifact {
        Artifact {
            id: format!("fiqa-test-{dimensions}"),
            dataset: "BEIR FiQA-2018".to_owned(),
            model: "text-embedding-3-small".to_owned(),
            dimensions,
            publication: Publication::Planned {
                repository: "owner/repository".to_owned(),
                prefix: format!("fixtures/fiqa/{dimensions}/{{manifest_sha256}}"),
                blocked_by: "test data must never be externally published".to_owned(),
            },
        }
    }

    pub(crate) fn write_fixture(directory: &Path, artifact: &Artifact) -> ToolResult<String> {
        fs::create_dir_all(directory)?;
        let value = 1.0 / (artifact.dimensions as f32).sqrt();
        let vector = vec![value; artifact.dimensions];
        let corpus_vectors = [vector.as_slice(), vector.as_slice()];
        let query_vectors = [vector.as_slice()];
        write_vectors(&directory.join("corpus.f32"), &corpus_vectors)?;
        write_vectors(&directory.join("queries.f32"), &query_vectors)?;
        atomic_write(&directory.join("corpus-ids.txt"), b"d1\nd2\n")?;
        atomic_write(&directory.join("query-ids.txt"), b"q1\n")?;
        atomic_write(
            &directory.join("qrels-test.tsv"),
            b"query-id\tcorpus-id\tscore\nq1\td1\t2\n",
        )?;
        let files = [
            (FixtureFileKind::CorpusVectors, "corpus.f32"),
            (FixtureFileKind::QueryVectors, "queries.f32"),
            (FixtureFileKind::CorpusIds, "corpus-ids.txt"),
            (FixtureFileKind::QueryIds, "query-ids.txt"),
            (FixtureFileKind::Qrels, "qrels-test.tsv"),
        ]
        .into_iter()
        .map(|(kind, relative)| {
            let path = directory.join(relative);
            Ok(FixtureFile {
                kind,
                path: relative.to_owned(),
                byte_count: fs::metadata(&path)?.len(),
                sha256: sha256_file(&path)?,
            })
        })
        .collect::<ToolResult<Vec<_>>>()?;
        let corpus = [("d1", "one"), ("d2", "two")];
        let queries = [("q1", "question")];
        let manifest = FixtureManifest {
            schema: FIXTURE_SCHEMA.to_owned(),
            artifact_id: artifact.id.clone(),
            dataset: artifact.dataset.clone(),
            source: SourceMetadata {
                url: "https://example.invalid/fiqa.zip".to_owned(),
                revision: "test-source".to_owned(),
                sha256: sha256_bytes(b"source"),
                license: "CC BY-SA 4.0".to_owned(),
            },
            preprocessing: "test preprocessing".to_owned(),
            selection: "all two test documents and one test query".to_owned(),
            corpus_input_sha256: input_sha256(corpus),
            query_input_sha256: input_sha256(queries),
            requested_model: artifact.model.clone(),
            returned_model: artifact.model.clone(),
            dimensions: artifact.dimensions,
            normalization: "l2".to_owned(),
            derivation: Some("independent test request".to_owned()),
            generator_revision: "3".repeat(40),
            representation: VectorRepresentation {
                scalar: "f32".to_owned(),
                endianness: "little".to_owned(),
            },
            counts: FixtureCounts {
                corpus: 2,
                queries: 1,
            },
            files,
        };
        manifest.write(directory.join("manifest.json"))?;
        sha256_file(&directory.join("manifest.json"))
    }

    fn write_vectors(path: &Path, vectors: &[&[f32]]) -> ToolResult<()> {
        let mut bytes = Vec::new();
        for vector in vectors {
            for value in *vector {
                bytes.write_all(&value.to_le_bytes())?;
            }
        }
        atomic_write(path, &bytes)
    }

    pub(crate) fn published_artifact(
        mut artifact: Artifact,
        manifest_sha256: String,
        revision: &str,
    ) -> Artifact {
        let Publication::Planned {
            repository, prefix, ..
        } = artifact.publication
        else {
            unreachable!();
        };
        artifact.publication = Publication::Published {
            repository,
            revision: revision.to_owned(),
            prefix: prefix.replace("{manifest_sha256}", &manifest_sha256),
            manifest_sha256,
        };
        artifact
    }
}
