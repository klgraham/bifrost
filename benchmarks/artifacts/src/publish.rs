use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use bifrost_benchmark_fixture::{Catalog, Publication, VerifiedFixture};
use hf_hub::{
    HFClient, HFError, HFRepositorySync,
    repository::{CommitOperation, RepoTreeEntry, RepoTypeDataset},
};

use crate::{
    ToolResult, error,
    util::{
        atomic_write, atomic_write_under, checked_relative_path, ensure_resolved_within_root,
        is_commit_sha, join_repo_path, prepare_mutation_root, reject_overlapping_paths,
        sha256_bytes, sha256_file,
    },
};

const MANIFEST_NAME: &str = "manifest.json";
const CARD_NAME: &str = "README.md";
const MANIFEST_PLACEHOLDER: &str = "{manifest_sha256}";

#[derive(Debug, Clone)]
pub struct PublishOptions {
    pub artifact_id: String,
    pub fixture: PathBuf,
    pub branch: String,
    pub card_template: PathBuf,
    pub workspace: PathBuf,
    pub update_catalog: Option<PathBuf>,
    pub allow_external_publication: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishReceipt {
    pub repository: String,
    pub revision: String,
    pub prefix: String,
    pub manifest_sha256: String,
    pub created_commit: bool,
    pub metadata_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteEntry {
    pub path: String,
    pub byte_count: u64,
    pub content_sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UploadFile {
    pub path_in_repo: String,
    pub local_path: PathBuf,
}

pub trait Publisher: Send + Sync {
    fn revision_sha(&self, repository: &str, revision: &str) -> ToolResult<String>;
    fn list_files(
        &self,
        repository: &str,
        revision: &str,
        prefix: &str,
    ) -> ToolResult<Vec<RemoteEntry>>;
    fn download_file(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        force: bool,
    ) -> ToolResult<Option<PathBuf>>;
    fn create_commit(
        &self,
        repository: &str,
        branch: &str,
        parent_commit: &str,
        message: &str,
        files: &[UploadFile],
    ) -> ToolResult<String>;
}

pub struct HfPublisher {
    repository_owner: String,
    repository_name: String,
    repository: HFRepositorySync<RepoTypeDataset>,
    cache_directory: PathBuf,
}

impl HfPublisher {
    pub fn new(
        repository: &str,
        endpoint: &str,
        cache_directory: impl Into<PathBuf>,
        token: String,
    ) -> ToolResult<Self> {
        if token.trim().is_empty() {
            return error("a nonempty Hugging Face token is required to publish");
        }
        let (owner, name) = split_repository(repository)?;
        let cache_directory = cache_directory.into();
        prepare_mutation_root(&cache_directory, "Hugging Face cache directory")?;
        let client = HFClient::builder()
            .endpoint(endpoint.to_owned())
            .cache_dir(cache_directory.clone())
            .token(token)
            .build_sync()?;
        let repository = client.dataset(owner.clone(), name.clone());
        Ok(Self {
            repository_owner: owner,
            repository_name: name,
            repository,
            cache_directory,
        })
    }

    fn require_repository(&self, repository: &str) -> ToolResult<()> {
        let expected = format!("{}/{}", self.repository_owner, self.repository_name);
        if repository != expected {
            return error(format!(
                "publisher is configured for {expected}, not {repository}"
            ));
        }
        Ok(())
    }
}

impl Publisher for HfPublisher {
    fn revision_sha(&self, repository: &str, revision: &str) -> ToolResult<String> {
        self.require_repository(repository)?;
        let info = self
            .repository
            .info()
            .revision(revision.to_owned())
            .send()?;
        info.sha.ok_or_else(|| {
            Box::new(crate::ToolError::new(format!(
                "Hugging Face did not return a commit SHA for {repository}@{revision}"
            ))) as Box<dyn std::error::Error + Send + Sync>
        })
    }

    fn list_files(
        &self,
        repository: &str,
        revision: &str,
        prefix: &str,
    ) -> ToolResult<Vec<RemoteEntry>> {
        self.require_repository(repository)?;
        let entries = match self
            .repository
            .list_tree()
            .revision(revision.to_owned())
            .path_in_repo(prefix.to_owned())
            .recursive(true)
            .expand(true)
            .send()
        {
            Ok(entries) => entries,
            Err(HFError::EntryNotFound { .. }) => return Ok(Vec::new()),
            Err(remote_error) => return Err(remote_error.into()),
        };
        Ok(entries
            .into_iter()
            .filter_map(|entry| match entry {
                RepoTreeEntry::File {
                    path, size, lfs, ..
                } => Some(RemoteEntry {
                    path,
                    byte_count: size,
                    content_sha256: lfs.and_then(|metadata| metadata.sha256),
                }),
                RepoTreeEntry::Directory { .. } => None,
            })
            .collect())
    }

    fn download_file(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        force: bool,
    ) -> ToolResult<Option<PathBuf>> {
        self.require_repository(repository)?;
        prepare_mutation_root(&self.cache_directory, "Hugging Face cache directory")?;
        let downloaded = match self
            .repository
            .download_file()
            .filename(path.to_owned())
            .revision(revision.to_owned())
            .force_download(force)
            .send()
        {
            Ok(path) => Some(path),
            Err(HFError::EntryNotFound { .. }) => None,
            Err(remote_error) => return Err(remote_error.into()),
        };
        if let Some(path) = &downloaded {
            ensure_resolved_within_root(&self.cache_directory, path, "downloaded Hub file")?;
        }
        Ok(downloaded)
    }

    fn create_commit(
        &self,
        repository: &str,
        branch: &str,
        parent_commit: &str,
        message: &str,
        files: &[UploadFile],
    ) -> ToolResult<String> {
        self.require_repository(repository)?;
        let operations = files
            .iter()
            .map(|file| CommitOperation::add_file(&file.path_in_repo, &file.local_path))
            .collect::<Vec<_>>();
        let commit = self
            .repository
            .create_commit()
            .operations(operations)
            .commit_message(message.to_owned())
            .revision(branch.to_owned())
            .parent_commit(parent_commit.to_owned())
            .send()?;
        commit.commit_oid.ok_or_else(|| {
            Box::new(crate::ToolError::new(
                "Hugging Face commit response did not contain commitOid",
            )) as Box<dyn std::error::Error + Send + Sync>
        })
    }
}

pub fn publish_artifact(
    catalog: &mut Catalog,
    options: &PublishOptions,
    publisher: &dyn Publisher,
) -> ToolResult<PublishReceipt> {
    if !options.allow_external_publication {
        return error(
            "publication is blocked until --allow-external-publication confirms the licensing, visibility, allowlist, and cost review",
        );
    }
    reject_overlapping_paths(
        &options.fixture,
        "validated fixture",
        &options.workspace,
        "publish workspace",
    )?;
    prepare_mutation_root(&options.workspace, "publish workspace")?;
    catalog.validate()?;
    let artifact_index = catalog
        .artifacts
        .iter()
        .position(|artifact| artifact.id == options.artifact_id)
        .ok_or_else(|| {
            crate::ToolError::new(format!(
                "artifact {:?} is not present in the catalog",
                options.artifact_id
            ))
        })?;
    let artifact = &catalog.artifacts[artifact_index];
    let verified = VerifiedFixture::load(&options.fixture, Some(artifact))?;
    let manifest_path = options.fixture.join(MANIFEST_NAME);
    let manifest_bytes = fs::read(&manifest_path)?;
    let canonical_manifest = verified.metadata.to_json_pretty()?;
    if manifest_bytes != canonical_manifest {
        return error(format!(
            "{} is valid JSON but is not the canonical v2 manifest serialization",
            manifest_path.display()
        ));
    }
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let target = publication_target(&artifact.publication, &manifest_sha256)?;
    let repository = target.repository;
    let prefix = target.prefix;
    let fixture_identity_sha256 = verified.metadata.identity_sha256();

    let card = render_card(
        &fs::read_to_string(&options.card_template)?,
        &verified.metadata,
        &manifest_sha256,
        &fixture_identity_sha256,
    )?;
    let card_path = options
        .workspace
        .join("cards")
        .join(&artifact.id)
        .join(&manifest_sha256)
        .join(CARD_NAME);
    atomic_write_under(&options.workspace, &card_path, card.as_bytes())?;

    let expected = expected_uploads(
        &options.fixture,
        &verified.metadata.files,
        &prefix,
        &card_path,
    )?;
    let (revision, created_commit) = if let Some(published_revision) = target.published_revision {
        verify_immutable_publication(
            publisher,
            &repository,
            &published_revision,
            &prefix,
            &expected,
        )?;
        (published_revision, false)
    } else {
        let parent = publisher.revision_sha(&repository, &options.branch)?;
        if !is_commit_sha(&parent) {
            return error(format!(
                "repository branch {} resolved to non-immutable revision {parent:?}",
                options.branch
            ));
        }
        let existing = publisher.list_files(&repository, &options.branch, &prefix)?;
        if existing.is_empty() {
            let revision = publisher.create_commit(
                &repository,
                &options.branch,
                &parent,
                &format!("Publish {}", artifact.id),
                &expected
                    .iter()
                    .map(|file| UploadFile {
                        path_in_repo: file.remote_path.clone(),
                        local_path: file.local_path.clone(),
                    })
                    .collect::<Vec<_>>(),
            )?;
            (revision, true)
        } else {
            verify_existing_publication(
                publisher,
                &repository,
                &options.branch,
                &expected,
                &existing,
            )?;
            (parent, false)
        }
    };
    verify_immutable_publication(publisher, &repository, &revision, &prefix, &expected)?;

    let publication = Publication::Published {
        repository: repository.clone(),
        revision: revision.clone(),
        prefix: prefix.clone(),
        manifest_sha256: manifest_sha256.clone(),
    };
    catalog.artifacts[artifact_index].publication = publication;
    catalog.validate()?;
    let metadata_revision = publish_repository_metadata(
        catalog,
        publisher,
        &repository,
        &options.branch,
        &options.workspace,
    )?;
    if let Some(path) = &options.update_catalog {
        let mut bytes = serde_json::to_vec_pretty(catalog)?;
        bytes.push(b'\n');
        atomic_write(path, &bytes)?;
    }
    Ok(PublishReceipt {
        repository,
        revision,
        prefix,
        manifest_sha256,
        created_commit,
        metadata_revision,
    })
}

#[derive(Debug)]
struct ExpectedFile {
    remote_path: String,
    local_path: PathBuf,
    byte_count: u64,
    sha256: String,
}

fn expected_uploads(
    fixture: &Path,
    files: &[bifrost_benchmark_fixture::FixtureFile],
    prefix: &str,
    card_path: &Path,
) -> ToolResult<Vec<ExpectedFile>> {
    let mut expected = Vec::with_capacity(files.len() + 2);
    expected.push(expected_file(
        fixture.join(MANIFEST_NAME),
        join_repo_path(prefix, MANIFEST_NAME)?,
    )?);
    for file in files {
        expected.push(expected_file(
            fixture.join(&file.path),
            join_repo_path(prefix, &file.path)?,
        )?);
    }
    expected.push(expected_file(
        card_path.to_owned(),
        join_repo_path(prefix, CARD_NAME)?,
    )?);
    let unique = expected
        .iter()
        .map(|file| file.remote_path.as_str())
        .collect::<HashSet<_>>();
    if unique.len() != expected.len() {
        return error("publication allowlist contains duplicate remote paths");
    }
    Ok(expected)
}

fn expected_file(local_path: PathBuf, remote_path: String) -> ToolResult<ExpectedFile> {
    let metadata = fs::metadata(&local_path)?;
    if !metadata.is_file() {
        return error(format!("{} is not a regular file", local_path.display()));
    }
    Ok(ExpectedFile {
        sha256: sha256_file(&local_path)?,
        local_path,
        remote_path,
        byte_count: metadata.len(),
    })
}

fn verify_existing_publication(
    publisher: &dyn Publisher,
    repository: &str,
    revision: &str,
    expected: &[ExpectedFile],
    existing: &[RemoteEntry],
) -> ToolResult<()> {
    let expected_paths = expected
        .iter()
        .map(|file| file.remote_path.as_str())
        .collect::<HashSet<_>>();
    let existing_paths = existing
        .iter()
        .map(|file| file.path.as_str())
        .collect::<HashSet<_>>();
    if expected_paths != existing_paths {
        return error(format!(
            "remote content-addressed prefix has a different allowlist; expected {expected_paths:?}, got {existing_paths:?}"
        ));
    }
    verify_remote_files(publisher, repository, revision, expected, existing)
}

fn verify_immutable_publication(
    publisher: &dyn Publisher,
    repository: &str,
    revision: &str,
    prefix: &str,
    expected: &[ExpectedFile],
) -> ToolResult<()> {
    if !is_commit_sha(revision) {
        return error(format!(
            "publisher returned non-immutable revision {revision:?}"
        ));
    }
    let resolved = publisher.revision_sha(repository, revision)?;
    if resolved != revision {
        return error(format!(
            "returned commit {revision} resolves to unexpected commit {resolved}"
        ));
    }
    let remote = publisher.list_files(repository, revision, prefix)?;
    let expected_paths = expected
        .iter()
        .map(|file| file.remote_path.as_str())
        .collect::<HashSet<_>>();
    let remote_paths = remote
        .iter()
        .map(|file| file.path.as_str())
        .collect::<HashSet<_>>();
    if expected_paths != remote_paths {
        return error("immutable publication does not contain the exact expected allowlist");
    }
    verify_remote_files(publisher, repository, revision, expected, &remote)
}

fn verify_remote_files(
    publisher: &dyn Publisher,
    repository: &str,
    revision: &str,
    expected: &[ExpectedFile],
    remote: &[RemoteEntry],
) -> ToolResult<()> {
    let by_path = remote
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<HashMap<_, _>>();
    for file in expected {
        let remote = by_path.get(file.remote_path.as_str()).ok_or_else(|| {
            crate::ToolError::new(format!(
                "remote publication is missing {}",
                file.remote_path
            ))
        })?;
        if remote.byte_count != file.byte_count {
            return error(format!(
                "remote {} has {} bytes; expected {}",
                file.remote_path, remote.byte_count, file.byte_count
            ));
        }
        if let Some(remote_sha256) = &remote.content_sha256
            && remote_sha256 != &file.sha256
        {
            return error(format!(
                "remote {} has SHA-256 {remote_sha256}; expected {}",
                file.remote_path, file.sha256
            ));
        }
        let path = publisher
            .download_file(repository, revision, &file.remote_path, false)?
            .ok_or_else(|| {
                crate::ToolError::new(format!("remote {} is missing", file.remote_path))
            })?;
        if sha256_file(&path)? != file.sha256 {
            let repaired = publisher
                .download_file(repository, revision, &file.remote_path, true)?
                .ok_or_else(|| {
                    crate::ToolError::new(format!("remote {} is missing", file.remote_path))
                })?;
            if sha256_file(&repaired)? != file.sha256 {
                return error(format!(
                    "remote {} content differs from the validated local file",
                    file.remote_path
                ));
            }
        }
    }
    Ok(())
}

struct PublicationTarget {
    repository: String,
    prefix: String,
    published_revision: Option<String>,
}

fn publication_target(
    publication: &Publication,
    manifest_sha256: &str,
) -> ToolResult<PublicationTarget> {
    let (repository, prefix, published_revision) = match publication {
        Publication::Planned {
            repository, prefix, ..
        } => {
            if prefix.matches(MANIFEST_PLACEHOLDER).count() != 1 {
                return error(format!(
                    "planned prefix must contain exactly one {MANIFEST_PLACEHOLDER} segment"
                ));
            }
            (
                repository.clone(),
                prefix.replace(MANIFEST_PLACEHOLDER, manifest_sha256),
                None,
            )
        }
        Publication::Published {
            repository,
            revision,
            prefix,
            manifest_sha256: published_manifest,
            ..
        } => {
            if published_manifest != manifest_sha256 {
                return error(format!(
                    "fixture manifest SHA-256 {manifest_sha256} differs from catalog publication {published_manifest}"
                ));
            }
            (repository.clone(), prefix.clone(), Some(revision.clone()))
        }
    };
    checked_relative_path(&prefix)?;
    if Path::new(&prefix)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(manifest_sha256)
    {
        return error(format!(
            "publication prefix must end with the canonical manifest SHA-256 {manifest_sha256}"
        ));
    }
    Ok(PublicationTarget {
        repository,
        prefix,
        published_revision,
    })
}

fn render_card(
    template: &str,
    manifest: &bifrost_benchmark_fixture::FixtureManifest,
    manifest_sha256: &str,
    fixture_identity_sha256: &str,
) -> ToolResult<String> {
    let dimensions = manifest.dimensions.to_string();
    let replacements = [
        ("{{source_url}}", manifest.source.url.as_str()),
        ("{{source_revision}}", manifest.source.revision.as_str()),
        ("{{source_sha256}}", manifest.source.sha256.as_str()),
        ("{{requested_model}}", manifest.requested_model.as_str()),
        ("{{returned_model}}", manifest.returned_model.as_str()),
        ("{{dimensions}}", dimensions.as_str()),
        ("{{preprocessing}}", manifest.preprocessing.as_str()),
        ("{{selection}}", manifest.selection.as_str()),
        (
            "{{generator_revision}}",
            manifest.generator_revision.as_str(),
        ),
        ("{{manifest_sha256}}", manifest_sha256),
        ("{{fixture_identity_sha256}}", fixture_identity_sha256),
    ];
    let mut card = template.to_owned();
    for (placeholder, value) in replacements {
        card = card.replace(placeholder, value);
    }
    if card.contains("{{") || card.contains("}}") {
        return error("dataset card template contains an unresolved placeholder");
    }
    Ok(card)
}

fn publish_repository_metadata(
    catalog: &Catalog,
    publisher: &dyn Publisher,
    repository: &str,
    branch: &str,
    workspace: &Path,
) -> ToolResult<String> {
    let parent = publisher.revision_sha(repository, branch)?;
    if !is_commit_sha(&parent) {
        return error(format!(
            "repository branch {branch} resolved to {parent:?}, not a commit SHA"
        ));
    }
    let mut catalog_bytes = serde_json::to_vec_pretty(catalog)?;
    catalog_bytes.push(b'\n');
    let root_card = render_root_card(catalog);
    let metadata_identity = sha256_bytes(
        [catalog_bytes.as_slice(), root_card.as_bytes()]
            .concat()
            .as_slice(),
    );
    let directory = workspace.join("metadata").join(metadata_identity);
    let catalog_path = directory.join("catalog.json");
    let card_path = directory.join(CARD_NAME);
    atomic_write_under(workspace, &catalog_path, &catalog_bytes)?;
    atomic_write_under(workspace, &card_path, root_card.as_bytes())?;
    let expected = [
        expected_file(card_path, CARD_NAME.to_owned())?,
        expected_file(catalog_path, "catalog.json".to_owned())?,
    ];
    let mut already_current = true;
    for file in &expected {
        let Some(path) = publisher.download_file(repository, branch, &file.remote_path, false)?
        else {
            already_current = false;
            continue;
        };
        if sha256_file(&path)? != file.sha256 {
            already_current = false;
        }
    }
    let revision = if already_current {
        parent
    } else {
        publisher.create_commit(
            repository,
            branch,
            &parent,
            "Update benchmark artifact catalog and dataset card",
            &expected
                .iter()
                .map(|file| UploadFile {
                    path_in_repo: file.remote_path.clone(),
                    local_path: file.local_path.clone(),
                })
                .collect::<Vec<_>>(),
        )?
    };
    if !is_commit_sha(&revision) || publisher.revision_sha(repository, &revision)? != revision {
        return error(format!(
            "metadata publication returned invalid commit {revision:?}"
        ));
    }
    for file in &expected {
        let path = publisher
            .download_file(repository, &revision, &file.remote_path, false)?
            .ok_or_else(|| {
                crate::ToolError::new(format!("metadata commit is missing {}", file.remote_path))
            })?;
        if sha256_file(&path)? != file.sha256 {
            let repaired = publisher
                .download_file(repository, &revision, &file.remote_path, true)?
                .ok_or_else(|| {
                    crate::ToolError::new(format!(
                        "metadata commit is missing {}",
                        file.remote_path
                    ))
                })?;
            if sha256_file(&repaired)? != file.sha256 {
                return error(format!(
                    "metadata commit contains unexpected {} content",
                    file.remote_path
                ));
            }
        }
    }
    Ok(revision)
}

fn render_root_card(catalog: &Catalog) -> String {
    let mut card = String::from(
        "---\nlicense: cc-by-sa-4.0\nlanguage:\n- en\npretty_name: Bifrost benchmark embedding fixtures\n---\n\n# Bifrost benchmark embedding fixtures\n\nThis dataset contains immutable, checksummed embedding fixtures for Bifrost's reproducible HNSW competitor benchmarks. The derived FiQA artifacts retain the source dataset's CC BY-SA 4.0 license.\n\n| artifact | dataset | model | dimensions | publication |\n|---|---|---|---:|---|\n",
    );
    for artifact in &catalog.artifacts {
        let publication = match &artifact.publication {
            Publication::Planned { blocked_by, .. } => format!("planned: {blocked_by}"),
            Publication::Published {
                revision, prefix, ..
            } => format!("`{revision}` at `{prefix}`"),
        };
        card.push_str(&format!(
            "| `{}` | {} | `{}` | {} | {} |\n",
            artifact.id, artifact.dataset, artifact.model, artifact.dimensions, publication
        ));
    }
    card.push_str(
        "\nEach fixture directory includes an immutable manifest with per-file SHA-256 checksums and provenance. The 384- and 1536-dimensional OpenAI fixtures come from independent requests; neither is truncated from the other. Review repository visibility, licensing, provenance, file sizes, and external publication cost before publishing a new artifact.\n",
    );
    card
}

fn split_repository(repository: &str) -> ToolResult<(String, String)> {
    let mut parts = repository.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty() || name.is_empty() || parts.next().is_some() {
        return error(format!(
            "Hugging Face repository must have owner/name form, got {repository:?}"
        ));
    }
    Ok((owner.to_owned(), name.to_owned()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use bifrost_benchmark_fixture::{ArtifactResolution, CATALOG_SCHEMA, Catalog, Publication};

    use super::*;
    use crate::{
        ToolError,
        test_support::{planned_artifact, write_fixture},
    };

    #[derive(Default)]
    struct FakeState {
        branch: String,
        revisions: HashMap<String, HashMap<String, Vec<u8>>>,
        commits: Vec<Vec<String>>,
        fail_commit_number: Option<usize>,
    }

    struct FakePublisher {
        downloads: PathBuf,
        state: Mutex<FakeState>,
    }

    impl FakePublisher {
        fn new(root: &Path) -> Self {
            let initial = "0".repeat(40);
            let mut revisions = HashMap::new();
            revisions.insert(initial.clone(), HashMap::new());
            Self {
                downloads: root.join("remote-downloads"),
                state: Mutex::new(FakeState {
                    branch: initial,
                    revisions,
                    commits: Vec::new(),
                    fail_commit_number: None,
                }),
            }
        }

        fn fail_commit_number(&self, number: usize) {
            self.state.lock().unwrap().fail_commit_number = Some(number);
        }

        fn corrupt_same_size(&self, revision: &str, suffix: &str) {
            let mut state = self.state.lock().unwrap();
            let files = state.revisions.get_mut(revision).unwrap();
            let (_, bytes) = files
                .iter_mut()
                .find(|(path, _)| path.ends_with(suffix))
                .unwrap();
            bytes[0] ^= 0xff;
        }
    }

    impl Publisher for FakePublisher {
        fn revision_sha(&self, _repository: &str, revision: &str) -> ToolResult<String> {
            let state = self.state.lock().unwrap();
            if revision == "main" {
                return Ok(state.branch.clone());
            }
            if state.revisions.contains_key(revision) {
                Ok(revision.to_owned())
            } else {
                Err(Box::new(ToolError::new(format!(
                    "unknown revision {revision}"
                ))))
            }
        }

        fn list_files(
            &self,
            _repository: &str,
            revision: &str,
            prefix: &str,
        ) -> ToolResult<Vec<RemoteEntry>> {
            let state = self.state.lock().unwrap();
            let revision = if revision == "main" {
                &state.branch
            } else {
                revision
            };
            let files = state
                .revisions
                .get(revision)
                .ok_or_else(|| ToolError::new(format!("unknown revision {revision}")))?;
            let prefix = format!("{}/", prefix.trim_end_matches('/'));
            Ok(files
                .iter()
                .filter(|(path, _)| path.starts_with(&prefix))
                .map(|(path, bytes)| RemoteEntry {
                    path: path.clone(),
                    byte_count: bytes.len() as u64,
                    content_sha256: None,
                })
                .collect())
        }

        fn download_file(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
            _force: bool,
        ) -> ToolResult<Option<PathBuf>> {
            let state = self.state.lock().unwrap();
            let revision = if revision == "main" {
                &state.branch
            } else {
                revision
            };
            let Some(bytes) = state
                .revisions
                .get(revision)
                .and_then(|files| files.get(path))
            else {
                return Ok(None);
            };
            let destination = self
                .downloads
                .join(revision)
                .join(format!("{}.download", sha256_bytes(path.as_bytes())));
            fs::create_dir_all(destination.parent().unwrap())?;
            fs::write(&destination, bytes)?;
            Ok(Some(destination))
        }

        fn create_commit(
            &self,
            _repository: &str,
            _branch: &str,
            parent_commit: &str,
            _message: &str,
            files: &[UploadFile],
        ) -> ToolResult<String> {
            let mut state = self.state.lock().unwrap();
            if state.branch != parent_commit {
                return Err(Box::new(ToolError::new("parent commit conflict")));
            }
            let commit_number = state.commits.len() + 1;
            if state.fail_commit_number == Some(commit_number) {
                state.fail_commit_number = None;
                return Err(Box::new(ToolError::new("injected commit failure")));
            }
            let mut contents = state.revisions.get(parent_commit).unwrap().clone();
            let mut paths = Vec::new();
            for file in files {
                contents.insert(file.path_in_repo.clone(), fs::read(&file.local_path)?);
                paths.push(file.path_in_repo.clone());
            }
            let revision = format!("{:040x}", state.commits.len() + 1);
            state.revisions.insert(revision.clone(), contents);
            state.branch = revision.clone();
            state.commits.push(paths);
            Ok(revision)
        }
    }

    fn setup() -> (tempfile::TempDir, Catalog, PublishOptions, FakePublisher) {
        let temp = tempfile::tempdir().unwrap();
        let artifact = planned_artifact(4);
        let fixture = temp.path().join("fixture");
        write_fixture(&fixture, &artifact).unwrap();
        let catalog = Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![artifact.clone()],
        };
        let options = PublishOptions {
            artifact_id: artifact.id,
            fixture,
            branch: "main".to_owned(),
            card_template: Path::new(env!("CARGO_MANIFEST_DIR")).join("dataset-card-template.md"),
            workspace: temp.path().join("publish-workspace"),
            update_catalog: None,
            allow_external_publication: true,
        };
        let publisher = FakePublisher::new(temp.path());
        (temp, catalog, options, publisher)
    }

    #[test]
    fn publication_uses_exact_artifact_and_metadata_allowlists() {
        let (_temp, mut catalog, options, publisher) = setup();
        let artifact = catalog.artifact(&options.artifact_id).unwrap();
        ArtifactResolution {
            artifact_id: artifact.id.clone(),
            repository: "owner/repository".to_owned(),
            revision: "f".repeat(40),
            manifest_sha256: sha256_file(&options.fixture.join(MANIFEST_NAME)).unwrap(),
        }
        .write(options.fixture.join(".bifrost-resolution.json"))
        .unwrap();
        let receipt = publish_artifact(&mut catalog, &options, &publisher).unwrap();
        assert!(receipt.created_commit);
        let state = publisher.state.lock().unwrap();
        assert_eq!(state.commits.len(), 2);
        let artifact_paths = &state.commits[0];
        assert_eq!(artifact_paths.len(), 7);
        assert!(
            artifact_paths
                .iter()
                .all(|path| path.starts_with(&receipt.prefix))
        );
        assert!(
            artifact_paths
                .iter()
                .all(|path| !path.contains(".bifrost-resolution"))
        );
        assert!(artifact_paths.iter().all(|path| !path.contains("source")));
        assert!(artifact_paths.iter().all(|path| !path.contains("batches")));
        let metadata_paths = state.commits[1].iter().cloned().collect::<HashSet<_>>();
        assert_eq!(
            metadata_paths,
            HashSet::from(["README.md".to_owned(), "catalog.json".to_owned()])
        );
        assert_eq!(receipt.revision, format!("{:040x}", 1));
        assert_eq!(receipt.metadata_revision, format!("{:040x}", 2));
    }

    #[test]
    fn repeated_publication_is_idempotent() {
        let (_temp, mut catalog, options, publisher) = setup();
        let first = publish_artifact(&mut catalog, &options, &publisher).unwrap();
        let second = publish_artifact(&mut catalog, &options, &publisher).unwrap();
        assert!(!second.created_commit);
        assert_eq!(second.revision, first.revision);
        assert_eq!(second.metadata_revision, first.metadata_revision);
        assert_eq!(publisher.state.lock().unwrap().commits.len(), 2);
    }

    #[test]
    fn retry_converges_after_artifact_commit_precedes_catalog_update() {
        let (temp, mut catalog, mut options, publisher) = setup();
        let catalog_path = temp.path().join("catalog.json");
        let mut original_bytes = serde_json::to_vec_pretty(&catalog).unwrap();
        original_bytes.push(b'\n');
        fs::write(&catalog_path, original_bytes).unwrap();
        options.update_catalog = Some(catalog_path.clone());
        publisher.fail_commit_number(2);

        let failure = publish_artifact(&mut catalog, &options, &publisher).unwrap_err();
        assert!(failure.to_string().contains("injected commit failure"));
        let mut retry_catalog = Catalog::load(&catalog_path).unwrap();
        assert!(matches!(
            retry_catalog
                .artifact(&options.artifact_id)
                .unwrap()
                .publication,
            Publication::Planned { .. }
        ));

        let receipt = publish_artifact(&mut retry_catalog, &options, &publisher).unwrap();

        assert!(!receipt.created_commit);
        let persisted = Catalog::load(&catalog_path).unwrap();
        let Publication::Published { revision, .. } = &persisted
            .artifact(&options.artifact_id)
            .unwrap()
            .publication
        else {
            panic!("catalog did not converge to Published");
        };
        assert_eq!(revision, &receipt.revision);
        assert_eq!(publisher.state.lock().unwrap().commits.len(), 2);
    }

    #[test]
    fn same_size_remote_corruption_is_rejected_without_hash_metadata() {
        let (_temp, mut catalog, options, publisher) = setup();
        let receipt = publish_artifact(&mut catalog, &options, &publisher).unwrap();
        publisher.corrupt_same_size(&receipt.revision, "corpus.f32");
        let failure = publish_artifact(&mut catalog, &options, &publisher).unwrap_err();
        assert!(failure.to_string().contains("content differs"));
    }

    #[test]
    fn publish_workspace_must_not_be_inside_the_fixture() {
        let (_temp, mut catalog, mut options, publisher) = setup();
        options.workspace = options.fixture.join("workspace");
        let failure = publish_artifact(&mut catalog, &options, &publisher).unwrap_err();
        assert!(failure.to_string().contains("non-nested"));
        assert!(publisher.state.lock().unwrap().commits.is_empty());
    }

    #[test]
    fn noncanonical_manifest_json_is_not_content_addressed_or_published() {
        let (_temp, mut catalog, options, publisher) = setup();
        let manifest_path = options.fixture.join(MANIFEST_NAME);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();

        let failure = publish_artifact(&mut catalog, &options, &publisher).unwrap_err();

        assert!(failure.to_string().contains("canonical v2 manifest"));
        assert!(publisher.state.lock().unwrap().commits.is_empty());
    }

    #[test]
    fn fixture_must_not_be_inside_the_publish_workspace() {
        let (temp, mut catalog, mut options, publisher) = setup();
        options.workspace = temp.path().join("parent-workspace");
        options.fixture = options.workspace.join("fixture");
        let artifact = catalog.artifact(&options.artifact_id).unwrap().clone();
        write_fixture(&options.fixture, &artifact).unwrap();
        let failure = publish_artifact(&mut catalog, &options, &publisher).unwrap_err();
        assert!(failure.to_string().contains("non-nested"));
        assert!(publisher.state.lock().unwrap().commits.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn card_workspace_symlink_cannot_escape_before_publication() {
        let (temp, mut catalog, options, publisher) = setup();
        fs::create_dir_all(&options.workspace).unwrap();
        let outside = temp.path().join("outside-cards");
        fs::create_dir(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        fs::write(&sentinel, b"unchanged").unwrap();
        std::os::unix::fs::symlink(&outside, options.workspace.join("cards")).unwrap();

        let failure = publish_artifact(&mut catalog, &options, &publisher).unwrap_err();

        assert!(failure.to_string().contains("refusing symlink"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
        assert!(publisher.state.lock().unwrap().commits.is_empty());
    }
}
