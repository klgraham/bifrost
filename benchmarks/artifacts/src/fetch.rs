use std::{
    fs,
    path::{Path, PathBuf},
};

use bifrost_benchmark_fixture::{
    Artifact, ArtifactResolution, Catalog, FixtureManifest, Publication, VerifiedFixture,
};
use hf_hub::{HFClient, HFRepositorySync, repository::RepoTypeDataset};

use crate::{
    ToolResult, error,
    util::{
        ArtifactLock, checked_relative_path, clean_stale_stages, copy_checked_under,
        create_dir_all_under, ensure_resolved_within_root, ensure_safe_descendant,
        install_directory, is_commit_sha, is_sha256, join_repo_path, make_tree_read_only,
        path_present, prepare_mutation_root, remove_tree, safe_name, sha256_file, unique_stage,
    },
};

const MANIFEST_NAME: &str = "manifest.json";
const RESOLUTION_NAME: &str = ".bifrost-resolution.json";

#[derive(Debug, Clone)]
pub struct FetchOptions {
    pub artifact_id: String,
    pub destination_root: PathBuf,
    pub offline: bool,
    pub repair: bool,
}

pub trait RemoteFileSource: Send + Sync {
    /// Resolve a file strictly from the configured local cache.
    fn cached_file(&self, repository: &str, revision: &str, path: &str) -> ToolResult<PathBuf>;

    /// Download a file, optionally bypassing an existing cache entry.
    fn download_file(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        force: bool,
    ) -> ToolResult<PathBuf>;
}

pub struct HfDownloader {
    repository_owner: String,
    repository_name: String,
    repository: HFRepositorySync<RepoTypeDataset>,
    cache_directory: PathBuf,
}

impl HfDownloader {
    pub fn new(
        repository: &str,
        endpoint: &str,
        cache_directory: impl Into<PathBuf>,
        token: Option<String>,
    ) -> ToolResult<Self> {
        let (owner, name) = split_repository(repository)?;
        let cache_directory = cache_directory.into();
        prepare_mutation_root(&cache_directory, "Hugging Face cache directory")?;
        let mut builder = HFClient::builder()
            .endpoint(endpoint.to_owned())
            .cache_dir(cache_directory.clone());
        if let Some(token) = token {
            if token.trim().is_empty() {
                return error("the configured Hugging Face token is empty");
            }
            builder = builder.token(token);
        }
        let client = builder.build_sync()?;
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
                "downloader is configured for {expected}, not {repository}"
            ));
        }
        Ok(())
    }
}

impl RemoteFileSource for HfDownloader {
    fn cached_file(&self, repository: &str, revision: &str, path: &str) -> ToolResult<PathBuf> {
        self.require_repository(repository)?;
        prepare_mutation_root(&self.cache_directory, "Hugging Face cache directory")?;
        let cached = self
            .repository
            .download_file()
            .filename(path.to_owned())
            .revision(revision.to_owned())
            .local_files_only(true)
            .send()?;
        ensure_resolved_within_root(&self.cache_directory, &cached, "cached Hub file")?;
        Ok(cached)
    }

    fn download_file(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        force: bool,
    ) -> ToolResult<PathBuf> {
        self.require_repository(repository)?;
        prepare_mutation_root(&self.cache_directory, "Hugging Face cache directory")?;
        let downloaded = self
            .repository
            .download_file()
            .filename(path.to_owned())
            .revision(revision.to_owned())
            .force_download(force)
            .send()?;
        ensure_resolved_within_root(&self.cache_directory, &downloaded, "downloaded Hub file")?;
        Ok(downloaded)
    }
}

pub fn fetch_artifact(
    catalog: &Catalog,
    options: &FetchOptions,
    source: &dyn RemoteFileSource,
) -> ToolResult<PathBuf> {
    catalog.validate()?;
    let artifact = catalog.artifact(&options.artifact_id).ok_or_else(|| {
        crate::ToolError::new(format!(
            "artifact {:?} is not present in the catalog",
            options.artifact_id
        ))
    })?;
    let PublishedArtifact {
        repository,
        revision,
        prefix,
        manifest_sha256,
    } = published_artifact(artifact)?;

    prepare_mutation_root(&options.destination_root, "fixture destination root")?;
    let _lock = ArtifactLock::acquire(&options.destination_root, &artifact.id)?;
    let destination = options
        .destination_root
        .join(safe_name(&artifact.id))
        .join(&revision);
    ensure_safe_descendant(
        &options.destination_root,
        &destination,
        "fixture destination",
    )?;
    if path_present(&destination) {
        match validate_resolved_fixture(&destination, artifact) {
            Ok(()) => {
                make_tree_read_only(&destination)?;
                return Ok(fs::canonicalize(destination)?);
            }
            Err(error) if !options.repair => {
                return Err(Box::new(crate::ToolError::new(format!(
                    "existing fixture {} is invalid: {error}; pass --repair with network access to replace it",
                    destination.display()
                ))));
            }
            Err(error) if options.offline => {
                return Err(Box::new(crate::ToolError::new(format!(
                    "existing fixture {} is invalid and cannot be repaired offline: {error}",
                    destination.display()
                ))));
            }
            Err(_) => {}
        }
    }

    let stage_root = options.destination_root.join(".staging");
    create_dir_all_under(
        &options.destination_root,
        &stage_root,
        "fixture staging root",
    )?;
    clean_stale_stages(&stage_root, &artifact.id)?;
    let stage = unique_stage(&stage_root, &artifact.id);
    create_dir_all_under(&stage_root, &stage, "fixture staging directory")?;

    let result = stage_fixture(
        source,
        artifact,
        &repository,
        &revision,
        &prefix,
        &manifest_sha256,
        &stage,
        options.offline,
        options.repair,
    )
    .and_then(|()| {
        make_tree_read_only(&stage)?;
        install_directory(&options.destination_root, &stage, &destination)?;
        Ok(())
    });
    if result.is_err() && path_present(&stage) {
        let _ = remove_tree(&stage);
    }
    result?;
    Ok(fs::canonicalize(destination)?)
}

#[allow(clippy::too_many_arguments)]
fn stage_fixture(
    source: &dyn RemoteFileSource,
    artifact: &Artifact,
    repository: &str,
    revision: &str,
    prefix: &str,
    manifest_sha256: &str,
    stage: &Path,
    offline: bool,
    repair: bool,
) -> ToolResult<()> {
    let remote_manifest = join_repo_path(prefix, MANIFEST_NAME)?;
    let cached_manifest = verified_remote_file(
        source,
        repository,
        revision,
        &remote_manifest,
        manifest_sha256,
        offline,
        repair,
    )?;
    let local_manifest = stage.join(MANIFEST_NAME);
    copy_checked_under(stage, &cached_manifest, &local_manifest, manifest_sha256)?;

    let manifest = FixtureManifest::load(&local_manifest)?;
    let mut seen_paths = std::collections::HashSet::new();
    for file in &manifest.files {
        if !seen_paths.insert(file.path.clone()) {
            return error(format!("manifest repeats fixture path {:?}", file.path));
        }
        let relative = checked_relative_path(&file.path)?;
        let remote = join_repo_path(prefix, &file.path)?;
        let cached = verified_remote_file(
            source,
            repository,
            revision,
            &remote,
            &file.sha256,
            offline,
            repair,
        )?;
        copy_checked_under(stage, &cached, &stage.join(relative), &file.sha256)?;
    }

    VerifiedFixture::load(stage, Some(artifact))?;
    let resolution = ArtifactResolution {
        artifact_id: artifact.id.clone(),
        repository: repository.to_owned(),
        revision: revision.to_owned(),
        manifest_sha256: manifest_sha256.to_owned(),
    };
    resolution.validate()?;
    resolution.write(stage.join(RESOLUTION_NAME))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verified_remote_file(
    source: &dyn RemoteFileSource,
    repository: &str,
    revision: &str,
    path: &str,
    expected_sha256: &str,
    offline: bool,
    repair: bool,
) -> ToolResult<PathBuf> {
    if !is_sha256(expected_sha256) {
        return error(format!("invalid SHA-256 for {path}: {expected_sha256:?}"));
    }
    let first = if offline {
        source.cached_file(repository, revision, path)
    } else {
        source.download_file(repository, revision, path, false)
    }?;
    if sha256_file(&first)? == expected_sha256 {
        return Ok(first);
    }
    if offline {
        return error(format!(
            "cached file {path} is corrupt; offline mode cannot repair it"
        ));
    }
    if !repair {
        return error(format!(
            "cached file {path} is corrupt; pass --repair to force a fresh download"
        ));
    }
    let repaired = source.download_file(repository, revision, path, true)?;
    let actual = sha256_file(&repaired)?;
    if actual != expected_sha256 {
        return error(format!(
            "fresh download for {path} has SHA-256 {actual}, expected {expected_sha256}"
        ));
    }
    Ok(repaired)
}

fn validate_resolved_fixture(directory: &Path, artifact: &Artifact) -> ToolResult<()> {
    VerifiedFixture::load(directory, Some(artifact))?;
    let resolution = ArtifactResolution::load(directory.join(RESOLUTION_NAME))?;
    resolution.validate()?;
    let expected = published_artifact(artifact)?;
    if resolution.artifact_id != artifact.id
        || resolution.repository != expected.repository
        || resolution.revision != expected.revision
        || resolution.manifest_sha256 != expected.manifest_sha256
    {
        return error(format!(
            "{} does not match the catalog publication",
            directory.join(RESOLUTION_NAME).display()
        ));
    }
    Ok(())
}

struct PublishedArtifact {
    repository: String,
    revision: String,
    prefix: String,
    manifest_sha256: String,
}

fn published_artifact(artifact: &Artifact) -> ToolResult<PublishedArtifact> {
    let Publication::Published {
        repository,
        revision,
        prefix,
        manifest_sha256,
    } = &artifact.publication
    else {
        return error(format!(
            "artifact {:?} is planned but has not been published",
            artifact.id
        ));
    };
    if !is_commit_sha(revision) {
        return error(format!(
            "artifact {:?} does not pin an immutable 40-hex revision",
            artifact.id
        ));
    }
    checked_relative_path(prefix)?;
    if !is_sha256(manifest_sha256) {
        return error(format!(
            "artifact {:?} has an invalid manifest SHA-256",
            artifact.id
        ));
    }
    Ok(PublishedArtifact {
        repository: repository.clone(),
        revision: revision.clone(),
        prefix: prefix.clone(),
        manifest_sha256: manifest_sha256.clone(),
    })
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
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
    };

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use bifrost_benchmark_fixture::{CATALOG_SCHEMA, Catalog, FixtureManifest};

    use super::*;
    use crate::{
        ToolError,
        test_support::{planned_artifact, published_artifact, write_fixture},
        util::remove_tree,
    };

    struct FakeSource {
        remote: PathBuf,
        cache: PathBuf,
        network_calls: AtomicUsize,
        cache_calls: AtomicUsize,
        force_calls: AtomicUsize,
    }

    impl FakeSource {
        fn new(root: &Path) -> Self {
            Self {
                remote: root.join("remote"),
                cache: root.join("cache"),
                network_calls: AtomicUsize::new(0),
                cache_calls: AtomicUsize::new(0),
                force_calls: AtomicUsize::new(0),
            }
        }

        fn cache_path(&self, path: &str) -> PathBuf {
            self.cache.join(path)
        }
    }

    impl RemoteFileSource for FakeSource {
        fn cached_file(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
        ) -> ToolResult<PathBuf> {
            self.cache_calls.fetch_add(1, Ordering::SeqCst);
            let cached = self.cache_path(path);
            if cached.is_file() {
                Ok(cached)
            } else {
                Err(Box::new(ToolError::new(format!("cache miss for {path}"))))
            }
        }

        fn download_file(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
            force: bool,
        ) -> ToolResult<PathBuf> {
            self.network_calls.fetch_add(1, Ordering::SeqCst);
            if force {
                self.force_calls.fetch_add(1, Ordering::SeqCst);
            }
            let remote = self.remote.join(path);
            if !remote.is_file() {
                return Err(Box::new(ToolError::new(format!("remote miss for {path}"))));
            }
            let cached = self.cache_path(path);
            if force || !cached.exists() {
                fs::create_dir_all(cached.parent().unwrap())?;
                fs::copy(remote, &cached)?;
            }
            Ok(cached)
        }
    }

    fn setup() -> (tempfile::TempDir, Catalog, FetchOptions, Arc<FakeSource>) {
        let temp = tempfile::tempdir().unwrap();
        let fixture = temp.path().join("fixture-source");
        let planned = planned_artifact(4);
        let manifest_sha256 = write_fixture(&fixture, &planned).unwrap();
        let revision = "a".repeat(40);
        let artifact = published_artifact(planned, manifest_sha256, &revision);
        let Publication::Published { prefix, .. } = &artifact.publication else {
            unreachable!();
        };
        let source = Arc::new(FakeSource::new(temp.path()));
        let remote_fixture = source.remote.join(prefix);
        fs::create_dir_all(&remote_fixture).unwrap();
        for entry in fs::read_dir(&fixture).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), remote_fixture.join(entry.file_name())).unwrap();
        }
        let catalog = Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![artifact.clone()],
        };
        let options = FetchOptions {
            artifact_id: artifact.id,
            destination_root: temp.path().join("destinations"),
            offline: false,
            repair: false,
        };
        (temp, catalog, options, source)
    }

    #[test]
    fn empty_cache_fails_offline_without_taking_network_path() {
        let (_temp, catalog, mut options, source) = setup();
        options.offline = true;
        let failure = fetch_artifact(&catalog, &options, source.as_ref()).unwrap_err();
        assert!(failure.to_string().contains("cache miss"));
        assert_eq!(source.network_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn online_then_offline_rebuild_uses_only_cached_files() {
        let (_temp, catalog, mut options, source) = setup();
        let first = fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        let online_calls = source.network_calls.load(Ordering::SeqCst);
        remove_tree(&first).unwrap();
        options.offline = true;
        let second = fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        assert!(second.is_dir());
        assert_eq!(source.network_calls.load(Ordering::SeqCst), online_calls);
        assert!(source.cache_calls.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn corrupt_cache_requires_and_honors_explicit_repair() {
        let (_temp, catalog, mut options, source) = setup();
        let first = fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        remove_tree(&first).unwrap();
        let artifact = catalog.artifact(&options.artifact_id).unwrap();
        let Publication::Published { prefix, .. } = &artifact.publication else {
            unreachable!();
        };
        let manifest =
            FixtureManifest::load(source.cache.join(prefix).join(MANIFEST_NAME)).unwrap();
        let corrupt = source.cache.join(prefix).join(&manifest.files[0].path);
        fs::write(
            &corrupt,
            vec![0_u8; fs::metadata(&corrupt).unwrap().len() as usize],
        )
        .unwrap();
        let failure = fetch_artifact(&catalog, &options, source.as_ref()).unwrap_err();
        assert!(failure.to_string().contains("--repair"));
        assert_eq!(source.force_calls.load(Ordering::SeqCst), 0);
        options.repair = true;
        let repaired = fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        assert!(repaired.is_dir());
        assert_eq!(source.force_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn incomplete_staging_directory_is_removed_before_fetch() {
        let (_temp, catalog, options, source) = setup();
        let stage_root = options.destination_root.join(".staging");
        let stale = unique_stage(&stage_root, &options.artifact_id);
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("partial"), b"partial").unwrap();
        fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        assert!(!stale.exists());
    }

    #[test]
    fn concurrent_fetches_share_one_artifact_lock() {
        let (_temp, catalog, options, source) = setup();
        let catalog = Arc::new(catalog);
        let options = Arc::new(options);
        let handles = (0..2)
            .map(|_| {
                let catalog = Arc::clone(&catalog);
                let options = Arc::clone(&options);
                let source = Arc::clone(&source);
                thread::spawn(move || fetch_artifact(&catalog, &options, source.as_ref()).unwrap())
            })
            .collect::<Vec<_>>();
        let paths = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(paths[0], paths[1]);
        let manifest = FixtureManifest::load(paths[0].join(MANIFEST_NAME)).unwrap();
        assert_eq!(
            source.network_calls.load(Ordering::SeqCst),
            manifest.files.len() + 1
        );
    }

    #[cfg(unix)]
    #[test]
    fn completed_fixture_tree_is_read_only() {
        let (_temp, catalog, options, source) = setup();
        let fixture = fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        let mut paths = vec![fixture.clone()];
        paths.extend(
            fs::read_dir(&fixture)
                .unwrap()
                .map(|entry| entry.unwrap().path()),
        );
        for path in paths {
            assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o222, 0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn repair_does_not_follow_symlinks_in_a_corrupt_destination() {
        let (temp, catalog, mut options, source) = setup();
        options.repair = true;
        let artifact = catalog.artifact(&options.artifact_id).unwrap();
        let Publication::Published { revision, .. } = &artifact.publication else {
            unreachable!();
        };
        let destination = options
            .destination_root
            .join(&options.artifact_id)
            .join(revision);
        fs::create_dir_all(&destination).unwrap();
        let outside = temp.path().join("outside.txt");
        fs::write(&outside, b"unchanged").unwrap();
        let before = fs::metadata(&outside).unwrap().permissions().mode();
        std::os::unix::fs::symlink(&outside, destination.join("manifest.json")).unwrap();
        fetch_artifact(&catalog, &options, source.as_ref()).unwrap();
        assert_eq!(fs::read(&outside).unwrap(), b"unchanged");
        assert_eq!(fs::metadata(&outside).unwrap().permissions().mode(), before);
    }

    #[cfg(unix)]
    #[test]
    fn artifact_directory_symlink_cannot_escape_the_destination_root() {
        let (temp, catalog, options, source) = setup();
        fs::create_dir_all(&options.destination_root).unwrap();
        let outside = temp.path().join("outside-directory");
        fs::create_dir(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        fs::write(&sentinel, b"unchanged").unwrap();
        std::os::unix::fs::symlink(
            &outside,
            options.destination_root.join(&options.artifact_id),
        )
        .unwrap();

        let failure = fetch_artifact(&catalog, &options, source.as_ref()).unwrap_err();

        assert!(failure.to_string().contains("refusing symlink"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
        assert_eq!(source.network_calls.load(Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[test]
    fn staging_root_symlink_cannot_escape_the_destination_root() {
        let (temp, catalog, options, source) = setup();
        fs::create_dir_all(&options.destination_root).unwrap();
        let outside = temp.path().join("outside-staging");
        fs::create_dir(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        fs::write(&sentinel, b"unchanged").unwrap();
        std::os::unix::fs::symlink(&outside, options.destination_root.join(".staging")).unwrap();

        let failure = fetch_artifact(&catalog, &options, source.as_ref()).unwrap_err();

        assert!(failure.to_string().contains("refusing symlink"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
        assert_eq!(source.network_calls.load(Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[test]
    fn lock_directory_symlink_is_rejected_without_touching_its_target() {
        let (temp, catalog, options, source) = setup();
        fs::create_dir_all(&options.destination_root).unwrap();
        let outside = temp.path().join("outside-locks");
        fs::create_dir(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        fs::write(&sentinel, b"unchanged").unwrap();
        std::os::unix::fs::symlink(&outside, options.destination_root.join(".locks")).unwrap();

        let failure = fetch_artifact(&catalog, &options, source.as_ref()).unwrap_err();

        assert!(failure.to_string().contains("refusing symlink"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
        assert_eq!(source.network_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn checked_in_catalog_contains_both_independent_dimensions() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog.json");
        let catalog = Catalog::load(path).unwrap();
        catalog.validate().unwrap();
        let dimensions = catalog
            .artifacts
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact.dimensions))
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(
            dimensions.get("fiqa-text-embedding-3-small-384"),
            Some(&384)
        );
        assert_eq!(
            dimensions.get("fiqa-text-embedding-3-small-1536"),
            Some(&1536)
        );
    }
}
