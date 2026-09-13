use std::{
    collections::VecDeque,
    env,
    ffi::OsString,
    fs,
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

use bifrost_benchmark_artifacts::{
    FetchOptions, FiqaOptions, FiqaSourceSpec, HfDownloader, HfPublisher, HttpFiqaSource,
    OpenAiEmbedder, PublishOptions, ToolError, ToolResult, fetch_artifact, prepare_fiqa,
    publish_artifact,
};
use bifrost_benchmark_fixture::{Artifact, Catalog, Publication, VerifiedFixture};

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";
const ARTIFACT_ROOT_ENV: &str = "BIFROST_BENCH_ARTIFACT_ROOT";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> ToolResult<()> {
    let mut args = Arguments::new(env::args().skip(1));
    let command = args.next().unwrap_or_else(|| "help".to_owned());
    match command.as_str() {
        "fetch" => run_fetch(&mut args),
        "verify" => run_verify(&mut args),
        "prepare-fiqa" => run_prepare_fiqa(&mut args),
        "publish" => run_publish(&mut args),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => Err(Box::new(ToolError::new(format!(
            "unknown command {command:?}; run with --help"
        )))),
    }
}

fn run_fetch(args: &mut Arguments) -> ToolResult<()> {
    let data_root = data_root()?;
    let mut catalog_path = default_catalog();
    let mut artifact_id = None;
    let mut destination_root = data_root.join("fixtures");
    let mut cache_directory = data_root.join("huggingface-cache");
    let mut endpoint = DEFAULT_HF_ENDPOINT.to_owned();
    let mut token_environment = "HF_TOKEN".to_owned();
    let mut offline = false;
    let mut repair = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--catalog" => catalog_path = args.path_value("--catalog")?,
            "--artifact" => artifact_id = Some(args.value("--artifact")?),
            "--destination-root" => destination_root = args.path_value("--destination-root")?,
            "--cache-dir" => cache_directory = args.path_value("--cache-dir")?,
            "--endpoint" => endpoint = args.value("--endpoint")?,
            "--token-env" => token_environment = args.value("--token-env")?,
            "--offline" => offline = true,
            "--repair" => repair = true,
            "--help" | "-h" => {
                print_fetch_help();
                return Ok(());
            }
            _ => return unknown_option("fetch", &argument),
        }
    }
    if offline && repair {
        return Err(Box::new(ToolError::new(
            "--offline and --repair cannot be used together",
        )));
    }
    reject_checkout_path(&destination_root, "fetch destination root")?;
    reject_checkout_path(&cache_directory, "Hugging Face cache directory")?;
    let artifact_id = artifact_id.ok_or_else(|| ToolError::new("fetch requires --artifact ID"))?;
    let catalog = Catalog::load(&catalog_path)?;
    catalog.validate()?;
    let artifact = catalog
        .artifact(&artifact_id)
        .ok_or_else(|| ToolError::new(format!("artifact {artifact_id:?} is not in the catalog")))?;
    let repository = published_repository(artifact)?;
    let token = env::var(&token_environment).ok();
    let downloader = HfDownloader::new(repository, &endpoint, cache_directory, token)?;
    let path = fetch_artifact(
        &catalog,
        &FetchOptions {
            artifact_id,
            destination_root,
            offline,
            repair,
        },
        &downloader,
    )?;
    println!("{}", path.display());
    Ok(())
}

fn run_verify(args: &mut Arguments) -> ToolResult<()> {
    let mut fixture = None;
    let mut catalog_path = default_catalog();
    let mut artifact_id = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--fixture" => fixture = Some(args.path_value("--fixture")?),
            "--catalog" => catalog_path = args.path_value("--catalog")?,
            "--artifact" => artifact_id = Some(args.value("--artifact")?),
            "--help" | "-h" => {
                print_verify_help();
                return Ok(());
            }
            _ => return unknown_option("verify", &argument),
        }
    }
    let fixture = fixture.ok_or_else(|| ToolError::new("verify requires --fixture PATH"))?;
    if let Some(artifact_id) = artifact_id {
        let catalog = Catalog::load(catalog_path)?;
        catalog.validate()?;
        let artifact = catalog.artifact(&artifact_id).ok_or_else(|| {
            ToolError::new(format!("artifact {artifact_id:?} is not in the catalog"))
        })?;
        VerifiedFixture::load(&fixture, Some(artifact))?;
    } else {
        VerifiedFixture::load(&fixture, None)?;
    }
    println!("{}", fs::canonicalize(fixture)?.display());
    Ok(())
}

fn run_prepare_fiqa(args: &mut Arguments) -> ToolResult<()> {
    let mut catalog_path = default_catalog();
    let mut artifact_id = None;
    let mut workspace = None;
    let mut output = None;
    let mut batch_size = 128;
    let mut repair_batches = false;
    let mut generator_revision = env::var("BIFROST_GENERATOR_REVISION").ok();
    let mut openai_endpoint = DEFAULT_OPENAI_ENDPOINT.to_owned();
    let mut api_key_environment = "OPENAI_API_KEY".to_owned();
    let mut source = FiqaSourceSpec::default();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--catalog" => catalog_path = args.path_value("--catalog")?,
            "--artifact" => artifact_id = Some(args.value("--artifact")?),
            "--workspace" => workspace = Some(args.path_value("--workspace")?),
            "--output" => output = Some(args.path_value("--output")?),
            "--batch-size" => batch_size = args.usize_value("--batch-size")?,
            "--repair-batches" => repair_batches = true,
            "--generator-revision" => {
                generator_revision = Some(args.value("--generator-revision")?)
            }
            "--openai-endpoint" => openai_endpoint = args.value("--openai-endpoint")?,
            "--api-key-env" => api_key_environment = args.value("--api-key-env")?,
            "--source-url" => source.url = args.value("--source-url")?,
            "--source-revision" => source.revision = args.value("--source-revision")?,
            "--source-sha256" => source.sha256 = args.value("--source-sha256")?,
            "--help" | "-h" => {
                print_prepare_help();
                return Ok(());
            }
            _ => return unknown_option("prepare-fiqa", &argument),
        }
    }
    let artifact_id =
        artifact_id.ok_or_else(|| ToolError::new("prepare-fiqa requires --artifact ID"))?;
    let generator_revision = generator_revision.ok_or_else(|| {
        ToolError::new(
            "prepare-fiqa requires --generator-revision SHA or BIFROST_GENERATOR_REVISION",
        )
    })?;
    let catalog = Catalog::load(catalog_path)?;
    catalog.validate()?;
    let artifact = catalog
        .artifact(&artifact_id)
        .ok_or_else(|| ToolError::new(format!("artifact {artifact_id:?} is not in the catalog")))?
        .clone();
    let default_root = if workspace.is_none() || output.is_none() {
        Some(data_root()?)
    } else {
        None
    };
    let workspace = workspace.unwrap_or_else(|| {
        default_root
            .as_ref()
            .expect("missing default artifact root")
            .join("workspaces")
            .join(&artifact_id)
    });
    let output = output.unwrap_or_else(|| {
        default_root
            .as_ref()
            .expect("missing default artifact root")
            .join("prepared")
            .join(&artifact_id)
    });
    reject_checkout_path(&workspace, "prepare workspace")?;
    reject_checkout_path(&output, "prepared fixture output")?;
    let source_client = HttpFiqaSource::new()?;
    let embedder = OpenAiEmbedder::new(openai_endpoint, api_key_environment)?;
    let path = prepare_fiqa(
        &FiqaOptions {
            artifact,
            workspace,
            output,
            batch_size,
            repair_batches,
            generator_revision,
            source,
        },
        &source_client,
        &embedder,
    )?;
    println!("{}", path.display());
    Ok(())
}

fn run_publish(args: &mut Arguments) -> ToolResult<()> {
    let data_root = data_root()?;
    let mut catalog_path = default_catalog();
    let mut artifact_id = None;
    let mut fixture = None;
    let mut branch = "main".to_owned();
    let mut endpoint = DEFAULT_HF_ENDPOINT.to_owned();
    let mut cache_directory = data_root.join("huggingface-cache");
    let mut token_environment = "HF_TOKEN".to_owned();
    let mut workspace = data_root.join("publish");
    let mut card_template = Path::new(env!("CARGO_MANIFEST_DIR")).join("dataset-card-template.md");
    let mut allow_external_publication = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--catalog" => catalog_path = args.path_value("--catalog")?,
            "--artifact" => artifact_id = Some(args.value("--artifact")?),
            "--fixture" => fixture = Some(args.path_value("--fixture")?),
            "--branch" => branch = args.value("--branch")?,
            "--endpoint" => endpoint = args.value("--endpoint")?,
            "--cache-dir" => cache_directory = args.path_value("--cache-dir")?,
            "--token-env" => token_environment = args.value("--token-env")?,
            "--workspace" => workspace = args.path_value("--workspace")?,
            "--card-template" => card_template = args.path_value("--card-template")?,
            "--allow-external-publication" => allow_external_publication = true,
            "--help" | "-h" => {
                print_publish_help();
                return Ok(());
            }
            _ => return unknown_option("publish", &argument),
        }
    }
    if !allow_external_publication {
        return Err(Box::new(ToolError::new(
            "publish requires --allow-external-publication after human review",
        )));
    }
    reject_checkout_path(&cache_directory, "Hugging Face cache directory")?;
    reject_checkout_path(&workspace, "publish workspace")?;
    let artifact_id =
        artifact_id.ok_or_else(|| ToolError::new("publish requires --artifact ID"))?;
    let fixture = fixture.ok_or_else(|| ToolError::new("publish requires --fixture PATH"))?;
    let mut catalog = Catalog::load(&catalog_path)?;
    catalog.validate()?;
    let artifact = catalog
        .artifact(&artifact_id)
        .ok_or_else(|| ToolError::new(format!("artifact {artifact_id:?} is not in the catalog")))?;
    let repository = publication_repository(artifact).to_owned();
    let token = env::var(&token_environment).map_err(|_| {
        ToolError::new(format!(
            "{token_environment} must be set to publish to Hugging Face"
        ))
    })?;
    let publisher = HfPublisher::new(&repository, &endpoint, cache_directory, token)?;
    let receipt = publish_artifact(
        &mut catalog,
        &PublishOptions {
            artifact_id,
            fixture,
            branch,
            card_template,
            workspace,
            update_catalog: Some(catalog_path),
            allow_external_publication,
        },
        &publisher,
    )?;
    eprintln!(
        "{} artifact commit {} at {}/{}; root metadata commit {}",
        if receipt.created_commit {
            "Published"
        } else {
            "Verified existing"
        },
        receipt.revision,
        receipt.repository,
        receipt.prefix,
        receipt.metadata_revision,
    );
    println!("{}", receipt.revision);
    Ok(())
}

struct Arguments {
    values: VecDeque<String>,
}

impl Arguments {
    fn new(values: impl Iterator<Item = String>) -> Self {
        Self {
            values: values.collect(),
        }
    }

    fn next(&mut self) -> Option<String> {
        self.values.pop_front()
    }

    fn value(&mut self, option: &str) -> ToolResult<String> {
        self.next()
            .ok_or_else(|| Box::new(ToolError::new(format!("{option} requires a value"))) as _)
    }

    fn path_value(&mut self, option: &str) -> ToolResult<PathBuf> {
        Ok(PathBuf::from(self.value(option)?))
    }

    fn usize_value(&mut self, option: &str) -> ToolResult<usize> {
        let value = self.value(option)?;
        value.parse().map_err(|_| {
            Box::new(ToolError::new(format!(
                "{option} must be a positive integer"
            ))) as _
        })
    }
}

fn default_catalog() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog.json")
}

fn data_root() -> ToolResult<PathBuf> {
    resolve_data_root(env::var_os(ARTIFACT_ROOT_ENV), dirs::cache_dir())
}

fn resolve_data_root(
    configured_root: Option<OsString>,
    system_cache_root: Option<PathBuf>,
) -> ToolResult<PathBuf> {
    let root = match configured_root {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        Some(_) => {
            return Err(Box::new(ToolError::new(format!(
                "{ARTIFACT_ROOT_ENV} must not be empty"
            ))));
        }
        None => system_cache_root
            .unwrap_or_else(env::temp_dir)
            .join("bifrost")
            .join("benchmark-artifacts"),
    };
    if !root.is_absolute() {
        return Err(Box::new(ToolError::new(format!(
            "{ARTIFACT_ROOT_ENV} and the system cache root must resolve to an absolute path"
        ))));
    }
    Ok(root)
}

fn checkout_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("artifacts crate must live under <checkout>/benchmarks/artifacts")
        .to_owned()
}

fn reject_checkout_path(path: &Path, label: &str) -> ToolResult<()> {
    let candidate = comparable_path(path)?;
    let checkout = comparable_path(&checkout_root())?;
    if candidate == checkout || candidate.starts_with(&checkout) {
        return Err(Box::new(ToolError::new(format!(
            "{label} ({}) must be outside the repository checkout ({})",
            candidate.display(),
            checkout.display()
        ))));
    }
    Ok(())
}

fn comparable_path(path: &Path) -> ToolResult<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        env::current_dir()?.join(path)
    };
    let mut missing = Vec::new();
    let mut existing = absolute.as_path();
    while !existing.exists() {
        let segment = existing.file_name().ok_or_else(|| {
            Box::new(ToolError::new(format!(
                "{} has no existing ancestor",
                path.display()
            ))) as Box<dyn std::error::Error + Send + Sync>
        })?;
        missing.push(segment.to_owned());
        existing = existing.parent().ok_or_else(|| {
            Box::new(ToolError::new(format!(
                "{} has no existing ancestor",
                path.display()
            ))) as Box<dyn std::error::Error + Send + Sync>
        })?;
    }
    let mut resolved = fs::canonicalize(existing)?;
    for segment in missing.iter().rev() {
        resolved.push(segment);
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
            Component::Normal(segment) => normalized.push(segment),
        }
    }
    normalized
}

fn published_repository(artifact: &Artifact) -> ToolResult<&str> {
    match &artifact.publication {
        Publication::Published { repository, .. } => Ok(repository),
        Publication::Planned { .. } => Err(Box::new(ToolError::new(format!(
            "artifact {:?} is planned and cannot be fetched",
            artifact.id
        )))),
    }
}

fn publication_repository(artifact: &Artifact) -> &str {
    match &artifact.publication {
        Publication::Planned { repository, .. } | Publication::Published { repository, .. } => {
            repository
        }
    }
}

fn unknown_option<T>(command: &str, option: &str) -> ToolResult<T> {
    Err(Box::new(ToolError::new(format!(
        "unknown {command} option {option:?}"
    ))))
}

fn print_help() {
    println!(
        "bifrost-benchmark-artifacts <COMMAND> [OPTIONS]\n\n\
         Commands:\n\
           fetch         Fetch and verify one published fixture\n\
           verify        Verify a local v2 fixture\n\
           prepare-fiqa  Prepare a resumable FiQA embedding fixture\n\
           publish       Explicitly publish a validated fixture\n\n\
         Generated data defaults to the OS cache directory. Override it with\n\
         BIFROST_BENCH_ARTIFACT_ROOT; generated paths inside this checkout are rejected.\n\n\
         Run `<COMMAND> --help` for command options."
    );
}

fn print_fetch_help() {
    println!(
        "fetch --artifact ID [OPTIONS]\n\n\
         --catalog PATH           Catalog JSON (default: checked-in catalog.json)\n\
         --destination-root PATH  Root for verified immutable fixtures\n\
         --cache-dir PATH         Explicit hf-hub cache directory\n\
         --endpoint URL           Explicit Hub endpoint\n\
         --token-env NAME         Token environment variable (default: HF_TOKEN)\n\
         --offline                Strict cache-only lookup; no network download path\n\
         --repair                 Force-refetch corrupt hf-hub cache entries\n\n\
         On success, the final stdout line is the resolved absolute fixture directory.\n\
         Progress and errors are written to stderr."
    );
}

fn print_verify_help() {
    println!(
        "verify --fixture PATH [--artifact ID] [--catalog PATH]\n\n\
         With --artifact, catalog identity and publication metadata are also checked."
    );
}

fn print_prepare_help() {
    println!(
        "prepare-fiqa --artifact ID --generator-revision SHA [OPTIONS]\n\n\
         --catalog PATH          Catalog JSON\n\
         --workspace PATH        Source and resumable batch cache directory\n\
         --output PATH           Final fixture directory, separate from workspace\n\
         --batch-size N          OpenAI inputs per request (default: 128)\n\
         --repair-batches       Explicitly replace malformed/mismatched cached batches\n\
         --openai-endpoint URL   Embeddings endpoint (default: OpenAI)\n\
         --api-key-env NAME      API-key environment variable (default: OPENAI_API_KEY)\n\
         --source-url URL        Override FiQA archive URL for instrumentation\n\
         --source-revision TEXT  Override recorded source revision\n\
         --source-sha256 HEX     Override expected source archive SHA-256\n\n\
         SHA must be the 40-character lowercase commit for the clean source used to build/run.\n\
         This is a maintainer-supplied provenance assertion; the CLI validates its syntax but\n\
         cannot reliably infer Git state. A complete valid output returns before source download,\n\
         credential lookup, or API calls."
    );
}

fn print_publish_help() {
    println!(
        "publish --artifact ID --fixture PATH --allow-external-publication [OPTIONS]\n\n\
         --catalog PATH          Catalog JSON\n\
         --branch NAME           Destination branch (default: main)\n\
         --endpoint URL          Explicit Hub endpoint\n\
         --cache-dir PATH        Explicit hf-hub cache directory\n\
         --token-env NAME        Required token environment variable\n\
         --workspace PATH        Generated-card workspace\n\
         --card-template PATH    Dataset card template\n\n\
         After remote verification, the selected catalog is updated atomically.\n\
         Publication never runs from tests or setup and requires the explicit review flag."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_root_defaults_to_the_external_system_cache() {
        let temp = tempfile::tempdir().unwrap();
        let root = resolve_data_root(None, Some(temp.path().to_owned())).unwrap();

        assert_eq!(root, temp.path().join("bifrost/benchmark-artifacts"));
        assert!(reject_checkout_path(&root, "test root").is_ok());
    }

    #[test]
    fn explicit_data_root_must_be_absolute() {
        let failure = resolve_data_root(Some(OsString::from("relative")), None).unwrap_err();
        assert!(failure.to_string().contains("absolute path"));
    }

    #[test]
    fn generated_paths_inside_the_checkout_are_rejected() {
        let checkout = checkout_root();
        assert!(reject_checkout_path(&checkout, "test root").is_err());
        assert!(reject_checkout_path(&checkout.join("generated/cache"), "test root").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn checkout_guard_resolves_symlink_ancestors() {
        let temp = tempfile::tempdir().unwrap();
        let alias = temp.path().join("checkout-alias");
        std::os::unix::fs::symlink(checkout_root(), &alias).unwrap();

        assert!(reject_checkout_path(&alias.join("generated"), "test root").is_err());
    }
}
