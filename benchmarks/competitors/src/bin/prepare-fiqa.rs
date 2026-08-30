use std::{
    collections::HashSet,
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use reqwest::{StatusCode, blocking::Client};
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

const FIQA_URL: &str =
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip";
const FIQA_MD5: &str = "17918ed23cd04fb15047f73e6c3bd9d9";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIMENSIONS: usize = 1536;
const DEFAULT_BATCH_SIZE: usize = 128;
const MAX_BATCH_INPUTS: usize = 2048;
const API_URL: &str = "https://api.openai.com/v1/embeddings";

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct Options {
    output: PathBuf,
    model: String,
    dimensions: usize,
    batch_size: usize,
    max_corpus: Option<usize>,
    max_queries: Option<usize>,
    download_only: bool,
}

#[derive(Deserialize)]
struct BeirRecord {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    dimensions: usize,
    encoding_format: &'static str,
}

#[derive(Debug, Deserialize, Serialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize, Serialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct EmbeddingUsage {
    prompt_tokens: u64,
    total_tokens: u64,
}

#[derive(Deserialize, Serialize)]
struct CachedBatch {
    requested_model: String,
    dimensions: usize,
    ids: Vec<String>,
    response: EmbeddingResponse,
}

fn main() -> Result<()> {
    let options = parse_options()?;
    fs::create_dir_all(&options.output)?;

    let archive_path = options.output.join("fiqa.zip");
    download_archive(&archive_path)?;
    verify_archive(&archive_path)?;

    let source_dir = options.output.join("source");
    fs::create_dir_all(&source_dir)?;
    let corpus_path = source_dir.join("corpus.jsonl");
    let queries_path = source_dir.join("queries.jsonl");
    let qrels_path = source_dir.join("qrels-test.tsv");
    extract_entry(&archive_path, "fiqa/corpus.jsonl", &corpus_path)?;
    extract_entry(&archive_path, "fiqa/queries.jsonl", &queries_path)?;
    extract_entry(&archive_path, "fiqa/qrels/test.tsv", &qrels_path)?;

    let corpus = load_corpus(&corpus_path, options.max_corpus)?;
    let test_query_ids = load_qrel_query_ids(&qrels_path)?;
    let queries = load_queries(&queries_path, &test_query_ids, options.max_queries)?;
    if options.download_only {
        println!(
            "Downloaded and verified BEIR FiQA-2018 at {}",
            options.output.display()
        );
        println!(
            "Validated {} corpus documents and {} test queries; no OpenAI API request was made.",
            corpus.len(),
            queries.len()
        );
        println!("Re-run without --download-only after setting OPENAI_API_KEY.");
        return Ok(());
    }

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY must be set to generate embeddings")?;
    let client = Client::builder()
        .timeout(Duration::from_secs(180))
        .user_agent("bifrost-fiqa-benchmark/1")
        .build()?;

    println!(
        "Preparing {} corpus embeddings and {} query embeddings with {}...",
        corpus.len(),
        queries.len(),
        options.model
    );

    let cache_root = options.output.join("cache");
    let corpus_tokens = embed_and_pack(
        &client,
        &api_key,
        &options,
        "corpus",
        &corpus,
        &cache_root.join("corpus"),
        &options.output.join("corpus.f32"),
        &options.output.join("corpus-ids.txt"),
    )?;
    let query_tokens = embed_and_pack(
        &client,
        &api_key,
        &options,
        "queries",
        &queries,
        &cache_root.join("queries"),
        &options.output.join("queries.f32"),
        &options.output.join("query-ids.txt"),
    )?;
    fs::copy(&qrels_path, options.output.join("qrels-test.tsv"))?;

    let created_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let manifest = format!(
        "format=hnsw-rs-embedding-fixture-v1\n\
         dataset=BEIR FiQA-2018\n\
         source_url={FIQA_URL}\n\
         source_md5={FIQA_MD5}\n\
         model={}\n\
         dimensions={}\n\
         corpus_count={}\n\
         query_count={}\n\
         input_tokens={}\n\
         created_unix={}\n",
        options.model,
        options.dimensions,
        corpus.len(),
        queries.len(),
        corpus_tokens + query_tokens,
        created_unix,
    );
    atomic_write(&options.output.join("manifest.txt"), manifest.as_bytes())?;

    println!(
        "FiQA embedding fixture ready at {}",
        options.output.display()
    );
    println!("Input tokens: {}", corpus_tokens + query_tokens);
    println!("Run the competitor benchmark with:");
    println!(
        "HNSW_BENCH_FIXTURE={} cargo run --release --manifest-path benchmarks/competitors/Cargo.toml",
        options.output.display()
    );
    Ok(())
}

fn parse_options() -> Result<Options> {
    let mut options = Options {
        output: Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("fiqa-text-embedding-3-small"),
        model: DEFAULT_MODEL.to_owned(),
        dimensions: DEFAULT_DIMENSIONS,
        batch_size: DEFAULT_BATCH_SIZE,
        max_corpus: None,
        max_queries: None,
        download_only: false,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => options.output = PathBuf::from(next_value(&mut args, "--output")?),
            "--model" => options.model = next_value(&mut args, "--model")?,
            "--dimensions" => {
                options.dimensions =
                    parse_usize(&next_value(&mut args, "--dimensions")?, "--dimensions")?
            }
            "--batch-size" => {
                options.batch_size =
                    parse_usize(&next_value(&mut args, "--batch-size")?, "--batch-size")?
            }
            "--max-corpus" => {
                options.max_corpus = Some(parse_usize(
                    &next_value(&mut args, "--max-corpus")?,
                    "--max-corpus",
                )?)
            }
            "--max-queries" => {
                options.max_queries = Some(parse_usize(
                    &next_value(&mut args, "--max-queries")?,
                    "--max-queries",
                )?)
            }
            "--download-only" => options.download_only = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if options.model.trim().is_empty() {
        return Err("--model must not be empty".into());
    }
    if options.dimensions == 0 {
        return Err("--dimensions must be positive".into());
    }
    if !(1..=MAX_BATCH_INPUTS).contains(&options.batch_size) {
        return Err(format!("--batch-size must be between 1 and {MAX_BATCH_INPUTS}").into());
    }
    if options.max_corpus == Some(0) || options.max_queries == Some(0) {
        return Err("record limits must be positive".into());
    }
    Ok(options)
}

fn print_help() {
    println!(
        "prepare-fiqa [OPTIONS]\n\n\
         Downloads BEIR FiQA-2018 and creates a resumable OpenAI embedding fixture.\n\n\
         Options:\n\
           --output PATH       Fixture directory\n\
           --model MODEL       Embedding model (default: {DEFAULT_MODEL})\n\
           --dimensions N      Output dimensions (default: {DEFAULT_DIMENSIONS})\n\
           --batch-size N      Inputs per API request (default: {DEFAULT_BATCH_SIZE})\n\
           --max-corpus N      Prepare only the first N sorted corpus records\n\
           --max-queries N     Prepare only the first N sorted queries\n\
           --download-only     Download and verify FiQA without calling OpenAI\n"
    );
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse()
        .map_err(|_| format!("{flag} must be an integer").into())
}

fn download_archive(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    println!("Downloading {FIQA_URL}...");
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .user_agent("bifrost-fiqa-benchmark/1")
        .build()?;
    let mut response = client.get(FIQA_URL).send()?.error_for_status()?;
    let temporary = path.with_extension("zip.part");
    let mut output = File::create(&temporary)?;
    std::io::copy(&mut response, &mut output)?;
    output.flush()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn verify_archive(path: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let actual = format!("{:x}", md5::compute(bytes));
    if actual != FIQA_MD5 {
        return Err(
            format!("FiQA archive checksum mismatch: expected {FIQA_MD5}, got {actual}").into(),
        );
    }
    Ok(())
}

fn extract_entry(archive_path: &Path, suffix: &str, output_path: &Path) -> Result<()> {
    if output_path.exists() {
        return Ok(());
    }
    let file = File::open(archive_path)?;
    let mut archive = ZipArchive::new(file)?;
    let index = (0..archive.len())
        .find(|index| {
            archive
                .by_index(*index)
                .is_ok_and(|entry| entry.name().ends_with(suffix))
        })
        .ok_or_else(|| format!("{suffix} not found in FiQA archive"))?;
    let mut entry = archive.by_index(index)?;
    let temporary = output_path.with_extension("part");
    let mut output = File::create(&temporary)?;
    std::io::copy(&mut entry, &mut output)?;
    output.flush()?;
    fs::rename(temporary, output_path)?;
    Ok(())
}

fn load_corpus(path: &Path, limit: Option<usize>) -> Result<Vec<(String, String)>> {
    let mut records = load_records(path)?;
    records.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    let raw_count = records.len();
    let mut records = records
        .into_iter()
        .filter_map(|record| {
            let text = match (record.title.trim(), record.text.trim()) {
                ("", "") => return None,
                ("", text) => text.to_owned(),
                (title, "") => title.to_owned(),
                (title, text) => format!("{title}\n\n{text}"),
            };
            Some((record.id, text))
        })
        .collect::<Vec<_>>();
    let skipped = raw_count - records.len();
    if skipped > 0 {
        println!("Skipping {skipped} empty FiQA corpus record(s).");
    }
    if let Some(limit) = limit {
        records.truncate(limit);
    }
    Ok(records)
}

fn load_queries(
    path: &Path,
    included_ids: &HashSet<String>,
    limit: Option<usize>,
) -> Result<Vec<(String, String)>> {
    let mut records = load_records(path)?;
    records.retain(|record| included_ids.contains(&record.id));
    if records.len() != included_ids.len() {
        return Err(format!(
            "{} contains {} of {} test-qrel queries",
            path.display(),
            records.len(),
            included_ids.len()
        )
        .into());
    }
    records.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    if let Some(limit) = limit {
        records.truncate(limit);
    }
    records
        .into_iter()
        .map(|record| {
            let text = record.text.trim();
            if text.is_empty() {
                return Err(format!("query {} is empty", record.id).into());
            }
            Ok((record.id, text.to_owned()))
        })
        .collect()
}

fn load_qrel_query_ids(path: &Path) -> Result<HashSet<String>> {
    let mut ids = HashSet::new();
    for (line_number, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line_number == 0 && line.to_ascii_lowercase().contains("query") {
            continue;
        }
        let mut columns = line.split('\t');
        let query_id = columns.next();
        let corpus_id = columns.next();
        let score = columns.next();
        if query_id.is_none() || corpus_id.is_none() || score.is_none() || columns.next().is_some()
        {
            return Err(format!(
                "{}:{}: expected query-id, corpus-id, score",
                path.display(),
                line_number + 1
            )
            .into());
        }
        ids.insert(query_id.expect("checked query ID").to_owned());
    }
    if ids.is_empty() {
        return Err(format!("{} contains no test queries", path.display()).into());
    }
    Ok(ids)
}

fn load_records(path: &Path) -> Result<Vec<BeirRecord>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(line_number, line)| {
            serde_json::from_str(&line?)
                .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1).into())
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn embed_and_pack(
    client: &Client,
    api_key: &str,
    options: &Options,
    label: &str,
    records: &[(String, String)],
    cache_dir: &Path,
    vectors_path: &Path,
    ids_path: &Path,
) -> Result<u64> {
    fs::create_dir_all(cache_dir)?;
    for (batch_index, batch) in records.chunks(options.batch_size).enumerate() {
        let start = batch_index * options.batch_size;
        let end = start + batch.len();
        let cache_path = cache_dir.join(format!("{start:08}-{end:08}.json"));
        if cache_path.exists() {
            let cached: CachedBatch = serde_json::from_reader(File::open(&cache_path)?)?;
            validate_cached_batch(&cached, options, batch)?;
            continue;
        }

        println!("Embedding {label} {start}..{end} of {}", records.len());
        let input = batch
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>();
        let response = request_embeddings(client, api_key, options, &input)?;
        let cached = CachedBatch {
            requested_model: options.model.clone(),
            dimensions: options.dimensions,
            ids: batch.iter().map(|(id, _)| id.clone()).collect(),
            response,
        };
        validate_cached_batch(&cached, options, batch)?;
        atomic_write(&cache_path, &serde_json::to_vec(&cached)?)?;
    }

    let vectors_temporary = vectors_path.with_extension("f32.part");
    let ids_temporary = ids_path.with_extension("txt.part");
    let mut vectors_output = BufWriter::new(File::create(&vectors_temporary)?);
    let mut ids_output = BufWriter::new(File::create(&ids_temporary)?);
    let mut total_tokens = 0_u64;
    for (batch_index, batch) in records.chunks(options.batch_size).enumerate() {
        let start = batch_index * options.batch_size;
        let end = start + batch.len();
        let cache_path = cache_dir.join(format!("{start:08}-{end:08}.json"));
        let mut cached: CachedBatch = serde_json::from_reader(File::open(cache_path)?)?;
        validate_cached_batch(&cached, options, batch)?;
        cached.response.data.sort_unstable_by_key(|item| item.index);
        for ((id, _), item) in batch.iter().zip(cached.response.data) {
            writeln!(ids_output, "{id}")?;
            for value in item.embedding {
                vectors_output.write_all(&value.to_le_bytes())?;
            }
        }
        total_tokens += cached.response.usage.total_tokens;
    }
    vectors_output.flush()?;
    ids_output.flush()?;
    fs::rename(vectors_temporary, vectors_path)?;
    fs::rename(ids_temporary, ids_path)?;
    Ok(total_tokens)
}

fn request_embeddings(
    client: &Client,
    api_key: &str,
    options: &Options,
    input: &[String],
) -> Result<EmbeddingResponse> {
    let request = EmbeddingRequest {
        model: &options.model,
        input,
        dimensions: options.dimensions,
        encoding_format: "float",
    };
    let mut last_error = String::new();
    for attempt in 0..6 {
        match client
            .post(API_URL)
            .bearer_auth(api_key)
            .json(&request)
            .send()
        {
            Ok(response) if response.status().is_success() => return Ok(response.json()?),
            Ok(response) => {
                let status = response.status();
                let retryable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                last_error = format!(
                    "OpenAI embeddings request failed with {status}: {}",
                    response.text()?
                );
                if !retryable {
                    return Err(last_error.into());
                }
            }
            Err(error) => last_error = format!("OpenAI embeddings request failed: {error}"),
        }
        if attempt < 5 {
            thread::sleep(Duration::from_secs(1_u64 << attempt));
        }
    }
    Err(last_error.into())
}

fn validate_cached_batch(
    cached: &CachedBatch,
    options: &Options,
    records: &[(String, String)],
) -> Result<()> {
    let expected_ids = records.iter().map(|(id, _)| id).collect::<Vec<_>>();
    if cached.requested_model != options.model
        || cached.dimensions != options.dimensions
        || cached.ids.iter().collect::<Vec<_>>() != expected_ids
    {
        return Err("cached embedding batch does not match the requested fixture".into());
    }
    if cached.response.data.len() != records.len() {
        return Err("OpenAI returned an unexpected embedding count".into());
    }
    let mut indexes = cached
        .response
        .data
        .iter()
        .map(|item| item.index)
        .collect::<Vec<_>>();
    indexes.sort_unstable();
    if indexes != (0..records.len()).collect::<Vec<_>>() {
        return Err("OpenAI returned invalid embedding indexes".into());
    }
    for item in &cached.response.data {
        if item.embedding.len() != options.dimensions {
            return Err(format!(
                "embedding {} has {} dimensions; expected {}",
                item.index,
                item.embedding.len(),
                options.dimensions
            )
            .into());
        }
        let norm = item
            .embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 0.01 {
            return Err(format!("embedding {} has invalid L2 norm {norm}", item.index).into());
        }
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("part");
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.flush()?;
    fs::rename(temporary, path)?;
    Ok(())
}
