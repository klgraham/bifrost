# Bifrost development environment

The environment uses Rust 1.89.0 from `rust-toolchain.toml` and the checked-in
Cargo lockfiles. The library's minimum supported Rust version is 1.89.
The competitor crate includes USearch's native C/C++ backend. On Linux,
`scripts/cargo.sh` selects GCC for compilation and linking. CI does not build or
run benchmark targets or benchmark support crates.

## Local setup

On macOS, install Xcode Command Line Tools and Python 3 first. On Ubuntu or
Debian, setup installs missing compiler tools with `apt-get`, using `sudo` when
needed. Setup installs Rust through rustup when missing. Downloads require
internet access and write to the user's Cargo and rustup caches.

Run from the repository root:

```bash
bash scripts/setup.sh
bash scripts/verify-environment.sh
```

Setup fetches every locked dependency graph and builds the library tests, the
distance benchmark, the competitor benchmark, and the Rust artifact tool.
Rerunning setup refreshes dependencies for the current checkout. Verification
runs formatting, Clippy, tests, and both synthetic benchmarks offline. No API
key, dataset download, or embedding API call is needed.

In Codex desktop's local environment settings for Bifrost, set the setup script
to `bash scripts/setup.sh`. Add actions with these commands:

| Action | Command |
| --- | --- |
| Verify environment | `bash scripts/verify-environment.sh` |
| Distance benchmark | `bash scripts/cargo.sh bench --locked --bench distance` |
| Competitor benchmark | `bash scripts/cargo.sh run --locked --release --manifest-path benchmarks/competitors/Cargo.toml` |
| Catalog artifact benchmark | `bash scripts/run-benchmark-artifact.sh ARTIFACT_ID` |

Codex can save local environment settings under `.codex` for sharing through Git.
See [OpenAI's local environment documentation](https://developers.openai.com/codex/app/local-environments).

Use `scripts/cargo.sh` for manual Cargo commands to preserve the Linux compiler
settings and find rustup's Cargo in a fresh shell. Setup's shell exports do not
need to persist between sessions.

## Codex cloud

The cloud environment is an account setting. Repository scripts do not register
it automatically. Make these files available on the GitHub branch before using
the commands below. Cloud cache preparation may use the default branch, so put
the setup files there before enabling this shared environment.

In [Codex environment settings](https://chatgpt.com/codex/settings/environments),
create or edit the environment for `klgraham/bifrost`:

| Field | Value |
| --- | --- |
| Name | `Bifrost` |
| Container image | `universal` |
| Setup mode | Manual |
| Setup script | `bash scripts/setup.sh` |
| Maintenance script | `bash scripts/setup.sh` |
| Agent internet access | Off for synthetic and already-prefetched artifact benchmarks |
| Environment variables | Optional `BIFROST_BENCH_ARTIFACT` to prefetch during setup |
| Secrets | None required |

Run the environment's setup test. Then start a cloud task on a branch containing
these files and ask it to run `bash scripts/verify-environment.sh`. Check for
passing tests and output tables for Bifrost, hnsw_rs, and USearch. A local run
does not verify the cloud container.

Setup runs with network access and fills the Cargo caches needed by offline
commands. If `BIFROST_BENCH_ARTIFACT` names a published catalog entry, setup
also fetches and verifies that fixture before agent internet access is disabled.
Maintenance repeats setup after the requested branch is checked out, covering
changes to every lockfile. Reset the environment cache if its installed state
becomes incompatible. These settings follow
[OpenAI's cloud environment documentation](https://learn.chatgpt.com/docs/environments/cloud-environment).

## Benchmark runs

The verification script uses 500 synthetic vectors, 64 dimensions, 10 queries,
and one timed repetition. Its output proves execution, not a stable performance
baseline. Run larger comparisons on a quiet machine and record the hardware,
compiler, parameters, and output. Cloud CPU allocation and contention can vary.

The distance benchmark defaults to one million iterations. Override with
`BIFROST_BENCH_ITERATIONS`. The competitor benchmark defaults to 10,000 vectors,
384 dimensions, 100 queries, and five repetitions. For example:

```bash
HNSW_BENCH_VECTORS=1000 HNSW_BENCH_QUERIES=20 \
  bash scripts/cargo.sh run --offline --locked --release \
  --manifest-path benchmarks/competitors/Cargo.toml
```

Other controls include `HNSW_BENCH_DIMENSIONS`, `HNSW_BENCH_REPETITIONS`,
`HNSW_BENCH_K`, `HNSW_BENCH_M`, `HNSW_BENCH_EF_CONSTRUCTION`,
`HNSW_BENCH_EF_SEARCH`, `HNSW_BENCH_EF_SEARCHES` as a comma-separated list,
and `HNSW_BENCH_SEED`.

For an existing local fixture, set `HNSW_BENCH_FIXTURE` to its directory and
set `HNSW_BENCH_OUTPUT_DIR` to a separate writable directory. The benchmark
refuses to write inside the fixture. For example:

```bash
HNSW_BENCH_FIXTURE=/path/to/fixture \
HNSW_BENCH_OUTPUT_DIR=target/benchmark-runs/local-fiqa \
  bash scripts/cargo.sh run --offline --locked --release \
  --manifest-path benchmarks/competitors/Cargo.toml
```

## Versioned benchmark artifacts

`benchmarks/artifacts/catalog.json` declares BEIR FiQA-2018 with
`text-embedding-3-small` at 384 and 1,536 dimensions. The entries remain
`planned` until their redistribution and cost review is complete and their
published records contain full Hugging Face commit and manifest hashes. A
planned entry fails with an actionable error instead of falling back to
embedding generation.

Fetch one published entry while network access is available:

```bash
bash scripts/prefetch-benchmark-artifact.sh \
  fiqa-text-embedding-3-small-384
```

The command prints the absolute verified fixture directory. Repeat the same
resolution with both Cargo and the resolver in strict offline mode:

```bash
BIFROST_BENCH_OFFLINE=1 \
  bash scripts/prefetch-benchmark-artifact.sh \
  fiqa-text-embedding-3-small-384
```

Run all three implementations against those exact bytes and save the Bifrost
index outside the read-only fixture cache:

```bash
BIFROST_BENCH_OFFLINE=1 \
  bash scripts/run-benchmark-artifact.sh \
  fiqa-text-embedding-3-small-384 \
  target/benchmark-runs/fiqa-384
```

The benchmark reports the artifact ID, repository, immutable commit, and
manifest SHA-256 before its results. Fetch and preparation time are outside
the benchmark process. The wrapper clears ambient vector, query, and dimension
limits so catalog runs always consume the complete published fixture. The
competitor crate has no OpenAI or Hugging Face dependency and never generates a
missing fixture.

Set `BIFROST_BENCH_ARTIFACT` when running `scripts/setup.sh` to prefetch during
local or cloud setup. These optional controls apply to the prefetch scripts:

| Variable | Purpose |
| --- | --- |
| `BIFROST_BENCH_ARTIFACT_ROOT` | Parent for all artifact-tool data; defaults to the OS user cache |
| `BIFROST_BENCH_FIXTURE_ROOT` | Complete verified fixture cache |
| `BIFROST_HF_CACHE_DIR` | Explicit `hf-hub` transport cache |
| `BIFROST_HF_ENDPOINT` | Hugging Face endpoint override |
| `BIFROST_HF_TOKEN_ENV` | Name of an optional read-token variable |
| `BIFROST_BENCH_OFFLINE=1` | Prohibit network resolution |
| `BIFROST_BENCH_FETCH_REPAIR=1` | Explicitly replace corrupt cached content online |

Do not combine offline and repair modes. A public repository needs no read
token; a restricted repository uses the variable named by
`BIFROST_HF_TOKEN_ENV`, which defaults to `HF_TOKEN`. Path overrides should be
absolute and must resolve outside the repository checkout. This keeps source,
transport caches, resumable batches, and completed fixtures out of Git.
This restriction applies to artifact-tool data. Benchmark indexes and reports
are disposable build outputs: the wrapper defaults them to the ignored
`target/benchmark-runs/` directory, or you can pass a separate external output
directory as its second argument.

## Prepare and publish

Preparation and publication are separate maintainer operations. They are
never invoked by setup, tests, or the competitor benchmark. The Rust tool
stores the verified FiQA source and resumable embedding batches in a workspace
outside the final fixture. Inspect its commands with:

```bash
bash scripts/artifact-cargo.sh run --locked --release -- --help
```

After approving OpenAI cost, generate one dimension with an explicit source,
artifact, and generator revision:

```bash
bash scripts/artifact-cargo.sh run --locked --release -- \
  prepare-fiqa \
  --artifact fiqa-text-embedding-3-small-384 \
  --generator-revision GIT_COMMIT \
  --workspace /path/to/artifact-workspace/fiqa-384 \
  --output /path/to/prepared/fiqa-384
```

Use the full lowercase 40-character commit that contains the generator source,
and run the generation from that clean checkout. The tool validates the SHA
format and records it throughout the resumable batch cache and final manifest;
the maintainer remains responsible for asserting that it identifies the binary
being run.

`OPENAI_API_KEY` is read only by `prepare-fiqa`. Valid completed batches are
reused. A malformed or mismatched cached batch fails unless the maintainer
adds `--repair-batches`, which explicitly permits a replacement request.

After completing the dataset card, licensing, visibility, file allowlist, and
publication review, create or verify the content-addressed Hugging Face version:

```bash
bash scripts/artifact-cargo.sh run --locked --release -- \
  publish \
  --artifact fiqa-text-embedding-3-small-384 \
  --fixture /path/to/prepared/fiqa-384 \
  --allow-external-publication
```

Publication also requires `HF_TOKEN` and an existing destination repository.
The explicit confirmation flag prevents tests or an accidental command from
publishing. After verifying both remote commits, the command atomically updates
the selected local catalog with the immutable artifact commit. A retry verifies
matching remote content rather than overwriting a content-addressed version. See
[Benchmark artifact design](benchmark-artifact-design.md) for schema,
publication layout, cache, and provenance decisions.
