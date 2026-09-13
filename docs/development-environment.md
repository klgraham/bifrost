# Bifrost development environment

The environment uses Rust 1.96.1 from `rust-toolchain.toml` and the two checked-in
Cargo lockfiles. The library's minimum supported Rust version remains 1.87.
The competitor crate includes USearch's native C/C++ backend. On Linux,
`scripts/cargo.sh` selects GCC for compilation and linking, matching CI.

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

Setup fetches both locked dependency graphs and builds the library tests, the
distance benchmark, the competitor benchmark, and the optional FiQA preparation
binary. Rerunning setup refreshes dependencies for the current checkout.
Verification runs formatting, Clippy, tests, and both synthetic benchmarks
offline. No API key, data download, or embedding API call is needed.

In Codex desktop's local environment settings for Bifrost, set the setup script
to `bash scripts/setup.sh`. Add actions with these commands:

| Action | Command |
| --- | --- |
| Verify environment | `bash scripts/verify-environment.sh` |
| Distance benchmark | `bash scripts/cargo.sh bench --locked --bench distance` |
| Competitor benchmark | `bash scripts/cargo.sh run --locked --release --manifest-path benchmarks/competitors/Cargo.toml` |

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
| Agent internet access | Off for the synthetic benchmarks and tests |
| Environment variables | None required |
| Secrets | None required |

Run the environment's setup test. Then start a cloud task on a branch containing
these files and ask it to run `bash scripts/verify-environment.sh`. Check for
passing tests and output tables for Bifrost, hnsw_rs, and USearch. A local run
does not verify the cloud container.

Setup runs with network access and fills the caches needed by offline commands.
Maintenance repeats setup after the requested branch is checked out, covering
changes to either lockfile. Reset the environment cache if its installed state
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

For existing embeddings, set `HNSW_BENCH_FIXTURE` to a fixture directory. The
benchmark writes a Bifrost snapshot under that directory's `indexes/` folder.
Keep generated datasets outside the repository.

The optional preparation binary downloads BEIR FiQA and normally calls OpenAI
to generate paid embeddings. Setup only compiles it. Inspect its options with:

```bash
bash scripts/cargo.sh run --offline --locked --release \
  --manifest-path benchmarks/competitors/Cargo.toml --features fiqa-prep \
  --bin prepare-fiqa -- --help
```

Run preparation separately when needed. `--download-only` downloads the corpus
without embeddings. Embedding generation needs `OPENAI_API_KEY` and network
access. Cloud secrets are available only during setup, so they are not available
to an ordinary agent-phase preparation command. Reusing an existing fixture
allows offline benchmark execution.

