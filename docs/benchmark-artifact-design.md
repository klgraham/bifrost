# Benchmark artifact design

Benchmark runs need stable input bytes. Embedding generation and publication have
different credentials, failure modes, and toolchain requirements, so Bifrost keeps
them outside the benchmark executable.

## Package boundaries

`benchmarks/fixture` owns the catalog and fixture schemas. It parses untrusted JSON,
checks paths and hashes, and returns verified vectors and IDs. The competitor
benchmark depends on this crate and has no Hugging Face or OpenAI dependency.

`benchmarks/artifacts` owns network and preparation operations. It fetches and
publishes through `hf-hub` 1.0.0 and prepares FiQA embeddings through the OpenAI
embeddings endpoint. The selected `hf-hub` dependency graph has a Rust 1.89
MSRV. Bifrost and both benchmark support crates use that MSRV.

The normal data flow is:

```text
catalog entry
  -> immutable Hugging Face commit
  -> per-process staging directory
  -> manifest and file verification
  -> atomic cache directory
  -> read-only fixture
  -> competitor benchmark
  -> separate writable run directory
```

The preparation flow writes source data and resumable API batches outside the final
fixture directory. The publisher accepts only files named by a verified fixture
manifest. API responses, source archives, cache files, and credentials cannot enter
the upload plan.

## Artifact matrix

The first catalog covers BEIR FiQA-2018 with `text-embedding-3-small` at 384 and
1,536 dimensions. Both variants use explicit OpenAI dimension requests. Neither
variant is derived by truncating another vector set.

The two dimensions match the benchmark configurations already recorded in this
repository: 384 is the default synthetic width, and 1,536 is the existing FiQA
preparation default and published evaluation width. Synthetic smoke runs remain
local and deterministic. They are not remote catalog entries.

Both FiQA entries remain `planned` until a maintainer approves the paid API calls,
confirms public redistribution terms, and publishes the files. Publication changes
an entry to `published` only after an immutable commit can be fetched and verified.

## Publication layout

The planned Hugging Face dataset repository is
`klgraham/bifrost-benchmark-fixtures`. One repository keeps the dataset card and
artifact catalog together. Each variant uses a separate prefix, and the final
published path ends with the fixture manifest SHA-256. The publisher first creates
or verifies that content-addressed artifact commit. It then writes the updated
catalog and matrix-level dataset card at the repository root in a second metadata
commit. The local catalog pins the full 40-character artifact commit returned by
Hugging Face, never a branch or `main`, so the metadata commit cannot make the
artifact identity self-referential.

A repository per dimension would isolate releases, but it would duplicate the
dataset card and make a benchmark matrix span several repositories. A single
moving repository path without content-addressed prefixes would be smaller, but it
would allow a retry to replace an existing artifact. The chosen layout gives each
variant an immutable identity while retaining one publication namespace.

## Cache ownership

Hugging Face owns its transport cache. Bifrost treats those files as untrusted
inputs because an existence-only cache hit does not prove content integrity.

The artifact resolver owns a second cache of complete fixtures. A process locks one
artifact identity, downloads into a unique staging directory on the same
filesystem, verifies every byte, writes resolution metadata, and renames the
directory into place. It removes write permissions from the completed fixture tree.
Offline mode reads only this completed cache. A corrupt cache entry fails until an
online `--repair` fetch safely replaces the read-only tree.

The benchmark never writes into either cache. `HNSW_BENCH_OUTPUT_DIR` must resolve
outside the fixture directory before the benchmark saves an index.

## Provenance and licensing

Each manifest records the source URL, source revision and SHA-256, dataset
license, split/subset selection, text preprocessing rule, corpus and query input
hashes, requested and returned model identifiers, dimensions, normalization,
derivation, generator revision, file representation, row counts, and per-file
SHA-256 values.

The BEIR Hugging Face cards label FiQA data and qrels as CC BY-SA 4.0. BEIR also
states that downstream users remain responsible for the source dataset rights.
The dataset card therefore carries attribution and a limitations section. The
publisher requires a maintainer to complete the redistribution review before the
catalog can move from `planned` to `published`.
