---
name: verify-bifrost
description: >-
  Verify Bifrost, a compact Rust HNSW library, through its public cargo/Rust
  API (HnswIndex insert, cosine-distance search, .hnsw v3 save/load). Use when
  proving library behavior after index, search, or persistence changes, or
  before treating `cargo test --all-features` as green.
---

# Verify Bifrost

This skill drives **Bifrost 0.2.2**, a library crate (`bifrost`) with no web UI,
CLI, or listening port. The user-facing surface is the public Rust API:
`Config`, `HnswIndex::{new, insert, search, build, save, len}`, `SearchHit`,
`load_file` / `LoadedHnsw`, and `bifrost::vector::{dot, cosine_distance,
cosine_similarity}`. Documented verification in `README.md` is
`cargo test --all-features`. Deletion, updates, concurrent mutation, filtering,
and quantization are **not** supported — do not invent those features.
Competitor benches under `benchmarks/competitors` and the OpenAI FiQA fixture
are out of scope (network, paid, extra native deps).

Repo root is this checkout. Discover it the way the helper does: walk from
the skill/script to `Cargo.toml`, or set `BIFROST_ROOT`. Read
`features/README.md` before driving.

Maintenance: keep this map honest with `/maintain-verification-skill`.

## Launch

There is no long-lived server. Launch means compile the test binaries once
into the crate's `target/` and prepare an isolated scratch directory for any
`.hnsw` files (`std::env::temp_dir()` honors `TMPDIR`).

```bash
export BIFROST_VERIFY_RUN_ID="${BIFROST_VERIFY_RUN_ID:-$(date +%Y%m%dT%H%M%S)-$$}"
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh launch --run-id "$BIFROST_VERIFY_RUN_ID"
```

Ready when the helper exits 0 and stdout contains both:

- `ok launch crate=bifrost-index@0.2.2`
- `Finished \`test\` profile`

Teardown is **Cleanup**, not a daemon kill. Cargo/rustc started by launch
finish before the helper returns; if a PID is recorded in the run scratch
(`cargo.pid`), cleanup kills **that PID only**.

Never drive `target/` leftovers from a different crate. Doctor first when
anything looks off.

## Doctor

Read-only. Answers "is this checkout worth driving?"

```bash
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh doctor
```

Require all of:

- `cargo` and `rustc` on `PATH`; `rustc` reports a version `>= 1.87.0`
  (`Cargo.toml` `rust-version = "1.87"`, edition 2024).
- Manifest package `name = "bifrost-index"`, `version = "0.2.2"`,
  `[lib] name = "bifrost"` so rustc still resolves `use bifrost::...`.
- `tests/interoperability.rs` and `tests/fixtures/v3.hex` exist.
- Decoded `v3.hex` is exactly 220 bytes with SHA-256
  `70feb12b392eb223c79db095aabb0500b64ba7f1ddf1a2f6030d29aa499466ca`
  (documented in `tests/fixtures/README.md`).
- No listening port is expected; a bound TCP port is **not** a health signal.

Stdout ends with a single `ok doctor ...` line. Non-zero exit means do not drive.

## Drive

Harness: `scripts/verify-bifrost.sh` wrapping `cargo test --all-features`.
Prefer the helper so `TMPDIR`, evidence paths, and filters stay consistent.
Drive from repo root after a successful doctor and launch for this `RUN_ID`.

```bash
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh drive <feature-id> --run-id "$BIFROST_VERIFY_RUN_ID"
```

Mapped feature IDs and the cargo filters they run (libtest substring after `--`):

| Feature file | feature-id | cargo invocation |
| --- | --- | --- |
| `features/insert-and-search.md` | `insert-and-search` | `cargo test --all-features --lib -- --nocapture insert_and_search cosine_distance_ranking_is_correct empty_and_single_vector_indexes` |
| `features/persist-v3.md` | `persist-v3` | `cargo test --all-features --test interoperability -- --nocapture` then `cargo test --all-features --lib -- --nocapture save_and_load_round_trip` |
| `features/api-errors.md` | `api-errors` | `cargo test --all-features --lib -- --nocapture dimension_mismatches_return_errors sparse_and_duplicate_external_ids invalid_configs_are_rejected` |
| `features/batch-build.md` | `batch-build` | `cargo test --all-features --lib -- --nocapture build_batch_and_search` |

Public-API shape these tests exercise (do not replace with private graph setters):

```rust
use bifrost::{load_file, Config, HnswIndex};

let mut index = HnswIndex::new(Config {
    dim: 4,
    rng_seed: Some(42),
    ..Config::default()
})?;
index.insert(100, &[1.0, 0.0, 0.0, 0.0])?;
let hits = index.search(&[0.98, 0.02, 0.0, 0.0], 10)?;
index.save(&path)?;
let loaded = load_file(&path)?;
```

Treat every test name and flag as literal. Do not add `--ignored`, do not
enable `fiqa-prep`, do not run `cargo bench` or `benchmarks/competitors`.

## Evidence

Proof artifacts live under the skill directory so cleanup cannot eat them:

`.cursor/skills/verify-bifrost/artifacts/<run-id>/` relative to the checkout.

Per drive, expect at least:

- `doctor.txt` — doctor transcript (copied at launch)
- `launch.txt` — compile transcript
- `<feature-id>/cargo-test.txt` — full cargo stdout+stderr
- `<feature-id>/meta.json` — `feature_id`, cargo filters, exit code, rustc/cargo
  versions, scratch path, UTC timestamp
- `run-state.env` — scratch path and any recorded child PID

Proof standards:

- Exercise the public API (the cargo tests above), not `Graph::insert_node` or
  other crate-private construction.
- Capture the command, the `test … ok` / failure lines, and the process exit
  code — not only a final "all passed" summary.
- For persistence, the interoperability writer test must reproduce all 220
  fixture bytes; record that `writer_matches_v3_fixture ... ok` and
  `loads_v3_fixture ... ok` both appear.
- Side effects: any `.hnsw` files must appear only under the run scratch
  `TMPDIR`, never under the repo or `$HOME`.
- Mocks are not used. Skip competitor/FiQA rather than stubbing OpenAI.

## Cleanup

```bash
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh cleanup --run-id "$BIFROST_VERIFY_RUN_ID"
```

Removes the disposable scratch directory (`TMPDIR` / `.hnsw` files) for this
run. If `scratch/cargo.pid` exists and that PID is still alive, send `TERM`
then `KILL` to **that PID only**. Never `pkill cargo`, never `killall rustc`,
never kill by process name.

Cleanup **must not** delete `artifacts/`. After cleanup, confirm
`artifacts/<run-id>/<feature-id>/cargo-test.txt` still exists.

Run cleanup after failed iterations too so crashed writers do not leave
`.hnsw` files in a shared `/tmp`.

## Helpers

`scripts/verify-bifrost.sh` is executable. Invoke it from the repo root (or
rely on `BIFROST_ROOT`):

```bash
scripts/verify-bifrost.sh          # from .cursor/skills/verify-bifrost/
# or
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh doctor
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh launch --run-id "$BIFROST_VERIFY_RUN_ID"
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh drive insert-and-search --run-id "$BIFROST_VERIFY_RUN_ID"
.cursor/skills/verify-bifrost/scripts/verify-bifrost.sh cleanup --run-id "$BIFROST_VERIFY_RUN_ID"
```

`--run-id` may be omitted when `BIFROST_VERIFY_RUN_ID` is already exported.
The script refuses to run if `Cargo.toml` is not this crate. It always sets
`TMPDIR` to the per-run scratch dir before invoking cargo.
