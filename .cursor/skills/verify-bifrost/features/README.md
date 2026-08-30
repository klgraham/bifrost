# Bifrost verification map

This directory is the maintained source for verifying the user-facing behavior
of the Bifrost Rust library. Read the index before driving the crate, then use
the matching feature file as the recipe.

## Baseline preconditions

- Work in `/Users/klogram/dev/klogram_labs/bifrost` (or `BIFROST_ROOT`).
- Toolchain: `cargo`/`rustc` ≥ 1.85 (crate `rust-version`). Current checkout
  was last proven on rustc 1.96.x.
- Run `scripts/verify-bifrost.sh doctor` and require crate `bifrost-index@0.2.0`
  plus the v3 fixture SHA-256
  `70feb12b392eb223c79db095aabb0500b64ba7f1ddf1a2f6030d29aa499466ca`.
- Run `scripts/verify-bifrost.sh launch --run-id $RUN_ID` so tests are compiled
  and `TMPDIR` is a disposable scratch directory.
- Never write `.hnsw` files into the repo. Never enable `fiqa-prep`. Never run
  `benchmarks/competitors`.
- Never drive deletion, in-place updates, or concurrent mutation — the crate
  does not support them.

## Driving conventions

- Start every recipe from a successful doctor + launch for this `RUN_ID`.
- Drive through `verify-bifrost.sh`, which wraps `cargo test --all-features`.
- Treat every test name and flag as literal.
- Vectors passed to `insert`/`search` must be unit-normalized (`1 - dot`
  cosine). `bifrost::vector::cosine_similarity` is for arbitrary vectors and
  is not a substitute for normalizing before insert.
- Restore nothing in the repo after a drive; tests use process-unique temp
  files under `TMPDIR`. Do not remove proof artifacts during cleanup.

## Proof and skip reporting

- Capture the cargo command, full transcript, and exit code.
- Mutation/persistence proof includes a second observation: loaded header /
  node / vector bytes, or the byte-for-byte fixture compare.
- Record the feature ID with every artifact under
  `.cursor/skills/verify-bifrost/artifacts/<run-id>/`.
- Report an unreachable path with the attempted cargo filter and the unmet
  precondition (missing rustc, fixture SHA mismatch, compile failure).
- Do not report competitor-bench or FiQA coverage as verified. Do not report a
  skipped entry point as verified through a different filter.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the
user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with verify-bifrost` starts with `Preconditions:` and uses
   labeled bullets that pair each user action with an exact command and
   observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only public API paths, cargo
filters, required state, commands, and observable proof.

## Features

- [Insert and search](./insert-and-search.md) covers sparse-ID insert, k-NN
  cosine search, empty/`k=0` results, and distance ranking.
- [Persist v3](./persist-v3.md) covers `.hnsw` v3 save/load against the 220-byte
  golden fixture and a save/load round trip.
- [API errors](./api-errors.md) covers dimension mismatch, duplicate external
  IDs, and invalid `Config` values.
- [Batch build](./batch-build.md) covers `HnswIndex::build` assigning dense
  IDs from `0`.
