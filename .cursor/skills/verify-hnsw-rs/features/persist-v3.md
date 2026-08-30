# Persist v3

Persist v3 lets a caller write a constructed index to a `.hnsw` file and load
it back as a memory-mapped v3 snapshot whose 220-byte golden encoding is
byte-for-byte stable.

## Sub-features

- `persist-save` writes a v3 file via `HnswIndex::save` / `save_file`.
- `persist-load-golden` loads `tests/fixtures/v3.hex` through `load_file` and
  reads header, nodes, and vectors.
- `persist-writer-golden` requires the writer to reproduce all 220 fixture bytes.
- `persist-round-trip` saves a three-node fixture index and reloads matching
  nodes, vectors, and per-layer edges.

## How to get to it (user POV)

- After inserts, call `index.save(&path)?` with `path` in a temp directory.
- Call `bifrost::load_file(&path)?` and read `loaded.header()`, `loaded.node(i)`,
  `loaded.vector(i)`.
- Follow the README persistence example (`dim: 2`, id `7`, vector `[1.0, 0.0]`).
- The checked-in golden is `tests/fixtures/v3.hex` (decoded SHA-256
  `70feb12b392eb223c79db095aabb0500b64ba7f1ddf1a2f6030d29aa499466ca`).

## Driving it with verify-hnsw-rs

Preconditions:

- `scripts/verify-hnsw-rs.sh doctor` exited 0, including the 220-byte fixture SHA.
- `scripts/verify-hnsw-rs.sh launch --run-id $RUN_ID` compiled tests.
- Scratch `TMPDIR` is the launch scratch dir so `.hnsw` files never land in the repo.

- **Load golden.** Drive persistence. Run
  `scripts/verify-hnsw-rs.sh drive persist-v3 --run-id $RUN_ID`.
  The helper runs
  `cargo test --all-features --test interoperability -- --nocapture`
  then
  `cargo test --all-features --lib -- --nocapture save_and_load_round_trip`.
  Exit code `0`. Transcript contains `test loads_v3_fixture ... ok`
  (`header().version` is `VERSION` (3), `node_count` is `3`, external ids
  `100`, `200`, `300`, vector 2 is `[-1.0, 0.0]`).
- **Writer matches golden.** Same drive. Transcript contains
  `test writer_matches_v3_fixture ... ok` (saved bytes equal decoded `v3.hex`).
- **Round trip.** Same drive. Transcript contains
  `test serialize::tests::save_and_load_round_trip ... ok`.
- **Proof.** Artifacts
  `artifacts/$RUN_ID/persist-v3/cargo-test.txt` and
  `artifacts/$RUN_ID/persist-v3/meta.json` with `exit_code` `0`.
  After cleanup, those files still exist and the repo working tree has no new
  `.hnsw` files.

## Gotchas

- The documented SHA-256 is of the **decoded** 220 bytes, not of `v3.hex` text.
- Loaded files stay memory-mapped; drop `LoadedHnsw` before deleting the path
  (the interoperability tests already `drop(loaded)` then `remove_file`).
- Valid v2 files are rewritten in place to v3 (atomic rename). This feature's
  golden is already v3; v2 migration is not required for a persist-v3 proof.
- `rng_seed` is not stored in the header (`Header::config` sets `rng_seed: None`).
  Do not expect a loaded file to reconstruct the original RNG.
- Isolation: tests name files
  `hnsw-rs-interop-<label>-<pid>-<time>.hnsw` under `temp_dir()`. A shared
  `/tmp` is not proof isolation — launch must set `TMPDIR` to the run scratch.
