#!/usr/bin/env bash
# Project-local harness for verifying the Bifrost public Rust API via cargo test.
# Invocation is documented in ../SKILL.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${HNSW_RS_ROOT:-$(cd "${SKILL_DIR}/../../.." && pwd)}"
MANIFEST="${REPO_ROOT}/Cargo.toml"
FIXTURE_HEX="${REPO_ROOT}/tests/fixtures/v3.hex"
ARTIFACTS_ROOT="${SKILL_DIR}/artifacts"
EXPECTED_SHA="70feb12b392eb223c79db095aabb0500b64ba7f1ddf1a2f6030d29aa499466ca"
EXPECTED_BYTES="220"
EXPECTED_NAME="bifrost"
EXPECTED_VERSION="0.2.0"
EXPECTED_RUST_VERSION="1.85"

usage() {
  cat <<'USAGE'
verify-hnsw-rs.sh <command> [args]

Commands:
  doctor                      Read-only toolchain, crate, and v3 fixture check
  launch [--run-id ID]        Compile tests; isolate TMPDIR for .hnsw files
  drive FEATURE [--run-id ID] Run one mapped feature via cargo test
  cleanup [--run-id ID]       Remove scratch (never artifacts); kill recorded PID only

Features:
  insert-and-search
  persist-v3
  api-errors
  batch-build

Environment:
  HNSW_RS_ROOT         Override repo root
  HNSW_VERIFY_RUN_ID   Default run id when --run-id is omitted
  HNSW_VERIFY_SCRATCH  Override scratch directory

Evidence:
  .cursor/skills/verify-hnsw-rs/artifacts/<run-id>/
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

parse_run_id() {
  RUN_ID="${HNSW_VERIFY_RUN_ID:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id)
        [[ $# -ge 2 ]] || die "--run-id requires a value"
        RUN_ID="$2"
        shift 2
        ;;
      --run-id=*)
        RUN_ID="${1#--run-id=}"
        shift
        ;;
      *)
        die "unexpected argument: $1"
        ;;
    esac
  done
  [[ -n "${RUN_ID}" ]] || RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  case "${RUN_ID}" in
    *[!A-Za-z0-9._-]*) die "run id must be alphanumeric plus . _ -" ;;
  esac
}

scratch_dir_for() {
  local id="$1"
  if [[ -n "${HNSW_VERIFY_SCRATCH:-}" ]]; then
    printf '%s\n' "${HNSW_VERIFY_SCRATCH}"
  else
    local base="${TMPDIR:-/tmp}"
    base="${base%/}"
    printf '%s\n' "${base}/hnsw-rs-verify-${id}"
  fi
}

run_dir_for() {
  printf '%s\n' "${ARTIFACTS_ROOT}/${1}"
}

require_repo() {
  [[ -f "${MANIFEST}" ]] || die "Cargo.toml not found at ${MANIFEST}"
  grep -q '^name = "bifrost"$' "${MANIFEST}" || die "${MANIFEST} is not crate bifrost"
}

cmd_doctor() {
  require_repo
  command -v cargo >/dev/null || die "cargo not on PATH"
  command -v rustc >/dev/null || die "rustc not on PATH"
  command -v python3 >/dev/null || die "python3 not on PATH"

  local cargo_v rustc_v
  cargo_v="$(cargo --version)"
  rustc_v="$(rustc --version)"

  python3 - "${MANIFEST}" "${FIXTURE_HEX}" "${EXPECTED_SHA}" "${EXPECTED_BYTES}" \
    "${EXPECTED_NAME}" "${EXPECTED_VERSION}" "${EXPECTED_RUST_VERSION}" "${rustc_v}" <<'PY'
import pathlib, re, sys

manifest, fixture, expected_sha, expected_bytes, expected_name, expected_version, expected_rv, rustc_line = sys.argv[1:]
text = pathlib.Path(manifest).read_text()

def field(key):
    match = re.search(rf'^{re.escape(key)} = "([^"]+)"', text, re.M)
    if not match:
        raise SystemExit(f"error: Cargo.toml missing {key}")
    return match.group(1)

name = field("name")
version = field("version")
rust_version = field("rust-version")
if name != expected_name:
    raise SystemExit(f"error: package name {name!r} != {expected_name!r}")
if version != expected_version:
    raise SystemExit(f"error: package version {version!r} != {expected_version!r}")
if rust_version != expected_rv:
    raise SystemExit(f"error: rust-version {rust_version!r} != {expected_rv!r}")
if "publish = false" not in text:
    raise SystemExit("error: expected publish = false")

def parse_semver(s):
    parts = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", s)
    if not parts:
        raise SystemExit(f"error: cannot parse semver {s!r}")
    return tuple(int(x or 0) for x in parts.groups())

rustc_match = re.search(r"rustc (\d+\.\d+\.\d+)", rustc_line)
if not rustc_match:
    raise SystemExit(f"error: cannot parse rustc version from {rustc_line!r}")
if parse_semver(rustc_match.group(1)) < parse_semver(expected_rv):
    raise SystemExit(
        f"error: rustc {rustc_match.group(1)} is older than rust-version {expected_rv}"
    )

fixture_path = pathlib.Path(fixture)
if not fixture_path.is_file():
    raise SystemExit(f"error: missing fixture {fixture}")
interop = fixture_path.parent.parent / "interoperability.rs"
if not interop.is_file():
    raise SystemExit(f"error: missing {interop}")

hex_text = "".join(fixture_path.read_text().split())
try:
    data = bytes.fromhex(hex_text)
except ValueError as exc:
    raise SystemExit(f"error: v3.hex is not valid hex: {exc}") from exc

import hashlib
digest = hashlib.sha256(data).hexdigest()
if str(len(data)) != expected_bytes:
    raise SystemExit(f"error: decoded fixture is {len(data)} bytes, expected {expected_bytes}")
if digest != expected_sha:
    raise SystemExit(f"error: fixture SHA-256 {digest} != {expected_sha}")

print(f"ok doctor crate={name}@{version} rust-version={rust_version} rustc={rustc_match.group(1)} fixture_bytes={len(data)} fixture_sha256={digest}")
PY

  echo "ok cargo=${cargo_v}"
  echo "ok rustc=${rustc_v}"
  echo "ok manifest=${MANIFEST}"
  echo "ok fixture=${FIXTURE_HEX}"
}

write_run_state() {
  local run_dir="$1" scratch="$2" cargo_pid="${3:-}"
  cat > "${run_dir}/run-state.env" <<STATE
RUN_ID=${RUN_ID}
REPO_ROOT=${REPO_ROOT}
SCRATCH=${scratch}
CARGO_PID=${cargo_pid}
SKILL_DIR=${SKILL_DIR}
STATE
}

run_cargo_isolated() {
  local scratch="$1" out="$2"
  shift 2
  mkdir -p "${scratch}" "$(dirname "${out}")"
  # Isolate .hnsw writers. Do not inherit a caller TMPDIR that points at the repo.
  local pid_file="${scratch}/cargo.pid"
  set +e
  (
    export TMPDIR="${scratch}"
    export CARGO_TERM_COLOR=never
    exec "$@"
  ) >"${out}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  wait "${pid}"
  local code=$?
  set -e
  rm -f "${pid_file}"
  return "${code}"
}

cmd_launch() {
  parse_run_id "$@"
  require_repo
  local run_dir scratch
  run_dir="$(run_dir_for "${RUN_ID}")"
  scratch="$(scratch_dir_for "${RUN_ID}")"
  mkdir -p "${run_dir}" "${scratch}"
  write_run_state "${run_dir}" "${scratch}" ""

  local doctor_out="${run_dir}/doctor.txt"
  if ! cmd_doctor >"${doctor_out}" 2>&1; then
    cat "${doctor_out}" >&2
    die "doctor failed; not compiling"
  fi

  local launch_out="${run_dir}/launch.txt"
  echo "run_id=${RUN_ID}" >"${launch_out}"
  echo "scratch=${scratch}" >>"${launch_out}"
  echo "command=cargo test --manifest-path ${MANIFEST} --all-features --no-run" >>"${launch_out}"
  if ! run_cargo_isolated "${scratch}" "${run_dir}/launch.cargo.txt" \
    cargo test --manifest-path "${MANIFEST}" --all-features --no-run; then
    cat "${run_dir}/launch.cargo.txt" >>"${launch_out}"
    cat "${run_dir}/launch.cargo.txt" >&2
    die "cargo test --no-run failed"
  fi
  cat "${run_dir}/launch.cargo.txt" >>"${launch_out}"
  grep -q 'Finished `test` profile' "${run_dir}/launch.cargo.txt" \
    || die "launch output missing Finished \`test\` profile"
  echo "ok launch crate=${EXPECTED_NAME}@${EXPECTED_VERSION} run_id=${RUN_ID} scratch=${scratch}"
  echo "ok launch evidence=${run_dir}/launch.txt"
}

feature_cargo_steps() {
  local feature="$1"
  case "${feature}" in
    insert-and-search)
      printf '%s\n' "cargo test --manifest-path ${MANIFEST} --all-features --lib -- --nocapture insert_and_search cosine_distance_ranking_is_correct empty_and_single_vector_indexes"
      ;;
    persist-v3)
      printf '%s\n' "cargo test --manifest-path ${MANIFEST} --all-features --test interoperability -- --nocapture"
      printf '%s\n' "cargo test --manifest-path ${MANIFEST} --all-features --lib -- --nocapture save_and_load_round_trip"
      ;;
    api-errors)
      printf '%s\n' "cargo test --manifest-path ${MANIFEST} --all-features --lib -- --nocapture dimension_mismatches_return_errors sparse_and_duplicate_external_ids invalid_configs_are_rejected"
      ;;
    batch-build)
      printf '%s\n' "cargo test --manifest-path ${MANIFEST} --all-features --lib -- --nocapture build_batch_and_search"
      ;;
    *)
      die "unknown feature '${feature}'. mapped: insert-and-search persist-v3 api-errors batch-build"
      ;;
  esac
}

cmd_drive() {
  local feature="${1:-}"
  [[ -n "${feature}" ]] || die "drive requires a feature id"
  shift
  parse_run_id "$@"
  require_repo
  local run_dir scratch feature_dir
  run_dir="$(run_dir_for "${RUN_ID}")"
  scratch="$(scratch_dir_for "${RUN_ID}")"
  feature_dir="${run_dir}/${feature}"
  [[ -d "${run_dir}" ]] || die "run ${RUN_ID} was not launched (missing ${run_dir})"
  [[ -d "${scratch}" ]] || die "scratch missing (${scratch}); re-run launch"
  mkdir -p "${feature_dir}"

  local transcript="${feature_dir}/cargo-test.txt"
  : >"${transcript}"
  local failed=0
  local step_idx=0
  while IFS= read -r spec; do
    step_idx=$((step_idx + 1))
    # shellcheck disable=SC2206
    local -a args
    # Re-split the documented cargo line.
    read -r -a args <<<"${spec}"
    {
      echo "===== step ${step_idx} ====="
      echo "command=${spec}"
      echo "TMPDIR=${scratch}"
    } >>"${transcript}"
    local step_out="${feature_dir}/step-${step_idx}.txt"
    if run_cargo_isolated "${scratch}" "${step_out}" "${args[@]}"; then
      cat "${step_out}" >>"${transcript}"
      echo "===== step ${step_idx} exit=0 =====" >>"${transcript}"
    else
      local code=$?
      cat "${step_out}" >>"${transcript}"
      echo "===== step ${step_idx} exit=${code} =====" >>"${transcript}"
      failed="${code}"
    fi
  done < <(feature_cargo_steps "${feature}")

  local rustc_v cargo_v
  rustc_v="$(rustc --version)"
  cargo_v="$(cargo --version)"
  python3 - "${feature_dir}/meta.json" "${feature}" "${RUN_ID}" "${failed}" \
    "${scratch}" "${rustc_v}" "${cargo_v}" "${REPO_ROOT}" <<'PY'
import json, sys, datetime
path, feature, run_id, exit_code, scratch, rustc, cargo, repo = sys.argv[1:]
json.dump(
    {
        "feature_id": feature,
        "run_id": run_id,
        "exit_code": int(exit_code),
        "scratch": scratch,
        "repo_root": repo,
        "rustc": rustc,
        "cargo": cargo,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harness": "verify-hnsw-rs.sh",
    },
    open(path, "w"),
    indent=2,
)
print()  # keep file close via context-less open; ok for this helper
PY

  echo "ok drive feature=${feature} run_id=${RUN_ID} exit=${failed} evidence=${transcript}"
  [[ "${failed}" -eq 0 ]] || exit "${failed}"
}

kill_recorded_pid() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 0
  local pid
  pid="$(tr -d '[:space:]' <"${pid_file}")"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    echo "ok cleanup sending TERM to recorded pid ${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
    local i
    for i in 1 2 3 4 5; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "ok cleanup sending KILL to recorded pid ${pid}"
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
}

cmd_cleanup() {
  parse_run_id "$@"
  local run_dir scratch
  run_dir="$(run_dir_for "${RUN_ID}")"
  if [[ -f "${run_dir}/run-state.env" ]]; then
    # shellcheck disable=SC1090
    source "${run_dir}/run-state.env"
    scratch="${SCRATCH}"
  else
    scratch="$(scratch_dir_for "${RUN_ID}")"
  fi
  if [[ -n "${scratch}" ]]; then
    kill_recorded_pid "${scratch}/cargo.pid"
    if [[ -d "${scratch}" ]]; then
      rm -rf "${scratch}"
      echo "ok cleanup removed scratch=${scratch}"
    else
      echo "ok cleanup scratch already absent=${scratch}"
    fi
  fi
  echo "ok cleanup evidence retained under ${run_dir}"
  echo "ok cleanup never deletes ${ARTIFACTS_ROOT}"
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    doctor) cmd_doctor "$@" ;;
    launch) cmd_launch "$@" ;;
    drive) cmd_drive "$@" ;;
    cleanup) cmd_cleanup "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      usage >&2
      die "unknown command ${cmd}"
      ;;
  esac
}

main "$@"
