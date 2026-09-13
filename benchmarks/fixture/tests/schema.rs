use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use bifrost_benchmark_fixture::{
    Artifact, ArtifactResolution, CATALOG_SCHEMA, Catalog, FIXTURE_SCHEMA, FixtureCounts,
    FixtureFile, FixtureFileKind, FixtureManifest, MANIFEST_FILE, Publication, RESOLUTION_FILE,
    SourceMetadata, VectorRepresentation, VerifiedFixture, input_sha256, sha256_hex,
};

static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn catalog_loads_two_dimensions_and_distinguishes_publication_states() {
    let directory = TestDirectory::new("catalog");
    let catalog = Catalog {
        schema: CATALOG_SCHEMA.to_owned(),
        artifacts: vec![planned_artifact(2), published_artifact(3, hex('b', 64))],
    };
    let path = directory.path().join("catalog.json");
    write_json(&path, &catalog);

    let loaded = Catalog::load(&path).expect("load catalog");
    loaded.validate().expect("validate catalog");
    assert_eq!(
        loaded.artifact("fiqa-2").map(|item| item.dimensions),
        Some(2)
    );
    assert_eq!(
        loaded.artifact("fiqa-3").map(|item| item.dimensions),
        Some(3)
    );
    assert!(matches!(
        &loaded
            .artifact("fiqa-2")
            .expect("planned artifact")
            .publication,
        Publication::Planned { .. }
    ));
    assert!(matches!(
        &loaded
            .artifact("fiqa-3")
            .expect("published artifact")
            .publication,
        Publication::Published { .. }
    ));
    assert!(loaded.artifact("missing").is_none());
}

#[test]
fn catalog_rejects_duplicate_ids_moving_revisions_and_path_traversal() {
    let duplicate = planned_artifact(2);
    let catalog = Catalog {
        schema: CATALOG_SCHEMA.to_owned(),
        artifacts: vec![duplicate.clone(), duplicate],
    };
    assert_error_contains(catalog.validate(), "duplicate artifact id fiqa-2");
    let directory = TestDirectory::new("invalid-catalog-load");
    let path = directory.path().join("catalog.json");
    write_json(&path, &catalog);
    assert_error_contains(Catalog::load(path), "duplicate artifact id fiqa-2");

    let mut moving = published_artifact(2, hex('a', 64));
    let Publication::Published { revision, .. } = &mut moving.publication else {
        panic!("published artifact")
    };
    *revision = "main".to_owned();
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![moving],
        }
        .validate(),
        "published revision must be exactly 40 lowercase hexadecimal characters",
    );

    let mut uppercase = published_artifact(2, hex('a', 64));
    let Publication::Published { revision, .. } = &mut uppercase.publication else {
        panic!("published artifact")
    };
    *revision = hex('A', 40);
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![uppercase],
        }
        .validate(),
        "lowercase hexadecimal",
    );

    let mut traversal = planned_artifact(2);
    let Publication::Planned { prefix, .. } = &mut traversal.publication else {
        panic!("planned artifact")
    };
    *prefix = "fixtures/../escape".to_owned();
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![traversal],
        }
        .validate(),
        "path traversal",
    );
}

#[test]
fn catalog_accepts_only_the_planned_manifest_hash_placeholder() {
    Catalog {
        schema: CATALOG_SCHEMA.to_owned(),
        artifacts: vec![planned_artifact(384)],
    }
    .validate()
    .expect("planned manifest hash placeholder");

    let mut published = published_artifact(384, hex('a', 64));
    let Publication::Published { prefix, .. } = &mut published.publication else {
        panic!("published artifact")
    };
    *prefix = "fixtures/fiqa/384/{manifest_sha256}".to_owned();
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![published],
        }
        .validate(),
        "unresolved manifest placeholder",
    );

    let mut misplaced = planned_artifact(384);
    let Publication::Planned { prefix, .. } = &mut misplaced.publication else {
        panic!("planned artifact")
    };
    *prefix = "fixtures/{manifest_sha256}/384".to_owned();
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![misplaced],
        }
        .validate(),
        "planned prefix must contain exactly one {manifest_sha256}",
    );

    let mut repeated = planned_artifact(384);
    let Publication::Planned { prefix, .. } = &mut repeated.publication else {
        panic!("planned artifact")
    };
    *prefix = "fixtures/{manifest_sha256}/{manifest_sha256}".to_owned();
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![repeated],
        }
        .validate(),
        "planned prefix must contain exactly one {manifest_sha256}",
    );

    let manifest_sha256 = hex('b', 64);
    let mut mismatched = published_artifact(384, manifest_sha256);
    let Publication::Published { prefix, .. } = &mut mismatched.publication else {
        panic!("published artifact")
    };
    *prefix = format!("fixtures/fiqa/384/{}", hex('c', 64));
    assert_error_contains(
        Catalog {
            schema: CATALOG_SCHEMA.to_owned(),
            artifacts: vec![mismatched],
        }
        .validate(),
        "published prefix must end with published manifest_sha256",
    );
}

#[test]
fn verified_fixture_loads_vectors_ids_qrels_and_resolution() {
    let directory = TestDirectory::new("verified");
    let manifest = write_valid_fixture(directory.path());
    let manifest_bytes = fs::read(directory.path().join(MANIFEST_FILE)).expect("manifest bytes");
    let expected = published_artifact(manifest.dimensions, sha256_hex(&manifest_bytes));
    let resolution = ArtifactResolution::from_artifact(&expected).expect("published resolution");
    resolution
        .write(directory.path().join(RESOLUTION_FILE))
        .expect("write resolution");

    let fixture = VerifiedFixture::load(directory.path(), Some(&expected)).expect("verify fixture");
    assert_eq!(fixture.corpus_vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(fixture.query_vectors, vec![vec![1.0, 0.0]]);
    assert_eq!(fixture.corpus_ids, vec!["d1", "d2"]);
    assert_eq!(fixture.query_ids, vec!["q1"]);
    assert_eq!(
        fixture.qrels_path,
        Some(directory.path().join("qrels-test.tsv"))
    );
    assert_eq!(fixture.metadata, manifest);
    assert_eq!(fixture.resolution, Some(resolution));
}

#[test]
fn fixture_rejects_wrong_schema_and_catalog_metadata() {
    let directory = TestDirectory::new("metadata");
    let mut manifest = write_valid_fixture(directory.path());
    manifest.schema = "bifrost-benchmark-fixture-v1".to_owned();
    assert_error_contains(
        manifest.validate_metadata(),
        "schema must be bifrost-benchmark-fixture-v2",
    );

    let mut manifest = write_valid_fixture(directory.path());
    manifest.generator_revision = "main".to_owned();
    assert_error_contains(
        manifest.validate_metadata(),
        "generator_revision must be exactly 40 lowercase hexadecimal characters",
    );

    let directory = TestDirectory::new("catalog-mismatch");
    write_valid_fixture(directory.path());
    let mut expected = planned_artifact(2);
    expected.model = "other-model".to_owned();
    assert_error_contains(
        VerifiedFixture::load(directory.path(), Some(&expected)),
        "does not match catalog model",
    );

    let mut expected = planned_artifact(3);
    expected.id = "fiqa-2".to_owned();
    assert_error_contains(
        VerifiedFixture::load(directory.path(), Some(&expected)),
        "dimensions do not match catalog dimensions",
    );
    expected.dimensions = 2;
    expected.dataset = "other-dataset".to_owned();
    assert_error_contains(
        VerifiedFixture::load(directory.path(), Some(&expected)),
        "does not match catalog dataset",
    );
}

#[test]
fn fixture_rejects_missing_extra_corrupt_and_bad_shape_files() {
    let directory = TestDirectory::new("missing");
    write_valid_fixture(directory.path());
    fs::remove_file(directory.path().join("queries.f32")).expect("remove query vectors");
    assert_error_contains(VerifiedFixture::load(directory.path(), None), "queries.f32");

    let directory = TestDirectory::new("extra");
    write_valid_fixture(directory.path());
    fs::write(directory.path().join("unexpected.txt"), b"extra").expect("write extra file");
    assert_error_contains(
        VerifiedFixture::load(directory.path(), None),
        "unlisted file",
    );

    let directory = TestDirectory::new("corrupt");
    write_valid_fixture(directory.path());
    fs::write(directory.path().join("corpus-ids.txt"), b"d1\nd9\n").expect("corrupt ID file");
    assert_error_contains(VerifiedFixture::load(directory.path(), None), "SHA-256");

    let directory = TestDirectory::new("shape");
    let mut manifest = write_valid_fixture(directory.path());
    manifest.dimensions = 3;
    assert_error_contains(manifest.validate_metadata(), "corpus shape requires");
}

#[test]
fn fixture_rejects_bad_vector_norms_and_duplicate_ids_after_hashing() {
    let directory = TestDirectory::new("norm");
    let mut manifest = write_valid_fixture(directory.path());
    replace_file(
        directory.path(),
        &mut manifest,
        FixtureFileKind::CorpusVectors,
        &vector_bytes(&[[2.0, 0.0], [0.0, 1.0]]),
    );
    manifest
        .write(directory.path().join(MANIFEST_FILE))
        .expect("rewrite manifest");
    assert_error_contains(
        VerifiedFixture::load(directory.path(), None),
        "invalid L2 norm",
    );

    let directory = TestDirectory::new("ids");
    let mut manifest = write_valid_fixture(directory.path());
    replace_file(
        directory.path(),
        &mut manifest,
        FixtureFileKind::CorpusIds,
        b"d1\nd1\n",
    );
    manifest
        .write(directory.path().join(MANIFEST_FILE))
        .expect("rewrite manifest");
    assert_error_contains(
        VerifiedFixture::load(directory.path(), None),
        "duplicate IDs",
    );
}

#[test]
fn fixture_rejects_semantically_invalid_qrels_after_hashing() {
    for (label, qrels, expected) in [
        (
            "unknown-document",
            b"query-id\tcorpus-id\tscore\nq1\tmissing\t1\n".as_slice(),
            "unknown corpus ID",
        ),
        (
            "duplicate-pair",
            b"query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\n".as_slice(),
            "repeats query/document pair",
        ),
        (
            "zero-score",
            b"query-id\tcorpus-id\tscore\nq1\td1\t0\n".as_slice(),
            "score must be a positive integer",
        ),
    ] {
        let directory = TestDirectory::new(label);
        let mut manifest = write_valid_fixture(directory.path());
        replace_file(
            directory.path(),
            &mut manifest,
            FixtureFileKind::Qrels,
            qrels,
        );
        manifest
            .write(directory.path().join(MANIFEST_FILE))
            .expect("rewrite manifest");
        assert_error_contains(VerifiedFixture::load(directory.path(), None), expected);
    }
}

#[test]
fn fixture_manifest_rejects_missing_roles_duplicate_paths_and_traversal() {
    let directory = TestDirectory::new("manifest-fields");
    let mut manifest = write_valid_fixture(directory.path());
    manifest
        .files
        .retain(|file| file.kind != FixtureFileKind::QueryIds);
    assert_error_contains(
        manifest.validate_metadata(),
        "missing fixture file kind query_ids",
    );

    let mut manifest = write_valid_fixture(directory.path());
    manifest
        .files
        .retain(|file| file.kind != FixtureFileKind::Qrels);
    assert_error_contains(
        manifest.validate_metadata(),
        "missing fixture file kind qrels",
    );

    let mut manifest = write_valid_fixture(directory.path());
    manifest.files[1].path = manifest.files[0].path.clone();
    assert_error_contains(manifest.validate_metadata(), "duplicate fixture file path");

    let mut manifest = write_valid_fixture(directory.path());
    manifest.files[0].path = "../corpus.f32".to_owned();
    assert_error_contains(manifest.validate_metadata(), "path traversal");
}

#[test]
fn fixture_rejects_manifest_and_resolution_hash_mismatches() {
    let directory = TestDirectory::new("manifest-hash");
    let manifest = write_valid_fixture(directory.path());
    let expected = published_artifact(manifest.dimensions, hex('0', 64));
    assert_error_contains(
        VerifiedFixture::load(directory.path(), Some(&expected)),
        "manifest.json SHA-256",
    );

    let directory = TestDirectory::new("resolution-hash");
    let manifest = write_valid_fixture(directory.path());
    let manifest_bytes = fs::read(directory.path().join(MANIFEST_FILE)).expect("manifest bytes");
    let expected = published_artifact(manifest.dimensions, sha256_hex(&manifest_bytes));
    let mut resolution = ArtifactResolution::from_artifact(&expected).expect("resolution");
    resolution.revision = hex('c', 40);
    resolution
        .write(directory.path().join(RESOLUTION_FILE))
        .expect("write resolution");
    assert_error_contains(
        VerifiedFixture::load(directory.path(), Some(&expected)),
        "does not match catalog revision",
    );
}

#[test]
fn input_and_fixture_identity_hash_every_generation_input() {
    let original_text = input_sha256([("stable-id", "original text")]);
    let changed_text = input_sha256([("stable-id", "changed text")]);
    assert_ne!(original_text, changed_text);
    assert_eq!(
        original_text,
        input_sha256([("stable-id", "original text")])
    );

    let directory = TestDirectory::new("identity");
    let base = write_valid_fixture(directory.path());
    let identity = base.identity_sha256();

    let mut changed = base.clone();
    changed.corpus_input_sha256 = changed_text;
    assert_ne!(identity, changed.identity_sha256());

    let mut changed = base.clone();
    changed.selection.push_str("; development subset");
    assert_ne!(identity, changed.identity_sha256());

    let mut changed = base.clone();
    changed.preprocessing.push_str("; lowercase");
    assert_ne!(identity, changed.identity_sha256());

    let mut changed = base.clone();
    changed.requested_model = "different-model".to_owned();
    assert_ne!(identity, changed.identity_sha256());

    let mut changed = base;
    changed.dimensions += 1;
    assert_ne!(identity, changed.identity_sha256());
}

fn planned_artifact(dimensions: usize) -> Artifact {
    Artifact {
        id: format!("fiqa-{dimensions}"),
        dataset: "BEIR FiQA-2018".to_owned(),
        model: "text-embedding-3-small".to_owned(),
        dimensions,
        publication: Publication::Planned {
            repository: "klgraham/bifrost-benchmark-fixtures".to_owned(),
            prefix: format!("fixtures/fiqa/{dimensions}/{{manifest_sha256}}"),
            blocked_by: "Generate and publish the fixture".to_owned(),
        },
    }
}

fn published_artifact(dimensions: usize, manifest_sha256: String) -> Artifact {
    Artifact {
        id: format!("fiqa-{dimensions}"),
        dataset: "BEIR FiQA-2018".to_owned(),
        model: "text-embedding-3-small".to_owned(),
        dimensions,
        publication: Publication::Published {
            repository: "klgraham/bifrost-benchmark-fixtures".to_owned(),
            revision: hex('a', 40),
            prefix: format!("fixtures/fiqa/{dimensions}/{manifest_sha256}"),
            manifest_sha256,
        },
    }
}

fn write_valid_fixture(directory: &Path) -> FixtureManifest {
    let corpus_vectors = vector_bytes(&[[1.0, 0.0], [0.0, 1.0]]);
    let query_vectors = vector_bytes(&[[1.0, 0.0]]);
    let corpus_ids = b"d1\nd2\n";
    let query_ids = b"q1\n";
    let qrels = b"query-id\tcorpus-id\tscore\nq1\td1\t1\n";

    let files = [
        (
            FixtureFileKind::CorpusVectors,
            "corpus.f32",
            corpus_vectors.as_slice(),
        ),
        (
            FixtureFileKind::QueryVectors,
            "queries.f32",
            query_vectors.as_slice(),
        ),
        (
            FixtureFileKind::CorpusIds,
            "corpus-ids.txt",
            corpus_ids.as_slice(),
        ),
        (
            FixtureFileKind::QueryIds,
            "query-ids.txt",
            query_ids.as_slice(),
        ),
        (FixtureFileKind::Qrels, "qrels-test.tsv", qrels.as_slice()),
    ]
    .into_iter()
    .map(|(kind, path, bytes)| {
        fs::write(directory.join(path), bytes).expect("write fixture file");
        fixture_file(kind, path, bytes)
    })
    .collect();

    let manifest = FixtureManifest {
        schema: FIXTURE_SCHEMA.to_owned(),
        artifact_id: "fiqa-2".to_owned(),
        dataset: "BEIR FiQA-2018".to_owned(),
        source: SourceMetadata {
            url: "https://example.test/fiqa.zip".to_owned(),
            revision: "fiqa-2018".to_owned(),
            sha256: sha256_hex(b"source archive"),
            license: "CC-BY-SA-4.0".to_owned(),
        },
        selection: "full corpus and test-qrel queries".to_owned(),
        preprocessing: "sort by stable ID; trim title and text".to_owned(),
        corpus_input_sha256: input_sha256([("d1", "one"), ("d2", "two")]),
        query_input_sha256: input_sha256([("q1", "one?")]),
        requested_model: "text-embedding-3-small".to_owned(),
        returned_model: "text-embedding-3-small".to_owned(),
        dimensions: 2,
        normalization: "l2".to_owned(),
        derivation: None,
        generator_revision: hex('1', 40),
        representation: VectorRepresentation {
            scalar: "f32".to_owned(),
            endianness: "little".to_owned(),
        },
        counts: FixtureCounts {
            corpus: 2,
            queries: 1,
        },
        files,
    };
    manifest
        .write(directory.join(MANIFEST_FILE))
        .expect("write manifest");
    manifest
}

fn replace_file(
    directory: &Path,
    manifest: &mut FixtureManifest,
    kind: FixtureFileKind,
    bytes: &[u8],
) {
    let file = manifest
        .files
        .iter_mut()
        .find(|file| file.kind == kind)
        .expect("fixture file kind");
    fs::write(directory.join(&file.path), bytes).expect("replace fixture file");
    file.byte_count = u64::try_from(bytes.len()).expect("test byte count");
    file.sha256 = sha256_hex(bytes);
}

fn fixture_file(kind: FixtureFileKind, path: &str, bytes: &[u8]) -> FixtureFile {
    FixtureFile {
        kind,
        path: path.to_owned(),
        byte_count: u64::try_from(bytes.len()).expect("test byte count"),
        sha256: sha256_hex(bytes),
    }
}

fn vector_bytes<const DIMENSIONS: usize>(vectors: &[[f32; DIMENSIONS]]) -> Vec<u8> {
    vectors
        .iter()
        .flat_map(|vector| vector.iter())
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_json(path: &Path, value: &impl serde::Serialize) {
    let mut bytes = serde_json::to_vec_pretty(value).expect("serialize JSON");
    bytes.push(b'\n');
    fs::write(path, bytes).expect("write JSON");
}

fn assert_error_contains<T: std::fmt::Debug>(
    result: bifrost_benchmark_fixture::Result<T>,
    expected: &str,
) {
    let error = result.expect_err("expected failure").to_string();
    assert!(
        error.contains(expected),
        "expected error containing {expected:?}, got {error:?}"
    );
}

fn hex(character: char, count: usize) -> String {
    std::iter::repeat_n(character, count).collect()
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let counter = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "bifrost-fixture-{label}-{}-{counter}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
