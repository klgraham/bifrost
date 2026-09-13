use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::{ToolResult, context, error};

static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct ArtifactLock {
    file: File,
}

impl ArtifactLock {
    pub(crate) fn acquire(root: &Path, artifact_id: &str) -> ToolResult<Self> {
        prepare_mutation_root(root, "artifact lock root")?;
        let lock_directory = root.join(".locks");
        create_dir_all_under(root, &lock_directory, "artifact lock directory")?;
        let path = lock_directory.join(format!("{}.lock", safe_name(artifact_id)));
        ensure_safe_descendant(root, &path, "artifact lock file")?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        FileExt::lock_exclusive(&file)?;
        Ok(Self { file })
    }
}

impl Drop for ArtifactLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

pub(crate) fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

pub(crate) fn prepare_mutation_root(root: &Path, label: &str) -> ToolResult<()> {
    match fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return error(format!(
                "refusing to use symlinked {label}: {}",
                root.display()
            ));
        }
        Ok(metadata) if !metadata.is_dir() => {
            return error(format!("{label} is not a directory: {}", root.display()));
        }
        Ok(_) => {}
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(root)?;
        }
        Err(source) => return Err(source.into()),
    }
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return error(format!(
            "refusing unsafe {label} after creation: {}",
            root.display()
        ));
    }
    Ok(())
}

pub(crate) fn ensure_safe_descendant(root: &Path, path: &Path, label: &str) -> ToolResult<()> {
    prepare_mutation_root(root, "mutation root")?;
    let root_absolute = absolute_normalized(root)?;
    let path_absolute = absolute_normalized(path)?;
    let relative = path_absolute.strip_prefix(&root_absolute).map_err(|_| {
        crate::ToolError::new(format!(
            "{label} {} escapes mutation root {}",
            path.display(),
            root.display()
        ))
    })?;
    let canonical_root = fs::canonicalize(&root_absolute)?;
    let mut current = root_absolute;
    for component in relative.components() {
        let Component::Normal(segment) = component else {
            return error(format!("{label} contains an unsafe path component"));
        };
        current.push(segment);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return error(format!(
                    "refusing symlink in {label}: {}",
                    current.display()
                ));
            }
            Ok(_) => {
                let resolved = fs::canonicalize(&current)?;
                if !resolved.starts_with(&canonical_root) {
                    return error(format!(
                        "{label} {} resolves outside mutation root {}",
                        path.display(),
                        root.display()
                    ));
                }
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => break,
            Err(source) => return Err(source.into()),
        }
    }
    Ok(())
}

pub(crate) fn ensure_resolved_within_root(root: &Path, path: &Path, label: &str) -> ToolResult<()> {
    prepare_mutation_root(root, "cache root")?;
    let canonical_root = fs::canonicalize(root)?;
    let canonical_path = fs::canonicalize(path)?;
    if !canonical_path.starts_with(&canonical_root) {
        return error(format!(
            "{label} {} resolves outside cache root {}",
            path.display(),
            root.display()
        ));
    }
    Ok(())
}

pub(crate) fn create_dir_all_under(root: &Path, directory: &Path, label: &str) -> ToolResult<()> {
    ensure_safe_descendant(root, directory, label)?;
    let root_absolute = absolute_normalized(root)?;
    let directory_absolute = absolute_normalized(directory)?;
    let relative = directory_absolute
        .strip_prefix(&root_absolute)
        .expect("containment checked above");
    let mut current = root_absolute;
    for component in relative.components() {
        let Component::Normal(segment) = component else {
            return error(format!("{label} contains an unsafe path component"));
        };
        current.push(segment);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return error(format!(
                    "refusing symlink in {label}: {}",
                    current.display()
                ));
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => {
                return error(format!(
                    "{label} component is not a directory: {}",
                    current.display()
                ));
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&current)?;
                        if metadata.file_type().is_symlink() || !metadata.is_dir() {
                            return error(format!(
                                "refusing raced {label} component: {}",
                                current.display()
                            ));
                        }
                    }
                    Err(source) => return Err(source.into()),
                }
            }
            Err(source) => return Err(source.into()),
        }
    }
    ensure_safe_descendant(root, directory, label)
}

pub(crate) fn unique_stage(parent: &Path, artifact_id: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let counter = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    stage_namespace(parent, artifact_id).join(format!("{}-{nanos}-{counter}", std::process::id()))
}

pub(crate) fn clean_stale_stages(stage_root: &Path, artifact_id: &str) -> ToolResult<()> {
    prepare_mutation_root(stage_root, "staging root")?;
    let namespace = stage_namespace(stage_root, artifact_id);
    let Ok(metadata) = fs::symlink_metadata(&namespace) else {
        return Ok(());
    };
    ensure_safe_descendant(stage_root, &namespace, "artifact staging namespace")?;
    if metadata.is_dir() {
        remove_tree(&namespace)?;
    } else {
        fs::remove_file(&namespace)?;
    }
    Ok(())
}

fn stage_namespace(parent: &Path, artifact_id: &str) -> PathBuf {
    parent.join(format!(
        "{}--{}",
        safe_name(artifact_id),
        sha256_bytes(artifact_id.as_bytes())
    ))
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> ToolResult<()> {
    let parent = path.parent().ok_or_else(|| {
        crate::ToolError::new(format!("{} has no parent directory", path.display()))
    })?;
    fs::create_dir_all(parent)?;
    let counter = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{}.{}.{counter}.part",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> ToolResult<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn atomic_write_under(root: &Path, path: &Path, bytes: &[u8]) -> ToolResult<()> {
    let parent = path.parent().ok_or_else(|| {
        crate::ToolError::new(format!("{} has no parent directory", path.display()))
    })?;
    create_dir_all_under(root, parent, "atomic write parent")?;
    ensure_safe_descendant(root, path, "atomic write target")?;
    atomic_write(path, bytes)
}

pub(crate) fn install_directory(root: &Path, stage: &Path, destination: &Path) -> ToolResult<()> {
    let parent = destination.parent().ok_or_else(|| {
        crate::ToolError::new(format!("{} has no parent directory", destination.display()))
    })?;
    context(
        create_dir_all_under(root, parent, "fixture destination parent"),
        "checking fixture destination parent",
    )?;
    context(
        ensure_safe_descendant(root, stage, "staged fixture"),
        "checking staged fixture path",
    )?;
    context(
        ensure_safe_descendant(root, destination, "fixture destination"),
        "checking fixture destination path",
    )?;
    let quarantine = parent.join(format!(
        ".{}.{}.replaced",
        destination
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        std::process::id()
    ));
    if path_present(&quarantine) {
        ensure_safe_descendant(root, &quarantine, "fixture replacement quarantine")?;
        if quarantine.is_dir() {
            remove_tree(&quarantine)?;
        } else {
            fs::remove_file(&quarantine)?;
        }
    }
    if path_present(destination) {
        ensure_safe_descendant(root, destination, "fixture destination")?;
        set_read_only(destination, false)?;
        context(
            fs::rename(destination, &quarantine),
            "moving old fixture to quarantine",
        )?;
    }
    ensure_safe_descendant(root, stage, "staged fixture")?;
    ensure_safe_descendant(root, destination, "fixture destination")?;
    set_read_only(stage, false)?;
    if let Err(rename_error) = fs::rename(stage, destination) {
        let _ = set_read_only(stage, true);
        if path_present(&quarantine)
            && !path_present(destination)
            && fs::rename(&quarantine, destination).is_ok()
        {
            let _ = set_read_only(destination, true);
        }
        return error(format!(
            "moving staged fixture into place failed: {rename_error}"
        ));
    }
    if let Err(permission_error) = set_read_only(destination, true) {
        let _ = set_read_only(destination, false);
        let _ = fs::rename(destination, stage);
        if path_present(&quarantine)
            && !path_present(destination)
            && fs::rename(&quarantine, destination).is_ok()
        {
            let _ = set_read_only(destination, true);
        }
        return error(format!(
            "making installed fixture immutable failed: {permission_error}"
        ));
    }
    context(sync_directory(parent), "syncing fixture destination parent")?;
    if path_present(&quarantine) {
        ensure_safe_descendant(root, &quarantine, "fixture replacement quarantine")?;
        if quarantine.is_dir() {
            remove_tree(&quarantine)?;
        } else {
            fs::remove_file(quarantine)?;
        }
    }
    Ok(())
}

pub(crate) fn make_tree_read_only(root: &Path) -> ToolResult<()> {
    let mut directories = Vec::new();
    for entry in walk_tree(root)? {
        if fs::symlink_metadata(&entry)?.file_type().is_symlink() {
            return error(format!(
                "refusing to make symlink read-only inside fixture tree: {}",
                entry.display()
            ));
        }
        if entry.is_dir() {
            directories.push(entry);
        } else {
            set_read_only(&entry, true)?;
        }
    }
    directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in directories {
        set_read_only(&directory, true)?;
    }
    set_read_only(root, true)
}

pub(crate) fn remove_tree(root: &Path) -> ToolResult<()> {
    if fs::symlink_metadata(root)?.file_type().is_symlink() {
        fs::remove_file(root)?;
        return Ok(());
    }
    make_tree_writable(root)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

fn make_tree_writable(root: &Path) -> ToolResult<()> {
    set_read_only(root, false)?;
    for entry in walk_tree(root)? {
        set_read_only(&entry, false)?;
    }
    Ok(())
}

fn walk_tree(root: &Path) -> ToolResult<Vec<PathBuf>> {
    let mut result = Vec::new();
    let root_metadata = fs::symlink_metadata(root)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Ok(result);
    }
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                pending.push(path.clone());
            }
            result.push(path);
        }
    }
    Ok(result)
}

#[cfg(unix)]
fn set_read_only(path: &Path, read_only: bool) -> ToolResult<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        if read_only {
            return error(format!(
                "refusing to change permissions through symlink {}",
                path.display()
            ));
        }
        return Ok(());
    }
    let current = metadata.permissions().mode();
    let mode = if read_only {
        current & !0o222
    } else if metadata.is_dir() {
        current | 0o700
    } else {
        current | 0o600
    };
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_read_only(path: &Path, read_only: bool) -> ToolResult<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        if read_only {
            return error(format!(
                "refusing to change permissions through symlink {}",
                path.display()
            ));
        }
        return Ok(());
    }
    let mut permissions = metadata.permissions();
    permissions.set_readonly(read_only);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> ToolResult<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> ToolResult<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_lower(&digest.finalize()))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(is_lower_hex)
}

pub(crate) fn is_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(is_lower_hex)
}

fn is_lower_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

pub(crate) fn checked_relative_path(value: &str) -> ToolResult<PathBuf> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return error(format!("unsafe repository-relative path: {value:?}"));
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return error(format!("unsafe repository-relative path: {value:?}"));
        }
    }
    Ok(path.to_owned())
}

pub(crate) fn reject_overlapping_paths(
    left: &Path,
    left_name: &str,
    right: &Path,
    right_name: &str,
) -> ToolResult<()> {
    let left = comparable_path(left)?;
    let right = comparable_path(right)?;
    if left == right || left.starts_with(&right) || right.starts_with(&left) {
        return error(format!(
            "{left_name} ({}) and {right_name} ({}) must be separate, non-nested directories",
            left.display(),
            right.display()
        ));
    }
    Ok(())
}

fn comparable_path(path: &Path) -> ToolResult<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut missing = Vec::new();
    let mut existing = absolute.as_path();
    while !existing.exists() {
        let name = existing.file_name().ok_or_else(|| {
            crate::ToolError::new(format!("{} has no existing ancestor", path.display()))
        })?;
        missing.push(name.to_owned());
        existing = existing.parent().ok_or_else(|| {
            crate::ToolError::new(format!("{} has no existing ancestor", path.display()))
        })?;
    }
    let mut resolved = fs::canonicalize(existing)?;
    for segment in missing.iter().rev() {
        resolved.push(segment);
    }
    Ok(normalize_path(&resolved))
}

fn absolute_normalized(path: &Path) -> ToolResult<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    Ok(normalize_path(&absolute))
}

pub(crate) fn path_present(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
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

pub(crate) fn join_repo_path(prefix: &str, path: &str) -> ToolResult<String> {
    checked_relative_path(prefix)?;
    checked_relative_path(path)?;
    Ok(format!(
        "{}/{}",
        prefix.trim_end_matches('/'),
        path.trim_start_matches('/')
    ))
}

pub(crate) fn copy_checked(
    source: &Path,
    destination: &Path,
    expected_sha256: &str,
) -> ToolResult<()> {
    let actual = sha256_file(source)?;
    if actual != expected_sha256 {
        return error(format!(
            "checksum mismatch for {}: expected {expected_sha256}, got {actual}",
            source.display()
        ));
    }
    let parent = destination.parent().ok_or_else(|| {
        crate::ToolError::new(format!("{} has no parent directory", destination.display()))
    })?;
    fs::create_dir_all(parent)?;
    fs::copy(source, destination)?;
    File::open(destination)?.sync_all()?;
    Ok(())
}

pub(crate) fn copy_checked_under(
    root: &Path,
    source: &Path,
    destination: &Path,
    expected_sha256: &str,
) -> ToolResult<()> {
    let parent = destination.parent().ok_or_else(|| {
        crate::ToolError::new(format!("{} has no parent directory", destination.display()))
    })?;
    create_dir_all_under(root, parent, "fixture copy parent")?;
    ensure_safe_descendant(root, destination, "fixture copy target")?;
    copy_checked(source, destination, expected_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_that_can_escape_a_repository_prefix() {
        for path in ["", "/absolute", "../escape", "nested/../../escape", "."] {
            assert!(checked_relative_path(path).is_err(), "accepted {path:?}");
        }
        assert_eq!(
            checked_relative_path("nested/file.json").unwrap(),
            Path::new("nested/file.json")
        );
    }

    #[test]
    fn identifies_only_full_commit_hashes() {
        assert!(is_commit_sha(&"a".repeat(40)));
        assert!(!is_commit_sha(&"a".repeat(39)));
        assert!(!is_commit_sha(&format!("{}g", "a".repeat(39))));
        assert!(!is_commit_sha(&"A".repeat(40)));
    }

    #[test]
    fn valid_dotted_artifact_ids_do_not_collide_with_dashes() {
        assert_eq!(safe_name("foo.bar"), "foo.bar");
        assert_ne!(safe_name("foo.bar"), safe_name("foo-bar"));
    }

    #[test]
    fn stale_stage_cleanup_is_scoped_to_the_exact_artifact_id() {
        let temp = tempfile::tempdir().unwrap();
        let stage_root = temp.path().join("staging");
        let foo_stage = unique_stage(&stage_root, "foo");
        let foo_bar_stage = unique_stage(&stage_root, "foo-bar");
        fs::create_dir_all(&foo_stage).unwrap();
        fs::create_dir_all(&foo_bar_stage).unwrap();
        fs::write(foo_bar_stage.join("keep"), b"keep").unwrap();

        clean_stale_stages(&stage_root, "foo").unwrap();

        assert!(!foo_stage.exists());
        assert_eq!(fs::read(foo_bar_stage.join("keep")).unwrap(), b"keep");
    }

    #[test]
    fn overlap_checks_resolve_existing_symlink_ancestors() {
        let temp = tempfile::tempdir().unwrap();
        let real = temp.path().join("real");
        fs::create_dir(&real).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real, temp.path().join("alias")).unwrap();
            assert!(
                reject_overlapping_paths(
                    &real.join("workspace"),
                    "workspace",
                    &temp.path().join("alias/workspace/output"),
                    "output",
                )
                .is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn cache_snapshot_symlinks_must_resolve_inside_the_cache_root() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let blob = cache.join("blobs/content");
        let snapshot = cache.join("snapshots/commit/file");
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        fs::write(&blob, b"content").unwrap();
        std::os::unix::fs::symlink("../../blobs/content", &snapshot).unwrap();

        ensure_resolved_within_root(&cache, &snapshot, "snapshot").unwrap();

        let outside = temp.path().join("outside");
        fs::write(&outside, b"outside").unwrap();
        let escaping = cache.join("snapshots/commit/escape");
        std::os::unix::fs::symlink(&outside, &escaping).unwrap();
        assert!(ensure_resolved_within_root(&cache, &escaping, "snapshot").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn removing_a_corrupt_tree_does_not_follow_or_chmod_symlinks() {
        let temp = tempfile::tempdir().unwrap();
        let outside = temp.path().join("outside");
        let corrupt = temp.path().join("corrupt");
        fs::create_dir(&outside).unwrap();
        fs::create_dir(&corrupt).unwrap();
        let outside_file = outside.join("keep.txt");
        fs::write(&outside_file, b"keep").unwrap();
        let before = fs::metadata(&outside_file).unwrap().permissions().mode();
        std::os::unix::fs::symlink(&outside, corrupt.join("escape")).unwrap();
        remove_tree(&corrupt).unwrap();
        assert_eq!(fs::read(&outside_file).unwrap(), b"keep");
        assert_eq!(
            fs::metadata(&outside_file).unwrap().permissions().mode(),
            before
        );
    }
}
