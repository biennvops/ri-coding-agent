use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
pub(crate) struct AtomicWriteOptions {
    pub(crate) preserve_permissions: bool,
    pub(crate) private: bool,
}

impl Default for AtomicWriteOptions {
    fn default() -> Self {
        Self {
            preserve_permissions: true,
            private: false,
        }
    }
}

/// Write a file through a same-directory temporary and replace the destination.
///
/// The temporary is created with `create_new`, so a stale temporary from an
/// earlier crash cannot cause a later write to overwrite it or fail due to a
/// predictable name collision. The destination is never removed before the
/// replacement operation.
pub(crate) fn atomic_write(
    destination: &Path,
    bytes: &[u8],
    options: AtomicWriteOptions,
) -> io::Result<()> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let existing_permissions = options
        .preserve_permissions
        .then(|| fs::metadata(destination).ok())
        .flatten()
        .map(|metadata| metadata.permissions());
    let temporary_path = temporary_path(parent);

    let result = (|| {
        let mut temporary = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        if options.private {
            set_private_file_permissions(&temporary)?;
        }
        temporary.write_all(bytes)?;
        if let Some(permissions) = existing_permissions {
            temporary.set_permissions(permissions)?;
        }
        temporary.sync_all()?;
        drop(temporary);

        replace_file(&temporary_path, destination)?;
        sync_parent_directory(parent)
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

pub(crate) fn temporary_path(parent: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".ri-temp-{}-{timestamp}-{counter}",
        std::process::id()
    ))
}

#[cfg(not(windows))]
fn replace_file(temporary: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(temporary, destination)
}

#[cfg(windows)]
fn replace_file(temporary: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const ERROR_FILE_NOT_FOUND: i32 = 2;
    const REPLACEFILE_WRITE_THROUGH: u32 = 1;
    const MOVEFILE_REPLACE_EXISTING: u32 = 1;
    const MOVEFILE_WRITE_THROUGH: u32 = 8;

    let temporary = wide_path(temporary);
    let destination = wide_path(destination);
    // SAFETY: both vectors are NUL-terminated UTF-16 paths owned for the
    // duration of the call; the optional backup and reserved pointers are
    // explicitly null as required by ReplaceFileW.
    let replaced = unsafe {
        ReplaceFileW(
            destination.as_ptr(),
            temporary.as_ptr(),
            std::ptr::null(),
            REPLACEFILE_WRITE_THROUGH,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if replaced != 0 {
        return Ok(());
    }

    let replace_error = io::Error::last_os_error();
    if replace_error.raw_os_error() != Some(ERROR_FILE_NOT_FOUND) {
        return Err(replace_error);
    }

    // ReplaceFileW requires an existing destination. MoveFileExW handles the
    // first write while retaining same-volume replacement semantics.
    // SAFETY: both vectors are NUL-terminated UTF-16 paths owned for the
    // duration of the call.
    let moved = unsafe {
        MoveFileExW(
            temporary.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved != 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn wide_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn ReplaceFileW(
        replaced_file_name: *const u16,
        replacement_file_name: *const u16,
        backup_file_name: *const u16,
        replace_flags: u32,
        exclude: *mut std::ffi::c_void,
        reserved: *mut std::ffi::c_void,
    ) -> i32;
    fn MoveFileExW(existing_file_name: *const u16, new_file_name: *const u16, flags: u32) -> i32;
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> io::Result<()> {
    Ok(())
}

fn set_private_file_permissions(file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_new_and_existing_files_without_a_missing_destination_window() {
        let root = unique_test_dir("atomic-replace");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("state.json");

        atomic_write(&path, "héllo".as_bytes(), AtomicWriteOptions::default()).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "héllo");

        atomic_write(
            &path,
            &vec![b'x'; 128 * 1024],
            AtomicWriteOptions::default(),
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap().len(), 128 * 1024);
        assert!(!fs::read_dir(&root).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".ri-temp-")));
        remove_test_dir(root);
    }

    #[cfg(unix)]
    #[test]
    fn replacement_preserves_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let root = unique_test_dir("atomic-permissions");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("file.txt");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();

        atomic_write(&path, "new".as_bytes(), AtomicWriteOptions::default()).unwrap();

        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
        remove_test_dir(root);
    }

    #[test]
    fn replacement_failure_removes_the_temporary_file() {
        let root = unique_test_dir("atomic-failure-cleanup");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("destination");
        fs::create_dir(&destination).unwrap();

        assert!(atomic_write(&destination, b"current", AtomicWriteOptions::default()).is_err());
        assert!(!fs::read_dir(&root).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".ri-temp-")));
        remove_test_dir(root);
    }

    #[test]
    fn stale_temporary_names_do_not_block_a_new_write() {
        let root = unique_test_dir("atomic-stale-temp");
        fs::create_dir_all(&root).unwrap();
        let stale = temporary_path(&root);
        fs::write(&stale, "stale").unwrap();
        let destination = root.join("file.txt");

        atomic_write(&destination, b"current", AtomicWriteOptions::default()).unwrap();

        assert_eq!(fs::read_to_string(&destination).unwrap(), "current");
        assert_eq!(fs::read_to_string(stale).unwrap(), "stale");
        remove_test_dir(root);
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn remove_test_dir(path: PathBuf) {
        let _ = fs::remove_dir_all(path);
    }
}
