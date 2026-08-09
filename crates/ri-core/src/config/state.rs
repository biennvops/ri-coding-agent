use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::fs::{atomic_write, AtomicWriteOptions};

use super::ModelRef;

pub const STATE_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecentModel {
    pub provider: String,
    pub model: String,
}

impl From<&ModelRef> for RecentModel {
    fn from(model: &ModelRef) -> Self {
        Self {
            provider: model.provider.clone(),
            model: model.model.clone(),
        }
    }
}

impl From<RecentModel> for ModelRef {
    fn from(model: RecentModel) -> Self {
        Self::new(model.provider, model.model)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRecentModel {
    #[serde(rename = "lastModel", default)]
    pub last_model: Option<RecentModel>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecentModelState {
    pub version: u32,
    #[serde(rename = "lastModel", default)]
    pub last_model: Option<RecentModel>,
    #[serde(default)]
    pub workspaces: BTreeMap<String, WorkspaceRecentModel>,
}

impl Default for RecentModelState {
    fn default() -> Self {
        Self {
            version: STATE_VERSION,
            last_model: None,
            workspaces: BTreeMap::new(),
        }
    }
}

impl RecentModelState {
    pub fn workspace_model(&self, workspace_id: &str) -> Option<&RecentModel> {
        self.workspaces
            .get(workspace_id)
            .and_then(|workspace| workspace.last_model.as_ref())
    }
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error("could not access recent state {path} while {operation}: {source}")]
    Io {
        path: PathBuf,
        operation: &'static str,
        source: io::Error,
    },

    #[error("could not parse recent state {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("unsupported ri state version {version}; expected {STATE_VERSION}")]
    UnsupportedVersion { version: u32 },
}

pub fn default_state_path() -> Option<PathBuf> {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".ri/agent/state.json"))
}

pub fn load_state(path: impl AsRef<Path>) -> Result<Option<RecentModelState>, StateError> {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Ok(None);
    }

    let lock = open_lock(&path, false)?;
    lock.lock_shared().map_err(|source| StateError::Io {
        path: lock_path(&path),
        operation: "locking",
        source,
    })?;
    let result = read_state(&path);
    let _ = lock.unlock();
    result.map(Some)
}

pub fn persist_recent_model(
    path: impl AsRef<Path>,
    workspace_id: &str,
    model: &ModelRef,
) -> Result<(), StateError> {
    let path = path.as_ref().to_path_buf();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    create_state_directory(parent, &path)?;

    let lock = open_lock(&path, true)?;
    lock.lock_exclusive().map_err(|source| StateError::Io {
        path: lock_path(&path),
        operation: "locking",
        source,
    })?;

    let mut state = if path.exists() {
        match read_state(&path) {
            Ok(state) => state,
            Err(StateError::Json { .. }) => {
                preserve_corrupt_state(&path)?;
                RecentModelState::default()
            }
            Err(error) => return Err(error),
        }
    } else {
        RecentModelState::default()
    };
    let recent = RecentModel::from(model);
    state.last_model = Some(recent.clone());
    state
        .workspaces
        .entry(workspace_id.to_owned())
        .or_default()
        .last_model = Some(recent);
    write_state_atomically(&path, &state)
}

fn read_state(path: &Path) -> Result<RecentModelState, StateError> {
    let mut source = String::new();
    File::open(path)
        .and_then(|mut file| file.read_to_string(&mut source))
        .map_err(|source| StateError::Io {
            path: path.to_path_buf(),
            operation: "reading",
            source,
        })?;
    let state: RecentModelState =
        serde_json::from_str(&source).map_err(|source| StateError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    if state.version != STATE_VERSION {
        return Err(StateError::UnsupportedVersion {
            version: state.version,
        });
    }
    Ok(state)
}

fn preserve_corrupt_state(path: &Path) -> Result<(), StateError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("state.json");
    let mut backup_path = path.with_file_name(format!("{file_name}.corrupt"));
    let mut suffix = 1;
    while backup_path.exists() {
        backup_path = path.with_file_name(format!("{file_name}.corrupt.{suffix}"));
        suffix += 1;
    }
    fs::rename(path, &backup_path).map_err(|source| StateError::Io {
        path: backup_path,
        operation: "preserving corrupt state",
        source,
    })
}

fn write_state_atomically(path: &Path, state: &RecentModelState) -> Result<(), StateError> {
    let mut source = serde_json::to_vec_pretty(state).expect("recent state is serializable");
    source.push(b'\n');
    atomic_write(
        path,
        &source,
        AtomicWriteOptions {
            preserve_permissions: true,
            private: true,
        },
    )
    .map_err(|source| StateError::Io {
        path: path.to_path_buf(),
        operation: "writing state",
        source,
    })
}

fn open_lock(path: &Path, create_parent: bool) -> Result<File, StateError> {
    if create_parent {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        create_state_directory(parent, path)?;
    }
    // The lock file is only a stable advisory-lock target. Its existence does
    // not indicate that another process currently owns the lock.
    let lock_path = lock_path(path);
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|source| StateError::Io {
            path: lock_path,
            operation: "opening lock",
            source,
        })
}

fn lock_path(path: &Path) -> PathBuf {
    let mut lock = path.as_os_str().to_os_string();
    lock.push(".lock");
    PathBuf::from(lock)
}

fn create_state_directory(parent: &Path, path: &Path) -> Result<(), StateError> {
    fs::create_dir_all(parent).map_err(|source| StateError::Io {
        path: path.to_path_buf(),
        operation: "creating state directory",
        source,
    })?;
    set_private_directory_permissions(parent, path)
}

fn set_private_directory_permissions(_path: &Path, _error_path: &Path) -> Result<(), StateError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(_path, fs::Permissions::from_mode(0o700)).map_err(|source| {
            StateError::Io {
                path: _error_path.to_path_buf(),
                operation: "setting state directory permissions",
                source,
            }
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_state_round_trips_global_and_workspace_models() {
        let root = unique_test_dir("state-round-trip");
        let path = root.join("state.json");
        let first_workspace = "a43bd19c5e1b4b2f89e114ad2b61ec33";
        let second_workspace = "b43bd19c5e1b4b2f89e114ad2b61ec33";
        let first_model = ModelRef::new("local", "qwen");
        let second_model = ModelRef::new("remote", "coder");

        persist_recent_model(&path, first_workspace, &first_model).expect("state should persist");
        persist_recent_model(&path, second_workspace, &second_model)
            .expect("state should preserve other workspaces");
        let state = load_state(&path)
            .expect("state should load")
            .expect("state should exist");

        assert_eq!(state.version, STATE_VERSION);
        assert_eq!(state.last_model, Some(RecentModel::from(&second_model)));
        assert_eq!(
            state.workspace_model(first_workspace),
            Some(&RecentModel::from(&first_model))
        );
        assert_eq!(
            state.workspace_model(second_workspace),
            Some(&RecentModel::from(&second_model))
        );
        remove_test_dir(root);
    }

    #[test]
    fn missing_state_is_normal() {
        let root = unique_test_dir("state-missing");
        let path = root.join("state.json");

        assert_eq!(load_state(&path).expect("missing state should load"), None);
        remove_test_dir(root);
    }

    #[test]
    fn state_write_failures_are_reported_without_partial_state() {
        let root = unique_test_dir("state-write-failure");
        fs::create_dir_all(&root).unwrap();
        let parent_file = root.join("not-a-directory");
        fs::write(&parent_file, "occupied").unwrap();
        let path = parent_file.join("state.json");

        let error = persist_recent_model(&path, "workspace", &ModelRef::new("p", "m"))
            .expect_err("a file cannot be used as the state directory");
        assert!(error.to_string().contains("state directory"));
        remove_test_dir(root);
    }

    #[test]
    fn malformed_or_unsupported_state_is_rejected() {
        let root = unique_test_dir("state-invalid");
        let path = root.join("state.json");
        fs::create_dir_all(&root).unwrap();

        fs::write(&path, "not json").unwrap();
        assert!(matches!(load_state(&path), Err(StateError::Json { .. })));

        let model = ModelRef::new("provider", "model");
        persist_recent_model(&path, "workspace", &model)
            .expect("malformed state should be recovered during persistence");
        let recovered = load_state(&path)
            .expect("recovered state should load")
            .expect("recovered state should exist");
        assert_eq!(recovered.version, STATE_VERSION);
        assert_eq!(recovered.last_model, Some(RecentModel::from(&model)));
        assert_eq!(
            fs::read_to_string(root.join("state.json.corrupt")).unwrap(),
            "not json"
        );

        let unsupported = r#"{"version":99,"lastModel":null,"workspaces":{}}"#;
        fs::write(&path, unsupported).unwrap();
        assert!(matches!(
            load_state(&path),
            Err(StateError::UnsupportedVersion { version: 99 })
        ));
        let error = persist_recent_model(&path, "workspace", &model)
            .expect_err("future state versions must not be overwritten");
        assert!(matches!(
            error,
            StateError::UnsupportedVersion { version: 99 }
        ));
        assert_eq!(fs::read_to_string(&path).unwrap(), unsupported);
        remove_test_dir(root);
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn remove_test_dir(path: PathBuf) {
        let _ = fs::remove_dir_all(path);
    }
}
