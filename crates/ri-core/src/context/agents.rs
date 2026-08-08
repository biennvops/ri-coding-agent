use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use thiserror::Error;

pub const MAX_CONTEXT_FILE_BYTES: usize = 128 * 1024;
pub const MAX_TOTAL_CONTEXT_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextFileKind {
    GlobalAgents,
    Agents,
    AgentsOverride,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedContextFile {
    pub path: PathBuf,
    pub kind: ContextFileKind,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextBundle {
    pub launch_cwd: PathBuf,
    pub project_root: PathBuf,
    pub files: Vec<LoadedContextFile>,
    enabled: bool,
}

impl ContextBundle {
    pub fn disabled(launch_cwd: PathBuf, project_root: PathBuf) -> Self {
        Self {
            launch_cwd,
            project_root,
            files: Vec::new(),
            enabled: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn diagnostic(&self) -> String {
        if !self.enabled {
            return "context: disabled".to_owned();
        }
        if self.files.is_empty() {
            return "context: no AGENTS files loaded".to_owned();
        }

        let mut diagnostic = String::from("context:\n");
        for file in &self.files {
            diagnostic.push_str("  ");
            diagnostic.push_str(&file.path.display().to_string());
            diagnostic.push('\n');
        }
        diagnostic.pop();
        diagnostic
    }
}

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("could not canonicalize {label} {path}: {source}")]
    Canonicalize {
        label: &'static str,
        path: PathBuf,
        source: io::Error,
    },

    #[error("{label} is not a directory: {path}")]
    NotDirectory { label: &'static str, path: PathBuf },

    #[error("project root {project_root} is not an ancestor of launch cwd {launch_cwd}")]
    InvalidHierarchy {
        launch_cwd: PathBuf,
        project_root: PathBuf,
    },

    #[error("could not {operation} context file {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },

    #[error("context file {path} is not a regular file")]
    NotAFile { path: PathBuf },

    #[error(
        "context file {path} is {bytes} bytes; maximum is {limit} bytes; reduce the file or run with --no-context"
    )]
    FileTooLarge {
        path: PathBuf,
        bytes: u64,
        limit: usize,
    },

    #[error(
        "AGENTS context exceeds {limit} bytes total ({bytes} bytes); reduce the instruction files or run with --no-context"
    )]
    TotalTooLarge { bytes: usize, limit: usize },

    #[error("context file {path} is not valid UTF-8: {source}")]
    InvalidUtf8 {
        path: PathBuf,
        source: std::string::FromUtf8Error,
    },
}

pub fn load_context(
    launch_cwd: impl AsRef<Path>,
    project_root: impl AsRef<Path>,
) -> Result<ContextBundle, ContextError> {
    load_context_with_home(launch_cwd, project_root, home_directory().as_deref())
}

pub fn load_context_with_home(
    launch_cwd: impl AsRef<Path>,
    project_root: impl AsRef<Path>,
    home: Option<&Path>,
) -> Result<ContextBundle, ContextError> {
    let launch_cwd = canonicalize_directory(launch_cwd.as_ref(), "launch cwd")?;
    let project_root = canonicalize_directory(project_root.as_ref(), "project root")?;
    if !launch_cwd.starts_with(&project_root) {
        return Err(ContextError::InvalidHierarchy {
            launch_cwd,
            project_root,
        });
    }

    let mut loader = ContextLoader {
        files: Vec::new(),
        seen_paths: HashSet::new(),
        total_bytes: 0,
    };

    if let Some(home) = home {
        loader.load_directory(&home.join(".ri/agent"), ContextFileKind::GlobalAgents)?;
    }

    let relative_cwd = launch_cwd
        .strip_prefix(&project_root)
        .expect("launch cwd was checked to be inside project root");
    let mut directory = project_root.clone();
    loader.load_directory(&directory, ContextFileKind::Agents)?;
    for component in relative_cwd.components() {
        directory.push(component);
        loader.load_directory(&directory, ContextFileKind::Agents)?;
    }

    Ok(ContextBundle {
        launch_cwd,
        project_root,
        files: loader.files,
        enabled: true,
    })
}

fn canonicalize_directory(path: &Path, label: &'static str) -> Result<PathBuf, ContextError> {
    let canonical = fs::canonicalize(path).map_err(|source| ContextError::Canonicalize {
        label,
        path: path.to_path_buf(),
        source,
    })?;
    if !canonical.is_dir() {
        return Err(ContextError::NotDirectory {
            label,
            path: canonical,
        });
    }
    Ok(canonical)
}

fn home_directory() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

struct ContextLoader {
    files: Vec<LoadedContextFile>,
    seen_paths: HashSet<PathBuf>,
    total_bytes: usize,
}

impl ContextLoader {
    fn load_directory(
        &mut self,
        directory: &Path,
        kind: ContextFileKind,
    ) -> Result<(), ContextError> {
        let override_path = directory.join("AGENTS.override.md");
        if self.load_candidate(&override_path, ContextFileKind::AgentsOverride)? {
            return Ok(());
        }
        self.load_candidate(&directory.join("AGENTS.md"), kind)?;
        Ok(())
    }

    fn load_candidate(&mut self, path: &Path, kind: ContextFileKind) -> Result<bool, ContextError> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(source) => {
                return Err(ContextError::Io {
                    operation: "inspect",
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        if metadata.file_type().is_symlink() {
            match fs::metadata(path) {
                Ok(metadata) if metadata.is_file() => {}
                Ok(_) => {
                    return Err(ContextError::NotAFile {
                        path: path.to_path_buf(),
                    });
                }
                Err(source) => {
                    return Err(ContextError::Io {
                        operation: "resolve",
                        path: path.to_path_buf(),
                        source,
                    });
                }
            }
        } else if !metadata.is_file() {
            return Err(ContextError::NotAFile {
                path: path.to_path_buf(),
            });
        }

        let canonical_path = fs::canonicalize(path).map_err(|source| ContextError::Io {
            operation: "canonicalize",
            path: path.to_path_buf(),
            source,
        })?;
        if !self.seen_paths.insert(canonical_path.clone()) {
            return Ok(true);
        }

        let bytes = metadata.len();
        if bytes > MAX_CONTEXT_FILE_BYTES as u64 {
            return Err(ContextError::FileTooLarge {
                path: canonical_path,
                bytes,
                limit: MAX_CONTEXT_FILE_BYTES,
            });
        }
        let total_with_file = self.total_bytes.saturating_add(bytes as usize);
        if total_with_file > MAX_TOTAL_CONTEXT_BYTES {
            return Err(ContextError::TotalTooLarge {
                bytes: total_with_file,
                limit: MAX_TOTAL_CONTEXT_BYTES,
            });
        }

        let mut file = File::open(path).map_err(|source| ContextError::Io {
            operation: "open",
            path: canonical_path.clone(),
            source,
        })?;
        let mut raw = Vec::with_capacity(bytes as usize);
        file.by_ref()
            .take((MAX_CONTEXT_FILE_BYTES as u64).saturating_add(1))
            .read_to_end(&mut raw)
            .map_err(|source| ContextError::Io {
                operation: "read",
                path: canonical_path.clone(),
                source,
            })?;
        if raw.len() > MAX_CONTEXT_FILE_BYTES {
            return Err(ContextError::FileTooLarge {
                path: canonical_path,
                bytes: raw.len() as u64,
                limit: MAX_CONTEXT_FILE_BYTES,
            });
        }
        self.total_bytes = self.total_bytes.saturating_add(raw.len());
        if self.total_bytes > MAX_TOTAL_CONTEXT_BYTES {
            return Err(ContextError::TotalTooLarge {
                bytes: self.total_bytes,
                limit: MAX_TOTAL_CONTEXT_BYTES,
            });
        }
        let content = String::from_utf8(raw).map_err(|source| ContextError::InvalidUtf8 {
            path: canonical_path.clone(),
            source,
        })?;
        self.files.push(LoadedContextFile {
            path: canonical_path,
            kind,
            content,
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_global_root_nested_and_cwd_context_in_order() {
        let home = unique_test_dir("agents-order-home");
        let root = unique_test_dir("agents-order-root");
        let nested = root.join("crates");
        let cwd = nested.join("foo");
        fs::create_dir_all(&cwd).unwrap();
        fs::create_dir_all(home.join(".ri/agent")).unwrap();
        fs::write(home.join(".ri/agent/AGENTS.md"), "global").unwrap();
        fs::write(root.join("AGENTS.md"), "root").unwrap();
        fs::write(nested.join("AGENTS.md"), "nested").unwrap();
        fs::write(cwd.join("AGENTS.md"), "cwd").unwrap();

        let bundle = load_context_with_home(&cwd, &root, Some(&home)).unwrap();

        assert_eq!(
            bundle
                .files
                .iter()
                .map(|file| file.content.as_str())
                .collect::<Vec<_>>(),
            ["global", "root", "nested", "cwd"]
        );
        assert!(bundle.files.iter().all(|file| file.path.is_absolute()));
        remove_test_dir(home);
        remove_test_dir(root);
    }

    #[test]
    fn global_override_replaces_global_agents() {
        let home = unique_test_dir("agents-global-override-home");
        let root = unique_test_dir("agents-global-override-root");
        fs::create_dir_all(home.join(".ri/agent")).unwrap();
        fs::create_dir_all(&root).unwrap();
        fs::write(home.join(".ri/agent/AGENTS.md"), "ignored global").unwrap();
        fs::write(home.join(".ri/agent/AGENTS.override.md"), "global override").unwrap();
        fs::write(root.join("AGENTS.md"), "project root").unwrap();

        let bundle = load_context_with_home(&root, &root, Some(&home)).unwrap();

        assert_eq!(
            bundle
                .files
                .iter()
                .map(|file| file.content.as_str())
                .collect::<Vec<_>>(),
            ["global override", "project root"]
        );
        assert_eq!(bundle.files[0].kind, ContextFileKind::AgentsOverride);
        remove_test_dir(home);
        remove_test_dir(root);
    }

    #[test]
    fn override_replaces_only_the_same_directory_file() {
        let root = unique_test_dir("agents-override");
        let nested = root.join("crates");
        let cwd = nested.join("foo");
        fs::create_dir_all(&cwd).unwrap();
        fs::write(root.join("AGENTS.md"), "root").unwrap();
        fs::write(nested.join("AGENTS.md"), "nested").unwrap();
        fs::write(cwd.join("AGENTS.md"), "ignored").unwrap();
        fs::write(cwd.join("AGENTS.override.md"), "override").unwrap();

        let bundle = load_context_with_home(&cwd, &root, None).unwrap();

        assert_eq!(
            bundle
                .files
                .iter()
                .map(|file| file.content.as_str())
                .collect::<Vec<_>>(),
            ["root", "nested", "override"]
        );
        assert_eq!(bundle.files[2].kind, ContextFileKind::AgentsOverride);
        remove_test_dir(root);
    }

    #[test]
    fn unrelated_sibling_context_is_not_loaded() {
        let root = unique_test_dir("agents-sibling");
        let cwd = root.join("nested");
        let sibling = root.join("sibling");
        fs::create_dir_all(&cwd).unwrap();
        fs::create_dir_all(&sibling).unwrap();
        fs::write(root.join("AGENTS.md"), "root").unwrap();
        fs::write(sibling.join("AGENTS.md"), "sibling").unwrap();

        let bundle = load_context_with_home(&cwd, &root, None).unwrap();

        assert_eq!(bundle.files.len(), 1);
        assert_eq!(bundle.files[0].content, "root");
        remove_test_dir(root);
    }

    #[test]
    fn missing_context_is_normal_and_diagnostic_is_explicit() {
        let root = unique_test_dir("agents-none");
        fs::create_dir_all(&root).unwrap();

        let bundle = load_context_with_home(&root, &root, None).unwrap();

        assert!(bundle.files.is_empty());
        assert_eq!(bundle.diagnostic(), "context: no AGENTS files loaded");
        remove_test_dir(root);
    }

    #[test]
    fn disabled_context_has_a_distinct_diagnostic() {
        let root = unique_test_dir("agents-disabled");
        fs::create_dir_all(&root).unwrap();

        let bundle = ContextBundle::disabled(root.clone(), root);

        assert!(!bundle.is_enabled());
        assert_eq!(bundle.diagnostic(), "context: disabled");
        remove_test_dir(bundle.launch_cwd);
    }

    #[test]
    fn accepts_a_file_at_the_per_file_limit() {
        let root = unique_test_dir("agents-file-limit-exact");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("AGENTS.md"), vec![b'x'; MAX_CONTEXT_FILE_BYTES]).unwrap();

        let bundle = load_context_with_home(&root, &root, None).unwrap();

        assert_eq!(bundle.files.len(), 1);
        assert_eq!(bundle.files[0].content.len(), MAX_CONTEXT_FILE_BYTES);
        remove_test_dir(root);
    }

    #[test]
    fn rejects_a_file_over_the_per_file_limit() {
        let root = unique_test_dir("agents-file-limit");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("AGENTS.md"),
            vec![b'x'; MAX_CONTEXT_FILE_BYTES + 1],
        )
        .unwrap();

        let error = load_context_with_home(&root, &root, None).unwrap_err();

        assert!(matches!(error, ContextError::FileTooLarge { .. }));
        assert!(error.to_string().contains("--no-context"));
        remove_test_dir(root);
    }

    #[test]
    fn rejects_context_over_the_total_limit() {
        let root = unique_test_dir("agents-total-limit");
        let nested = root.join("nested");
        let cwd = nested.join("foo");
        fs::create_dir_all(&cwd).unwrap();
        fs::write(root.join("AGENTS.md"), vec![b'x'; MAX_CONTEXT_FILE_BYTES]).unwrap();
        fs::write(nested.join("AGENTS.md"), vec![b'y'; MAX_CONTEXT_FILE_BYTES]).unwrap();
        fs::write(cwd.join("AGENTS.md"), "z").unwrap();

        let error = load_context_with_home(&cwd, &root, None).unwrap_err();

        assert!(matches!(error, ContextError::TotalTooLarge { .. }));
        assert!(error.to_string().contains("256000") || error.to_string().contains("262144"));
        remove_test_dir(root);
    }

    #[test]
    fn rejects_invalid_utf8_without_truncating_or_replacing_it() {
        let root = unique_test_dir("agents-utf8");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("AGENTS.md"), [0xff, 0xfe]).unwrap();

        let error = load_context_with_home(&root, &root, None).unwrap_err();

        assert!(matches!(error, ContextError::InvalidUtf8 { .. }));
        assert!(error.to_string().contains("not valid UTF-8"));
        remove_test_dir(root);
    }

    #[cfg(unix)]
    #[test]
    fn canonical_paths_prevent_duplicate_context_from_symlink_aliases() {
        use std::os::unix::fs::symlink;

        let home = unique_test_dir("agents-dedup-home");
        let root = unique_test_dir("agents-dedup-root");
        let cwd = root.join("nested");
        fs::create_dir_all(&cwd).unwrap();
        fs::create_dir_all(home.join(".ri/agent")).unwrap();
        fs::write(root.join("AGENTS.md"), "same").unwrap();
        symlink(root.join("AGENTS.md"), home.join(".ri/agent/AGENTS.md")).unwrap();

        let bundle = load_context_with_home(&cwd, &root, Some(&home)).unwrap();

        assert_eq!(bundle.files.len(), 1);
        assert_eq!(bundle.files[0].content, "same");
        remove_test_dir(home);
        remove_test_dir(root);
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn remove_test_dir(path: PathBuf) {
        fs::remove_dir_all(path).unwrap();
    }
}
