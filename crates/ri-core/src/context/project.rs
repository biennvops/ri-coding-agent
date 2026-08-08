use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectLayout {
    pub launch_cwd: PathBuf,
    pub project_root: PathBuf,
}

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("could not canonicalize launch cwd {path}: {source}")]
    Canonicalize { path: PathBuf, source: io::Error },

    #[error("launch cwd is not a directory: {path}")]
    NotDirectory { path: PathBuf },

    #[error("could not inspect project marker {path}: {source}")]
    MarkerIo { path: PathBuf, source: io::Error },
}

pub fn discover_project(start: impl AsRef<Path>) -> Result<ProjectLayout, ProjectError> {
    let launch_cwd = canonicalize_launch_cwd(start)?;
    let project_root = discover_project_root_from_canonical(&launch_cwd)?;
    Ok(ProjectLayout {
        launch_cwd,
        project_root,
    })
}

pub fn canonicalize_launch_cwd(start: impl AsRef<Path>) -> Result<PathBuf, ProjectError> {
    let path = start.as_ref().to_path_buf();
    let launch_cwd = fs::canonicalize(&path).map_err(|source| ProjectError::Canonicalize {
        path: path.clone(),
        source,
    })?;
    if !launch_cwd.is_dir() {
        return Err(ProjectError::NotDirectory { path: launch_cwd });
    }
    Ok(launch_cwd)
}

pub fn discover_project_root(start: impl AsRef<Path>) -> Result<PathBuf, ProjectError> {
    let launch_cwd = canonicalize_launch_cwd(start)?;
    discover_project_root_from_canonical(&launch_cwd)
}

fn discover_project_root_from_canonical(launch_cwd: &Path) -> Result<PathBuf, ProjectError> {
    let mut current = launch_cwd;
    loop {
        if has_git_marker(current)? {
            return Ok(current.to_path_buf());
        }

        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent;
    }

    Ok(launch_cwd.to_path_buf())
}

fn has_git_marker(directory: &Path) -> Result<bool, ProjectError> {
    let marker = directory.join(".git");
    match fs::metadata(&marker) {
        Ok(metadata) => Ok(metadata.is_file() || metadata.is_dir()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(ProjectError::MarkerIo {
            path: marker,
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_directory_with_git_is_project_root() {
        let root = unique_test_dir("project-root");
        fs::create_dir_all(root.join(".git")).unwrap();

        let layout = discover_project(&root).unwrap();

        assert_eq!(layout.launch_cwd, fs::canonicalize(&root).unwrap());
        assert_eq!(layout.project_root, layout.launch_cwd);
        remove_test_dir(root);
    }

    #[test]
    fn nested_launch_directory_uses_nearest_git_ancestor() {
        let outer = unique_test_dir("project-nearest");
        let inner = outer.join("workspace");
        let launch = inner.join("crates/foo");
        fs::create_dir_all(&launch).unwrap();
        fs::create_dir_all(outer.join(".git")).unwrap();
        fs::create_dir_all(inner.join(".git")).unwrap();

        let layout = discover_project(&launch).unwrap();

        assert_eq!(layout.project_root, fs::canonicalize(&inner).unwrap());
        remove_test_dir(outer);
    }

    #[test]
    fn git_file_marks_a_project_root() {
        let root = unique_test_dir("project-worktree");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join(".git"), "gitdir: /some/worktree\n").unwrap();

        assert_eq!(
            discover_project_root(&root).unwrap(),
            fs::canonicalize(&root).unwrap()
        );
        remove_test_dir(root);
    }

    #[test]
    fn missing_git_uses_launch_directory() {
        let root = unique_test_dir("project-no-git");
        let launch = root.join("nested");
        fs::create_dir_all(&launch).unwrap();

        assert_eq!(
            discover_project_root(&launch).unwrap(),
            fs::canonicalize(&launch).unwrap()
        );
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
