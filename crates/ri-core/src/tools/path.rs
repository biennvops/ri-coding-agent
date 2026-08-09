use std::fs;
use std::path::{Path, PathBuf};

use super::{ToolContext, ToolError};

pub(crate) fn resolve_existing(
    context: &ToolContext,
    requested: &str,
) -> Result<PathBuf, ToolError> {
    let candidate = candidate_path(context, requested)?;
    let canonical = fs::canonicalize(&candidate).map_err(|error| {
        ToolError::Failed(format!("could not resolve path {requested:?}: {error}"))
    })?;
    ensure_inside(context, &canonical, requested)?;
    Ok(canonical)
}

pub(crate) fn resolve_for_write(
    context: &ToolContext,
    requested: &str,
) -> Result<PathBuf, ToolError> {
    let requested_is_absolute = Path::new(requested).is_absolute();
    let candidate = candidate_path(context, requested)?;
    let lexical = normalize_lexical(&candidate)
        .ok_or_else(|| ToolError::Failed(format!("path {requested:?} escapes the workspace")))?;
    if !requested_is_absolute {
        ensure_inside(context, &lexical, requested)?;
    }

    match fs::symlink_metadata(&candidate) {
        Ok(_) => {
            let canonical = fs::canonicalize(&candidate).map_err(|error| {
                ToolError::Failed(format!("could not resolve path {requested:?}: {error}"))
            })?;
            ensure_inside(context, &canonical, requested)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = nearest_existing_parent(&candidate).ok_or_else(|| {
                ToolError::Failed(format!(
                    "could not resolve parent directory for {requested:?}"
                ))
            })?;
            let canonical_parent = fs::canonicalize(parent).map_err(|error| {
                ToolError::Failed(format!(
                    "could not resolve parent directory for {requested:?}: {error}"
                ))
            })?;
            ensure_inside(context, &canonical_parent, requested)?;
        }
        Err(error) => {
            return Err(ToolError::Failed(format!(
                "could not inspect path {requested:?}: {error}"
            )));
        }
    }

    Ok(candidate)
}

fn candidate_path(context: &ToolContext, requested: &str) -> Result<PathBuf, ToolError> {
    if requested.trim().is_empty() {
        return Err(ToolError::InvalidArguments(
            "path must not be empty".to_owned(),
        ));
    }

    let path = Path::new(requested);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(context.workspace_root.join(path))
    }
}

fn normalize_lexical(path: &Path) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
            std::path::Component::Normal(component) => normalized.push(component),
        }
    }
    Some(normalized)
}

fn nearest_existing_parent(path: &Path) -> Option<&Path> {
    let mut current = path;
    loop {
        if fs::symlink_metadata(current).is_ok() {
            return Some(current);
        }
        current = current.parent()?;
    }
}

fn ensure_inside(context: &ToolContext, path: &Path, requested: &str) -> Result<(), ToolError> {
    if path_is_inside(&context.workspace_root, path) {
        Ok(())
    } else {
        Err(ToolError::Failed(format!(
            "path {requested:?} escapes the workspace"
        )))
    }
}

#[cfg(not(windows))]
fn path_is_inside(root: &Path, path: &Path) -> bool {
    path.starts_with(root)
}

#[cfg(windows)]
fn path_is_inside(root: &Path, path: &Path) -> bool {
    let mut root_components = root.components();
    let mut path_components = path.components();
    root_components.all(|root_component| {
        path_components.next().is_some_and(|path_component| {
            components_equal_ignore_case(root_component, path_component)
        })
    })
}

#[cfg(windows)]
fn components_equal_ignore_case(
    left: std::path::Component<'_>,
    right: std::path::Component<'_>,
) -> bool {
    use std::path::Component;

    match (left, right) {
        (Component::Prefix(left), Component::Prefix(right)) => left
            .as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy()),
        (Component::RootDir, Component::RootDir) => true,
        (Component::CurDir, Component::CurDir) => true,
        (Component::ParentDir, Component::ParentDir) => true,
        (Component::Normal(left), Component::Normal(right)) => left
            .to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn rejects_parent_escape_and_accepts_nested_new_path() {
        let root = unique_test_dir("path");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();

        assert!(resolve_for_write(&context, "../outside.txt").is_err());
        assert!(resolve_for_write(&context, "missing/../../outside.txt").is_err());
        assert!(resolve_for_write(&context, "nested/file.txt").is_ok());
        assert!(
            resolve_for_write(&context, &root.join("absolute.txt").display().to_string()).is_ok()
        );
        assert!(resolve_for_write(
            &context,
            &root
                .parent()
                .unwrap()
                .join("outside.txt")
                .display()
                .to_string()
        )
        .is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_use_case_insensitive_component_containment() {
        let root = unique_test_dir("windows-paths");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let differently_cased = root.to_string_lossy().to_uppercase();

        assert!(
            resolve_for_write(&context, &format!(r"{differently_cased}\nested\file.txt")).is_ok()
        );
        assert!(resolve_for_write(&context, r"Z:\outside.txt").is_err());
        assert!(resolve_for_write(&context, r"nested\..\..\outside.txt").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = unique_test_dir("symlink");
        let outside = unique_test_dir("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, root.join("link")).unwrap();
        let context = ToolContext::new(&root).unwrap();

        assert!(resolve_existing(&context, "link/secret.txt").is_err());
        assert!(resolve_for_write(&context, "link/new.txt").is_err());

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
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
}
