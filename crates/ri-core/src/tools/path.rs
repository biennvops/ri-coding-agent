use std::fs;
use std::path::{Path, PathBuf};

use super::{ToolContext, ToolError};

pub(crate) fn resolve_existing(
    context: &ToolContext,
    requested: &str,
) -> Result<PathBuf, ToolError> {
    let candidate = candidate_path(context, requested)?;
    fs::canonicalize(&candidate).map_err(|error| {
        ToolError::Failed(format!("could not resolve path {requested:?}: {error}"))
    })
}

pub(crate) fn resolve_for_write(
    context: &ToolContext,
    requested: &str,
) -> Result<PathBuf, ToolError> {
    let candidate = candidate_path(context, requested)?;
    normalize_lexical(&candidate)
        .ok_or_else(|| ToolError::Failed(format!("could not resolve path {requested:?}")))
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn resolves_workspace_relative_parent_and_absolute_paths() {
        let root = unique_test_dir("path");
        let outside = unique_test_dir("path-outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let outside_file = outside.join("existing.txt");
        fs::write(&outside_file, "outside").unwrap();
        let context = ToolContext::new(&root).unwrap();

        assert_eq!(
            resolve_for_write(&context, "nested/file.txt").unwrap(),
            context.workspace_root.join("nested/file.txt")
        );
        assert_eq!(
            resolve_for_write(&context, "../outside.txt").unwrap(),
            context.workspace_root.parent().unwrap().join("outside.txt")
        );
        assert_eq!(
            resolve_for_write(&context, &outside.join("new.txt").display().to_string()).unwrap(),
            outside.join("new.txt")
        );
        assert_eq!(
            resolve_existing(&context, &outside_file.display().to_string()).unwrap(),
            fs::canonicalize(&outside_file).unwrap()
        );

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn resolves_symlinks_outside_workspace() {
        use std::os::unix::fs::symlink;

        let root = unique_test_dir("symlink");
        let outside = unique_test_dir("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, root.join("link")).unwrap();
        let context = ToolContext::new(&root).unwrap();

        assert_eq!(
            resolve_existing(&context, "link/secret.txt").unwrap(),
            fs::canonicalize(outside.join("secret.txt")).unwrap()
        );
        assert!(resolve_for_write(&context, "link/new.txt").is_ok());

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
