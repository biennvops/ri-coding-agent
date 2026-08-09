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
    ensure_canonical_inside(context, &canonical, requested)?;
    Ok(canonical)
}

pub(crate) fn resolve_for_write(
    context: &ToolContext,
    requested: &str,
) -> Result<PathBuf, ToolError> {
    let candidate = candidate_path(context, requested)?;
    let lexical = normalize_lexical(&candidate)
        .ok_or_else(|| ToolError::Failed(format!("path {requested:?} escapes the workspace")))?;
    if !Path::new(requested).is_absolute() {
        ensure_lexically_inside(context, &lexical, requested)?;
    }

    match fs::symlink_metadata(&lexical) {
        Ok(_) => {
            let canonical = fs::canonicalize(&lexical).map_err(|error| {
                ToolError::Failed(format!("could not resolve path {requested:?}: {error}"))
            })?;
            ensure_canonical_inside(context, &canonical, requested)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = nearest_existing_parent(&lexical).ok_or_else(|| {
                ToolError::Failed(format!(
                    "could not resolve parent directory for {requested:?}"
                ))
            })?;
            let canonical_parent = fs::canonicalize(parent).map_err(|error| {
                ToolError::Failed(format!(
                    "could not resolve parent directory for {requested:?}: {error}"
                ))
            })?;
            ensure_canonical_inside(context, &canonical_parent, requested)?;
        }
        Err(error) => {
            return Err(ToolError::Failed(format!(
                "could not inspect path {requested:?}: {error}"
            )));
        }
    }

    Ok(lexical)
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

fn ensure_lexically_inside(
    context: &ToolContext,
    path: &Path,
    requested: &str,
) -> Result<(), ToolError> {
    if lexical_path_is_inside(&context.workspace_root, path) {
        Ok(())
    } else {
        Err(ToolError::Failed(format!(
            "path {requested:?} escapes the workspace"
        )))
    }
}

fn ensure_canonical_inside(
    context: &ToolContext,
    path: &Path,
    requested: &str,
) -> Result<(), ToolError> {
    if path.starts_with(&context.workspace_root) {
        Ok(())
    } else {
        Err(ToolError::Failed(format!(
            "path {requested:?} escapes the workspace"
        )))
    }
}

#[cfg(not(windows))]
fn lexical_path_is_inside(root: &Path, path: &Path) -> bool {
    path.starts_with(root)
}

#[cfg(windows)]
fn lexical_path_is_inside(root: &Path, path: &Path) -> bool {
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
        (Component::Prefix(left), Component::Prefix(right)) => {
            prefixes_equal_ignore_case(left.kind(), right.kind())
        }
        (Component::RootDir, Component::RootDir) => true,
        (Component::CurDir, Component::CurDir) => true,
        (Component::ParentDir, Component::ParentDir) => true,
        (Component::Normal(left), Component::Normal(right)) => left
            .to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy()),
        _ => false,
    }
}

#[cfg(windows)]
fn prefixes_equal_ignore_case(left: std::path::Prefix<'_>, right: std::path::Prefix<'_>) -> bool {
    use std::path::Prefix;

    match (left, right) {
        (Prefix::Disk(left), Prefix::Disk(right))
        | (Prefix::Disk(left), Prefix::VerbatimDisk(right))
        | (Prefix::VerbatimDisk(left), Prefix::Disk(right))
        | (Prefix::VerbatimDisk(left), Prefix::VerbatimDisk(right)) => {
            left.eq_ignore_ascii_case(&right)
        }
        (Prefix::UNC(left_server, left_share), Prefix::UNC(right_server, right_share))
        | (Prefix::UNC(left_server, left_share), Prefix::VerbatimUNC(right_server, right_share))
        | (Prefix::VerbatimUNC(left_server, left_share), Prefix::UNC(right_server, right_share))
        | (
            Prefix::VerbatimUNC(left_server, left_share),
            Prefix::VerbatimUNC(right_server, right_share),
        ) => {
            left_server
                .to_string_lossy()
                .eq_ignore_ascii_case(&right_server.to_string_lossy())
                && left_share
                    .to_string_lossy()
                    .eq_ignore_ascii_case(&right_share.to_string_lossy())
        }
        (Prefix::Verbatim(left), Prefix::Verbatim(right))
        | (Prefix::DeviceNS(left), Prefix::DeviceNS(right)) => left
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
        assert!(resolve_for_write(
            &context,
            &context
                .workspace_root
                .join("absolute.txt")
                .display()
                .to_string()
        )
        .is_ok());
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

    #[cfg(windows)]
    #[test]
    fn canonical_containment_does_not_fold_normal_components() {
        let root = unique_test_dir("canonical-case");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let name = context
            .workspace_root
            .file_name()
            .unwrap()
            .to_string_lossy();
        let differently_cased = name
            .chars()
            .map(|character| character.to_ascii_uppercase())
            .collect::<String>();
        assert_ne!(name, differently_cased);
        let sibling = context
            .workspace_root
            .parent()
            .unwrap()
            .join(differently_cased)
            .join("file.txt");

        assert!(ensure_canonical_inside(&context, &sibling, "sibling").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn allows_alias_that_resolves_inside_workspace() {
        use std::os::unix::fs::symlink;

        let root = unique_test_dir("alias-inside");
        let alias = root.with_file_name(format!("{}-alias", root.file_name().unwrap().display()));
        fs::create_dir_all(&root).unwrap();
        symlink(&root, &alias).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let requested = alias.join("nested/file.txt");

        assert!(resolve_for_write(&context, &requested.display().to_string()).is_ok());
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(alias).unwrap();
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
