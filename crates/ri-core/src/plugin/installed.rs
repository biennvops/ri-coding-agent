use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::manifest::validate_plugin_id_value;
use super::{load_plugin_manifest, LoadedPluginManifest, PluginManifestError};

pub const PLUGIN_MANIFEST_FILENAME: &str = "plugin.json";

pub fn default_plugins_dir() -> Option<PathBuf> {
    crate::fs::home_directory().map(|home| home.join(".ri/agent/plugins"))
}

#[derive(Debug, thiserror::Error)]
pub enum InstalledPluginError {
    #[error("invalid plugin id {id:?}: {message}")]
    InvalidId { id: String, message: String },
    #[error("plugin {id:?} is not installed; expected manifest at {path}")]
    NotInstalled { id: String, path: PathBuf },
    #[error("could not load installed plugin {id:?} from {path}: {source}")]
    InvalidManifest {
        id: String,
        path: PathBuf,
        #[source]
        source: PluginManifestError,
    },
    #[error("installed plugin directory {directory} contains manifest id {actual:?}, expected {expected:?}")]
    IdentityMismatch {
        directory: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("plugin {id:?} was selected more than once")]
    DuplicateSelection { id: String },
}

pub fn resolve_installed_plugins(
    root: impl AsRef<Path>,
    ids: &[String],
) -> Result<Vec<LoadedPluginManifest>, InstalledPluginError> {
    let mut seen = HashSet::new();
    let mut manifests = Vec::new();
    for id in ids {
        validate_plugin_id_value(id).map_err(|message| InstalledPluginError::InvalidId {
            id: id.clone(),
            message: message.into(),
        })?;
        if !seen.insert(id) {
            return Err(InstalledPluginError::DuplicateSelection { id: id.clone() });
        }
        let directory = root.as_ref().join(id);
        let path = directory.join(PLUGIN_MANIFEST_FILENAME);
        let loaded = load_plugin_manifest(&path).map_err(|source| {
            if matches!(&source, PluginManifestError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound) {
                InstalledPluginError::NotInstalled { id: id.clone(), path: path.clone() }
            } else {
                InstalledPluginError::InvalidManifest { id: id.clone(), path: path.clone(), source }
            }
        })?;
        if loaded.manifest.id != *id {
            return Err(InstalledPluginError::IdentityMismatch {
                directory,
                expected: id.clone(),
                actual: loaded.manifest.id,
            });
        }
        manifests.push(loaded);
    }
    Ok(manifests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::Fixture;

    #[test]
    fn resolves_only_selected_ids_in_order() {
        let fixture = Fixture::scripted(false, vec![]);
        let root = fixture.load().directory;
        for id in ["a", "b"] {
            std::fs::create_dir(root.join(id)).unwrap();
            let mut manifest = fixture.load().manifest;
            manifest.id = id.into();
            std::fs::write(
                root.join(id).join("plugin.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
        }
        std::fs::create_dir(root.join("broken")).unwrap();
        std::fs::write(root.join("broken/plugin.json"), "invalid").unwrap();
        assert!(resolve_installed_plugins(&root, &[]).unwrap().is_empty());
        let loaded = resolve_installed_plugins(&root, &["b".into(), "a".into()]).unwrap();
        assert_eq!(
            loaded
                .iter()
                .map(|m| m.manifest.id.as_str())
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
        assert!(matches!(
            resolve_installed_plugins(&root, &["a".into(), "a".into()]),
            Err(InstalledPluginError::DuplicateSelection { .. })
        ));
        assert!(matches!(
            resolve_installed_plugins(&root, &["a/b".into()]),
            Err(InstalledPluginError::InvalidId { .. })
        ));
        assert!(matches!(
            resolve_installed_plugins(&root, &["broken".into()]),
            Err(InstalledPluginError::InvalidManifest { .. })
        ));
        assert!(
            matches!(resolve_installed_plugins(&root, &["missing".into()]), Err(InstalledPluginError::NotInstalled { path, .. }) if path == root.join("missing/plugin.json"))
        );
        std::fs::copy(root.join("a/plugin.json"), root.join("b/plugin.json")).unwrap();
        assert!(matches!(
            resolve_installed_plugins(&root, &["b".into()]),
            Err(InstalledPluginError::IdentityMismatch { .. })
        ));
    }
}
