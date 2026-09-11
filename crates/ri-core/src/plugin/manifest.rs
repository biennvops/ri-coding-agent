use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PLUGIN_MANIFEST_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PluginManifest {
    pub manifest_version: u32,
    pub id: String,
    pub name: String,
    pub version: String,
    pub protocol_version: String,
    pub entrypoint: PluginEntrypoint,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PluginEntrypoint {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct LoadedPluginManifest {
    pub manifest: PluginManifest,
    pub path: PathBuf,
    pub directory: PathBuf,
}

#[derive(Debug, Error)]
pub enum PluginManifestError {
    #[error("manifest {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("manifest JSON {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("manifest {path}: unsupported manifest version {value}")]
    UnsupportedManifestVersion { path: PathBuf, value: u32 },
    #[error("manifest {path}: unsupported protocol version {value}")]
    UnsupportedProtocolVersion { path: PathBuf, value: String },
    #[error("manifest {path}: invalid {field}: {value:?}")]
    InvalidField {
        path: PathBuf,
        field: &'static str,
        value: String,
    },
}

impl PluginManifest {
    pub(crate) fn validate(&self, path: &Path) -> Result<(), PluginManifestError> {
        if self.manifest_version != PLUGIN_MANIFEST_VERSION {
            return Err(PluginManifestError::UnsupportedManifestVersion {
                path: path.into(),
                value: self.manifest_version,
            });
        }
        if self.protocol_version != "ri.plugin.v1" {
            return Err(PluginManifestError::UnsupportedProtocolVersion {
                path: path.into(),
                value: self.protocol_version.clone(),
            });
        }
        let valid_id = !self.id.is_empty()
            && self.id.len() <= 128
            && self
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b".-_".contains(&b))
            && self
                .id
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
        for (field, value, valid) in [
            ("id", &self.id, valid_id),
            ("name", &self.name, !self.name.trim().is_empty()),
            ("version", &self.version, !self.version.trim().is_empty()),
            (
                "entrypoint.command",
                &self.entrypoint.command,
                !self.entrypoint.command.trim().is_empty(),
            ),
        ] {
            if !valid {
                return Err(PluginManifestError::InvalidField {
                    path: path.into(),
                    field,
                    value: value.clone(),
                });
            }
        }
        Ok(())
    }
}

pub fn load_plugin_manifest(
    path: impl AsRef<Path>,
) -> Result<LoadedPluginManifest, PluginManifestError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|source| PluginManifestError::Io {
        path: path.into(),
        source,
    })?;
    let path = path
        .canonicalize()
        .map_err(|source| PluginManifestError::Io {
            path: path.into(),
            source,
        })?;
    let manifest: PluginManifest =
        serde_json::from_reader(file).map_err(|source| PluginManifestError::Json {
            path: path.clone(),
            source,
        })?;
    manifest.validate(&path)?;
    let directory = path
        .parent()
        .expect("canonical file has a parent")
        .to_owned();
    Ok(LoadedPluginManifest {
        manifest,
        path,
        directory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn example() -> Value {
        json!({"manifestVersion":1,"id":"dev.example.echo","name":"Echo","version":"0.1.0","protocolVersion":"ri.plugin.v1","entrypoint":{"command":"./echo"}})
    }

    #[test]
    fn loads_valid_v1_manifest() {
        let path = std::env::temp_dir().join(format!(
            "ri-manifest-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, example().to_string()).unwrap();
        let loaded = load_plugin_manifest(&path).unwrap();
        assert_eq!(loaded.path, path.canonicalize().unwrap());
        assert_eq!(loaded.directory, loaded.path.parent().unwrap());
        let m = loaded.manifest;
        assert_eq!(m.manifest_version, 1);
        assert_eq!(m.protocol_version, "ri.plugin.v1");
        assert_eq!(m.id, "dev.example.echo");
        assert_eq!(m.name, "Echo");
        assert_eq!(m.version, "0.1.0");
        assert_eq!(m.entrypoint.command, "./echo");
        assert!(m.entrypoint.args.is_empty());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_fields_and_versions() {
        for (field, value) in [
            ("manifestVersion", json!(2)),
            ("protocolVersion", json!("v2")),
            ("id", json!("")),
            ("id", json!("Upper")),
            ("id", json!(".abc")),
            ("id", json!("a".repeat(129))),
            ("name", json!(" ")),
            ("version", json!("")),
        ] {
            let mut value_json = example();
            value_json[field] = value;
            let manifest: PluginManifest = serde_json::from_value(value_json).unwrap();
            assert!(
                manifest.validate(Path::new("plugin.json")).is_err(),
                "{field}"
            );
        }
        let mut value = example();
        value["entrypoint"]["command"] = json!(" ");
        assert!(serde_json::from_value::<PluginManifest>(value)
            .unwrap()
            .validate(Path::new("plugin.json"))
            .is_err());
    }

    #[test]
    fn rejects_unknown_manifest_fields() {
        for nested in [false, true] {
            let mut value = example();
            if nested {
                value["entrypoint"]["typo"] = json!(true);
            } else {
                value["typo"] = json!(true);
            }
            assert!(serde_json::from_value::<PluginManifest>(value).is_err());
        }
    }
}
