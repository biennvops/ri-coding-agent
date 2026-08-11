use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};
use serde_json::Value;
use thiserror::Error;

use super::ConfigWarning;
use super::ThinkingLevel;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedSettings {
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
    pub default_thinking_level: Option<ThinkingLevel>,
    pub context: ContextSettings,
    pub compaction: CompactionSettings,
}

pub type Settings = ResolvedSettings;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextSettings {
    pub enabled: bool,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionSettings {
    pub enabled: bool,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SettingsLoad {
    pub settings: ResolvedSettings,
    pub warnings: Vec<ConfigWarning>,
}

#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("could not read settings {path}: {source}")]
    Io { path: PathBuf, source: io::Error },

    #[error("could not parse settings {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("invalid settings {path}: {message}")]
    Invalid { path: String, message: String },
}

pub fn default_settings_path() -> Option<PathBuf> {
    home_directory().map(|home| home.join(".ri/agent/settings.json"))
}

pub fn project_settings_path(project_root: impl AsRef<Path>) -> PathBuf {
    project_root.as_ref().join(".ri/settings.json")
}

pub fn load_default_settings(
    project_root: impl AsRef<Path>,
) -> Result<SettingsLoad, SettingsError> {
    let project_path = project_settings_path(project_root);
    let global_path = default_settings_path();
    load_settings_from_paths(global_path.as_deref(), Some(&project_path))
}

pub fn load_settings_from_paths(
    global_path: Option<&Path>,
    project_path: Option<&Path>,
) -> Result<SettingsLoad, SettingsError> {
    let mut load = SettingsLoad {
        settings: ResolvedSettings::default(),
        warnings: Vec::new(),
    };
    if let Some(path) = global_path {
        if let Some(raw) = read_settings(path)? {
            apply_raw_settings(&mut load, raw, path)?;
        }
    }
    if let Some(path) = project_path {
        if let Some(raw) = read_settings(path)? {
            apply_raw_settings(&mut load, raw, path)?;
        }
    }
    Ok(load)
}

fn read_settings(path: &Path) -> Result<Option<RawSettings>, SettingsError> {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SettingsError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let raw = serde_json::from_str(&source).map_err(|source| SettingsError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Some(raw))
}

fn apply_raw_settings(
    load: &mut SettingsLoad,
    raw: RawSettings,
    source_path: &Path,
) -> Result<(), SettingsError> {
    for key in raw.extra.keys() {
        load.warnings.push(ConfigWarning {
            path: format!("settings.{key}"),
            message: "unknown field".to_owned(),
        });
    }
    for key in raw.context.extra.keys() {
        load.warnings.push(ConfigWarning {
            path: format!("settings.context.{key}"),
            message: "unknown field".to_owned(),
        });
    }
    for key in raw.compaction.extra.keys() {
        load.warnings.push(ConfigWarning {
            path: format!("settings.compaction.{key}"),
            message: "unknown field".to_owned(),
        });
    }

    let default_provider = optional_string(
        raw.default_provider,
        &format_path(source_path, "defaultProvider"),
        "defaultProvider",
    )?;
    let default_model = optional_string(
        raw.default_model,
        &format_path(source_path, "defaultModel"),
        "defaultModel",
    )?;
    let default_thinking_level = optional_thinking_level(
        raw.default_thinking_level,
        &format_path(source_path, "defaultThinkingLevel"),
    )?;
    let context_enabled = optional_bool(
        raw.context.enabled,
        &format_path(source_path, "context.enabled"),
        "context.enabled",
    )?;
    let compaction_enabled = optional_bool(
        raw.compaction.enabled,
        &format_path(source_path, "compaction.enabled"),
        "compaction.enabled",
    )?;

    if default_provider.is_some() {
        load.settings.default_provider = default_provider;
    }
    if default_model.is_some() {
        load.settings.default_model = default_model;
    }
    if default_thinking_level.is_some() {
        load.settings.default_thinking_level = default_thinking_level;
    }
    if let Some(enabled) = context_enabled {
        load.settings.context.enabled = enabled;
    }
    if let Some(enabled) = compaction_enabled {
        load.settings.compaction.enabled = enabled;
    }
    Ok(())
}

fn optional_string(
    value: RawField<String>,
    path: &str,
    field: &str,
) -> Result<Option<String>, SettingsError> {
    let value = match value {
        RawField::Missing => return Ok(None),
        RawField::Null => {
            return Err(SettingsError::Invalid {
                path: path.to_owned(),
                message: format!("{field} must be a non-empty string"),
            });
        }
        RawField::Value(value) => value,
    };
    if value.trim().is_empty() {
        return Err(SettingsError::Invalid {
            path: path.to_owned(),
            message: format!("{field} must be a non-empty string"),
        });
    }
    Ok(Some(value))
}

fn optional_thinking_level(
    value: RawField<String>,
    path: &str,
) -> Result<Option<ThinkingLevel>, SettingsError> {
    let value = match value {
        RawField::Missing => return Ok(None),
        RawField::Null => {
            return Err(SettingsError::Invalid {
                path: path.to_owned(),
                message: "defaultThinkingLevel must be a valid thinking level".to_owned(),
            });
        }
        RawField::Value(value) => value,
    };
    value
        .parse()
        .map(Some)
        .map_err(|error: super::ThinkingLevelError| SettingsError::Invalid {
            path: path.to_owned(),
            message: error.to_string(),
        })
}

fn optional_bool(
    value: RawField<bool>,
    path: &str,
    field: &str,
) -> Result<Option<bool>, SettingsError> {
    match value {
        RawField::Missing => Ok(None),
        RawField::Null => Err(SettingsError::Invalid {
            path: path.to_owned(),
            message: format!("{field} must be a boolean"),
        }),
        RawField::Value(value) => Ok(Some(value)),
    }
}

fn format_path(source_path: &Path, field: &str) -> String {
    format!("{}.{field}", source_path.display())
}

fn home_directory() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[derive(Debug, Deserialize, Default)]
struct RawSettings {
    #[serde(rename = "defaultProvider", default)]
    default_provider: RawField<String>,
    #[serde(rename = "defaultModel", default)]
    default_model: RawField<String>,
    #[serde(rename = "defaultThinkingLevel", default)]
    default_thinking_level: RawField<String>,
    #[serde(default)]
    context: RawContextSettings,
    #[serde(default)]
    compaction: RawCompactionSettings,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Default)]
enum RawField<T> {
    #[default]
    Missing,
    Null,
    Value(T),
}

impl<'de, T> Deserialize<'de> for RawField<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Value(value),
            None => Self::Null,
        })
    }
}

#[derive(Debug, Deserialize, Default)]
struct RawContextSettings {
    #[serde(default)]
    enabled: RawField<bool>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Default)]
struct RawCompactionSettings {
    #[serde(default)]
    enabled: RawField<bool>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_settings_keep_builtin_defaults() {
        let root = unique_test_dir("settings-missing");
        fs::create_dir_all(&root).unwrap();
        let global = root.join("global.json");
        let project = root.join("project.json");

        let load = load_settings_from_paths(Some(&global), Some(&project)).unwrap();

        assert_eq!(load.settings, ResolvedSettings::default());
        assert!(load.warnings.is_empty());
        remove_test_dir(root);
    }

    #[test]
    fn project_settings_override_global_values_and_merge_context() {
        let root = unique_test_dir("settings-merge");
        fs::create_dir_all(&root).unwrap();
        let global = root.join("global.json");
        let project = root.join("project.json");
        fs::write(
            &global,
            r#"{
                "defaultProvider": "remote",
                "defaultModel": "general",
                "context": {"enabled": true}
            }"#,
        )
        .unwrap();
        fs::write(
            &project,
            r#"{
                "defaultModel": "coding",
                "context": {"enabled": false}
            }"#,
        )
        .unwrap();

        let load = load_settings_from_paths(Some(&global), Some(&project)).unwrap();

        assert_eq!(load.settings.default_provider.as_deref(), Some("remote"));
        assert_eq!(load.settings.default_model.as_deref(), Some("coding"));
        assert!(!load.settings.context.enabled);
        remove_test_dir(root);
    }

    #[test]
    fn compaction_setting_merges_with_project_precedence() {
        let root = unique_test_dir("settings-compaction");
        fs::create_dir_all(&root).unwrap();
        let global = root.join("global.json");
        let project = root.join("project.json");
        fs::write(&global, r#"{"compaction":{"enabled":false}}"#).unwrap();
        fs::write(&project, r#"{"compaction":{"enabled":true}}"#).unwrap();

        let load = load_settings_from_paths(Some(&global), Some(&project)).unwrap();

        assert!(load.settings.compaction.enabled);
        remove_test_dir(root);
    }

    #[test]
    fn max_default_thinking_level_is_accepted() {
        let root = unique_test_dir("settings-thinking-max");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("settings.json");
        fs::write(&path, r#"{"defaultThinkingLevel":"max"}"#).unwrap();

        let load = load_settings_from_paths(Some(&path), None).unwrap();

        assert_eq!(
            load.settings.default_thinking_level,
            Some(ThinkingLevel::Max)
        );
        remove_test_dir(root);
    }

    #[test]
    fn unknown_fields_are_warnings_with_stable_paths() {
        let root = unique_test_dir("settings-warning");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("settings.json");
        fs::write(
            &path,
            r#"{
                "future": true,
                "context": {"futureEnabledMode": "later"}
            }"#,
        )
        .unwrap();

        let load = load_settings_from_paths(Some(&path), None).unwrap();

        assert_eq!(
            load.warnings,
            vec![
                ConfigWarning {
                    path: "settings.future".to_owned(),
                    message: "unknown field".to_owned(),
                },
                ConfigWarning {
                    path: "settings.context.futureEnabledMode".to_owned(),
                    message: "unknown field".to_owned(),
                },
            ]
        );
        remove_test_dir(root);
    }

    #[test]
    fn malformed_json_names_the_source_path() {
        let root = unique_test_dir("settings-json");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("settings.json");
        fs::write(&path, "{ not json").unwrap();

        let error = load_settings_from_paths(Some(&path), None).unwrap_err();

        assert!(error.to_string().contains(&path.display().to_string()));
        assert!(matches!(error, SettingsError::Json { .. }));
        remove_test_dir(root);
    }

    #[test]
    fn empty_model_selections_are_rejected() {
        let root = unique_test_dir("settings-invalid-values");
        fs::create_dir_all(&root).unwrap();
        for (field, source) in [
            ("defaultProvider", r#"{"defaultProvider":"  "}"#),
            ("defaultModel", r#"{"defaultModel":""}"#),
        ] {
            let path = root.join(format!("{field}.json"));
            fs::write(&path, source).unwrap();
            let error = load_settings_from_paths(Some(&path), None).unwrap_err();
            assert!(error.to_string().contains(field));
        }
        remove_test_dir(root);
    }

    #[test]
    fn known_values_with_wrong_types_are_rejected() {
        let root = unique_test_dir("settings-invalid-type");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("settings.json");
        fs::write(&path, r#"{"context":{"enabled":"yes"}}"#).unwrap();

        let error = load_settings_from_paths(Some(&path), None).unwrap_err();

        assert!(matches!(error, SettingsError::Json { .. }));
        assert!(error.to_string().contains(&path.display().to_string()));
        remove_test_dir(root);
    }

    #[test]
    fn null_known_values_are_not_treated_as_unset() {
        let root = unique_test_dir("settings-null");
        fs::create_dir_all(&root).unwrap();
        for (field, source) in [
            ("defaultProvider", r#"{"defaultProvider":null}"#),
            ("defaultModel", r#"{"defaultModel":null}"#),
            ("context.enabled", r#"{"context":{"enabled":null}}"#),
        ] {
            let path = root.join(format!("{field}.json"));
            fs::write(&path, source).unwrap();
            let error = load_settings_from_paths(Some(&path), None).unwrap_err();
            assert!(error.to_string().contains(field));
        }
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

    use std::fs;
    use std::path::PathBuf;
}
