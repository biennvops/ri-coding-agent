use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiKind {
    OpenAiResponses,
    OpenAiCompletions,
}

impl FromStr for ApiKind {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "openai-responses" => Ok(Self::OpenAiResponses),
            "openai-completions" => Ok(Self::OpenAiCompletions),
            other => Err(ConfigError::Invalid(format!(
                "unsupported model API {other:?}; expected openai-responses or openai-completions"
            ))),
        }
    }
}

impl fmt::Display for ApiKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OpenAiResponses => "openai-responses",
            Self::OpenAiCompletions => "openai-completions",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
}

impl ModelRef {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    pub fn display_name(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compatibility {
    pub supports_developer_role: bool,
    pub supports_reasoning_effort: bool,
}

impl Default for Compatibility {
    fn default() -> Self {
        Self {
            supports_developer_role: true,
            supports_reasoning_effort: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostMetadata {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModel {
    pub model_ref: ModelRef,
    pub name: String,
    pub base_url: String,
    pub api: ApiKind,
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub auth_header: bool,
    pub compatibility: Compatibility,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
    pub cost: CostMetadata,
    pub sampling_params: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigWarning {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("could not read models config {path}: {source}")]
    Io { path: PathBuf, source: io::Error },

    #[error("could not parse models config {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("invalid model configuration: {0}")]
    Invalid(String),

    #[error("environment variable {variable} referenced by {path} is not set")]
    MissingEnv { variable: String, path: String },
}

#[derive(Clone, Debug, Default)]
pub struct ModelCatalog {
    models: Vec<ResolvedModel>,
    warnings: Vec<ConfigWarning>,
}

impl ModelCatalog {
    pub fn from_json(path: impl Into<PathBuf>, source: &str) -> Result<Self, ConfigError> {
        let path = path.into();
        let raw: RawModelsFile =
            serde_json::from_str(source).map_err(|source| ConfigError::Json {
                path: path.clone(),
                source,
            })?;
        Self::from_raw(path, raw)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref().to_path_buf();
        let source = fs::read_to_string(&path).map_err(|source| ConfigError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_json(path, &source)
    }

    pub fn warnings(&self) -> &[ConfigWarning] {
        &self.warnings
    }

    pub fn model_refs(&self) -> impl Iterator<Item = ModelRef> + '_ {
        self.models.iter().map(|model| model.model_ref.clone())
    }

    pub fn models(&self) -> &[ResolvedModel] {
        &self.models
    }

    pub fn resolve(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<ResolvedModel, ConfigError> {
        let (provider, model) = split_selection(provider, model)?;
        if model.is_none() {
            if let Some(provider) = provider {
                return self
                    .models
                    .iter()
                    .find(|candidate| candidate.model_ref.provider == provider)
                    .cloned()
                    .ok_or_else(|| ConfigError::Invalid(selection_error(Some(provider), None)));
            }
            return self
                .models
                .first()
                .cloned()
                .ok_or_else(|| ConfigError::Invalid(selection_error(None, None)));
        }

        let candidates = self.models.iter().filter(|candidate| {
            provider.is_none_or(|provider| candidate.model_ref.provider == provider)
                && model.is_none_or(|model| {
                    candidate.model_ref.model == model || candidate.name == model
                })
        });
        let matches: Vec<&ResolvedModel> = candidates.collect();

        match matches.as_slice() {
            [candidate] => Ok((*candidate).clone()),
            [] => Err(ConfigError::Invalid(selection_error(provider, model))),
            _ => Err(ConfigError::Invalid(format!(
                "model {model:?} is ambiguous; specify a provider"
            ))),
        }
    }

    fn from_raw(path: PathBuf, raw: RawModelsFile) -> Result<Self, ConfigError> {
        let mut warnings = Vec::new();
        for key in raw.extra.keys() {
            warnings.push(ConfigWarning {
                path: key.clone(),
                message: "unknown top-level field".to_owned(),
            });
        }

        let mut models = Vec::new();
        for (provider_id, provider) in raw.providers {
            if provider_id.trim().is_empty() {
                return Err(ConfigError::Invalid(
                    "provider ID must not be empty".to_owned(),
                ));
            }
            for key in provider.extra.keys() {
                warnings.push(ConfigWarning {
                    path: format!("providers.{provider_id}.{key}"),
                    message: "unknown provider field".to_owned(),
                });
            }

            let base_url = interpolate_env(
                &provider.base_url,
                &format!("providers.{provider_id}.baseUrl"),
            )?;
            if base_url.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "provider {provider_id:?} baseUrl must not be empty"
                )));
            }
            let base_url_url = reqwest::Url::parse(&base_url).map_err(|error| {
                ConfigError::Invalid(format!(
                    "provider {provider_id:?} baseUrl must be a valid URL: {error}"
                ))
            })?;
            if !matches!(base_url_url.scheme(), "http" | "https") || base_url_url.host().is_none() {
                return Err(ConfigError::Invalid(format!(
                    "provider {provider_id:?} baseUrl must be an HTTP or HTTPS URL"
                )));
            }
            let api = provider
                .api
                .as_deref()
                .ok_or_else(|| {
                    ConfigError::Invalid(format!(
                        "provider {provider_id:?} is missing its api field"
                    ))
                })?
                .parse::<ApiKind>()?;
            let api_key = provider
                .api_key
                .as_deref()
                .map(|value| interpolate_env(value, &format!("providers.{provider_id}.apiKey")))
                .transpose()?;
            let headers = provider
                .headers
                .into_iter()
                .map(|(name, value)| {
                    let value = interpolate_env(
                        &value,
                        &format!("providers.{provider_id}.headers.{name}"),
                    )?;
                    Ok((name, value))
                })
                .collect::<Result<BTreeMap<_, _>, ConfigError>>()?;
            let provider_compat = provider.compat.resolve();

            for model in provider.models {
                for key in model.extra.keys() {
                    warnings.push(ConfigWarning {
                        path: format!("providers.{provider_id}.models.{}.{}", model.id, key),
                        message: "unknown model field".to_owned(),
                    });
                }
                let model_id =
                    interpolate_env(&model.id, &format!("providers.{provider_id}.models.id"))?;
                if model_id.trim().is_empty() {
                    return Err(ConfigError::Invalid(format!(
                        "provider {provider_id:?} model ID must not be empty"
                    )));
                }
                if model.context_window == Some(0) {
                    return Err(ConfigError::Invalid(format!(
                        "model {provider_id}/{model_id:?} contextWindow must be greater than zero"
                    )));
                }
                if model.max_tokens == Some(0) {
                    return Err(ConfigError::Invalid(format!(
                        "model {provider_id}/{model_id:?} maxTokens must be greater than zero"
                    )));
                }
                let name = model.name.clone().unwrap_or_else(|| model_id.clone());
                let api = model
                    .api
                    .as_deref()
                    .map(str::parse)
                    .transpose()?
                    .unwrap_or(api);
                let compatibility = Compatibility {
                    supports_developer_role: model
                        .compat
                        .supports_developer_role
                        .unwrap_or(provider_compat.supports_developer_role),
                    supports_reasoning_effort: model
                        .compat
                        .supports_reasoning_effort
                        .unwrap_or(provider_compat.supports_reasoning_effort),
                };
                let cost = model.cost.map(CostMetadata::from).unwrap_or_default();

                models.push(ResolvedModel {
                    model_ref: ModelRef::new(&provider_id, &model_id),
                    name,
                    base_url: base_url.clone(),
                    api,
                    api_key: api_key.clone(),
                    headers: headers.clone(),
                    auth_header: provider.auth_header.unwrap_or(true),
                    compatibility,
                    reasoning: model.reasoning,
                    input: model.input,
                    context_window: model.context_window,
                    max_tokens: model.max_tokens,
                    cost,
                    sampling_params: model.sampling_params,
                });
            }
        }

        if models.is_empty() {
            warnings.push(ConfigWarning {
                path: path.display().to_string(),
                message: "models config contains no selectable models".to_owned(),
            });
        }

        Ok(Self { models, warnings })
    }
}

pub fn default_models_path() -> Option<PathBuf> {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".ri/agent/models.json"))
}

pub fn load_default_models() -> Result<Option<ModelCatalog>, ConfigError> {
    let path = env::var_os("RI_MODELS_FILE")
        .map(PathBuf::from)
        .or_else(default_models_path);
    let Some(path) = path else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    ModelCatalog::load(path).map(Some)
}

pub fn interpolate_env(value: &str, path: &str) -> Result<String, ConfigError> {
    let characters: Vec<char> = value.chars().collect();
    let mut output = String::with_capacity(value.len());
    let mut index = 0;

    while index < characters.len() {
        if characters[index] != '$' {
            output.push(characters[index]);
            index += 1;
            continue;
        }

        if characters.get(index + 1) == Some(&'$') {
            output.push('$');
            index += 2;
            continue;
        }

        let (name, consumed) = if characters.get(index + 1) == Some(&'{') {
            let Some(end) = characters[index + 2..]
                .iter()
                .position(|character| *character == '}')
            else {
                output.push('$');
                index += 1;
                continue;
            };
            let end = index + 2 + end;
            let name: String = characters[index + 2..end].iter().collect();
            (name, end + 1 - index)
        } else {
            let mut end = index + 1;
            while end < characters.len()
                && (characters[end].is_ascii_alphanumeric() || characters[end] == '_')
            {
                end += 1;
            }
            if end == index + 1 {
                output.push('$');
                index += 1;
                continue;
            }
            let name: String = characters[index + 1..end].iter().collect();
            (name, end - index)
        };

        if name.is_empty() {
            return Err(ConfigError::Invalid(format!(
                "empty environment variable reference at {path}"
            )));
        }
        let replacement = env::var(&name).map_err(|_| ConfigError::MissingEnv {
            variable: name.clone(),
            path: path.to_owned(),
        })?;
        output.push_str(&replacement);
        index += consumed;
    }

    Ok(output)
}

fn split_selection<'a>(
    provider: Option<&'a str>,
    model: Option<&'a str>,
) -> Result<(Option<&'a str>, Option<&'a str>), ConfigError> {
    if let Some(model) = model {
        if provider.is_none() {
            if let Some((provider, model)) = model.split_once('/') {
                return Ok((Some(provider), Some(model)));
            }
        }
    }
    Ok((provider, model))
}

fn selection_error(provider: Option<&str>, model: Option<&str>) -> String {
    match (provider, model) {
        (Some(provider), Some(model)) => {
            format!("model {provider}/{model:?} was not found in models.json")
        }
        (Some(provider), None) => {
            format!("provider {provider:?} has no selectable model in models.json")
        }
        (None, Some(model)) => format!("model {model:?} was not found in models.json"),
        (None, None) => "models.json contains no selectable model".to_owned(),
    }
}

#[derive(Debug, Deserialize, Default)]
struct RawModelsFile {
    #[serde(default)]
    providers: BTreeMap<String, RawProvider>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Default)]
struct RawProvider {
    #[serde(rename = "baseUrl", default)]
    base_url: String,
    #[serde(default)]
    api: Option<String>,
    #[serde(rename = "apiKey", default)]
    api_key: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(rename = "authHeader", default)]
    auth_header: Option<bool>,
    #[serde(default)]
    compat: RawCompatibility,
    #[serde(default)]
    models: Vec<RawModel>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Default)]
struct RawModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    input: Vec<String>,
    #[serde(rename = "contextWindow", default)]
    context_window: Option<u64>,
    #[serde(rename = "maxTokens", default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    cost: Option<RawCost>,
    #[serde(default)]
    compat: RawCompatibility,
    #[serde(rename = "samplingParams", alias = "sampling", default)]
    sampling_params: BTreeMap<String, Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Default)]
struct RawCompatibility {
    #[serde(rename = "supportsDeveloperRole", default)]
    supports_developer_role: Option<bool>,
    #[serde(rename = "supportsReasoningEffort", default)]
    supports_reasoning_effort: Option<bool>,
}

impl RawCompatibility {
    fn resolve(&self) -> Compatibility {
        Compatibility {
            supports_developer_role: self.supports_developer_role.unwrap_or(true),
            supports_reasoning_effort: self.supports_reasoning_effort.unwrap_or(true),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawCost {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(rename = "cacheRead", default)]
    cache_read: f64,
    #[serde(rename = "cacheWrite", default)]
    cache_write: f64,
}

impl From<RawCost> for CostMetadata {
    fn from(cost: RawCost) -> Self {
        Self {
            input: cost.input,
            output: cost.output,
            cache_read: cost.cache_read,
            cache_write: cost.cache_write,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_models_and_resolves_selection() {
        let source = r#"
        {
          "providers": {
            "custom": {
              "baseUrl": "https://example.test/v1",
              "api": "openai-completions",
              "apiKey": "literal-key",
              "models": [{
                "id": "coding",
                "name": "Coding Model",
                "reasoning": true,
                "contextWindow": 100000,
                "maxTokens": 4096,
                "compat": {"supportsDeveloperRole": true}
              }]
            }
          }
        }
        "#;

        let catalog = ModelCatalog::from_json("models.json", source).expect("config should parse");
        let model = catalog
            .resolve(Some("custom"), Some("coding"))
            .expect("model should resolve");
        assert_eq!(model.api, ApiKind::OpenAiCompletions);
        assert_eq!(model.model_ref.display_name(), "custom/coding");
        assert!(model.reasoning);
        assert!(model.compatibility.supports_developer_role);

        let multiple = ModelCatalog::from_json(
            "models.json",
            r#"{
              "providers": {
                "p": {
                  "baseUrl": "https://example.test",
                  "api": "openai-responses",
                  "models": [{"id": "first"}, {"id": "second"}]
                }
              }
            }"#,
        )
        .expect("multiple models should parse");
        assert_eq!(
            multiple.resolve(None, None).unwrap().model_ref.model,
            "first"
        );
        assert_eq!(
            multiple.resolve(Some("p"), None).unwrap().model_ref.model,
            "first"
        );
        assert_eq!(
            multiple
                .resolve(None, Some("p/second"))
                .unwrap()
                .model_ref
                .model,
            "second"
        );
    }

    #[test]
    fn interpolates_plain_and_braced_environment_variables() {
        unsafe { env::set_var("RI_CONFIG_TEST_TOKEN", "secret") };
        assert_eq!(
            interpolate_env(
                "Bearer $RI_CONFIG_TEST_TOKEN / ${RI_CONFIG_TEST_TOKEN}",
                "apiKey"
            )
            .expect("variable should resolve"),
            "Bearer secret / secret"
        );
        unsafe { env::remove_var("RI_CONFIG_TEST_TOKEN") };
    }

    #[test]
    fn unknown_fields_become_warnings() {
        let catalog = ModelCatalog::from_json(
            "models.json",
            r#"{
              "providers": {
                "p": {
                  "baseUrl": "https://example.test",
                  "api": "openai-responses",
                  "futureProviderField": true,
                  "models": [{"id": "m", "futureModelField": 1}]
                }
              },
              "futureTopLevelField": true
            }"#,
        )
        .expect("config should parse");

        assert_eq!(catalog.warnings().len(), 3);
    }

    #[test]
    fn missing_environment_variable_is_actionable() {
        let error = interpolate_env("$RI_CONFIG_MISSING_VARIABLE", "headers.X-Key")
            .expect_err("missing variable should fail");
        assert!(error.to_string().contains("RI_CONFIG_MISSING_VARIABLE"));
        assert!(error.to_string().contains("headers.X-Key"));
    }
}
