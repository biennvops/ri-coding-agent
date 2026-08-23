use ri_core::{ConfigError, ModelCatalog, RecentModelState, ResolvedModel, ResolvedSettings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelSelectionSource {
    Cli,
    Settings,
    WorkspaceRecent,
    GlobalRecent,
    FirstConfigured,
}

#[derive(Debug)]
pub(crate) struct ModelSelection {
    pub model: ResolvedModel,
    pub source: ModelSelectionSource,
    pub warnings: Vec<String>,
}

pub(crate) fn resolve_model(
    catalog: &ModelCatalog,
    cli_provider: Option<&str>,
    cli_model: Option<&str>,
    settings: &ResolvedSettings,
    recent_state: Option<&RecentModelState>,
    workspace_id: &str,
) -> Result<ModelSelection, ConfigError> {
    let mut warnings = Vec::new();

    if cli_provider.is_some() || cli_model.is_some() {
        let (provider, model) = cli_selection(cli_provider, cli_model, settings);
        return Ok(ModelSelection {
            model: catalog.resolve(provider, model)?,
            source: ModelSelectionSource::Cli,
            warnings,
        });
    }

    if settings.default_provider.is_some() || settings.default_model.is_some() {
        return Ok(ModelSelection {
            model: catalog.resolve(
                settings.default_provider.as_deref(),
                settings.default_model.as_deref(),
            )?,
            source: ModelSelectionSource::Settings,
            warnings,
        });
    }

    if let Some(recent) = recent_state.and_then(|state| state.workspace_model(workspace_id)) {
        match catalog.resolve(Some(&recent.provider), Some(&recent.model)) {
            Ok(model) => {
                return Ok(ModelSelection {
                    model,
                    source: ModelSelectionSource::WorkspaceRecent,
                    warnings,
                });
            }
            Err(error) => warnings.push(stale_warning("workspace", recent, error)),
        }
    }

    if let Some(recent) = recent_state.and_then(|state| state.last_model.as_ref()) {
        match catalog.resolve(Some(&recent.provider), Some(&recent.model)) {
            Ok(model) => {
                return Ok(ModelSelection {
                    model,
                    source: ModelSelectionSource::GlobalRecent,
                    warnings,
                });
            }
            Err(error) => warnings.push(stale_warning("global", recent, error)),
        }
    }

    Ok(ModelSelection {
        model: catalog.resolve(None, None)?,
        source: ModelSelectionSource::FirstConfigured,
        warnings,
    })
}

fn cli_selection<'a>(
    provider: Option<&'a str>,
    model: Option<&'a str>,
    settings: &'a ResolvedSettings,
) -> (Option<&'a str>, Option<&'a str>) {
    let model = model.or(settings.default_model.as_deref());
    let provider = provider.or_else(|| {
        if model.is_some_and(|model| model.contains('/')) {
            None
        } else {
            settings.default_provider.as_deref()
        }
    });
    (provider, model)
}

fn stale_warning(scope: &str, recent: &ri_core::RecentModel, error: ConfigError) -> String {
    format!(
        "recent {scope} model {}/{} is unavailable; ignoring it: {error}",
        recent.provider, recent.model
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ri_core::{ModelRef, RecentModel, RecentModelState, WorkspaceRecentModel};
    use std::collections::BTreeMap;

    #[test]
    fn precedence_is_cli_then_settings_then_workspace_then_global_then_first() {
        let catalog = catalog();
        let state = RecentModelState {
            last_model: Some(RecentModel {
                provider: "provider".to_owned(),
                model: "global".to_owned(),
            }),
            workspaces: BTreeMap::from([(
                "workspace".to_owned(),
                WorkspaceRecentModel {
                    last_model: Some(RecentModel {
                        provider: "provider".to_owned(),
                        model: "workspace".to_owned(),
                    }),
                    ..WorkspaceRecentModel::default()
                },
            )]),
            ..RecentModelState::default()
        };
        let no_settings = ResolvedSettings::default();

        let first = resolve_model(&catalog, None, None, &no_settings, None, "workspace")
            .unwrap()
            .model
            .model_ref;
        assert_eq!(first, ModelRef::new("provider", "first"));

        let global = resolve_model(&catalog, None, None, &no_settings, Some(&state), "other")
            .unwrap()
            .model
            .model_ref;
        assert_eq!(global, ModelRef::new("provider", "global"));

        let workspace = resolve_model(
            &catalog,
            None,
            None,
            &no_settings,
            Some(&state),
            "workspace",
        )
        .unwrap()
        .model
        .model_ref;
        assert_eq!(workspace, ModelRef::new("provider", "workspace"));

        let settings = ResolvedSettings {
            default_model: Some("settings".to_owned()),
            ..ResolvedSettings::default()
        };
        let settings_model =
            resolve_model(&catalog, None, None, &settings, Some(&state), "workspace")
                .unwrap()
                .model
                .model_ref;
        assert_eq!(settings_model, ModelRef::new("provider", "settings"));

        let cli_model = resolve_model(
            &catalog,
            Some("provider"),
            Some("cli"),
            &settings,
            Some(&state),
            "workspace",
        )
        .unwrap()
        .model
        .model_ref;
        assert_eq!(cli_model, ModelRef::new("provider", "cli"));
    }

    #[test]
    fn qualified_cli_model_does_not_inherit_settings_provider() {
        let catalog = catalog_with_second_provider();
        let settings = ResolvedSettings {
            default_provider: Some("provider".to_owned()),
            ..ResolvedSettings::default()
        };

        let selected = resolve_model(
            &catalog,
            None,
            Some("other/cli"),
            &settings,
            None,
            "workspace",
        )
        .unwrap();

        assert_eq!(selected.model.model_ref, ModelRef::new("other", "cli"));
    }

    #[test]
    fn stale_recent_models_warn_and_fall_back() {
        let catalog = catalog();
        let state = RecentModelState {
            last_model: Some(RecentModel {
                provider: "provider".to_owned(),
                model: "removed".to_owned(),
            }),
            workspaces: BTreeMap::from([(
                "workspace".to_owned(),
                WorkspaceRecentModel {
                    last_model: Some(RecentModel {
                        provider: "provider".to_owned(),
                        model: "also-removed".to_owned(),
                    }),
                    ..WorkspaceRecentModel::default()
                },
            )]),
            ..RecentModelState::default()
        };

        let selection = resolve_model(
            &catalog,
            None,
            None,
            &ResolvedSettings::default(),
            Some(&state),
            "workspace",
        )
        .unwrap();

        assert_eq!(
            selection.model.model_ref,
            ModelRef::new("provider", "first")
        );
        assert_eq!(selection.warnings.len(), 2);
        assert!(selection.warnings[0].contains("workspace"));
        assert!(selection.warnings[1].contains("global"));
    }

    fn catalog() -> ModelCatalog {
        ModelCatalog::from_json(
            "models.json",
            r#"{
                "providers": {
                    "provider": {
                        "baseUrl": "https://example.test",
                        "api": "openai-responses",
                        "models": [
                            {"id": "first"},
                            {"id": "settings"},
                            {"id": "workspace"},
                            {"id": "global"},
                            {"id": "cli"}
                        ]
                    }
                }
            }"#,
        )
        .unwrap()
    }

    fn catalog_with_second_provider() -> ModelCatalog {
        ModelCatalog::from_json(
            "models.json",
            r#"{
                "providers": {
                    "provider": {
                        "baseUrl": "https://example.test",
                        "api": "openai-responses",
                        "models": [{"id": "first"}]
                    },
                    "other": {
                        "baseUrl": "https://example.test",
                        "api": "openai-responses",
                        "models": [{"id": "cli"}]
                    }
                }
            }"#,
        )
        .unwrap()
    }
}
