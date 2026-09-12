use std::sync::Arc;
use std::time::Duration;

use tokio::time::timeout;

use super::{
    builtin_plugins, ExternalToolError, ExternalToolSet, LoadedPluginManifest, PluginProcess,
    PluginProcessError, PluginRegistry,
};
use crate::tools::builtin_tool_registry;

const PLUGIN_TOOLS_TIMEOUT: Duration = Duration::from_secs(5);

pub struct PluginHost {
    registry: PluginRegistry,
    processes: Vec<PluginProcess>,
    active_ids: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum PluginActivationError {
    #[error("could not start plugin {plugin_id:?}: {source}")]
    Start {
        plugin_id: String,
        #[source]
        source: PluginProcessError,
    },
    #[error("could not load tools from plugin {plugin_id:?}: {source}")]
    Tools {
        plugin_id: String,
        #[source]
        source: ExternalToolError,
    },
    #[error("could not load tools from plugin {plugin_id:?}: tools/list timed out")]
    ToolsTimeout { plugin_id: String },
    #[error("plugin activation failed: {source}; cleanup failures: {cleanup:?}")]
    Cleanup {
        #[source]
        source: Box<PluginActivationError>,
        cleanup: Vec<PluginShutdownFailure>,
    },
}

#[derive(Debug)]
pub struct PluginShutdownFailure {
    pub plugin_id: String,
    pub message: String,
}

impl PluginHost {
    pub fn builtin_only() -> Self {
        Self {
            registry: builtin_plugins(),
            processes: Vec::new(),
            active_ids: Vec::new(),
        }
    }

    pub async fn activate(
        manifests: Vec<LoadedPluginManifest>,
    ) -> Result<Self, PluginActivationError> {
        let mut host = Self::builtin_only();
        let mut registry = builtin_tool_registry();
        let result = async {
            for manifest in manifests {
                let plugin_id = manifest.manifest.id.clone();
                let process = PluginProcess::start(manifest).await.map_err(|source| {
                    PluginActivationError::Start {
                        plugin_id: plugin_id.clone(),
                        source,
                    }
                })?;
                host.processes.push(process);
                let process = host.processes.last().expect("just started process");
                let tools = timeout(PLUGIN_TOOLS_TIMEOUT, ExternalToolSet::load(process))
                    .await
                    .map_err(|_| PluginActivationError::ToolsTimeout {
                        plugin_id: plugin_id.clone(),
                    })?
                    .map_err(|source| PluginActivationError::Tools {
                        plugin_id: plugin_id.clone(),
                        source,
                    })?;
                tools.register_into(&mut registry).map_err(|source| {
                    PluginActivationError::Tools {
                        plugin_id: plugin_id.clone(),
                        source,
                    }
                })?;
                host.active_ids.push(plugin_id);
            }
            Ok(())
        }
        .await;
        if let Err(source) = result {
            let cleanup = host.shutdown().await;
            return Err(if cleanup.is_empty() {
                source
            } else {
                PluginActivationError::Cleanup {
                    source: Box::new(source),
                    cleanup,
                }
            });
        }
        host.registry = PluginRegistry::new(Arc::new(registry));
        Ok(host)
    }

    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    pub fn active_ids(&self) -> &[String] {
        &self.active_ids
    }

    pub async fn shutdown(self) -> Vec<PluginShutdownFailure> {
        let mut failures = Vec::new();
        for process in self.processes.into_iter().rev() {
            let plugin_id = process.manifest().id.clone();
            if let Err(error) = process.shutdown().await {
                failures.push(PluginShutdownFailure {
                    plugin_id,
                    message: error.to_string(),
                });
            }
        }
        failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::Fixture;
    use serde_json::json;
    use std::path::Path;

    fn fixture(id: &str, tool: &str, log: &Path) -> Fixture {
        let fixture = Fixture::scripted(
            true,
            vec![(
                "tools/list",
                json!({}),
                json!({"result":{"tools":[{"name":tool,"inputSchema":{}}]}}),
            )],
        );
        let mut loaded = fixture.load();
        loaded.manifest.id = id.into();
        std::fs::write(&loaded.path, serde_json::to_vec(&loaded.manifest).unwrap()).unwrap();
        fixture.change_script(|script| {
            let script = script.replace("test.echo", id);
            #[cfg(unix)]
            {
                format!("{script}printf '%s\\n' '{id}' >> '{}'\n", log.display())
            }
            #[cfg(windows)]
            {
                script.replace(
                    "exit /b 0",
                    &format!("echo {id}>>\"{}\"\r\nexit /b 0", log.display()),
                )
            }
        });
        fixture
    }

    #[tokio::test]
    async fn builtin_only_has_no_processes() {
        let host = PluginHost::builtin_only();
        assert_eq!(
            host.registry().tools().names(),
            ["read", "write", "edit", "bash"]
        );
        assert!(host.active_ids().is_empty());
        assert!(host.shutdown().await.is_empty());
    }

    #[tokio::test]
    async fn activation_order_and_reverse_shutdown() {
        let base = Fixture::scripted(false, vec![]);
        let log = base.load().directory.join("shutdown.log");
        let a = fixture("plugin-a", "alpha", &log);
        let b = fixture("plugin-b", "beta", &log);
        let host = PluginHost::activate(vec![a.load(), b.load()])
            .await
            .unwrap();
        assert_eq!(
            host.registry().tools().names(),
            ["read", "write", "edit", "bash", "alpha", "beta"]
        );
        assert_eq!(host.active_ids(), ["plugin-a", "plugin-b"]);
        assert!(!log.exists());
        assert!(host.shutdown().await.is_empty());
        assert_eq!(
            std::fs::read_to_string(log)
                .unwrap()
                .lines()
                .collect::<Vec<_>>(),
            ["plugin-b", "plugin-a"]
        );
    }

    #[tokio::test]
    async fn collision_cleans_up_every_started_process() {
        let base = Fixture::scripted(false, vec![]);
        let log = base.load().directory.join("shutdown.log");
        let a = fixture("plugin-a", "alpha", &log);
        let b = fixture("plugin-b", "read", &log);
        assert!(
            matches!(PluginHost::activate(vec![a.load(), b.load()]).await, Err(PluginActivationError::Tools { plugin_id, .. }) if plugin_id == "plugin-b")
        );
        assert_eq!(
            std::fs::read_to_string(log)
                .unwrap()
                .lines()
                .collect::<Vec<_>>(),
            ["plugin-b", "plugin-a"]
        );
    }

    #[tokio::test]
    async fn shutdown_collects_every_failure_and_activation_preserves_cleanup_errors() {
        let base = Fixture::scripted(false, vec![]);
        let log = base.load().directory.join("shutdown.log");
        for collision in [false, true] {
            let a = fixture("plugin-a", "alpha", &log);
            let b = fixture("plugin-b", if collision { "read" } else { "beta" }, &log);
            for fixture in [&a, &b] {
                fixture.change_script(|s| {
                    s.replace(
                        r#""result":null"#,
                        r#""error":{"code":-32000,"message":"shutdown failed"}"#,
                    )
                });
            }
            let failures = match PluginHost::activate(vec![a.load(), b.load()]).await {
                Ok(host) => {
                    assert!(!collision);
                    host.shutdown().await
                }
                Err(PluginActivationError::Cleanup { source, cleanup }) => {
                    assert!(collision);
                    assert!(matches!(*source, PluginActivationError::Tools { .. }));
                    cleanup
                }
                Err(error) => panic!("unexpected activation failure: {error}"),
            };
            assert_eq!(
                failures
                    .iter()
                    .map(|f| f.plugin_id.as_str())
                    .collect::<Vec<_>>(),
                ["plugin-b", "plugin-a"]
            );
            assert!(failures
                .iter()
                .all(|f| f.message.contains("shutdown failed")));
        }
    }

    #[tokio::test]
    async fn stalled_tools_list_times_out_and_cleans_up_in_reverse_order() {
        let base = Fixture::scripted(false, vec![]);
        let log = base.load().directory.join("shutdown.log");
        let a = fixture("plugin-a", "alpha", &log);
        let b = fixture("plugin-b", "beta", &log);
        b.change_script(|script| {
            #[cfg(unix)]
            let newline = "\n";
            #[cfg(windows)]
            let newline = "\r\n";
            script
                .lines()
                .filter(|line| !line.contains("inputSchema"))
                .collect::<Vec<_>>()
                .join(newline)
                + newline
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(9),
            PluginHost::activate(vec![a.load(), b.load()]),
        )
        .await
        .expect("capability loading must have a deadline");
        let error = result
            .err()
            .expect("stalled tools/list must fail activation");
        assert!(error.to_string().contains("plugin-b"));
        assert!(error.to_string().contains("tools/list timed out"));
        assert_eq!(
            std::fs::read_to_string(log)
                .unwrap()
                .lines()
                .collect::<Vec<_>>(),
            ["plugin-b", "plugin-a"]
        );
    }

    #[tokio::test]
    async fn startup_failure_cleans_up_previous_processes() {
        let base = Fixture::scripted(false, vec![]);
        let log = base.load().directory.join("shutdown.log");
        let a = fixture("plugin-a", "alpha", &log);
        let b = fixture("plugin-b", "beta", &log);
        b.change_script(|s| s.replace("ri.plugin.v1", "invalid.protocol"));
        assert!(
            matches!(PluginHost::activate(vec![a.load(), b.load()]).await, Err(PluginActivationError::Start { plugin_id, .. }) if plugin_id == "plugin-b")
        );
        assert_eq!(std::fs::read_to_string(log).unwrap().trim(), "plugin-a");
    }
}
