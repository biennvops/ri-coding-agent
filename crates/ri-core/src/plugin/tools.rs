use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use serde_json::Value;

use super::process::{PluginClient, PluginProcess, PluginProcessError};
use super::protocol::{PluginToolDefinition, ToolCallParams, ToolCallResult, ToolsListResult};
use crate::model::ToolDefinition;
use crate::tools::{
    Tool, ToolContext, ToolError, ToolEventSender, ToolExecutionResult, ToolRegistry,
    ToolRegistryError,
};

pub const MAX_EXTERNAL_TOOLS_PER_PLUGIN: usize = 128;
pub const MAX_EXTERNAL_TOOL_NAME_BYTES: usize = 64;
pub const MAX_EXTERNAL_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ExternalToolError {
    #[error("plugin {plugin_id:?} returned invalid tool {name:?}: {message}")]
    InvalidDefinition {
        plugin_id: String,
        name: String,
        message: String,
    },
    #[error("plugin {plugin_id:?} returned duplicate tool {name:?}")]
    DuplicateDefinition { plugin_id: String, name: String },
    #[error("plugin {plugin_id:?} exposes too many tools: {count} > {max}")]
    TooManyTools {
        plugin_id: String,
        count: usize,
        max: usize,
    },
    #[error("plugin {plugin_id:?} request failed: {source}")]
    Process {
        plugin_id: String,
        #[source]
        source: PluginProcessError,
    },
    #[error("plugin {plugin_id:?} returned invalid tools/list result: {message}")]
    InvalidListResult { plugin_id: String, message: String },
    #[error(transparent)]
    Registry(#[from] ToolRegistryError),
}

fn validate_definitions(
    plugin_id: &str,
    tools: Vec<PluginToolDefinition>,
) -> Result<Vec<ToolDefinition>, ExternalToolError> {
    if tools.len() > MAX_EXTERNAL_TOOLS_PER_PLUGIN {
        return Err(ExternalToolError::TooManyTools {
            plugin_id: plugin_id.into(),
            count: tools.len(),
            max: MAX_EXTERNAL_TOOLS_PER_PLUGIN,
        });
    }
    let mut names = HashSet::new();
    tools.into_iter().map(|tool| {
        let invalid = |message: &str| ExternalToolError::InvalidDefinition {
            plugin_id: plugin_id.into(), name: tool.name.clone(), message: message.into(),
        };
        if tool.name.is_empty() || tool.name.len() > MAX_EXTERNAL_TOOL_NAME_BYTES
            || !tool.name.as_bytes()[0].is_ascii_alphanumeric()
            || !tool.name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
            return Err(invalid("name must be 1..=64 ASCII letters, digits, underscores or hyphens, starting with a letter or digit"));
        }
        if let Some(description) = &tool.description {
            if description.trim().is_empty() || description.len() > MAX_EXTERNAL_TOOL_DESCRIPTION_BYTES {
                return Err(invalid("description must be nonblank and at most 16 KiB"));
            }
        }
        if !tool.input_schema.is_object() {
            return Err(invalid("inputSchema must be a JSON object"));
        }
        if !names.insert(tool.name.clone()) {
            return Err(ExternalToolError::DuplicateDefinition { plugin_id: plugin_id.into(), name: tool.name });
        }
        Ok(ToolDefinition { name: tool.name, description: tool.description, parameters: tool.input_schema })
    }).collect()
}

/// An immutable snapshot loaded explicitly from one plugin's tool capability.
/// Keep the owning `PluginProcess` alive until these tools are no longer needed.
#[derive(Clone)]
pub struct ExternalToolSet {
    plugin_id: String,
    tools: Vec<Arc<dyn Tool>>,
}

impl ExternalToolSet {
    pub async fn load(process: &PluginProcess) -> Result<Self, ExternalToolError> {
        let plugin_id = process.manifest().id.clone();
        let mut tools = Vec::new();
        if process.capabilities().tools {
            let response = process
                .request("tools/list", serde_json::json!({}))
                .await
                .map_err(|source| ExternalToolError::Process {
                    plugin_id: plugin_id.clone(),
                    source,
                })?;
            let result: ToolsListResult = serde_json::from_value(response).map_err(|error| {
                ExternalToolError::InvalidListResult {
                    plugin_id: plugin_id.clone(),
                    message: error.to_string(),
                }
            })?;
            for definition in validate_definitions(&plugin_id, result.tools)? {
                tools.push(Arc::new(ExternalTool {
                    plugin_id: plugin_id.clone(),
                    definition,
                    client: process.client(),
                }) as Arc<dyn Tool>);
            }
        }
        Ok(Self { plugin_id, tools })
    }

    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    pub fn names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect()
    }

    pub fn register_into(&self, registry: &mut ToolRegistry) -> Result<(), ExternalToolError> {
        let mut names: HashSet<_> = registry.names().into_iter().collect();
        for name in self.names() {
            if !names.insert(name.clone()) {
                return Err(ToolRegistryError::DuplicateTool { name }.into());
            }
        }
        for tool in &self.tools {
            registry.register(tool.clone())?;
        }
        Ok(())
    }
}

struct ExternalTool {
    plugin_id: String,
    definition: ToolDefinition,
    client: PluginClient,
}

#[async_trait]
impl Tool for ExternalTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
        _events: ToolEventSender,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionResult, ToolError> {
        let started = Instant::now();
        let params = serde_json::to_value(ToolCallParams {
            name: self.definition.name.clone(),
            arguments,
        })
        .expect("serializable tool call params");
        let failed = |message: String| {
            ToolError::Failed(format!(
                "plugin {:?} tool {:?}: {message}",
                self.plugin_id, self.definition.name
            ))
        };
        // Host cancellation drops the pending request; remote computation may continue.
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ToolError::Cancelled),
            response = self.client.request("tools/call", params) => response.map_err(|error| failed(error.to_string()))?,
        };
        let response: ToolCallResult =
            serde_json::from_value(response).map_err(|error| failed(error.to_string()))?;
        let mut result = if response.is_error {
            ToolExecutionResult::failure(response.content)
        } else {
            ToolExecutionResult::success(response.content)
        };
        result.metadata.duration = started.elapsed();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn descriptor(name: &str) -> PluginToolDefinition {
        PluginToolDefinition {
            name: name.into(),
            description: Some("Echo".into()),
            input_schema: json!({"type":"object", "custom":true}),
        }
    }

    use super::super::process::tests::Fixture;

    fn list(names: &[&str]) -> Value {
        serde_json::to_value(super::super::protocol::ToolsListResult {
            tools: names.iter().map(|name| descriptor(name)).collect(),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn loads_once_preserves_order_and_registers_atomically() {
        for names in [
            vec![],
            vec!["foo", "bar"],
            vec!["search"],
            vec!["search", "read"],
        ] {
            let fixture = Fixture::scripted(
                true,
                vec![("tools/list", json!({}), json!({"result":list(&names)}))],
            );
            let process = PluginProcess::start(fixture.load()).await.unwrap();
            let set = ExternalToolSet::load(&process).await.unwrap();
            assert_eq!(set.plugin_id(), "test.echo");
            assert_eq!(set.names(), names);
            assert_eq!(set.tools().len(), names.len());
            let mut empty = ToolRegistry::new();
            set.register_into(&mut empty).unwrap();
            assert_eq!(empty.names(), names);
            let mut registry = crate::tools::builtin_tool_registry();
            let before = registry.definitions();
            if names.contains(&"read") {
                assert!(matches!(
                    set.register_into(&mut registry),
                    Err(ExternalToolError::Registry(_))
                ));
                assert_eq!(registry.definitions(), before);
            } else {
                set.register_into(&mut registry).unwrap();
                let mut expected = vec!["read", "write", "edit", "bash"];
                expected.extend(names);
                assert_eq!(registry.names(), expected);
            }
            process.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn absent_capability_skips_list_and_invalid_lists_fail() {
        let fixture = Fixture::scripted(false, vec![]);
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        assert!(ExternalToolSet::load(&process)
            .await
            .unwrap()
            .tools()
            .is_empty());
        process.shutdown().await.unwrap();
        for response in [
            json!({"result":null}),
            json!({"result":list(&["echo", "echo"])}),
            json!({"result":list(&["bad.name"])}),
            json!({"error":{"code":-1,"message":"no list"}}),
        ] {
            let fixture = Fixture::scripted(true, vec![("tools/list", json!({}), response)]);
            let process = PluginProcess::start(fixture.load()).await.unwrap();
            let error = ExternalToolSet::load(&process).await.err().unwrap();
            assert!(error.to_string().contains("test.echo"));
            process.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn executes_through_registry_and_maps_results() {
        for response in [
            json!({"result":{"content":"hello","isError":false}}),
            json!({"result":{"content":"no such item","isError":true}}),
            json!({"error":{"code":-1,"message":"RPC failed"}}),
            json!({"result":{"content":42}}),
        ] {
            let fixture = Fixture::scripted(
                true,
                vec![
                    ("tools/list", json!({}), json!({"result":list(&["echo"])})),
                    (
                        "tools/call",
                        json!({"name":"echo","arguments":{"text":"hello"}}),
                        response.clone(),
                    ),
                ],
            );
            let process = PluginProcess::start(fixture.load()).await.unwrap();
            let set = ExternalToolSet::load(&process).await.unwrap();
            let mut registry = ToolRegistry::new();
            set.register_into(&mut registry).unwrap();
            let arguments = json!({"text":"hello"});
            let presentation = registry.presentation("echo", &arguments);
            assert_eq!(presentation.summary, "echo");
            assert_eq!(presentation.preview[0].text, arguments.to_string());
            let result = registry
                .execute(
                    "echo",
                    arguments,
                    &ToolContext::new(std::env::temp_dir()).unwrap(),
                    tokio::sync::mpsc::channel(1).0,
                    CancellationToken::new(),
                )
                .await;
            if let Some(content) = response["result"]["content"].as_str() {
                let result = result.unwrap();
                assert_eq!(result.model_content, content);
                let mut metadata = crate::tools::ToolExecutionMetadata::success();
                metadata.success = !response["result"]["isError"].as_bool().unwrap();
                assert!(result.metadata.duration > std::time::Duration::ZERO);
                metadata.duration = result.metadata.duration;
                assert_eq!(result.metadata, metadata);
            } else {
                assert!(
                    matches!(&result, Err(ToolError::Failed(message)) if message.contains("test.echo") && message.contains("echo"))
                );
            }
            process.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn cancellation_ignores_late_response_and_keeps_transport_usable() {
        let fixture = Fixture::scripted(
            true,
            vec![
                ("tools/list", json!({}), json!({"result":list(&["echo"])})),
                (
                    "tools/call",
                    json!({"name":"echo","arguments":{}}),
                    json!({"result":{"content":"late"}}),
                ),
                ("ping", json!({}), json!({"result":"pong"})),
            ],
        );
        fixture.change_script(|script| {
            let late = json!({"jsonrpc":"2.0","id":3,"result":{"content":"late"}}).to_string();
            let ping = json!({"jsonrpc":"2.0","id":4,"result":"pong"}).to_string();
            let notification =
                json!({"jsonrpc":"2.0","method":"call/received","params":{}}).to_string();
            #[cfg(unix)]
            let script = script
                .replace(
                    &format!("printf '%s\\n' '{late}'"),
                    &format!("printf '%s\\n' '{notification}'"),
                )
                .replace(
                    &format!("printf '%s\\n' '{ping}'"),
                    &format!("printf '%s\\n' '{late}'\nprintf '%s\\n' '{ping}'"),
                );
            #[cfg(windows)]
            let script = script
                .replace(&format!("echo {late}"), &format!("echo {notification}"))
                .replace(
                    &format!("echo {ping}"),
                    &format!("echo {late}\r\necho {ping}"),
                );
            script
        });
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let set = ExternalToolSet::load(&process).await.unwrap();
        let cancel = CancellationToken::new();
        let context = ToolContext::new(std::env::temp_dir()).unwrap();
        let tool = &set.tools()[0];
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            tokio::join!(
                tool.execute(
                    json!({}),
                    &context,
                    tokio::sync::mpsc::channel(1).0,
                    cancel.clone()
                ),
                async {
                    assert_eq!(
                        process.recv_notification().await.unwrap().method,
                        "call/received"
                    );
                    cancel.cancel();
                }
            )
            .0
        })
        .await
        .unwrap();
        assert!(matches!(result, Err(ToolError::Cancelled)));
        assert_eq!(process.request("ping", json!({})).await.unwrap(), "pong");
        process.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pre_cancelled_call_sends_nothing_and_transport_errors_identify_tool() {
        let fixture = Fixture::scripted(
            true,
            vec![("tools/list", json!({}), json!({"result":list(&["echo"])}))],
        );
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let set = ExternalToolSet::load(&process).await.unwrap();
        let context = ToolContext::new(std::env::temp_dir()).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = set.tools()[0]
            .execute(json!({}), &context, tokio::sync::mpsc::channel(1).0, cancel)
            .await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
        process.shutdown().await.unwrap();
        let result = set.tools()[0]
            .execute(
                json!({}),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await;
        assert!(
            matches!(result, Err(ToolError::Failed(message)) if message.contains("test.echo") && message.contains("echo"))
        );

        let fixture = Fixture::scripted(
            true,
            vec![
                ("tools/list", json!({}), json!({"result":list(&["echo"])})),
                (
                    "tools/call",
                    json!({"name":"echo","arguments":{}}),
                    json!({"result":"malformed-marker"}),
                ),
            ],
        );
        fixture.change_script(|script| {
            script.replace(
                &json!({"jsonrpc":"2.0","id":3,"result":"malformed-marker"}).to_string(),
                "not JSON",
            )
        });
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let set = ExternalToolSet::load(&process).await.unwrap();
        let result = set.tools()[0]
            .execute(
                json!({}),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await;
        assert!(
            matches!(result, Err(ToolError::Failed(message)) if message.contains("test.echo") && message.contains("echo"))
        );
        assert!(process.shutdown().await.is_err());
    }

    #[test]
    fn validates_names_without_rewriting_definitions() {
        for name in ["snake_case", "hyphen-name", "0Mixed", &"a".repeat(64)] {
            let tool = descriptor(name);
            let definitions = validate_definitions("test.echo", vec![tool.clone()]).unwrap();
            assert_eq!(definitions[0].name, name);
            assert_eq!(definitions[0].description, tool.description);
            assert_eq!(definitions[0].parameters, tool.input_schema);
        }
        for name in ["", &"a".repeat(65), "a.b", "a b", "_abc", "-abc", "écho"] {
            assert!(matches!(
                validate_definitions("test.echo", vec![descriptor(name)]),
                Err(ExternalToolError::InvalidDefinition { .. })
            ));
        }
    }

    #[test]
    fn validates_descriptions_schemas_duplicates_and_count() {
        for description in ["", " \n\t", &"é".repeat(8193)] {
            let mut tool = descriptor("echo");
            tool.description = Some(description.into());
            assert!(matches!(
                validate_definitions("p", vec![tool]),
                Err(ExternalToolError::InvalidDefinition { .. })
            ));
        }
        for description in [None, Some("é".repeat(8192))] {
            let mut tool = descriptor("echo");
            tool.description = description;
            assert!(validate_definitions("p", vec![tool]).is_ok());
        }
        for schema in [
            Value::Null,
            json!([]),
            json!(true),
            json!("object"),
            json!(42),
        ] {
            let mut tool = descriptor("echo");
            tool.input_schema = schema;
            assert!(matches!(
                validate_definitions("p", vec![tool]),
                Err(ExternalToolError::InvalidDefinition { .. })
            ));
        }
        assert!(matches!(
            validate_definitions("p", vec![descriptor("echo"); 2]),
            Err(ExternalToolError::DuplicateDefinition { .. })
        ));
        let tools = (0..128).map(|i| descriptor(&format!("tool{i}"))).collect();
        assert_eq!(validate_definitions("p", tools).unwrap().len(), 128);
        assert!(matches!(
            validate_definitions("p", vec![descriptor("echo"); 129]),
            Err(ExternalToolError::TooManyTools {
                count: 129,
                max: 128,
                ..
            })
        ));
    }
}
