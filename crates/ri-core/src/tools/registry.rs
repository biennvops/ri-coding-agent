use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::{
    Tool, ToolCallPresentation, ToolContext, ToolError, ToolEventSender, ToolExecutionResult,
};

#[derive(Clone)]
struct RegisteredTool {
    name: String,
    tool: Arc<dyn Tool>,
}

#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<RegisteredTool>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolRegistryError {
    #[error("tool {name:?} is already registered")]
    DuplicateTool { name: String },
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("names", &self.names())
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolRegistryError> {
        let name = tool.definition().name;
        if self.tools.iter().any(|registered| registered.name == name) {
            return Err(ToolRegistryError::DuplicateTool { name });
        }
        self.tools.push(RegisteredTool { name, tool });
        Ok(())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|registered| registered.tool.definition())
            .collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|registered| registered.name.clone())
            .collect()
    }

    pub fn presentation(&self, name: &str, arguments: &Value) -> ToolCallPresentation {
        self.tools
            .iter()
            .find(|tool| tool.name == name)
            .map_or_else(
                || ToolCallPresentation::fallback(name, arguments),
                |registered| registered.tool.presentation(arguments),
            )
    }

    pub async fn execute(
        &self,
        name: &str,
        arguments: Value,
        context: &ToolContext,
        events: ToolEventSender,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionResult, ToolError> {
        let Some(registered) = self.tools.iter().find(|tool| tool.name == name) else {
            return Err(ToolError::UnknownTool {
                name: name.to_owned(),
                available: if self.tools.is_empty() {
                    "(none)".to_owned()
                } else {
                    self.names().join(", ")
                },
            });
        };
        registered
            .tool
            .execute(arguments, context, events, cancel)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tools::tests::EchoTool;

    #[test]
    fn registration_preserves_order_and_rejects_duplicates() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool("foo"))).unwrap();
        registry.register(Arc::new(EchoTool("bar"))).unwrap();
        assert_eq!(registry.names(), ["foo", "bar"]);
        assert_eq!(
            registry
                .definitions()
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["foo", "bar"]
        );

        let mut registry = crate::tools::builtin_tool_registry();
        let before = registry.definitions();
        for tool in [
            Arc::new(EchoTool("read")) as Arc<dyn Tool>,
            Arc::new(crate::tools::read::ReadTool),
        ] {
            assert_eq!(
                registry.register(tool),
                Err(ToolRegistryError::DuplicateTool {
                    name: "read".to_owned()
                })
            );
            assert_eq!(registry.definitions(), before);
        }
    }

    #[tokio::test]
    async fn empty_registry_is_valid_and_unknown_call_is_recoverable() {
        let registry = ToolRegistry::new();
        assert!(registry.names().is_empty());
        assert!(registry.definitions().is_empty());
        assert_eq!(registry.presentation("nope", &Value::Null).summary, "nope");
        let error = registry
            .execute(
                "nope",
                Value::Null,
                &ToolContext::new(std::env::temp_dir()).unwrap(),
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(&error, ToolError::UnknownTool { name, available } if name == "nope" && available == "(none)")
        );
        assert_eq!(
            error.to_string(),
            "unknown tool \"nope\"; available tools: (none)"
        );
    }

    #[tokio::test]
    async fn custom_tool_is_visible_presented_and_executable() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool("echo"))).unwrap();
        assert_eq!(registry.names(), ["echo"]);
        assert_eq!(registry.definitions(), [EchoTool("echo").definition()]);
        let arguments = serde_json::json!({"text": "hello"});
        assert_eq!(
            registry.presentation("echo", &arguments).summary,
            "Echo: hello"
        );
        let result = registry
            .execute(
                "echo",
                arguments,
                &ToolContext::new(std::env::temp_dir()).unwrap(),
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(result, ToolExecutionResult::success("hello"));
    }

    #[test]
    fn exposes_exactly_the_builtin_tools_and_strict_schemas() {
        let registry = crate::tools::builtin_tool_registry();
        assert_eq!(registry.names(), ["read", "write", "edit", "bash"]);
        assert_eq!(registry.definitions().len(), 4);
        for definition in registry.definitions() {
            assert_eq!(definition.parameters["additionalProperties"], false);
        }
    }

    #[tokio::test]
    async fn unknown_tool_is_recoverable() {
        let registry = crate::tools::builtin_tool_registry();
        let context = crate::tools::ToolContext::new(std::env::temp_dir()).unwrap();
        let error = registry
            .execute(
                "nope",
                Value::Object(Default::default()),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("nope"));
        assert!(error.to_string().contains("read, write, edit, bash"));
    }
}
