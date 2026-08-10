use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::bash::BashTool;
use super::edit::EditTool;
use super::read::ReadTool;
use super::write::WriteTool;
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

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: vec![
                RegisteredTool {
                    name: "read".to_owned(),
                    tool: Arc::new(ReadTool),
                },
                RegisteredTool {
                    name: "write".to_owned(),
                    tool: Arc::new(WriteTool),
                },
                RegisteredTool {
                    name: "edit".to_owned(),
                    tool: Arc::new(EditTool),
                },
                RegisteredTool {
                    name: "bash".to_owned(),
                    tool: Arc::new(BashTool),
                },
            ],
        }
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
                available: self.names().join(", "),
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

    #[test]
    fn exposes_exactly_the_builtin_tools_and_strict_schemas() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.names(), ["read", "write", "edit", "bash"]);
        assert_eq!(registry.definitions().len(), 4);
        for definition in registry.definitions() {
            assert_eq!(definition.parameters["additionalProperties"], false);
        }
    }

    #[tokio::test]
    async fn unknown_tool_is_recoverable() {
        let registry = ToolRegistry::new();
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
