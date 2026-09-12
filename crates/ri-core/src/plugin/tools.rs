use std::collections::HashSet;

use serde_json::Value;

use super::process::PluginProcessError;
use super::protocol::PluginToolDefinition;
use crate::model::ToolDefinition;
use crate::tools::ToolRegistryError;

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
