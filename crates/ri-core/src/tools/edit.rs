use std::fs;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::path::resolve_existing;
use super::write::atomic_replace;
use super::MAX_EDIT_FILE_BYTES;
use super::{
    bounded_preview, Tool, ToolCallPresentation, ToolContext, ToolError, ToolEventSender,
    ToolExecutionResult,
};

pub(crate) struct EditTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditArguments {
    path: String,
    old_text: String,
    new_text: String,
}

impl EditArguments {
    fn parse(arguments: &Value) -> Result<Self, ToolError> {
        serde_json::from_value(arguments.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))
    }
}

#[async_trait]
impl Tool for EditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".to_owned(),
            description: Some(
                "Replace exactly one matching text range in a workspace file.".to_owned(),
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        }
    }

    fn presentation(&self, arguments: &Value) -> ToolCallPresentation {
        let Ok(arguments) = EditArguments::parse(arguments) else {
            return ToolCallPresentation::fallback("edit", arguments);
        };
        let old_lines = arguments.old_text.lines().count();
        let new_lines = arguments.new_text.lines().count();
        ToolCallPresentation {
            summary: format!("edit {}", arguments.path),
            preview: Some(bounded_preview(
                arguments
                    .old_text
                    .lines()
                    .map(|line| (Some('-'), line))
                    .chain(arguments.new_text.lines().map(|line| (Some('+'), line))),
                old_lines + new_lines,
            )),
        }
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
        _events: ToolEventSender,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionResult, ToolError> {
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }
        let arguments = EditArguments::parse(&arguments)?;
        if arguments.old_text.is_empty() {
            return Err(ToolError::InvalidArguments(
                "old_text must not be empty".to_owned(),
            ));
        }

        let path = resolve_existing(context, &arguments.path)?;
        let metadata = fs::metadata(&path).map_err(|error| {
            ToolError::Failed(format!("could not inspect {}: {error}", arguments.path))
        })?;
        if !metadata.is_file() {
            return Err(ToolError::Failed(format!(
                "path is not a regular file: {}",
                arguments.path
            )));
        }
        if metadata.len() > MAX_EDIT_FILE_BYTES {
            return Err(ToolError::Failed(format!(
                "file is too large to edit safely ({} bytes; maximum is {} bytes)",
                metadata.len(),
                MAX_EDIT_FILE_BYTES
            )));
        }

        let content = fs::read_to_string(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::InvalidData {
                ToolError::Failed(format!(
                    "{} is not valid UTF-8 text (binary files are not supported)",
                    arguments.path
                ))
            } else {
                ToolError::Failed(format!("could not read {}: {error}", arguments.path))
            }
        })?;
        let matches = content.match_indices(&arguments.old_text).count();
        match matches {
            0 => {
                return Err(ToolError::Failed(format!(
                    "old_text was not found in {}",
                    arguments.path
                )))
            }
            count if count > 1 => {
                return Err(ToolError::Failed(format!(
                    "old_text matched {count} locations in {}; provide more surrounding context",
                    arguments.path
                )))
            }
            _ => {}
        }
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        let replacement_size = content
            .len()
            .saturating_sub(arguments.old_text.len())
            .saturating_add(arguments.new_text.len());
        if replacement_size > MAX_EDIT_FILE_BYTES as usize {
            return Err(ToolError::Failed(format!(
                "edited file would be too large ({replacement_size} bytes; maximum is {} bytes)",
                MAX_EDIT_FILE_BYTES
            )));
        }

        let start = content
            .find(&arguments.old_text)
            .expect("the unique edit match should still exist");
        let end = start + arguments.old_text.len();
        let mut replacement = String::with_capacity(replacement_size);
        replacement.push_str(&content[..start]);
        replacement.push_str(&arguments.new_text);
        replacement.push_str(&content[end..]);
        let bytes_written = atomic_replace(&path, replacement.as_bytes())?;

        Ok(ToolExecutionResult::success(format!(
            "replaced 1 occurrence in {} ({bytes_written} bytes)",
            arguments.path
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn presents_single_line_replacement_as_diff() {
        let presentation = EditTool.presentation(&json!({
            "path": "src/foo.rs",
            "old_text": "let a = 1;",
            "new_text": "let a = 2;"
        }));

        assert_eq!(presentation.summary, "edit src/foo.rs");
        assert_eq!(
            presentation.preview.as_deref(),
            Some("-let a = 1;\n+let a = 2;")
        );
    }

    #[test]
    fn presents_and_bounds_multiline_replacement() {
        let old_text = (1..=15)
            .map(|line| format!("old {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let new_text = (1..=15)
            .map(|line| format!("new {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let presentation = EditTool.presentation(&json!({
            "path": "src/foo.rs",
            "old_text": old_text,
            "new_text": new_text
        }));
        let preview = presentation.preview.unwrap();

        assert!(preview.contains("-old 1"));
        assert!(preview.contains("+new 1"));
        assert!(preview.contains("… 10 more lines …"));
        assert!(preview.lines().count() <= super::super::MAX_TOOL_PREVIEW_LINES + 1);
        assert!(preview.len() <= super::super::MAX_TOOL_PREVIEW_BYTES);
    }

    #[tokio::test]
    async fn replaces_one_exact_multiline_match() {
        let root = unique_test_dir("edit");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file.txt"), "before\nold\ntext\nafter").unwrap();
        let context = ToolContext::new(&root).unwrap();

        let result = EditTool
            .execute(
                json!({
                    "path":"file.txt",
                    "old_text":"old\ntext",
                    "new_text":"new\ntext"
                }),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(result.model_content.contains("replaced 1 occurrence"));
        assert_eq!(
            fs::read_to_string(root.join("file.txt")).unwrap(),
            "before\nnew\ntext\nafter"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_zero_and_multiple_matches() {
        let root = unique_test_dir("edit-errors");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file.txt"), "same\nsame").unwrap();
        let context = ToolContext::new(&root).unwrap();
        let sender = tokio::sync::mpsc::channel(1).0;

        let multiple = EditTool
            .execute(
                json!({"path":"file.txt", "old_text":"same", "new_text":"other"}),
                &context,
                sender.clone(),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(multiple.to_string().contains("matched 2"));

        let missing = EditTool
            .execute(
                json!({"path":"file.txt", "old_text":"missing", "new_text":"other"}),
                &context,
                sender,
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("not found"));
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
