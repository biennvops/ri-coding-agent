use std::fs;
use std::path::{Path, PathBuf};

use crate::fs::{atomic_write, temporary_path as atomic_temporary_path, AtomicWriteOptions};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::path::resolve_for_write;
use super::{Tool, ToolContext, ToolError, ToolEventSender, ToolExecutionResult};

pub(crate) struct WriteTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArguments {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".to_owned(),
            description: Some("Write complete UTF-8 text to a workspace file.".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
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
        let arguments: WriteArguments = serde_json::from_value(arguments)
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        let path = resolve_for_write(context, &arguments.path)?;
        let existed = path.exists();
        let bytes_written = atomic_replace(&path, arguments.content.as_bytes())?;
        let action = if existed { "replaced" } else { "created" };
        Ok(ToolExecutionResult::success(format!(
            "{action} {} ({bytes_written} bytes)",
            arguments.path
        )))
    }
}

pub(crate) fn atomic_replace(path: &Path, content: &[u8]) -> Result<usize, ToolError> {
    let parent = path.parent().ok_or_else(|| {
        ToolError::Failed(format!(
            "could not determine parent directory for {}",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        ToolError::Failed(format!(
            "could not create parent directory for {}: {error}",
            path.display()
        ))
    })?;

    atomic_write(path, content, AtomicWriteOptions::default()).map_err(|error| {
        ToolError::Failed(format!(
            "could not atomically replace {}: {error}",
            path.display()
        ))
    })?;

    Ok(content.len())
}

pub(crate) fn temporary_path(parent: &Path) -> PathBuf {
    atomic_temporary_path(parent)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[tokio::test]
    async fn creates_replaces_and_creates_nested_files() {
        let root = unique_test_dir("write");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let sender = tokio::sync::mpsc::channel(1).0;

        let first = WriteTool
            .execute(
                json!({"path":"nested/file.txt", "content":"héllo"}),
                &context,
                sender.clone(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(first.model_content.contains("created"));
        assert_eq!(
            fs::read_to_string(root.join("nested/file.txt")).unwrap(),
            "héllo"
        );

        let second = WriteTool
            .execute(
                json!({"path":"nested/file.txt", "content":"updated"}),
                &context,
                sender,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(second.model_content.contains("replaced"));
        assert_eq!(
            fs::read_to_string(root.join("nested/file.txt")).unwrap(),
            "updated"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_workspace_escape() {
        let root = unique_test_dir("write-escape");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = WriteTool
            .execute(
                json!({"path":"../outside.txt", "content":"nope"}),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_absolute_escape_through_missing_components() {
        let root = unique_test_dir("write-absolute-escape");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let workspace = context.workspace_root.clone();
        let outside = workspace
            .parent()
            .unwrap()
            .join(format!("ri-write-outside-{}", std::process::id()));
        let requested = workspace
            .join("missing")
            .join("..")
            .join("..")
            .join(outside.file_name().unwrap())
            .join("file.txt");

        let result = WriteTool
            .execute(
                json!({
                    "path": requested.to_string_lossy(),
                    "content": "nope"
                }),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await;

        assert!(result.is_err());
        assert!(!outside.join("file.txt").exists());
        fs::remove_dir_all(root).unwrap();
        let _ = fs::remove_dir_all(outside);
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
