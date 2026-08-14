use std::fs;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::path::resolve_existing;
use super::{
    BoundedText, Tool, ToolCallPresentation, ToolContext, ToolError, ToolEventSender,
    ToolExecutionResult, ToolOutputKind, ToolSummaryKind,
};
use super::{MAX_READ_FILE_BYTES, MAX_READ_LINES, MAX_TOOL_OUTPUT_BYTES};

const DEFAULT_READ_LIMIT: usize = 200;

pub(crate) struct ReadTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArguments {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl ReadArguments {
    fn parse(arguments: &Value) -> Result<Self, ToolError> {
        serde_json::from_value(arguments.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".to_owned(),
            description: Some("Read text from a workspace file with numbered lines.".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based line to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to return"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    fn presentation(&self, arguments: &Value) -> ToolCallPresentation {
        let Ok(arguments) = ReadArguments::parse(arguments) else {
            return ToolCallPresentation::fallback("read", arguments);
        };
        let mut summary = format!("read {}", arguments.path);
        let summary_kind = if arguments.offset.is_some() || arguments.limit.is_some() {
            let start = arguments.offset.unwrap_or(1);
            let limit = arguments.limit.unwrap_or(DEFAULT_READ_LIMIT);
            let end = start.saturating_add(limit).saturating_sub(1);
            let range_start = summary.len();
            summary.push_str(&format!(" · lines {start}–{end}"));
            ToolSummaryKind::Range { start: range_start }
        } else {
            ToolSummaryKind::Normal
        };
        ToolCallPresentation {
            summary,
            summary_kind,
            output_kind: ToolOutputKind::NumberedLines,
            preview: Vec::new(),
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
        let arguments = ReadArguments::parse(&arguments)?;
        let offset = arguments.offset.unwrap_or(1);
        let requested_limit = arguments.limit.unwrap_or(DEFAULT_READ_LIMIT);
        if offset == 0 || requested_limit == 0 {
            return Err(ToolError::InvalidArguments(
                "offset and limit must be at least 1".to_owned(),
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
        if metadata.len() > MAX_READ_FILE_BYTES {
            return Err(ToolError::Failed(format!(
                "file is too large to read safely ({} bytes; maximum is {} bytes)",
                metadata.len(),
                MAX_READ_FILE_BYTES
            )));
        }

        let bytes = fs::read(&path).map_err(|error| {
            ToolError::Failed(format!("could not read {}: {error}", arguments.path))
        })?;
        let content = String::from_utf8(bytes).map_err(|_| {
            ToolError::Failed(format!(
                "{} is not valid UTF-8 text (binary files are not supported)",
                arguments.path
            ))
        })?;

        let total_lines = content.split_inclusive('\n').count();
        if total_lines == 0 {
            return Ok(ToolExecutionResult::success(format!(
                "{} is empty.",
                arguments.path
            )));
        }
        if offset > total_lines {
            return Err(ToolError::Failed(format!(
                "offset {offset} is outside {} with {total_lines} lines",
                arguments.path
            )));
        }

        let limit = requested_limit.min(MAX_READ_LINES);
        let end_line = offset
            .saturating_add(limit)
            .saturating_sub(1)
            .min(total_lines);
        let mut output = BoundedText::new(MAX_TOOL_OUTPUT_BYTES);
        for (index, line) in content.split_inclusive('\n').enumerate() {
            let line_number = index + 1;
            if line_number < offset {
                continue;
            }
            if line_number > end_line {
                break;
            }
            output.push(&format!("{line_number} | "));
            output.push(trim_line_ending(line));
            output.push("\n");
        }

        let mut result = output.render();
        if end_line < total_lines {
            result.push_str(&format!(
                "\n[Showing lines {offset}-{end_line} of {total_lines}. Continue with offset={}.]",
                end_line + 1
            ));
        }
        if requested_limit > MAX_READ_LINES {
            result.push_str(&format!("\n[Read limit capped at {MAX_READ_LINES} lines.]"));
        }
        if output.is_truncated() {
            result.push_str("\n[Read output was truncated by the byte limit.]");
        }

        Ok(ToolExecutionResult {
            model_content: result,
            metadata: super::ToolExecutionMetadata {
                truncated: output.is_truncated(),
                ..super::ToolExecutionMetadata::success()
            },
        })
    }
}

fn trim_line_ending(line: &str) -> &str {
    line.strip_suffix('\n')
        .unwrap_or(line)
        .strip_suffix('\r')
        .unwrap_or_else(|| line.strip_suffix('\n').unwrap_or(line))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn presents_path_and_requested_line_range() {
        let presentation = ReadTool.presentation(&json!({
            "path": "src/foo.rs",
            "offset": 10,
            "limit": 20
        }));

        assert_eq!(presentation.summary, "read src/foo.rs · lines 10–29");
        assert_eq!(
            presentation.summary_kind,
            ToolSummaryKind::Range {
                start: "read src/foo.rs".len()
            }
        );
        assert_eq!(presentation.output_kind, ToolOutputKind::NumberedLines);
        assert!(presentation.preview.is_empty());
        assert!(!presentation.summary.contains("{\"path\":"));
    }

    #[tokio::test]
    async fn reads_numbered_ranges_and_reports_more_content() {
        let root = unique_test_dir("read");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file.txt"), "one\ntwo\nthree\nfour\n").unwrap();
        let context = ToolContext::new(&root).unwrap();

        let result = ReadTool
            .execute(
                json!({"path":"file.txt", "offset":2, "limit":2}),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(
            result.model_content,
            "2 | two\n3 | three\n\n[Showing lines 2-3 of 4. Continue with offset=4.]"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_binary_and_escape_paths() {
        let root = unique_test_dir("read-errors");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("binary"), [0xff, 0xfe]).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let sender = tokio::sync::mpsc::channel(1).0;

        assert!(ReadTool
            .execute(
                json!({"path":"binary"}),
                &context,
                sender.clone(),
                CancellationToken::new()
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("UTF-8"));
        assert!(ReadTool
            .execute(
                json!({"path":"../outside"}),
                &context,
                sender,
                CancellationToken::new()
            )
            .await
            .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
