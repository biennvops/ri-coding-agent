use std::fs;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::path::resolve_existing;
use super::{
    format_size, Tool, ToolCallPresentation, ToolContext, ToolError, ToolEventSender,
    ToolExecutionResult, ToolOutputKind, ToolSummaryKind,
};
use super::{MAX_READ_FILE_BYTES, MAX_TOOL_OUTPUT_BYTES, MAX_TOOL_OUTPUT_LINES};

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
            description: Some(
                "Read text from a file with numbered lines. Output is truncated to 2000 lines or 50KB (whichever is hit first); use offset/limit and continue with offset until complete."
                    .to_owned(),
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the workspace or absolute"
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

        let limit = requested_limit.min(MAX_TOOL_OUTPUT_LINES);
        let end_line = offset
            .saturating_add(limit)
            .saturating_sub(1)
            .min(total_lines);
        let mut output = String::new();
        let mut last_shown = offset - 1;
        let mut byte_limited = false;
        for (index, line) in content.split_inclusive('\n').enumerate() {
            let line_number = index + 1;
            if line_number < offset {
                continue;
            }
            if line_number > end_line {
                break;
            }
            let rendered = format!("{line_number} | {}\n", trim_line_ending(line));
            if output.len() + rendered.len() > MAX_TOOL_OUTPUT_BYTES {
                byte_limited = true;
                break;
            }
            output.push_str(&rendered);
            last_shown = line_number;
        }

        if last_shown < offset {
            let line_size = format_size(
                trim_line_ending(
                    content
                        .split_inclusive('\n')
                        .nth(offset - 1)
                        .unwrap_or_default(),
                )
                .len(),
            );
            return Ok(ToolExecutionResult {
                model_content: format!(
                    "[Line {offset} is {line_size}, exceeds {} limit. Use bash: sed -n '{offset}p' {} | head -c {MAX_TOOL_OUTPUT_BYTES}]",
                    format_size(MAX_TOOL_OUTPUT_BYTES),
                    arguments.path
                ),
                metadata: super::ToolExecutionMetadata {
                    truncated: true,
                    ..super::ToolExecutionMetadata::success()
                },
            });
        }

        let mut result = output;
        if byte_limited {
            result.push_str(&format!(
                "\n[Showing lines {offset}-{last_shown} of {total_lines} ({} limit). Continue with offset={}.]",
                format_size(MAX_TOOL_OUTPUT_BYTES),
                last_shown + 1
            ));
        } else if end_line < total_lines {
            result.push_str(&format!(
                "\n[Showing lines {offset}-{end_line} of {total_lines}. Continue with offset={}.]",
                end_line + 1
            ));
        }
        if requested_limit > MAX_TOOL_OUTPUT_LINES {
            result.push_str(&format!(
                "\n[Read limit capped at {MAX_TOOL_OUTPUT_LINES} lines.]"
            ));
        }

        Ok(ToolExecutionResult {
            model_content: result,
            metadata: super::ToolExecutionMetadata {
                truncated: byte_limited,
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

    async fn read_file(name: &str, content: &str, arguments: Value) -> ToolExecutionResult {
        let root = unique_test_dir(name);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file.txt"), content).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = ReadTool
            .execute(
                arguments,
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        fs::remove_dir_all(root).unwrap();
        result
    }

    #[tokio::test]
    async fn line_cap_matches_the_shared_tool_limit() {
        let total = MAX_TOOL_OUTPUT_LINES + 500;
        let content = (1..=total).map(|n| format!("{n}\n")).collect::<String>();
        let result = read_file(
            "read-line-cap",
            &content,
            json!({"path": "file.txt", "limit": total}),
        )
        .await;

        assert!(result.model_content.contains(&format!(
            "\n{MAX_TOOL_OUTPUT_LINES} | {MAX_TOOL_OUTPUT_LINES}\n"
        )));
        assert!(!result
            .model_content
            .contains(&format!("{} | ", MAX_TOOL_OUTPUT_LINES + 1)));
        assert!(result.model_content.contains(&format!(
            "[Showing lines 1-{MAX_TOOL_OUTPUT_LINES} of {total}. Continue with offset={}.]",
            MAX_TOOL_OUTPUT_LINES + 1
        )));
        assert!(result.model_content.contains(&format!(
            "[Read limit capped at {MAX_TOOL_OUTPUT_LINES} lines.]"
        )));
        assert!(!result.metadata.truncated);
    }

    #[tokio::test]
    async fn byte_cap_keeps_whole_lines_and_says_where_to_continue() {
        let line = "x".repeat(99);
        let content = (0..1000).map(|_| format!("{line}\n")).collect::<String>();
        let result = read_file(
            "read-byte-cap",
            &content,
            json!({"path": "file.txt", "limit": 1000}),
        )
        .await;

        assert!(result.metadata.truncated);
        assert!(result.model_content.len() < MAX_TOOL_OUTPUT_BYTES + 256);
        let notice = result.model_content.lines().last().unwrap();
        let shown: usize = notice
            .strip_prefix("[Showing lines 1-")
            .and_then(|rest| rest.split(' ').next())
            .and_then(|n| n.parse().ok())
            .expect("byte-limit notice");
        assert!(notice.contains("of 1000 (50.0KB limit)"));
        assert!(notice.ends_with(&format!("Continue with offset={}.]", shown + 1)));
        assert!(result
            .model_content
            .contains(&format!("\n{shown} | {line}\n")));
        assert!(!result
            .model_content
            .contains(&format!("\n{} | ", shown + 1)));
    }

    #[tokio::test]
    async fn line_longer_than_the_byte_cap_points_at_bash() {
        let content = format!("short\n{}\nafter\n", "y".repeat(MAX_TOOL_OUTPUT_BYTES * 2));
        let result = read_file(
            "read-huge-line",
            &content,
            json!({"path": "file.txt", "offset": 2}),
        )
        .await;

        assert!(result.metadata.truncated);
        assert!(result
            .model_content
            .starts_with("[Line 2 is 100.0KB, exceeds 50.0KB limit."));
        assert!(result
            .model_content
            .contains("sed -n '2p' file.txt | head -c 51200"));
    }

    #[tokio::test]
    async fn rejects_binary_files() {
        let root = unique_test_dir("read-errors");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("binary"), [0xff, 0xfe]).unwrap();
        let context = ToolContext::new(&root).unwrap();

        assert!(ReadTool
            .execute(
                json!({"path":"binary"}),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new()
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("UTF-8"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn reads_absolute_paths_outside_the_workspace() {
        let root = unique_test_dir("read-workspace");
        let outside = unique_test_dir("read-outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("log.txt"), "spilled\n").unwrap();
        let context = ToolContext::new(&root).unwrap();

        let result = ReadTool
            .execute(
                json!({"path": outside.join("log.txt").to_string_lossy()}),
                &context,
                tokio::sync::mpsc::channel(1).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.model_content, "1 | spilled\n");
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
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
