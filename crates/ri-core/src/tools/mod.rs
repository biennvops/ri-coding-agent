use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

mod bash;
mod edit;
mod path;
mod read;
mod registry;
mod write;

pub use bash::DEFAULT_BASH_TIMEOUT_MS;
pub use registry::ToolRegistry;

pub const MAX_TOOL_OUTPUT_BYTES: usize = 1024 * 1024;
pub const MAX_TOOL_PREVIEW_LINES: usize = 20;
pub const MAX_TOOL_PREVIEW_BYTES: usize = 16 * 1024;
pub const MAX_READ_LINES: usize = 1_000;
pub const MAX_READ_FILE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_EDIT_FILE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct ToolContext {
    pub workspace_root: PathBuf,
}

impl ToolContext {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Result<Self, ToolError> {
        let workspace_root = std::fs::canonicalize(workspace_root.into()).map_err(|error| {
            ToolError::Failed(format!("could not resolve workspace root: {error}"))
        })?;
        if !workspace_root.is_dir() {
            return Err(ToolError::Failed(format!(
                "workspace root is not a directory: {}",
                workspace_root.display()
            )));
        }
        Ok(Self { workspace_root })
    }

    pub fn from_current_dir() -> Result<Self, ToolError> {
        Self::new(std::env::current_dir().map_err(|error| {
            ToolError::Failed(format!("could not determine workspace root: {error}"))
        })?)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolOutputStream {
    Stdout,
    Stderr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolEvent {
    Output {
        stream: ToolOutputStream,
        chunk: String,
    },
}

pub type ToolEventSender = mpsc::Sender<ToolEvent>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecutionMetadata {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub truncated: bool,
    pub duration: Duration,
    pub full_output_path: Option<PathBuf>,
}

impl ToolExecutionMetadata {
    pub fn success() -> Self {
        Self {
            success: true,
            exit_code: None,
            timed_out: false,
            cancelled: false,
            truncated: false,
            duration: Duration::ZERO,
            full_output_path: None,
        }
    }

    pub fn failure() -> Self {
        Self {
            success: false,
            ..Self::success()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecutionResult {
    pub model_content: String,
    pub metadata: ToolExecutionMetadata,
}

impl ToolExecutionResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            model_content: content.into(),
            metadata: ToolExecutionMetadata::success(),
        }
    }

    pub fn failure(content: impl Into<String>) -> Self {
        Self {
            model_content: content.into(),
            metadata: ToolExecutionMetadata::failure(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCallPresentation {
    pub summary: String,
    pub preview: Option<String>,
}

impl ToolCallPresentation {
    pub(crate) fn fallback(name: &str, arguments: &Value) -> Self {
        let arguments = serde_json::to_string(arguments)
            .unwrap_or_else(|_| "<arguments unavailable>".to_owned());
        Self {
            summary: name.to_owned(),
            preview: Some(bounded_preview(
                arguments.lines().map(|line| (None, line)),
                arguments.lines().count(),
            )),
        }
    }
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("{0}")]
    Failed(String),
    #[error("tool execution cancelled")]
    Cancelled,
    #[error("unknown tool {name:?}; available tools: {available}")]
    UnknownTool { name: String, available: String },
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> crate::model::ToolDefinition;

    fn presentation(&self, arguments: &Value) -> ToolCallPresentation {
        ToolCallPresentation::fallback(&self.definition().name, arguments)
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
        events: ToolEventSender,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionResult, ToolError>;
}

pub(crate) fn bounded_preview<'a>(
    lines: impl Iterator<Item = (Option<char>, &'a str)>,
    total_lines: usize,
) -> String {
    bounded_preview_with_limits(
        lines,
        total_lines,
        MAX_TOOL_PREVIEW_LINES,
        MAX_TOOL_PREVIEW_BYTES,
    )
}

pub(crate) fn bounded_preview_with_limits<'a>(
    lines: impl Iterator<Item = (Option<char>, &'a str)>,
    total_lines: usize,
    max_lines: usize,
    max_bytes: usize,
) -> String {
    const TRUNCATED_MARKER: &str = "… content preview truncated …";
    const MARKER_RESERVE: usize = 64;

    if total_lines == 0 {
        return "(empty content)".to_owned();
    }

    let mut preview = String::new();
    let mut shown = 0usize;
    for (prefix, line) in lines.take(max_lines) {
        let separator_bytes = usize::from(!preview.is_empty());
        let prefix_bytes = usize::from(prefix.is_some());
        let content_limit = max_bytes.saturating_sub(MARKER_RESERVE);
        let available = content_limit
            .saturating_sub(preview.len())
            .saturating_sub(separator_bytes)
            .saturating_sub(prefix_bytes);
        if separator_bytes == 1 {
            preview.push('\n');
        }
        if let Some(prefix) = prefix {
            preview.push(prefix);
        }
        if line.len() > available {
            preview.push_str(prefix_at_byte_boundary(line, available));
            preview.push('\n');
            preview.push_str(TRUNCATED_MARKER);
            return preview;
        }
        preview.push_str(line);
        shown += 1;
    }

    if shown < total_lines {
        if !preview.is_empty() {
            preview.push('\n');
        }
        preview.push_str(&format!("… {} more lines …", total_lines - shown));
    }
    preview
}

#[derive(Debug)]
pub(crate) struct BoundedText {
    limit: usize,
    total_bytes: usize,
    truncated: bool,
    head: String,
    tail: String,
}

impl BoundedText {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(2),
            total_bytes: 0,
            truncated: false,
            head: String::new(),
            tail: String::new(),
        }
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub(crate) fn is_truncated(&self) -> bool {
        self.truncated
    }

    pub(crate) fn push(&mut self, text: &str) {
        self.total_bytes = self.total_bytes.saturating_add(text.len());
        if !self.truncated && self.head.len().saturating_add(text.len()) <= self.limit {
            self.head.push_str(text);
            return;
        }

        let head_limit = self.limit / 2;
        let tail_limit = self.limit.saturating_sub(head_limit);
        if !self.truncated {
            let previous_head = std::mem::take(&mut self.head);
            self.head = prefix_at_byte_boundary(&previous_head, head_limit).to_owned();
            self.tail = if text.len() >= tail_limit {
                suffix_at_byte_boundary(text, tail_limit).to_owned()
            } else {
                let mut tail =
                    suffix_at_byte_boundary(&previous_head, tail_limit.saturating_sub(text.len()))
                        .to_owned();
                tail.push_str(text);
                tail
            };
        } else {
            self.tail = if text.len() >= tail_limit {
                suffix_at_byte_boundary(text, tail_limit).to_owned()
            } else {
                let mut tail =
                    suffix_at_byte_boundary(&self.tail, tail_limit.saturating_sub(text.len()))
                        .to_owned();
                tail.push_str(text);
                tail
            };
        }
        self.truncated = true;
    }

    pub(crate) fn retained(&self) -> &str {
        &self.head
    }

    pub(crate) fn render(&self) -> String {
        if !self.truncated {
            return self.head.clone();
        }
        format!("{}\n[… output truncated …]\n{}", self.head, self.tail)
    }
}

pub(crate) fn prefix_at_byte_boundary(text: &str, limit: usize) -> &str {
    let limit = limit.min(text.len());
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

pub(crate) fn suffix_at_byte_boundary(text: &str, limit: usize) -> &str {
    let limit = limit.min(text.len());
    let mut start = text.len().saturating_sub(limit);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_text_retains_head_and_tail() {
        let mut output = BoundedText::new(10);
        output.push("0123456789");
        output.push("abcdefghij");

        assert!(output.is_truncated());
        assert_eq!(output.total_bytes(), 20);
        assert!(output.render().contains("01234"));
        assert!(output.render().contains("fghij"));
        assert!(output.render().is_char_boundary(output.render().len()));
    }

    #[test]
    fn bounded_text_does_not_split_unicode() {
        let mut output = BoundedText::new(8);
        output.push("🦀🦀🦀🦀");
        output.push("終わり");

        assert!(std::str::from_utf8(output.render().as_bytes()).is_ok());
    }
}
