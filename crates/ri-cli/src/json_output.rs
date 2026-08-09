use std::io::Write;

use anyhow::Result;
use ri_core::{
    AgentEvent, ContextUsage, ModelAssistantItem, ModelRef, SessionInfo, StopReason,
    ToolOutputStream, Usage,
};
use serde::Serialize;
use serde_json::json;

pub(crate) const JSON_EVENT_VERSION: u32 = 1;

#[derive(Serialize)]
struct JsonEnvelope<'a, T> {
    version: u32,
    seq: u64,
    #[serde(rename = "type")]
    event_type: &'a str,
    data: T,
}

pub(crate) struct JsonEmitter<W> {
    writer: W,
    next_seq: u64,
}

impl<W: Write> JsonEmitter<W> {
    pub(crate) fn new(writer: W) -> Self {
        Self {
            writer,
            next_seq: 0,
        }
    }

    pub(crate) fn emit<T: Serialize>(&mut self, event_type: &'static str, data: T) -> Result<()> {
        let envelope = JsonEnvelope {
            version: JSON_EVENT_VERSION,
            seq: self.next_seq,
            event_type,
            data,
        };
        serde_json::to_writer(&mut self.writer, &envelope)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.next_seq = self.next_seq.saturating_add(1);
        Ok(())
    }

    pub(crate) fn emit_agent_event(&mut self, event: &AgentEvent) -> Result<()> {
        match event {
            AgentEvent::TurnStarted => self.emit("turn_started", json!({})),
            AgentEvent::AssistantMessageStarted => {
                self.emit("assistant_message_started", json!({}))
            }
            AgentEvent::AssistantTextDelta { text, .. } => {
                self.emit("assistant_text_delta", TextData { text })
            }
            AgentEvent::AssistantRefusalDelta { text, .. } => {
                self.emit("assistant_refusal_delta", TextData { text })
            }
            AgentEvent::AssistantThinkingDelta { item_id, text } => self.emit(
                "assistant_reasoning_delta",
                ReasoningData {
                    item_id: item_id.as_deref(),
                    kind: "summary",
                    text,
                },
            ),
            AgentEvent::AssistantThinkingContentDelta { item_id, text } => self.emit(
                "assistant_reasoning_delta",
                ReasoningData {
                    item_id: item_id.as_deref(),
                    kind: "content",
                    text,
                },
            ),
            AgentEvent::AssistantMessageFinished { items } => self.emit(
                "assistant_message_finished",
                MessageFinishedData {
                    item_count: items.len(),
                    items: items.iter().map(AssistantItemData::from).collect(),
                },
            ),
            AgentEvent::ToolExecutionStarted {
                call_id,
                name,
                arguments,
            } => self.emit(
                "tool_started",
                ToolStartedData {
                    call_id,
                    name,
                    arguments,
                },
            ),
            AgentEvent::ToolExecutionOutput {
                call_id,
                stream,
                chunk,
            } => self.emit(
                "tool_output",
                ToolOutputData {
                    call_id,
                    stream: tool_stream_name(stream),
                    chunk,
                },
            ),
            AgentEvent::ToolExecutionFinished {
                call_id,
                name,
                result,
            } => self.emit(
                "tool_finished",
                ToolFinishedData {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    success: result.metadata.success,
                    exit_code: result.metadata.exit_code,
                    timed_out: result.metadata.timed_out,
                    cancelled: result.metadata.cancelled,
                    truncated: result.metadata.truncated,
                    duration_ms: result.metadata.duration.as_millis() as u64,
                    full_output_path: result
                        .metadata
                        .full_output_path
                        .as_ref()
                        .map(|path| path.display().to_string()),
                },
            ),
            AgentEvent::UsageUpdated(usage) => self.emit("usage", UsageData::from(usage)),
            AgentEvent::ContextUsageUpdated(usage) => {
                self.emit("context_usage", ContextUsageData::from(usage))
            }
            AgentEvent::CompactionStarted { automatic } => self.emit(
                "compaction_started",
                CompactionStartedData {
                    automatic: *automatic,
                },
            ),
            AgentEvent::CompactionFinished {
                automatic,
                before_tokens,
                after_tokens,
            } => self.emit(
                "compaction_finished",
                CompactionFinishedData {
                    automatic: *automatic,
                    before_tokens: *before_tokens,
                    after_tokens: *after_tokens,
                },
            ),
            AgentEvent::CompactionFailed { message } => {
                self.emit("compaction_failed", MessageData { message })
            }
            AgentEvent::SessionChanged { info } => {
                self.emit("session_changed", SessionChangedData::from(info))
            }
            AgentEvent::SessionLoaded { info, .. } => {
                self.emit("session_changed", SessionChangedData::from(info))
            }
            AgentEvent::TurnFinished { reason } => self.emit(
                "turn_finished",
                TurnFinishedData {
                    reason: stop_reason_name(reason),
                },
            ),
            AgentEvent::Error(error) => self.emit(
                "error",
                ErrorData {
                    message: &error.message,
                    fatal: true,
                },
            ),
            AgentEvent::AssistantTextItem { .. }
            | AgentEvent::AssistantRefusalItem { .. }
            | AgentEvent::AssistantThinkingItem { .. }
            | AgentEvent::ToolCallDelta { .. }
            | AgentEvent::ContextLimitsUpdated(_)
            | AgentEvent::ModelChanged(_) => Ok(()),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ModelData {
    pub(crate) provider: String,
    pub(crate) model: String,
}

impl From<&ModelRef> for ModelData {
    fn from(model: &ModelRef) -> Self {
        Self {
            provider: model.provider.clone(),
            model: model.model.clone(),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct SessionData {
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) ephemeral: bool,
}

impl From<Option<&SessionInfo>> for SessionData {
    fn from(info: Option<&SessionInfo>) -> Self {
        match info {
            Some(info) => Self {
                id: Some(info.id.to_string()),
                name: info.name.clone(),
                ephemeral: false,
            },
            None => Self {
                id: None,
                name: None,
                ephemeral: true,
            },
        }
    }
}

#[derive(Serialize)]
pub(crate) struct RunStartedData {
    pub(crate) model: ModelData,
    pub(crate) workspace: String,
    pub(crate) session: SessionData,
}

impl RunStartedData {
    pub(crate) fn new(model: &ModelRef, workspace: String, session: Option<&SessionInfo>) -> Self {
        Self {
            model: ModelData::from(model),
            workspace,
            session: SessionData::from(session),
        }
    }
}

#[derive(Serialize)]
struct TextData<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct ReasoningData<'a> {
    #[serde(rename = "itemId")]
    item_id: Option<&'a str>,
    kind: &'static str,
    text: &'a str,
}

#[derive(Serialize)]
struct MessageFinishedData<'a> {
    #[serde(rename = "itemCount")]
    item_count: usize,
    items: Vec<AssistantItemData<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AssistantItemData<'a> {
    #[serde(rename = "text")]
    Text { content: &'a str },
    #[serde(rename = "refusal")]
    Refusal { content: &'a str },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(rename = "itemId")]
        item_id: Option<&'a str>,
        summary: &'a str,
        content: &'a str,
        #[serde(rename = "encryptedContent")]
        encrypted_content: Option<&'a str>,
    },
    #[serde(rename = "tool_call")]
    ToolCall {
        index: usize,
        #[serde(rename = "callId")]
        call_id: Option<&'a str>,
        #[serde(rename = "itemId")]
        item_id: Option<&'a str>,
        name: Option<&'a str>,
        arguments: &'a str,
    },
}

impl<'a> From<&'a ModelAssistantItem> for AssistantItemData<'a> {
    fn from(item: &'a ModelAssistantItem) -> Self {
        match item {
            ModelAssistantItem::Text { content } => Self::Text { content },
            ModelAssistantItem::Refusal { content } => Self::Refusal { content },
            ModelAssistantItem::Reasoning(thinking) => Self::Reasoning {
                item_id: thinking.item_id.as_deref(),
                summary: &thinking.summary,
                content: &thinking.content,
                encrypted_content: thinking.encrypted_content.as_deref(),
            },
            ModelAssistantItem::ToolCall(call) => Self::ToolCall {
                index: call.index,
                call_id: call.call_id.as_deref(),
                item_id: call.item_id.as_deref(),
                name: call.name.as_deref(),
                arguments: &call.arguments,
            },
        }
    }
}

#[derive(Serialize)]
struct ToolStartedData<'a> {
    #[serde(rename = "callId")]
    call_id: &'a str,
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
struct ToolOutputData<'a> {
    #[serde(rename = "callId")]
    call_id: &'a str,
    stream: &'static str,
    chunk: &'a str,
}

#[derive(Serialize)]
struct ToolFinishedData {
    #[serde(rename = "callId")]
    call_id: String,
    name: String,
    success: bool,
    #[serde(rename = "exitCode")]
    exit_code: Option<i32>,
    #[serde(rename = "timedOut")]
    timed_out: bool,
    cancelled: bool,
    truncated: bool,
    #[serde(rename = "durationMs")]
    duration_ms: u64,
    #[serde(rename = "fullOutputPath")]
    full_output_path: Option<String>,
}

#[derive(Serialize)]
struct UsageData {
    #[serde(rename = "inputTokens")]
    input_tokens: Option<u64>,
    #[serde(rename = "outputTokens")]
    output_tokens: Option<u64>,
    #[serde(rename = "totalTokens")]
    total_tokens: Option<u64>,
    #[serde(rename = "cacheReadTokens")]
    cache_read_tokens: Option<u64>,
    #[serde(rename = "cacheWriteTokens")]
    cache_write_tokens: Option<u64>,
}

impl From<&Usage> for UsageData {
    fn from(usage: &Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            cache_write_tokens: usage.cache_write_tokens,
        }
    }
}

#[derive(Serialize)]
struct ContextUsageData {
    #[serde(rename = "inputTokens")]
    input_tokens: u64,
    estimated: bool,
    #[serde(rename = "contextWindow")]
    context_window: Option<u64>,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: Option<u64>,
}

impl From<&ContextUsage> for ContextUsageData {
    fn from(usage: &ContextUsage) -> Self {
        Self {
            input_tokens: usage.current_tokens(),
            estimated: matches!(usage.source, ri_core::UsageSource::Estimated),
            context_window: usage.context_window,
            max_output_tokens: usage.max_output_tokens,
        }
    }
}

#[derive(Serialize)]
struct CompactionStartedData {
    automatic: bool,
}

#[derive(Serialize)]
struct CompactionFinishedData {
    automatic: bool,
    #[serde(rename = "beforeTokens")]
    before_tokens: u64,
    #[serde(rename = "afterTokens")]
    after_tokens: u64,
}

#[derive(Serialize)]
struct MessageData<'a> {
    message: &'a str,
}

#[derive(Serialize)]
struct SessionChangedData {
    id: String,
    name: Option<String>,
    #[serde(rename = "messageCount")]
    message_count: usize,
}

impl From<&SessionInfo> for SessionChangedData {
    fn from(info: &SessionInfo) -> Self {
        Self {
            id: info.id.to_string(),
            name: info.name.clone(),
            message_count: info.message_count,
        }
    }
}

#[derive(Serialize)]
struct TurnFinishedData {
    reason: &'static str,
}

#[derive(Serialize)]
struct ErrorData<'a> {
    message: &'a str,
    fatal: bool,
}

fn tool_stream_name(stream: &ToolOutputStream) -> &'static str {
    match stream {
        ToolOutputStream::Stdout => "stdout",
        ToolOutputStream::Stderr => "stderr",
    }
}

pub(crate) fn stop_reason_name(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Stop => "stop",
        StopReason::ToolCalls => "tool_calls",
        StopReason::Length => "length",
        StopReason::ContentFilter => "content_filter",
        StopReason::Incomplete => "incomplete",
        StopReason::Cancelled => "cancelled",
        StopReason::Error => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ri_core::{
        ModelLimits, ModelThinking, ModelToolCall, ToolExecutionMetadata, ToolExecutionResult,
    };
    use serde_json::Value;
    use std::time::Duration;

    #[test]
    fn emitter_writes_versioned_single_line_events_with_monotonic_sequences() {
        let mut output = Vec::new();
        let mut emitter = JsonEmitter::new(&mut output);
        emitter
            .emit("run_started", json!({"ok": true}))
            .expect("event should serialize");
        emitter
            .emit("run_finished", json!({"success": true}))
            .expect("event should serialize");

        let records: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["version"], JSON_EVENT_VERSION);
        assert_eq!(records[0]["seq"], 0);
        assert_eq!(records[1]["seq"], 1);
        assert_eq!(records[0]["type"], "run_started");
        assert_eq!(records[1]["type"], "run_finished");
    }

    #[test]
    fn agent_event_wire_names_and_fields_are_explicit() {
        let mut output = Vec::new();
        let mut emitter = JsonEmitter::new(&mut output);
        let mut metadata = ToolExecutionMetadata::success();
        metadata.exit_code = Some(0);
        metadata.duration = Duration::from_millis(12);
        for event in [
            AgentEvent::TurnStarted,
            AgentEvent::AssistantMessageStarted,
            AgentEvent::AssistantTextDelta {
                index: None,
                text: "hello".to_owned(),
            },
            AgentEvent::AssistantRefusalDelta {
                index: None,
                text: "no".to_owned(),
            },
            AgentEvent::AssistantThinkingDelta {
                item_id: Some("item".to_owned()),
                text: "thinking".to_owned(),
            },
            AgentEvent::ToolExecutionStarted {
                call_id: "call".to_owned(),
                name: "bash".to_owned(),
                arguments: "{}".to_owned(),
            },
            AgentEvent::ToolExecutionOutput {
                call_id: "call".to_owned(),
                stream: ToolOutputStream::Stdout,
                chunk: "out\n".to_owned(),
            },
            AgentEvent::ToolExecutionFinished {
                call_id: "call".to_owned(),
                name: "bash".to_owned(),
                result: ToolExecutionResult {
                    model_content: "ok".to_owned(),
                    metadata,
                },
            },
            AgentEvent::UsageUpdated(Usage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                total_tokens: Some(3),
                cache_read_tokens: None,
                cache_write_tokens: None,
            }),
            AgentEvent::ContextUsageUpdated(ContextUsage::estimated(
                4,
                ModelLimits {
                    context_window: Some(10),
                    max_output_tokens: Some(2),
                },
            )),
            AgentEvent::CompactionStarted { automatic: true },
            AgentEvent::CompactionFinished {
                automatic: true,
                before_tokens: 10,
                after_tokens: 5,
            },
            AgentEvent::CompactionFailed {
                message: "failed".to_owned(),
            },
            AgentEvent::TurnFinished {
                reason: StopReason::Stop,
            },
            AgentEvent::Error(ri_core::AgentError::new("bad")),
        ] {
            emitter
                .emit_agent_event(&event)
                .expect("agent event should serialize");
        }

        let types: Vec<String> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(
            types,
            [
                "turn_started",
                "assistant_message_started",
                "assistant_text_delta",
                "assistant_refusal_delta",
                "assistant_reasoning_delta",
                "tool_started",
                "tool_output",
                "tool_finished",
                "usage",
                "context_usage",
                "compaction_started",
                "compaction_finished",
                "compaction_failed",
                "turn_finished",
                "error",
            ]
        );
    }

    #[test]
    fn reasoning_delta_wire_kinds_preserve_summary_and_content() {
        let mut output = Vec::new();
        let mut emitter = JsonEmitter::new(&mut output);
        emitter
            .emit_agent_event(&AgentEvent::AssistantThinkingDelta {
                item_id: Some("reasoning".to_owned()),
                text: "summary".to_owned(),
            })
            .expect("summary delta should serialize");
        emitter
            .emit_agent_event(&AgentEvent::AssistantThinkingContentDelta {
                item_id: Some("reasoning".to_owned()),
                text: "content".to_owned(),
            })
            .expect("content delta should serialize");

        let records: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records[0]["type"], "assistant_reasoning_delta");
        assert_eq!(records[0]["data"]["itemId"], "reasoning");
        assert_eq!(records[0]["data"]["kind"], "summary");
        assert_eq!(records[0]["data"]["text"], "summary");
        assert_eq!(records[1]["data"]["kind"], "content");
        assert_eq!(records[1]["data"]["text"], "content");
    }

    #[test]
    fn assistant_message_finished_contains_normalized_items() {
        let mut output = Vec::new();
        let mut emitter = JsonEmitter::new(&mut output);
        emitter
            .emit_agent_event(&AgentEvent::AssistantMessageFinished {
                items: vec![
                    ri_core::ModelAssistantItem::Text {
                        content: "hello".to_owned(),
                    },
                    ri_core::ModelAssistantItem::Refusal {
                        content: "no".to_owned(),
                    },
                    ri_core::ModelAssistantItem::Reasoning(ModelThinking {
                        item_id: Some("reasoning".to_owned()),
                        summary: "summary".to_owned(),
                        content: "content".to_owned(),
                        encrypted_content: Some("encrypted".to_owned()),
                    }),
                    ri_core::ModelAssistantItem::ToolCall(ModelToolCall {
                        index: 3,
                        call_id: Some("call".to_owned()),
                        item_id: Some("item".to_owned()),
                        name: Some("read".to_owned()),
                        arguments: "{}".to_owned(),
                    }),
                ],
            })
            .expect("finished message should serialize");

        let record: Value = serde_json::from_str(String::from_utf8(output).unwrap().trim())
            .expect("event should be valid JSON");
        assert_eq!(record["type"], "assistant_message_finished");
        assert_eq!(record["data"]["itemCount"], 4);
        assert_eq!(
            record["data"]["items"][0],
            json!({"type": "text", "content": "hello"})
        );
        assert_eq!(
            record["data"]["items"][1],
            json!({"type": "refusal", "content": "no"})
        );
        assert_eq!(
            record["data"]["items"][2],
            json!({
                "type": "reasoning",
                "itemId": "reasoning",
                "summary": "summary",
                "content": "content",
                "encryptedContent": "encrypted"
            })
        );
        assert_eq!(
            record["data"]["items"][3],
            json!({
                "type": "tool_call",
                "index": 3,
                "callId": "call",
                "itemId": "item",
                "name": "read",
                "arguments": "{}"
            })
        );
    }

    #[test]
    fn stop_reasons_have_stable_wire_names() {
        assert_eq!(stop_reason_name(&StopReason::ToolCalls), "tool_calls");
        assert_eq!(
            stop_reason_name(&StopReason::ContentFilter),
            "content_filter"
        );
        assert_eq!(stop_reason_name(&StopReason::Incomplete), "incomplete");
    }
}
