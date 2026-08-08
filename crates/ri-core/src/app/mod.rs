use unicode_segmentation::UnicodeSegmentation;

use crate::agent::{AgentError, AgentEvent};
use crate::config::ModelRef;
use crate::model::{ModelAssistantItem, ModelMessage, StopReason};
use crate::session::SessionInfo;
use crate::tools::{ToolExecutionMetadata, ToolOutputStream};

const MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES: usize = 256 * 1024;
const MAX_TOOL_TRANSCRIPT_ARGUMENT_BYTES: usize = 16 * 1024;
const TOOL_OUTPUT_MARKER: &str = "\n[… output truncated …]\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptMessage {
    pub role: MessageRole,
    pub content: String,
    pub thinking: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Finished(ToolExecutionMetadata),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolTranscriptEntry {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub output: String,
    pub output_truncated: bool,
    pub status: ToolStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptEntry {
    Message(TranscriptMessage),
    Tool(ToolTranscriptEntry),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StreamingAssistant {
    content: String,
    thinking: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AppState {
    messages: Vec<TranscriptMessage>,
    entries: Vec<TranscriptEntry>,
    streaming_assistant: Option<StreamingAssistant>,
    input: String,
    cursor: usize,
    turn_active: bool,
    last_error: Option<String>,
    last_stop_reason: Option<StopReason>,
    active_model: Option<ModelRef>,
    session_info: Option<SessionInfo>,
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn messages(&self) -> &[TranscriptMessage] {
        &self.messages
    }

    pub fn transcript_entries(&self) -> &[TranscriptEntry] {
        &self.entries
    }

    pub fn streaming_assistant(&self) -> Option<(&str, &str)> {
        self.streaming_assistant
            .as_ref()
            .map(|assistant| (assistant.content.as_str(), assistant.thinking.as_str()))
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn set_cursor(&mut self, cursor: usize) {
        if !self.turn_active && cursor <= self.input.len() && self.input.is_char_boundary(cursor) {
            self.cursor = cursor;
        }
    }

    pub fn is_turn_active(&self) -> bool {
        self.turn_active
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn last_stop_reason(&self) -> Option<&StopReason> {
        self.last_stop_reason.as_ref()
    }

    pub fn active_model(&self) -> Option<&ModelRef> {
        self.active_model.as_ref()
    }

    pub fn session_info(&self) -> Option<&SessionInfo> {
        self.session_info.as_ref()
    }

    pub fn replace_history(&mut self, history: &[ModelMessage]) {
        self.messages.clear();
        self.entries.clear();
        self.streaming_assistant = None;
        self.input.clear();
        self.cursor = 0;
        self.turn_active = false;
        self.last_error = None;
        self.last_stop_reason = None;
        for message in history {
            self.append_semantic_message(message);
        }
    }

    pub fn set_session_info(&mut self, info: Option<SessionInfo>) {
        self.session_info = info;
    }

    pub fn insert_text(&mut self, text: &str) {
        if self.turn_active || text.is_empty() {
            return;
        }
        self.input.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    pub fn insert_newline(&mut self) {
        self.insert_text("\n");
    }

    pub fn backspace(&mut self) {
        if self.turn_active || self.cursor == 0 {
            return;
        }
        let previous = previous_grapheme_boundary(&self.input, self.cursor);
        self.input.drain(previous..self.cursor);
        self.cursor = previous;
    }

    pub fn delete(&mut self) {
        if self.turn_active || self.cursor == self.input.len() {
            return;
        }
        let next = next_grapheme_boundary(&self.input, self.cursor);
        self.input.drain(self.cursor..next);
    }

    pub fn move_left(&mut self) {
        if !self.turn_active {
            self.cursor = previous_grapheme_boundary(&self.input, self.cursor);
        }
    }

    pub fn move_right(&mut self) {
        if !self.turn_active {
            self.cursor = next_grapheme_boundary(&self.input, self.cursor);
        }
    }

    pub fn move_home(&mut self) {
        if !self.turn_active {
            self.cursor = line_start(&self.input, self.cursor);
        }
    }

    pub fn move_end(&mut self) {
        if !self.turn_active {
            self.cursor = line_end(&self.input, self.cursor);
        }
    }

    pub fn take_input(&mut self) -> Option<String> {
        if self.turn_active || self.input.trim().is_empty() {
            return None;
        }
        let text = std::mem::take(&mut self.input);
        self.cursor = 0;
        Some(text)
    }

    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.push_message(TranscriptMessage {
            role: MessageRole::System,
            content: content.into(),
            thinking: None,
        });
    }

    pub fn submit_input(&mut self) -> Option<String> {
        let text = self.take_input()?;
        self.finalize_streaming_assistant();
        self.push_message(TranscriptMessage {
            role: MessageRole::User,
            content: text.clone(),
            thinking: None,
        });
        self.turn_active = true;
        self.last_error = None;
        self.last_stop_reason = None;
        Some(text)
    }

    pub fn reduce(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TurnStarted => {
                self.turn_active = true;
                self.last_error = None;
                self.last_stop_reason = None;
            }
            AgentEvent::AssistantMessageStarted => {
                self.finalize_streaming_assistant();
                self.streaming_assistant = Some(StreamingAssistant::default());
            }
            AgentEvent::AssistantTextDelta { text, .. } => {
                self.streaming_assistant
                    .get_or_insert_with(StreamingAssistant::default)
                    .content
                    .push_str(&text);
            }
            AgentEvent::AssistantThinkingDelta { text, .. } => {
                self.streaming_assistant
                    .get_or_insert_with(StreamingAssistant::default)
                    .thinking
                    .push_str(&text);
            }
            AgentEvent::AssistantRefusalDelta { text, .. } => {
                self.streaming_assistant
                    .get_or_insert_with(StreamingAssistant::default)
                    .content
                    .push_str(&text);
            }
            AgentEvent::AssistantRefusalItem { content, .. } => {
                if let Some(content) = content {
                    let assistant = self
                        .streaming_assistant
                        .get_or_insert_with(StreamingAssistant::default);
                    if assistant.content.is_empty() {
                        assistant.content = content;
                    }
                }
            }
            AgentEvent::AssistantMessageFinished { items } => {
                let assistant = self
                    .streaming_assistant
                    .get_or_insert_with(StreamingAssistant::default);
                if assistant.content.is_empty() {
                    assistant.content = items
                        .iter()
                        .find_map(|item| match item {
                            ModelAssistantItem::Text { content }
                            | ModelAssistantItem::Refusal { content } => Some(content.clone()),
                            ModelAssistantItem::Reasoning(_) | ModelAssistantItem::ToolCall(_) => {
                                None
                            }
                        })
                        .unwrap_or_default();
                }
                if assistant.thinking.is_empty() {
                    assistant.thinking = items
                        .iter()
                        .find_map(|item| match item {
                            ModelAssistantItem::Reasoning(thinking) => {
                                (!thinking.content.is_empty()).then_some(thinking.content.clone())
                            }
                            ModelAssistantItem::Text { .. }
                            | ModelAssistantItem::Refusal { .. }
                            | ModelAssistantItem::ToolCall(_) => None,
                        })
                        .unwrap_or_default();
                }
                self.finalize_streaming_assistant();
            }
            AgentEvent::ToolExecutionStarted {
                call_id,
                name,
                arguments,
            } => {
                self.entries
                    .push(TranscriptEntry::Tool(ToolTranscriptEntry {
                        call_id,
                        name,
                        arguments: truncate_text(&arguments, MAX_TOOL_TRANSCRIPT_ARGUMENT_BYTES),
                        output: String::new(),
                        output_truncated: false,
                        status: ToolStatus::Running,
                    }));
            }
            AgentEvent::ToolExecutionOutput {
                call_id,
                stream,
                chunk,
            } => {
                if let Some(TranscriptEntry::Tool(tool)) = self.entries.iter_mut().rev().find(
                    |entry| matches!(entry, TranscriptEntry::Tool(tool) if tool.call_id == call_id),
                ) {
                    if matches!(stream, ToolOutputStream::Stderr) && tool.output.is_empty() {
                        append_tool_output(tool, "[stderr]\n");
                    }
                    append_tool_output(tool, &chunk);
                }
            }
            AgentEvent::ToolExecutionFinished {
                call_id,
                name: _,
                result,
            } => {
                if let Some(TranscriptEntry::Tool(tool)) = self.entries.iter_mut().rev().find(
                    |entry| matches!(entry, TranscriptEntry::Tool(tool) if tool.call_id == call_id),
                ) {
                    let live_output_truncated = tool.output_truncated;
                    tool.output.clear();
                    tool.output_truncated = false;
                    append_tool_output(tool, &result.model_content);
                    tool.output_truncated |= live_output_truncated || result.metadata.truncated;
                    tool.status = ToolStatus::Finished(result.metadata);
                }
            }
            AgentEvent::TurnFinished { reason } => {
                self.turn_active = false;
                self.last_stop_reason = Some(reason);
            }
            AgentEvent::AssistantTextItem { .. }
            | AgentEvent::AssistantThinkingContentDelta { .. }
            | AgentEvent::AssistantThinkingItem { .. }
            | AgentEvent::ToolCallDelta { .. }
            | AgentEvent::UsageUpdated(_) => {}
            AgentEvent::ModelChanged(model) => {
                self.active_model = Some(model);
            }
            AgentEvent::SessionChanged { info } => {
                self.session_info = Some(info);
            }
            AgentEvent::SessionLoaded { info, history } => {
                self.session_info = Some(info);
                self.replace_history(&history);
            }
            AgentEvent::Error(AgentError { message }) => {
                self.last_error = Some(message);
            }
        }
    }

    fn push_message(&mut self, message: TranscriptMessage) {
        self.messages.push(message.clone());
        self.entries.push(TranscriptEntry::Message(message));
    }

    fn append_semantic_message(&mut self, message: &ModelMessage) {
        match message {
            ModelMessage::User { content } => self.push_message(TranscriptMessage {
                role: MessageRole::User,
                content: content.clone(),
                thinking: None,
            }),
            ModelMessage::Assistant { items } => {
                let content = items
                    .iter()
                    .filter_map(|item| match item {
                        ModelAssistantItem::Text { content }
                        | ModelAssistantItem::Refusal { content } => Some(content.as_str()),
                        ModelAssistantItem::Reasoning(_) | ModelAssistantItem::ToolCall(_) => None,
                    })
                    .collect::<String>();
                let thinking = items
                    .iter()
                    .filter_map(|item| match item {
                        ModelAssistantItem::Reasoning(thinking) => {
                            if thinking.content.is_empty() {
                                (!thinking.summary.is_empty()).then_some(thinking.summary.as_str())
                            } else {
                                Some(thinking.content.as_str())
                            }
                        }
                        ModelAssistantItem::Text { .. }
                        | ModelAssistantItem::Refusal { .. }
                        | ModelAssistantItem::ToolCall(_) => None,
                    })
                    .collect::<String>();
                if !content.is_empty() || !thinking.is_empty() {
                    self.push_message(TranscriptMessage {
                        role: MessageRole::Assistant,
                        content,
                        thinking: (!thinking.is_empty()).then_some(thinking),
                    });
                }
                for item in items {
                    let ModelAssistantItem::ToolCall(call) = item else {
                        continue;
                    };
                    let (Some(call_id), Some(name)) = (call.call_id.clone(), call.name.clone())
                    else {
                        continue;
                    };
                    self.entries
                        .push(TranscriptEntry::Tool(ToolTranscriptEntry {
                            call_id,
                            name,
                            arguments: truncate_text(
                                &call.arguments,
                                MAX_TOOL_TRANSCRIPT_ARGUMENT_BYTES,
                            ),
                            output: String::new(),
                            output_truncated: false,
                            status: ToolStatus::Running,
                        }));
                }
            }
            ModelMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
            } => {
                let found = self.entries.iter_mut().rev().find_map(|entry| match entry {
                    TranscriptEntry::Tool(tool) if tool.call_id == *tool_call_id => Some(tool),
                    _ => None,
                });
                if let Some(tool) = found {
                    tool.output.clear();
                    tool.output_truncated = false;
                    append_tool_output(tool, content);
                    tool.status = ToolStatus::Finished(ToolExecutionMetadata::success());
                } else {
                    let mut tool = ToolTranscriptEntry {
                        call_id: tool_call_id.clone(),
                        name: tool_name.clone(),
                        arguments: String::new(),
                        output: String::new(),
                        output_truncated: false,
                        status: ToolStatus::Finished(ToolExecutionMetadata::success()),
                    };
                    append_tool_output(&mut tool, content);
                    self.entries.push(TranscriptEntry::Tool(tool));
                }
            }
            ModelMessage::System { .. } | ModelMessage::Developer { .. } => {}
        }
    }

    fn finalize_streaming_assistant(&mut self) {
        let Some(assistant) = self.streaming_assistant.take() else {
            return;
        };
        if !assistant.content.is_empty() || !assistant.thinking.is_empty() {
            self.push_message(TranscriptMessage {
                role: MessageRole::Assistant,
                content: assistant.content,
                thinking: (!assistant.thinking.is_empty()).then_some(assistant.thinking),
            });
        }
    }
}

fn append_tool_output(tool: &mut ToolTranscriptEntry, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    if !tool.output_truncated
        && tool.output.len().saturating_add(chunk.len()) <= MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES
    {
        tool.output.push_str(chunk);
        return;
    }

    let head_limit = MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES / 2;
    let tail_limit = MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES - head_limit;
    if !tool.output_truncated {
        let previous = std::mem::take(&mut tool.output);
        let head = prefix_at_boundary(&previous, head_limit).to_owned();
        let tail = if chunk.len() >= tail_limit {
            suffix_at_boundary(chunk, tail_limit).to_owned()
        } else {
            let mut tail = suffix_at_boundary(&previous, tail_limit - chunk.len()).to_owned();
            tail.push_str(chunk);
            tail
        };
        tool.output = format!("{head}{TOOL_OUTPUT_MARKER}{tail}");
        tool.output_truncated = true;
        return;
    }

    let Some((head, old_tail)) = tool.output.split_once(TOOL_OUTPUT_MARKER) else {
        tool.output = truncate_text(&tool.output, MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES);
        return;
    };
    let tail = if chunk.len() >= tail_limit {
        suffix_at_boundary(chunk, tail_limit).to_owned()
    } else {
        let mut tail = suffix_at_boundary(old_tail, tail_limit - chunk.len()).to_owned();
        tail.push_str(chunk);
        tail
    };
    tool.output = format!("{head}{TOOL_OUTPUT_MARKER}{tail}");
}

fn truncate_text(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    let head_limit = limit / 2;
    let tail_limit = limit - head_limit;
    format!(
        "{}\n[… text truncated …]\n{}",
        prefix_at_boundary(text, head_limit),
        suffix_at_boundary(text, tail_limit)
    )
}

fn prefix_at_boundary(text: &str, limit: usize) -> &str {
    let mut end = limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn suffix_at_boundary(text: &str, limit: usize) -> &str {
    let mut start = text.len().saturating_sub(limit);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn previous_grapheme_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .grapheme_indices(true)
        .next_back()
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn next_grapheme_boundary(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .grapheme_indices(true)
        .nth(1)
        .map(|(index, _)| cursor + index)
        .unwrap_or(text.len())
}

fn line_start(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0)
}

fn line_end(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .find('\n')
        .map(|index| cursor + index)
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelAssistantItem;

    #[test]
    fn assistant_boundaries_finalize_messages_inside_one_turn() {
        let mut state = AppState::new();
        state.insert_text("inspect this");
        assert_eq!(state.submit_input().as_deref(), Some("inspect this"));

        state.reduce(AgentEvent::TurnStarted);
        state.reduce(AgentEvent::AssistantMessageStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: "hello ".to_owned(),
        });
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: "world".to_owned(),
        });
        state.reduce(AgentEvent::AssistantMessageFinished {
            items: vec![ModelAssistantItem::Text {
                content: "hello world".to_owned(),
            }],
        });
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: "{\"path\":\"src/main.rs\"}".to_owned(),
        });
        assert!(state.is_turn_active());
        assert_eq!(state.messages().len(), 2);
        assert!(matches!(
            state.transcript_entries()[2],
            TranscriptEntry::Tool(_)
        ));
        state.reduce(AgentEvent::AssistantMessageStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: "done".to_owned(),
        });
        state.reduce(AgentEvent::AssistantMessageFinished { items: vec![] });
        state.reduce(AgentEvent::TurnFinished {
            reason: StopReason::Stop,
        });

        assert!(!state.is_turn_active());
        assert_eq!(state.messages().len(), 3);
        assert_eq!(state.messages()[1].content, "hello world");
        assert_eq!(state.messages()[2].content, "done");
        assert!(state.streaming_assistant().is_none());
    }

    #[test]
    fn cancellation_does_not_make_turn_finished_an_assistant_boundary() {
        let mut state = AppState::new();
        state.insert_text("cancel me");
        state.submit_input();
        state.reduce(AgentEvent::TurnStarted);
        state.reduce(AgentEvent::AssistantMessageStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: "partial".to_owned(),
        });
        state.reduce(AgentEvent::TurnFinished {
            reason: StopReason::Cancelled,
        });

        assert!(!state.is_turn_active());
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.streaming_assistant(), Some(("partial", "")));
        assert_eq!(state.last_stop_reason(), Some(&StopReason::Cancelled));
    }

    #[test]
    fn finished_tool_result_reconciles_the_live_transcript() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            arguments: "{}".to_owned(),
        });
        state.reduce(AgentEvent::ToolExecutionOutput {
            call_id: "call-1".to_owned(),
            stream: ToolOutputStream::Stdout,
            chunk: "partial live output".to_owned(),
        });
        state.reduce(AgentEvent::ToolExecutionFinished {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            result: ToolExecutionResultForTest::result_with_content("authoritative result"),
        });

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0] else {
            panic!("expected tool entry");
        };
        assert_eq!(tool.output, "authoritative result");
        assert!(matches!(tool.status, ToolStatus::Finished(_)));
    }

    #[test]
    fn tool_output_is_bounded_and_finishes_with_metadata() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            arguments: "{}".to_owned(),
        });
        state.reduce(AgentEvent::ToolExecutionOutput {
            call_id: "call-1".to_owned(),
            stream: ToolOutputStream::Stdout,
            chunk: "x".repeat(MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES + 1),
        });
        state.reduce(AgentEvent::ToolExecutionFinished {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            result: ToolExecutionResultForTest::result(),
        });

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0] else {
            panic!("expected tool entry");
        };
        assert!(tool.output_truncated);
        assert!(matches!(tool.status, ToolStatus::Finished(_)));
        assert!(tool.output.len() <= MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES + TOOL_OUTPUT_MARKER.len());
    }

    struct ToolExecutionResultForTest;

    impl ToolExecutionResultForTest {
        fn result() -> crate::tools::ToolExecutionResult {
            Self::result_with_content("done")
        }

        fn result_with_content(content: &str) -> crate::tools::ToolExecutionResult {
            crate::tools::ToolExecutionResult {
                model_content: content.to_owned(),
                metadata: ToolExecutionMetadata::success(),
            }
        }
    }

    #[test]
    fn semantic_history_rebuilds_messages_and_historical_tools() {
        let call = crate::model::ModelToolCall {
            index: 0,
            call_id: Some("call-1".to_owned()),
            item_id: Some("item-1".to_owned()),
            name: Some("read".to_owned()),
            arguments: r#"{"path":"note.txt"}"#.to_owned(),
        };
        let history = vec![
            ModelMessage::user("inspect note"),
            ModelMessage::Assistant {
                items: vec![
                    ModelAssistantItem::Reasoning(crate::model::ModelThinking {
                        item_id: Some("reasoning-1".to_owned()),
                        summary: "summary".to_owned(),
                        content: "thinking".to_owned(),
                        encrypted_content: Some("encrypted".to_owned()),
                    }),
                    ModelAssistantItem::Text {
                        content: "I will inspect it.".to_owned(),
                    },
                    ModelAssistantItem::ToolCall(call),
                ],
            },
            ModelMessage::ToolResult {
                tool_call_id: "call-1".to_owned(),
                tool_name: "read".to_owned(),
                content: "1 | note".to_owned(),
            },
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "The note is present.".to_owned(),
                }],
            },
        ];
        let mut state = AppState::new();
        state.replace_history(&history);

        assert_eq!(state.messages().len(), 3);
        assert_eq!(state.messages()[0].role, MessageRole::User);
        assert_eq!(state.messages()[1].content, "I will inspect it.");
        assert_eq!(state.messages()[1].thinking.as_deref(), Some("thinking"));
        assert_eq!(state.messages()[2].content, "The note is present.");
        assert_eq!(state.transcript_entries().len(), 4);
        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[2] else {
            panic!("expected historical tool entry");
        };
        assert_eq!(tool.call_id, "call-1");
        assert_eq!(tool.output, "1 | note");
        assert!(matches!(tool.status, ToolStatus::Finished(_)));
    }

    #[test]
    fn editor_supports_multiline_unicode_and_grapheme_editing() {
        let mut state = AppState::new();
        state.insert_text("one\ntwo");
        state.move_home();
        state.move_right();
        state.move_right();
        assert_eq!(&state.input()[..state.cursor()], "one\ntw");

        state.move_end();
        state.insert_text(" 🦀");
        state.backspace();
        assert_eq!(state.input(), "one\ntwo ");

        state.insert_text("👨‍👩‍👧‍👦");
        state.backspace();
        assert_eq!(state.input(), "one\ntwo ");
    }
}
