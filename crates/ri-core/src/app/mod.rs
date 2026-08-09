use std::ops::Index;

use unicode_segmentation::UnicodeSegmentation;

use crate::agent::{AgentError, AgentEvent};
use crate::config::ModelRef;
use crate::context::ContextUsage;
use crate::model::{ModelAssistantItem, ModelLimits, ModelMessage, StopReason, Usage};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TranscriptEntryId(u64);

impl TranscriptEntryId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptEntryState {
    pub id: TranscriptEntryId,
    pub revision: u64,
    pub entry: TranscriptEntry,
}

pub struct TranscriptMessages<'a> {
    entries: &'a [TranscriptEntryState],
    indices: &'a [usize],
}

impl<'a> TranscriptMessages<'a> {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn iter(
        &'a self,
    ) -> impl DoubleEndedIterator<Item = &'a TranscriptMessage> + ExactSizeIterator + 'a {
        self.indices
            .iter()
            .map(move |&index| match &self.entries[index].entry {
                TranscriptEntry::Message(message) => message,
                TranscriptEntry::Tool(_) => unreachable!("message index points to a tool entry"),
            })
    }
}

impl Index<usize> for TranscriptMessages<'_> {
    type Output = TranscriptMessage;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.entries[self.indices[index]].entry {
            TranscriptEntry::Message(message) => message,
            TranscriptEntry::Tool(_) => unreachable!("message index points to a tool entry"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingAssistantState {
    pub id: TranscriptEntryId,
    pub revision: u64,
    pub content: String,
    pub thinking: String,
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
pub struct AppState {
    message_entry_indices: Vec<usize>,
    entries: Vec<TranscriptEntryState>,
    streaming_assistant: Option<StreamingAssistantState>,
    next_transcript_entry_id: u64,
    transcript_epoch: u64,
    transcript_revision: u64,
    pending_transcript_changes: Vec<TranscriptEntryId>,
    input: String,
    cursor: usize,
    input_revision: u64,
    turn_active: bool,
    compaction_active: bool,
    last_error: Option<String>,
    last_stop_reason: Option<StopReason>,
    active_model: Option<ModelRef>,
    session_info: Option<SessionInfo>,
    context_usage: ContextUsage,
    latest_usage: Option<Usage>,
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn messages(&self) -> TranscriptMessages<'_> {
        TranscriptMessages {
            entries: &self.entries,
            indices: &self.message_entry_indices,
        }
    }

    pub fn transcript_entries(&self) -> &[TranscriptEntryState] {
        &self.entries
    }

    pub fn transcript_epoch(&self) -> u64 {
        self.transcript_epoch
    }

    pub fn transcript_revision(&self) -> u64 {
        self.transcript_revision
    }

    pub fn pending_transcript_changes(&self) -> &[TranscriptEntryId] {
        &self.pending_transcript_changes
    }

    pub fn acknowledge_transcript_changes(&mut self) {
        self.pending_transcript_changes.clear();
    }

    pub fn streaming_assistant_state(&self) -> Option<&StreamingAssistantState> {
        self.streaming_assistant.as_ref()
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

    pub fn input_revision(&self) -> u64 {
        self.input_revision
    }

    pub fn set_cursor(&mut self, cursor: usize) {
        if !self.is_busy() && cursor <= self.input.len() && self.input.is_char_boundary(cursor) {
            self.cursor = cursor;
        }
    }

    pub fn is_turn_active(&self) -> bool {
        self.turn_active
    }

    pub fn is_compaction_active(&self) -> bool {
        self.compaction_active
    }

    pub fn set_compaction_active(&mut self, active: bool) {
        if !self.turn_active {
            self.compaction_active = active;
        }
    }

    pub fn is_busy(&self) -> bool {
        self.turn_active || self.compaction_active
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

    pub fn context_usage(&self) -> ContextUsage {
        self.context_usage
    }

    pub fn latest_usage(&self) -> Option<&Usage> {
        self.latest_usage.as_ref()
    }

    pub fn set_context_limits(&mut self, limits: ModelLimits) {
        self.context_usage.context_window = limits.context_window;
        self.context_usage.max_output_tokens = limits.max_output_tokens;
        self.context_usage.input_tokens = None;
        self.context_usage.source = crate::context::UsageSource::Estimated;
    }

    pub fn replace_history(&mut self, history: &[ModelMessage]) {
        self.transcript_epoch = self.transcript_epoch.wrapping_add(1);
        self.pending_transcript_changes.clear();
        self.message_entry_indices = Vec::new();
        self.entries = Vec::new();
        self.streaming_assistant = None;
        self.input.clear();
        self.cursor = 0;
        self.input_revision = self.input_revision.wrapping_add(1);
        self.turn_active = false;
        self.compaction_active = false;
        self.last_error = None;
        self.last_stop_reason = None;
        self.context_usage.input_tokens = None;
        self.context_usage.estimated_input_tokens = 0;
        self.context_usage.source = crate::context::UsageSource::Estimated;
        self.latest_usage = None;
        for message in history {
            self.append_semantic_message(message);
        }
        self.pending_transcript_changes.clear();
    }

    pub fn set_session_info(&mut self, info: Option<SessionInfo>) {
        self.session_info = info;
    }

    pub fn insert_text(&mut self, text: &str) {
        if self.is_busy() || text.is_empty() {
            return;
        }
        self.input.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn insert_newline(&mut self) {
        self.insert_text("\n");
    }

    pub fn backspace(&mut self) {
        if self.is_busy() || self.cursor == 0 {
            return;
        }
        let previous = previous_grapheme_boundary(&self.input, self.cursor);
        self.input.drain(previous..self.cursor);
        self.cursor = previous;
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn delete(&mut self) {
        if self.is_busy() || self.cursor == self.input.len() {
            return;
        }
        let next = next_grapheme_boundary(&self.input, self.cursor);
        self.input.drain(self.cursor..next);
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn move_left(&mut self) {
        if !self.is_busy() {
            self.cursor = previous_grapheme_boundary(&self.input, self.cursor);
        }
    }

    pub fn move_right(&mut self) {
        if !self.is_busy() {
            self.cursor = next_grapheme_boundary(&self.input, self.cursor);
        }
    }

    pub fn move_home(&mut self) {
        if !self.is_busy() {
            self.cursor = line_start(&self.input, self.cursor);
        }
    }

    pub fn move_end(&mut self) {
        if !self.is_busy() {
            self.cursor = line_end(&self.input, self.cursor);
        }
    }

    pub fn take_input(&mut self) -> Option<String> {
        if self.is_busy() || self.input.trim().is_empty() {
            return None;
        }
        let text = std::mem::take(&mut self.input);
        self.cursor = 0;
        self.input_revision = self.input_revision.wrapping_add(1);
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
                self.start_streaming_assistant();
            }
            AgentEvent::AssistantTextDelta { text, .. } => {
                self.append_streaming_content(&text);
            }
            AgentEvent::AssistantThinkingDelta { text, .. } => {
                self.append_streaming_thinking(&text);
            }
            AgentEvent::AssistantRefusalDelta { text, .. } => {
                self.append_streaming_content(&text);
            }
            AgentEvent::AssistantRefusalItem { content, .. } => {
                if let Some(content) = content {
                    self.ensure_streaming_assistant();
                    let (id, changed) = {
                        let assistant = self.streaming_assistant.as_mut().expect("streaming assistant");
                        if assistant.content.is_empty() && !content.is_empty() {
                            assistant.content = content;
                            assistant.revision = assistant.revision.wrapping_add(1);
                            (assistant.id, true)
                        } else {
                            (assistant.id, false)
                        }
                    };
                    if changed {
                        self.mark_transcript_changed(id);
                    }
                }
            }
            AgentEvent::AssistantMessageFinished { items } => {
                let fallback_content = items.iter().find_map(|item| match item {
                    ModelAssistantItem::Text { content }
                    | ModelAssistantItem::Refusal { content } => Some(content.clone()),
                    ModelAssistantItem::Reasoning(_) | ModelAssistantItem::ToolCall(_) => None,
                });
                let fallback_thinking = items.iter().find_map(|item| match item {
                    ModelAssistantItem::Reasoning(thinking) => {
                        (!thinking.content.is_empty()).then_some(thinking.content.clone())
                    }
                    ModelAssistantItem::Text { .. }
                    | ModelAssistantItem::Refusal { .. }
                    | ModelAssistantItem::ToolCall(_) => None,
                });
                self.ensure_streaming_assistant();
                let (id, changed) = {
                    let assistant = self
                        .streaming_assistant
                        .as_mut()
                        .expect("streaming assistant");
                    let mut changed = false;
                    if assistant.content.is_empty() {
                        if let Some(content) = fallback_content.filter(|content| !content.is_empty()) {
                            assistant.content = content;
                            changed = true;
                        }
                    }
                    if assistant.thinking.is_empty() {
                        if let Some(thinking) = fallback_thinking.filter(|thinking| !thinking.is_empty()) {
                            assistant.thinking = thinking;
                            changed = true;
                        }
                    }
                    if changed {
                        assistant.revision = assistant.revision.wrapping_add(1);
                    }
                    (assistant.id, changed)
                };
                if changed {
                    self.mark_transcript_changed(id);
                }
                self.finalize_streaming_assistant();
            }
            AgentEvent::ToolExecutionStarted {
                call_id,
                name,
                arguments,
            } => {
                self.push_entry(TranscriptEntry::Tool(ToolTranscriptEntry {
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
                if let Some(index) = self.entries.iter().rposition(
                    |entry| matches!(&entry.entry, TranscriptEntry::Tool(tool) if tool.call_id == call_id),
                ) {
                    let mut changed = !chunk.is_empty();
                    if let TranscriptEntry::Tool(tool) = &mut self.entries[index].entry {
                        if matches!(stream, ToolOutputStream::Stderr) && tool.output.is_empty() {
                            append_tool_output(tool, "[stderr]\n");
                            changed = true;
                        }
                        append_tool_output(tool, &chunk);
                    }
                    if changed {
                        self.mark_entry_changed(index);
                    }
                }
            }
            AgentEvent::ToolExecutionFinished {
                call_id,
                name: _,
                result,
            } => {
                if let Some(index) = self.entries.iter().rposition(
                    |entry| matches!(&entry.entry, TranscriptEntry::Tool(tool) if tool.call_id == call_id),
                ) {
                    if let TranscriptEntry::Tool(tool) = &mut self.entries[index].entry {
                        let live_output_truncated = tool.output_truncated;
                        tool.output.clear();
                        tool.output_truncated = false;
                        append_tool_output(tool, &result.model_content);
                        tool.output_truncated |= live_output_truncated || result.metadata.truncated;
                        tool.status = ToolStatus::Finished(result.metadata);
                    }
                    self.mark_entry_changed(index);
                }
            }
            AgentEvent::TurnFinished { reason } => {
                self.turn_active = false;
                self.last_stop_reason = Some(reason);
            }
            AgentEvent::CompactionStarted { .. } => {
                self.compaction_active = true;
                self.last_error = None;
            }
            AgentEvent::CompactionFinished {
                before_tokens,
                after_tokens,
                ..
            } => {
                self.compaction_active = false;
                self.push_message(TranscriptMessage {
                    role: MessageRole::System,
                    content: format!(
                        "context compacted · ~{} → ~{} tokens",
                        compact_token_count(before_tokens),
                        compact_token_count(after_tokens)
                    ),
                    thinking: None,
                });
            }
            AgentEvent::CompactionFailed { message } => {
                self.compaction_active = false;
                self.last_error = Some(message.clone());
                self.push_message(TranscriptMessage {
                    role: MessageRole::System,
                    content: message,
                    thinking: None,
                });
            }
            AgentEvent::AssistantTextItem { .. }
            | AgentEvent::AssistantThinkingContentDelta { .. }
            | AgentEvent::AssistantThinkingItem { .. }
            | AgentEvent::ToolCallDelta { .. } => {}
            AgentEvent::UsageUpdated(usage) => {
                self.latest_usage = Some(usage.clone());
                if let Some(input_tokens) = usage.input_tokens {
                    self.context_usage.input_tokens = Some(input_tokens);
                    self.context_usage.source = crate::context::UsageSource::Provider;
                }
            }
            AgentEvent::ContextUsageUpdated(usage) => {
                self.context_usage = usage;
            }
            AgentEvent::ContextLimitsUpdated(limits) => {
                self.set_context_limits(limits);
            }
            AgentEvent::ModelChanged(model) => {
                self.active_model = Some(model);
                self.context_usage.input_tokens = None;
                self.context_usage.source = crate::context::UsageSource::Estimated;
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
        let id = self.allocate_transcript_entry_id();
        self.push_message_with_identity(message, id, 0);
    }

    fn push_message_with_identity(
        &mut self,
        message: TranscriptMessage,
        id: TranscriptEntryId,
        revision: u64,
    ) {
        let entry_index = self.entries.len();
        self.push_entry_with_identity(TranscriptEntry::Message(message), id, revision);
        self.message_entry_indices.push(entry_index);
    }

    fn push_entry(&mut self, entry: TranscriptEntry) -> TranscriptEntryId {
        let id = self.allocate_transcript_entry_id();
        self.push_entry_with_identity(entry, id, 0);
        id
    }

    fn push_entry_with_identity(
        &mut self,
        entry: TranscriptEntry,
        id: TranscriptEntryId,
        revision: u64,
    ) {
        self.entries.push(TranscriptEntryState {
            id,
            revision,
            entry,
        });
        self.mark_transcript_changed(id);
    }

    fn allocate_transcript_entry_id(&mut self) -> TranscriptEntryId {
        self.next_transcript_entry_id = self.next_transcript_entry_id.wrapping_add(1);
        TranscriptEntryId(self.next_transcript_entry_id)
    }

    fn new_streaming_assistant(&mut self) -> StreamingAssistantState {
        StreamingAssistantState {
            id: self.allocate_transcript_entry_id(),
            revision: 0,
            content: String::new(),
            thinking: String::new(),
        }
    }

    fn ensure_streaming_assistant(&mut self) {
        if self.streaming_assistant.is_none() {
            self.start_streaming_assistant();
        }
    }

    fn start_streaming_assistant(&mut self) {
        let assistant = self.new_streaming_assistant();
        self.mark_transcript_changed(assistant.id);
        self.streaming_assistant = Some(assistant);
    }

    fn append_streaming_content(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.ensure_streaming_assistant();
        let id = {
            let assistant = self
                .streaming_assistant
                .as_mut()
                .expect("streaming assistant");
            assistant.content.push_str(text);
            assistant.revision = assistant.revision.wrapping_add(1);
            assistant.id
        };
        self.mark_transcript_changed(id);
    }

    fn append_streaming_thinking(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.ensure_streaming_assistant();
        let id = {
            let assistant = self
                .streaming_assistant
                .as_mut()
                .expect("streaming assistant");
            assistant.thinking.push_str(text);
            assistant.revision = assistant.revision.wrapping_add(1);
            assistant.id
        };
        self.mark_transcript_changed(id);
    }

    fn mark_entry_changed(&mut self, index: usize) {
        let Some(entry) = self.entries.get_mut(index) else {
            return;
        };
        entry.revision = entry.revision.wrapping_add(1);
        let id = entry.id;
        self.mark_transcript_changed(id);
    }

    fn mark_transcript_changed(&mut self, id: TranscriptEntryId) {
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        if self.pending_transcript_changes.last().copied() != Some(id) {
            self.pending_transcript_changes.push(id);
        }
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
                    self.push_entry(TranscriptEntry::Tool(ToolTranscriptEntry {
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
                let found = self.entries.iter().rposition(|entry| {
                    matches!(&entry.entry, TranscriptEntry::Tool(tool) if tool.call_id == *tool_call_id)
                });
                if let Some(index) = found {
                    if let TranscriptEntry::Tool(tool) = &mut self.entries[index].entry {
                        tool.output.clear();
                        tool.output_truncated = false;
                        append_tool_output(tool, content);
                        tool.status = ToolStatus::Finished(ToolExecutionMetadata::success());
                    }
                    self.mark_entry_changed(index);
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
                    self.push_entry(TranscriptEntry::Tool(tool));
                }
            }
            ModelMessage::System { .. } | ModelMessage::Developer { .. } => {}
        }
    }

    fn finalize_streaming_assistant(&mut self) {
        let Some(assistant) = self.streaming_assistant.take() else {
            return;
        };
        self.mark_transcript_changed(assistant.id);
        if !assistant.content.is_empty() || !assistant.thinking.is_empty() {
            self.push_message_with_identity(
                TranscriptMessage {
                    role: MessageRole::Assistant,
                    content: assistant.content,
                    thinking: (!assistant.thinking.is_empty()).then_some(assistant.thinking),
                },
                assistant.id,
                assistant.revision.wrapping_add(1),
            );
        }
    }
}

fn compact_token_count(value: u64) -> String {
    if value < 1_000 {
        value.to_string()
    } else if value < 1_000_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    } else {
        format!("{:.1}m", value as f64 / 1_000_000.0)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
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
            state.transcript_entries()[2].entry,
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

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0].entry else {
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

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0].entry else {
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
        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[2].entry else {
            panic!("expected historical tool entry");
        };
        assert_eq!(tool.call_id, "call-1");
        assert_eq!(tool.output, "1 | note");
        assert!(matches!(tool.status, ToolStatus::Finished(_)));
    }

    #[test]
    fn transcript_entries_have_stable_ids_and_targeted_revisions() {
        let mut state = AppState::new();
        state.add_system_message("static");
        let static_id = state.transcript_entries()[0].id;
        let static_revision = state.transcript_entries()[0].revision;

        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            arguments: "{}".to_owned(),
        });
        let tool_id = state.transcript_entries()[1].id;
        state.reduce(AgentEvent::ToolExecutionOutput {
            call_id: "call-1".to_owned(),
            stream: ToolOutputStream::Stdout,
            chunk: "one".to_owned(),
        });

        assert_ne!(static_id, tool_id);
        assert_eq!(state.transcript_entries()[0].id, static_id);
        assert_eq!(state.transcript_entries()[0].revision, static_revision);
        assert_eq!(state.transcript_entries()[1].id, tool_id);
        assert_eq!(state.transcript_entries()[1].revision, 1);
        state.reduce(AgentEvent::ToolExecutionOutput {
            call_id: "call-1".to_owned(),
            stream: ToolOutputStream::Stdout,
            chunk: String::new(),
        });
        assert_eq!(state.transcript_entries()[1].revision, 1);

        state.acknowledge_transcript_changes();
        state.reduce(AgentEvent::AssistantMessageStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: "stream".to_owned(),
        });
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: " more".to_owned(),
        });
        assert_eq!(state.pending_transcript_changes().len(), 1);
        let streaming = state
            .streaming_assistant_state()
            .expect("streaming assistant should have an id");
        let streaming_id = streaming.id;
        let streaming_revision = streaming.revision;
        state.reduce(AgentEvent::AssistantMessageFinished { items: vec![] });
        let finalized = state
            .transcript_entries()
            .last()
            .expect("streaming assistant should finalize");
        assert_eq!(finalized.id, streaming_id);
        assert!(finalized.revision > streaming_revision);

        let epoch = state.transcript_epoch();
        state.replace_history(&[ModelMessage::user("replacement")]);
        assert_eq!(state.transcript_epoch(), epoch.wrapping_add(1));
        assert_ne!(state.transcript_entries()[0].id, static_id);
    }

    #[test]
    fn message_view_derives_from_transcript_entries_without_copying_text() {
        let mut state = AppState::new();
        state.add_system_message("system");
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: "{}".to_owned(),
        });
        state.add_system_message("later");

        let messages = state.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "system");
        assert_eq!(messages[1].content, "later");
        assert_eq!(messages.iter().count(), 2);
    }

    #[test]
    fn context_usage_reducer_keeps_provider_and_estimated_signals() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::ContextLimitsUpdated(
            crate::model::ModelLimits {
                context_window: Some(200_000),
                max_output_tokens: Some(32_000),
            },
        ));
        state.reduce(AgentEvent::ContextUsageUpdated(ContextUsage::estimated(
            42_000,
            crate::model::ModelLimits {
                context_window: Some(200_000),
                max_output_tokens: Some(32_000),
            },
        )));
        assert_eq!(state.context_usage().estimated_input_tokens, 42_000);
        assert_eq!(state.context_usage().input_tokens, None);
        state.reduce(AgentEvent::UsageUpdated(crate::model::Usage {
            input_tokens: Some(43_000),
            output_tokens: Some(100),
            ..crate::model::Usage::default()
        }));
        assert_eq!(state.context_usage().input_tokens, Some(43_000));
        assert_eq!(state.context_usage().context_window, Some(200_000));
        assert_eq!(
            state.latest_usage().and_then(|usage| usage.output_tokens),
            Some(100)
        );
        state.reduce(AgentEvent::ModelChanged(crate::config::ModelRef {
            provider: "test".to_owned(),
            model: "new-model".to_owned(),
        }));
        assert_eq!(state.context_usage().input_tokens, None);
        assert_eq!(
            state.context_usage().source,
            crate::context::UsageSource::Estimated
        );
    }

    #[test]
    fn compaction_busy_state_disables_editor_until_finished() {
        let mut state = AppState::new();
        state.insert_text("/compact");
        state.set_compaction_active(true);
        assert!(state.is_busy());
        assert_eq!(state.take_input(), None);
        state.reduce(AgentEvent::CompactionFinished {
            automatic: false,
            before_tokens: 142_000,
            after_tokens: 71_000,
        });
        assert!(!state.is_busy());
        assert!(state
            .messages()
            .iter()
            .any(|message| { message.content.contains("context compacted") }));
    }

    #[test]
    fn editor_revision_changes_for_text_edits_but_not_cursor_motion() {
        let mut state = AppState::new();
        let initial = state.input_revision();
        state.insert_text("text");
        let after_insert = state.input_revision();
        assert!(after_insert > initial);
        state.move_left();
        assert_eq!(state.input_revision(), after_insert);
        state.backspace();
        assert!(state.input_revision() > after_insert);
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
