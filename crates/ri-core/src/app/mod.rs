use std::ops::Index;

use unicode_segmentation::UnicodeSegmentation;

use crate::agent::{AgentError, AgentEvent};
use crate::config::{ModelRef, ThinkingLevel};
use crate::context::ContextUsage;
use crate::model::{ModelAssistantItem, ModelLimits, ModelMessage, StopReason, Usage};
use crate::session::SessionInfo;
use crate::tools::{
    ToolCallPresentation, ToolExecutionMetadata, ToolOutputStream, ToolPreviewKind,
    ToolPreviewLine, ToolRegistry, MAX_TOOL_PREVIEW_BYTES,
};

const MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES: usize = 256 * 1024;
const TOOL_OUTPUT_MARKER: &str = "\n[… output truncated …]\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserMessageStatus {
    Delivered,
    Queued,
    Recovered,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptMessage {
    pub role: MessageRole,
    pub content: String,
    pub thinking: Option<String>,
    pub user_status: UserMessageStatus,
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
pub struct ToolOutputChunk {
    pub stream: Option<ToolOutputStream>,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolTranscriptEntry {
    pub call_id: String,
    pub name: String,
    pub summary: String,
    pub preview: Vec<ToolPreviewLine>,
    pub output: String,
    pub output_chunks: Vec<ToolOutputChunk>,
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
    thinking_level: Option<ThinkingLevel>,
    git_branch: Option<String>,
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
        if cursor <= self.input.len() && self.input.is_char_boundary(cursor) {
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

    pub fn thinking_level(&self) -> Option<ThinkingLevel> {
        self.thinking_level
    }

    pub fn set_thinking_level(&mut self, level: Option<ThinkingLevel>) {
        self.thinking_level = level;
    }

    pub fn git_branch(&self) -> Option<&str> {
        self.git_branch.as_deref()
    }

    pub fn set_git_branch(&mut self, branch: Option<String>) {
        self.git_branch = branch;
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
        if text.is_empty() {
            return;
        }
        self.input.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn set_input(&mut self, text: String) {
        if self.input == text {
            return;
        }
        self.input = text;
        self.cursor = self.input.len();
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn insert_newline(&mut self) {
        self.insert_text("\n");
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let previous = previous_grapheme_boundary(&self.input, self.cursor);
        self.input.drain(previous..self.cursor);
        self.cursor = previous;
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn delete(&mut self) {
        if self.cursor == self.input.len() {
            return;
        }
        let next = next_grapheme_boundary(&self.input, self.cursor);
        self.input.drain(self.cursor..next);
        self.input_revision = self.input_revision.wrapping_add(1);
    }

    pub fn move_left(&mut self) {
        self.cursor = previous_grapheme_boundary(&self.input, self.cursor);
    }

    pub fn move_right(&mut self) {
        self.cursor = next_grapheme_boundary(&self.input, self.cursor);
    }

    pub fn move_home(&mut self) {
        self.cursor = line_start(&self.input, self.cursor);
    }

    pub fn move_end(&mut self) {
        self.cursor = line_end(&self.input, self.cursor);
    }

    pub fn take_input(&mut self) -> Option<String> {
        if self.input.trim().is_empty() {
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
            user_status: UserMessageStatus::Delivered,
        });
    }

    pub fn submit_input(&mut self) -> Option<String> {
        if self.is_busy() {
            return None;
        }
        let text = self.take_input()?;
        self.finalize_streaming_assistant();
        self.push_message(TranscriptMessage {
            role: MessageRole::User,
            content: text.clone(),
            thinking: None,
            user_status: UserMessageStatus::Delivered,
        });
        self.turn_active = true;
        self.last_error = None;
        self.last_stop_reason = None;
        Some(text)
    }

    pub fn queue_input(&mut self) -> Option<String> {
        if !self.turn_active || self.compaction_active {
            return None;
        }
        let text = self.take_input()?;
        self.push_message(TranscriptMessage {
            role: MessageRole::User,
            content: text.clone(),
            thinking: None,
            user_status: UserMessageStatus::Queued,
        });
        Some(text)
    }

    pub fn reduce(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TurnStarted => {
                self.turn_active = true;
                self.last_error = None;
                self.last_stop_reason = None;
            }
            AgentEvent::SteeringMessageDelivered { text } => {
                if !self.update_oldest_queued_message(UserMessageStatus::Delivered) {
                    self.push_message(TranscriptMessage {
                        role: MessageRole::User,
                        content: text,
                        thinking: None,
                        user_status: UserMessageStatus::Delivered,
                    });
                }
            }
            AgentEvent::SteeringMessagesRecovered { messages } => {
                for text in messages {
                    if !self.update_oldest_queued_message(UserMessageStatus::Recovered) {
                        self.push_message(TranscriptMessage {
                            role: MessageRole::User,
                            content: text,
                            thinking: None,
                            user_status: UserMessageStatus::Recovered,
                        });
                    }
                }
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
                let presentation = tool_call_presentation(&name, &arguments);
                self.push_entry(TranscriptEntry::Tool(ToolTranscriptEntry {
                    call_id,
                    name,
                    summary: presentation.summary,
                    preview: presentation.preview,
                    output: String::new(),
                    output_chunks: Vec::new(),
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
                    let changed = !chunk.is_empty();
                    if let TranscriptEntry::Tool(tool) = &mut self.entries[index].entry {
                        append_tool_output(tool, &chunk);
                        append_tool_output_chunk(tool, Some(stream), &chunk);
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
                        if tool.output_chunks.is_empty() {
                            append_tool_output_chunk(tool, None, &result.model_content);
                        }
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
                    user_status: UserMessageStatus::Delivered,
                });
            }
            AgentEvent::CompactionFailed { message } => {
                self.compaction_active = false;
                self.last_error = Some(message.clone());
                self.push_message(TranscriptMessage {
                    role: MessageRole::System,
                    content: message,
                    thinking: None,
                    user_status: UserMessageStatus::Delivered,
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
                self.last_error = Some(message.clone());
                self.push_message(TranscriptMessage {
                    role: MessageRole::System,
                    content: format!("error: {message}"),
                    thinking: None,
                    user_status: UserMessageStatus::Delivered,
                });
            }
        }
    }

    fn update_oldest_queued_message(&mut self, status: UserMessageStatus) -> bool {
        let Some(index) = self.entries.iter().position(|entry| {
            matches!(
                &entry.entry,
                TranscriptEntry::Message(TranscriptMessage {
                    role: MessageRole::User,
                    user_status: UserMessageStatus::Queued,
                    ..
                })
            )
        }) else {
            return false;
        };
        if let TranscriptEntry::Message(message) = &mut self.entries[index].entry {
            message.user_status = status;
        }
        self.mark_entry_changed(index);
        true
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
                user_status: UserMessageStatus::Delivered,
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
                        user_status: UserMessageStatus::Delivered,
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
                    let presentation = tool_call_presentation(&name, &call.arguments);
                    self.push_entry(TranscriptEntry::Tool(ToolTranscriptEntry {
                        call_id,
                        name,
                        summary: presentation.summary,
                        preview: presentation.preview,
                        output: String::new(),
                        output_chunks: Vec::new(),
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
                        tool.output_chunks.clear();
                        tool.output_truncated = false;
                        append_tool_output(tool, content);
                        append_tool_output_chunk(tool, None, content);
                        tool.status = ToolStatus::Finished(ToolExecutionMetadata::success());
                    }
                    self.mark_entry_changed(index);
                } else {
                    let mut tool = ToolTranscriptEntry {
                        call_id: tool_call_id.clone(),
                        name: tool_name.clone(),
                        summary: tool_name.clone(),
                        preview: Vec::new(),
                        output: String::new(),
                        output_chunks: Vec::new(),
                        output_truncated: false,
                        status: ToolStatus::Finished(ToolExecutionMetadata::success()),
                    };
                    append_tool_output(&mut tool, content);
                    append_tool_output_chunk(&mut tool, None, content);
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
                    user_status: UserMessageStatus::Delivered,
                },
                assistant.id,
                assistant.revision.wrapping_add(1),
            );
        }
    }
}

fn append_tool_output_chunk(
    tool: &mut ToolTranscriptEntry,
    stream: Option<ToolOutputStream>,
    text: &str,
) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = tool.output_chunks.last_mut() {
        if last.stream == stream {
            last.text.push_str(text);
        } else {
            tool.output_chunks.push(ToolOutputChunk {
                stream,
                text: text.to_owned(),
            });
        }
    } else {
        tool.output_chunks.push(ToolOutputChunk {
            stream,
            text: text.to_owned(),
        });
    }

    let total = tool
        .output_chunks
        .iter()
        .map(|chunk| chunk.text.len())
        .sum::<usize>();
    if total <= MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES {
        return;
    }

    let retained = MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES.saturating_sub(TOOL_OUTPUT_MARKER.len());
    let head_limit = retained / 2;
    let tail_limit = retained - head_limit;
    let mut bounded = tool_output_prefix(&tool.output_chunks, head_limit);
    bounded.push(ToolOutputChunk {
        stream: None,
        text: TOOL_OUTPUT_MARKER.to_owned(),
    });
    bounded.extend(tool_output_suffix(&tool.output_chunks, tail_limit));
    tool.output_chunks = bounded;
}

fn tool_output_prefix(chunks: &[ToolOutputChunk], limit: usize) -> Vec<ToolOutputChunk> {
    let mut remaining = limit;
    let mut result = Vec::new();
    for chunk in chunks {
        if remaining == 0 {
            break;
        }
        let text = prefix_at_boundary(&chunk.text, remaining.min(chunk.text.len()));
        if !text.is_empty() {
            result.push(ToolOutputChunk {
                stream: chunk.stream,
                text: text.to_owned(),
            });
            remaining = remaining.saturating_sub(text.len());
        }
        if text.len() < chunk.text.len() {
            break;
        }
    }
    result
}

fn tool_output_suffix(chunks: &[ToolOutputChunk], limit: usize) -> Vec<ToolOutputChunk> {
    let mut remaining = limit;
    let mut result = Vec::new();
    for chunk in chunks.iter().rev() {
        if remaining == 0 {
            break;
        }
        let text = suffix_at_boundary(&chunk.text, remaining.min(chunk.text.len()));
        if !text.is_empty() {
            result.push(ToolOutputChunk {
                stream: chunk.stream,
                text: text.to_owned(),
            });
            remaining = remaining.saturating_sub(text.len());
        }
        if text.len() < chunk.text.len() {
            break;
        }
    }
    result.reverse();
    result
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

fn tool_call_presentation(name: &str, arguments: &str) -> ToolCallPresentation {
    match serde_json::from_str(arguments) {
        Ok(arguments) => ToolRegistry::new().presentation(name, &arguments),
        Err(_) => ToolCallPresentation {
            summary: name.to_owned(),
            preview: truncate_text(arguments, MAX_TOOL_PREVIEW_BYTES)
                .lines()
                .map(|text| ToolPreviewLine {
                    kind: if text.starts_with('…') || text.starts_with("[…") {
                        ToolPreviewKind::Dim
                    } else {
                        ToolPreviewKind::Normal
                    },
                    text: text.to_owned(),
                })
                .collect(),
        },
    }
}

fn truncate_text(text: &str, limit: usize) -> String {
    const MARKER: &str = "\n[… text truncated …]\n";

    if text.len() <= limit {
        return text.to_owned();
    }
    if limit <= MARKER.len() {
        return prefix_at_boundary(MARKER, limit).to_owned();
    }
    let retained = limit - MARKER.len();
    let head_limit = retained / 2;
    let tail_limit = retained - head_limit;
    format!(
        "{}{}{}",
        prefix_at_boundary(text, head_limit),
        MARKER,
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
    fn live_tool_output_preserves_bounded_stream_order_after_completion() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"test"}"#.to_owned(),
        });
        for (stream, chunk) in [
            (ToolOutputStream::Stdout, "out 1\n"),
            (ToolOutputStream::Stderr, "err\n"),
            (ToolOutputStream::Stdout, "out 2\n"),
        ] {
            state.reduce(AgentEvent::ToolExecutionOutput {
                call_id: "call-1".to_owned(),
                stream,
                chunk: chunk.to_owned(),
            });
        }
        state.reduce(AgentEvent::ToolExecutionFinished {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            result: ToolExecutionResultForTest::result_with_content("authoritative result"),
        });

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0].entry else {
            panic!("expected tool entry");
        };
        assert_eq!(
            tool.output_chunks
                .iter()
                .map(|chunk| (chunk.stream, chunk.text.as_str()))
                .collect::<Vec<_>>(),
            [
                (Some(ToolOutputStream::Stdout), "out 1\n"),
                (Some(ToolOutputStream::Stderr), "err\n"),
                (Some(ToolOutputStream::Stdout), "out 2\n"),
            ]
        );
        assert!(
            tool.output_chunks
                .iter()
                .map(|chunk| chunk.text.len())
                .sum::<usize>()
                <= MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES
        );
    }

    #[test]
    fn write_and_edit_previews_survive_tool_completion() {
        let cases = [
            (
                "write",
                r#"{"path":"src/foo.rs","content":"one\ntwo\nthree\n"}"#,
                "write src/foo.rs",
                "one\ntwo\nthree",
            ),
            (
                "edit",
                r#"{"path":"src/foo.rs","old_text":"let a = 1;","new_text":"let a = 2;"}"#,
                "edit src/foo.rs",
                "-let a = 1;\n+let a = 2;",
            ),
        ];

        for (name, arguments, summary, preview) in cases {
            let mut state = AppState::new();
            state.reduce(AgentEvent::ToolExecutionStarted {
                call_id: name.to_owned(),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            });
            state.reduce(AgentEvent::ToolExecutionFinished {
                call_id: name.to_owned(),
                name: name.to_owned(),
                result: ToolExecutionResultForTest::result_with_content("completed result"),
            });

            let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0].entry else {
                panic!("expected tool entry");
            };
            assert_eq!(tool.summary, summary);
            assert_eq!(
                tool.preview
                    .iter()
                    .map(|line| line.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                preview
            );
            assert_eq!(tool.output, "completed result");
            assert!(matches!(tool.status, ToolStatus::Finished(_)));
        }
    }

    #[test]
    fn failed_edit_preserves_proposed_change_and_error() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "edit-1".to_owned(),
            name: "edit".to_owned(),
            arguments: r#"{"path":"src/foo.rs","old_text":"old","new_text":"new"}"#.to_owned(),
        });
        state.reduce(AgentEvent::ToolExecutionFinished {
            call_id: "edit-1".to_owned(),
            name: "edit".to_owned(),
            result: crate::tools::ToolExecutionResult::failure(
                "old_text matched 3 locations; provide more surrounding context",
            ),
        });

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0].entry else {
            panic!("expected tool entry");
        };
        assert_eq!(
            tool.preview
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            "-old\n+new"
        );
        assert!(tool.output.contains("matched 3 locations"));
        assert!(matches!(
            tool.status,
            ToolStatus::Finished(ref metadata) if !metadata.success
        ));
    }

    #[test]
    fn malformed_tool_arguments_have_a_strictly_bounded_fallback_preview() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "malformed".to_owned(),
            name: "write".to_owned(),
            arguments: "x".repeat(MAX_TOOL_PREVIEW_BYTES * 2),
        });

        let TranscriptEntry::Tool(tool) = &state.transcript_entries()[0].entry else {
            panic!("expected tool entry");
        };
        let preview = tool
            .preview
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(tool.summary, "write");
        assert!(preview.contains("[… text truncated …]"));
        assert!(preview.len() <= MAX_TOOL_PREVIEW_BYTES);
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
        assert!(
            tool.output_chunks
                .iter()
                .map(|chunk| chunk.text.len())
                .sum::<usize>()
                <= MAX_TOOL_TRANSCRIPT_OUTPUT_BYTES
        );
        assert!(tool
            .output_chunks
            .iter()
            .any(|chunk| chunk.text == TOOL_OUTPUT_MARKER));
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
        assert_eq!(tool.summary, "read note.txt");
        assert!(tool.preview.is_empty());
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
    fn queued_user_messages_transition_without_duplicate_transcript_entries() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::TurnStarted);
        state.insert_text("A");
        assert_eq!(state.queue_input().as_deref(), Some("A"));
        state.insert_text("B");
        assert_eq!(state.queue_input().as_deref(), Some("B"));
        assert_eq!(state.messages().len(), 2);
        assert!(state
            .messages()
            .iter()
            .all(|message| message.user_status == UserMessageStatus::Queued));

        state.reduce(AgentEvent::SteeringMessageDelivered {
            text: "A".to_owned(),
        });
        assert_eq!(state.messages().len(), 2);
        assert_eq!(
            state
                .messages()
                .iter()
                .map(|message| message.user_status)
                .collect::<Vec<_>>(),
            [UserMessageStatus::Delivered, UserMessageStatus::Queued]
        );

        state.reduce(AgentEvent::SteeringMessagesRecovered {
            messages: vec!["B".to_owned()],
        });
        assert_eq!(state.messages().len(), 2);
        assert_eq!(
            state.messages()[1].user_status,
            UserMessageStatus::Recovered
        );
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
    fn editor_remains_usable_during_turns_and_compaction() {
        let mut state = AppState::new();
        state.insert_text("abc");
        state.reduce(AgentEvent::TurnStarted);
        state.insert_text("d");
        state.move_left();
        state.backspace();
        state.insert_newline();
        state.delete();
        assert_eq!(state.input(), "ab\n");
        assert_eq!(state.take_input().as_deref(), Some("ab\n"));

        state.reduce(AgentEvent::TurnFinished {
            reason: StopReason::Stop,
        });
        state.set_compaction_active(true);
        state.insert_text("during compaction");
        assert_eq!(state.input(), "during compaction");
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
    fn agent_errors_remain_available_and_are_appended_to_the_transcript() {
        let mut state = AppState::new();
        let message = "provider returned HTTP 400:\n{\"error\":\"bad request\"}";

        state.reduce(AgentEvent::Error(AgentError::new(message)));

        assert_eq!(state.last_error(), Some(message));
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].role, MessageRole::System);
        assert_eq!(state.messages()[0].content, format!("error: {message}"));
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
    fn replacing_editor_input_moves_the_cursor_and_revision() {
        let mut state = AppState::new();
        state.insert_text("old");
        let revision = state.input_revision();

        state.set_input("/model ".to_owned());

        assert_eq!(state.input(), "/model ");
        assert_eq!(state.cursor(), state.input().len());
        assert!(state.input_revision() > revision);
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
