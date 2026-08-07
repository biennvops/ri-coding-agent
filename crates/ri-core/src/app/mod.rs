use unicode_segmentation::UnicodeSegmentation;

use crate::agent::{AgentError, AgentEvent};
use crate::config::ModelRef;
use crate::model::StopReason;

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StreamingAssistant {
    content: String,
    thinking: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AppState {
    messages: Vec<TranscriptMessage>,
    streaming_assistant: Option<StreamingAssistant>,
    input: String,
    cursor: usize,
    turn_active: bool,
    last_error: Option<String>,
    last_stop_reason: Option<StopReason>,
    active_model: Option<ModelRef>,
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn messages(&self) -> &[TranscriptMessage] {
        &self.messages
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
        self.messages.push(TranscriptMessage {
            role: MessageRole::System,
            content: content.into(),
            thinking: None,
        });
    }

    pub fn submit_input(&mut self) -> Option<String> {
        let text = self.take_input()?;
        self.messages.push(TranscriptMessage {
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
                self.streaming_assistant = Some(StreamingAssistant::default());
                self.last_error = None;
                self.last_stop_reason = None;
            }
            AgentEvent::AssistantTextDelta { text } => {
                self.streaming_assistant
                    .get_or_insert_with(StreamingAssistant::default)
                    .content
                    .push_str(&text);
            }
            AgentEvent::AssistantThinkingDelta { text } => {
                self.streaming_assistant
                    .get_or_insert_with(StreamingAssistant::default)
                    .thinking
                    .push_str(&text);
            }
            AgentEvent::TurnFinished { reason } => {
                if let Some(assistant) = self.streaming_assistant.take() {
                    if !assistant.content.is_empty() || !assistant.thinking.is_empty() {
                        self.messages.push(TranscriptMessage {
                            role: MessageRole::Assistant,
                            content: assistant.content,
                            thinking: (!assistant.thinking.is_empty())
                                .then_some(assistant.thinking),
                        });
                    }
                }
                self.turn_active = false;
                self.last_stop_reason = Some(reason);
            }
            AgentEvent::ToolCallDelta { .. } | AgentEvent::UsageUpdated(_) => {}
            AgentEvent::ModelChanged(model) => {
                self.active_model = Some(model);
            }
            AgentEvent::Error(AgentError { message }) => {
                self.last_error = Some(message);
            }
        }
    }
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

    #[test]
    fn reducer_keeps_streaming_deltas_in_one_assistant_message() {
        let mut state = AppState::new();
        state.insert_text("inspect this");
        assert_eq!(state.submit_input().as_deref(), Some("inspect this"));

        state.reduce(AgentEvent::TurnStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            text: "hello ".to_owned(),
        });
        assert_eq!(state.messages().len(), 1);
        state.reduce(AgentEvent::AssistantTextDelta {
            text: "world".to_owned(),
        });
        state.reduce(AgentEvent::TurnFinished {
            reason: StopReason::Stop,
        });

        assert!(!state.is_turn_active());
        assert_eq!(state.messages().len(), 2);
        assert_eq!(state.messages()[1].role, MessageRole::Assistant);
        assert_eq!(state.messages()[1].content, "hello world");
        assert!(state.streaming_assistant().is_none());
    }

    #[test]
    fn cancellation_finalizes_partial_stream_and_returns_to_idle() {
        let mut state = AppState::new();
        state.insert_text("cancel me");
        state.submit_input();
        state.reduce(AgentEvent::TurnStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            text: "partial".to_owned(),
        });
        state.reduce(AgentEvent::TurnFinished {
            reason: StopReason::Cancelled,
        });

        assert!(!state.is_turn_active());
        assert_eq!(state.messages()[1].content, "partial");
        assert_eq!(state.last_stop_reason(), Some(&StopReason::Cancelled));
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
