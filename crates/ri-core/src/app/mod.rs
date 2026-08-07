use crate::agent::{AgentError, AgentEvent};
use crate::model::StopReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
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

    pub fn is_turn_active(&self) -> bool {
        self.turn_active
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn last_stop_reason(&self) -> Option<&StopReason> {
        self.last_stop_reason.as_ref()
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
        let previous = previous_char_boundary(&self.input, self.cursor);
        self.input.drain(previous..self.cursor);
        self.cursor = previous;
    }

    pub fn delete(&mut self) {
        if self.turn_active || self.cursor == self.input.len() {
            return;
        }
        let next = next_char_boundary(&self.input, self.cursor);
        self.input.drain(self.cursor..next);
    }

    pub fn move_left(&mut self) {
        if !self.turn_active {
            self.cursor = previous_char_boundary(&self.input, self.cursor);
        }
    }

    pub fn move_right(&mut self) {
        if !self.turn_active {
            self.cursor = next_char_boundary(&self.input, self.cursor);
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

    pub fn move_up(&mut self) {
        if !self.turn_active {
            self.cursor = vertical_cursor(&self.input, self.cursor, -1);
        }
    }

    pub fn move_down(&mut self) {
        if !self.turn_active {
            self.cursor = vertical_cursor(&self.input, self.cursor, 1);
        }
    }

    pub fn submit_input(&mut self) -> Option<String> {
        if self.turn_active || self.input.trim().is_empty() {
            return None;
        }

        let text = std::mem::take(&mut self.input);
        self.cursor = 0;
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
            AgentEvent::Error(AgentError { message }) => {
                self.last_error = Some(message);
            }
        }
    }
}

fn previous_char_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .char_indices()
        .next_back()
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn next_char_boundary(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .char_indices()
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

fn vertical_cursor(text: &str, cursor: usize, direction: isize) -> usize {
    let current_start = line_start(text, cursor);
    let column = text[current_start..cursor].chars().count();

    if direction < 0 {
        if current_start == 0 {
            return cursor;
        }
        let previous_end = current_start - 1;
        let previous_start = line_start(text, previous_end);
        return char_offset(text, previous_start, previous_end, column);
    }

    let current_end = line_end(text, cursor);
    if current_end == text.len() {
        return cursor;
    }
    let next_start = current_end + 1;
    let next_end = line_end(text, next_start);
    char_offset(text, next_start, next_end, column)
}

fn char_offset(text: &str, start: usize, end: usize, column: usize) -> usize {
    text[start..end]
        .char_indices()
        .nth(column)
        .map(|(index, _)| start + index)
        .unwrap_or(end)
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
    fn editor_supports_multiline_unicode_and_vertical_movement() {
        let mut state = AppState::new();
        state.insert_text("one\ntwo");
        state.move_home();
        state.move_right();
        state.move_right();
        state.move_up();
        assert_eq!(&state.input()[..state.cursor()], "on");
        state.move_down();
        assert_eq!(&state.input()[..state.cursor()], "one\ntw");

        state.move_end();
        state.insert_text(" 🦀");
        state.backspace();
        assert_eq!(state.input(), "one\ntwo ");
    }
}
