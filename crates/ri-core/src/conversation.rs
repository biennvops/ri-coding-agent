use std::collections::HashSet;

use crate::model::{ModelAssistantItem, ModelMessage};

pub const COMPACTION_SUMMARY_START: &str = "The following is an automatically generated summary of earlier conversation history. Treat it as prior working context, not as a new user request.\n\n<conversation-summary>\n";
pub const COMPACTION_SUMMARY_END: &str = "\n</conversation-summary>";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionSummary {
    pub content: String,
}

impl CompactionSummary {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn as_message(&self) -> ModelMessage {
        ModelMessage::Developer {
            content: self.as_prompt_content(),
        }
    }

    pub fn as_prompt_content(&self) -> String {
        format!(
            "{COMPACTION_SUMMARY_START}{}{COMPACTION_SUMMARY_END}",
            self.content
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConversationHistory {
    summary: Option<CompactionSummary>,
    messages: Vec<ModelMessage>,
}

impl ConversationHistory {
    pub fn new(summary: Option<CompactionSummary>, messages: Vec<ModelMessage>) -> Self {
        Self { summary, messages }
    }

    pub fn from_provider_messages(mut messages: Vec<ModelMessage>) -> Self {
        let summary = match messages.first() {
            Some(ModelMessage::Developer { content })
                if content.starts_with(COMPACTION_SUMMARY_START)
                    && content.ends_with(COMPACTION_SUMMARY_END) =>
            {
                let content = content
                    .strip_prefix(COMPACTION_SUMMARY_START)
                    .and_then(|content| content.strip_suffix(COMPACTION_SUMMARY_END))
                    .unwrap_or_default()
                    .to_owned();
                messages.remove(0);
                Some(CompactionSummary::new(content))
            }
            _ => None,
        };
        Self { summary, messages }
    }

    pub fn summary(&self) -> Option<&CompactionSummary> {
        self.summary.as_ref()
    }

    pub fn messages(&self) -> &[ModelMessage] {
        &self.messages
    }

    pub fn into_messages(self) -> Vec<ModelMessage> {
        self.messages
    }

    pub fn provider_messages(&self) -> Vec<ModelMessage> {
        let mut messages =
            Vec::with_capacity(self.messages.len() + usize::from(self.summary.is_some()));
        if let Some(summary) = &self.summary {
            messages.push(summary.as_message());
        }
        messages.extend(self.messages.iter().cloned());
        messages
    }

    pub fn push(&mut self, message: ModelMessage) {
        self.messages.push(message);
    }

    pub fn replace(&mut self, summary: CompactionSummary, messages: Vec<ModelMessage>) {
        self.summary = Some(summary);
        self.messages = messages;
    }

    pub fn clear(&mut self) {
        self.summary = None;
        self.messages.clear();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistorySegment {
    pub messages: Vec<ModelMessage>,
    pub has_user_message: bool,
    pub safe_to_compact: bool,
}

/// Groups a conversation at user-turn boundaries. A complete assistant tool
/// batch and all of its results stay in the same segment as their user turn.
pub fn segment_history(messages: &[ModelMessage]) -> Vec<HistorySegment> {
    let mut segments = Vec::new();
    let mut current = Vec::new();

    for message in messages {
        if matches!(message, ModelMessage::User { .. }) && !current.is_empty() {
            segments.push(make_segment(std::mem::take(&mut current)));
        }
        current.push(message.clone());
    }
    if !current.is_empty() {
        segments.push(make_segment(current));
    }
    segments
}

fn make_segment(messages: Vec<ModelMessage>) -> HistorySegment {
    let has_user_message = messages
        .iter()
        .any(|message| matches!(message, ModelMessage::User { .. }));
    let safe_to_compact = resolved_tool_interactions(&messages);
    HistorySegment {
        messages,
        has_user_message,
        safe_to_compact,
    }
}

fn resolved_tool_interactions(messages: &[ModelMessage]) -> bool {
    let mut pending = HashSet::new();

    for message in messages {
        match message {
            ModelMessage::Assistant { items } => {
                if !pending.is_empty() {
                    return false;
                }
                for item in items {
                    let ModelAssistantItem::ToolCall(call) = item else {
                        continue;
                    };
                    let Some(call_id) = call.call_id.as_deref() else {
                        return false;
                    };
                    if !pending.insert(call_id.to_owned()) {
                        return false;
                    }
                }
            }
            ModelMessage::ToolResult { tool_call_id, .. } => {
                if !pending.remove(tool_call_id) {
                    return false;
                }
            }
            ModelMessage::User { .. } => {
                if !pending.is_empty() {
                    return false;
                }
            }
            ModelMessage::System { .. } | ModelMessage::Developer { .. } => {}
        }
    }

    pending.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelAssistantItem, ModelToolCall};

    #[test]
    fn summary_projects_as_developer_context_and_round_trips() {
        let history = ConversationHistory::new(
            Some(CompactionSummary::new("remember alpha")),
            vec![ModelMessage::user("continue")],
        );
        let projected = history.provider_messages();
        assert!(matches!(
            &projected[0],
            ModelMessage::Developer { content } if content.contains("remember alpha")
        ));
        assert_eq!(
            ConversationHistory::from_provider_messages(projected),
            history
        );
    }

    #[test]
    fn segmentation_keeps_tool_calls_and_results_together() {
        let call = ModelToolCall {
            index: 0,
            call_id: Some("call-a".to_owned()),
            item_id: None,
            name: Some("read".to_owned()),
            arguments: "{}".to_owned(),
        };
        let history = vec![
            ModelMessage::user("first"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::ToolCall(call)],
            },
            ModelMessage::ToolResult {
                tool_call_id: "call-a".to_owned(),
                tool_name: "read".to_owned(),
                content: "done".to_owned(),
            },
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "finished".to_owned(),
                }],
            },
            ModelMessage::user("second"),
        ];
        let segments = segment_history(&history);
        assert_eq!(segments.len(), 2);
        assert!(segments[0].safe_to_compact);
        assert_eq!(segments[0].messages.len(), 4);
    }

    #[test]
    fn unresolved_tool_batch_is_not_safe_to_compact() {
        let history = vec![ModelMessage::Assistant {
            items: vec![ModelAssistantItem::ToolCall(ModelToolCall {
                index: 0,
                call_id: Some("call-a".to_owned()),
                item_id: None,
                name: Some("read".to_owned()),
                arguments: "{}".to_owned(),
            })],
        }];
        assert!(!segment_history(&history)[0].safe_to_compact);
    }
}
