use crate::model::{ModelAssistantItem, ModelMessage, ModelToolCall, ToolDefinition};

use super::super::model::{ModelLimits, ModelRequest};

pub const AUTO_COMPACTION_TRIGGER_PERCENT: u64 = 80;
pub const AUTO_COMPACTION_TARGET_PERCENT: u64 = 50;
pub const DEFAULT_RESERVED_OUTPUT_TOKENS: u64 = 4_096;
pub const COMPACTION_MAX_OUTPUT_TOKENS: u64 = 4_096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UsageSource {
    #[default]
    Estimated,
    Provider,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextUsage {
    pub input_tokens: Option<u64>,
    pub estimated_input_tokens: u64,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub source: UsageSource,
}

impl ContextUsage {
    pub fn estimated(tokens: u64, limits: ModelLimits) -> Self {
        Self {
            input_tokens: None,
            estimated_input_tokens: tokens,
            context_window: limits.context_window,
            max_output_tokens: limits.max_output_tokens,
            source: UsageSource::Estimated,
        }
    }

    pub fn current_tokens(self) -> u64 {
        self.input_tokens.unwrap_or(self.estimated_input_tokens)
    }
}

pub trait TokenEstimator {
    fn estimate_messages(&self, messages: &[ModelMessage]) -> u64;

    fn estimate_tools(&self, tools: &[ToolDefinition]) -> u64 {
        tools.iter().map(estimate_tool).sum()
    }

    fn estimate_request(&self, request: &ModelRequest) -> u64 {
        self.estimate_messages(&request.messages) + self.estimate_tools(&request.tools)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConservativeTokenEstimator;

impl TokenEstimator for ConservativeTokenEstimator {
    fn estimate_messages(&self, messages: &[ModelMessage]) -> u64 {
        messages.iter().map(estimate_message).sum()
    }
}

pub type GenericTokenEstimator = ConservativeTokenEstimator;

pub fn input_budget(limits: ModelLimits) -> Option<u64> {
    let context_window = limits.context_window?;
    if context_window == 0 {
        return Some(0);
    }
    let reserve = limits
        .max_output_tokens
        .unwrap_or_else(|| (context_window / 5).min(DEFAULT_RESERVED_OUTPUT_TOKENS))
        .min(context_window.saturating_sub(1));
    Some(context_window.saturating_sub(reserve))
}

pub fn automatic_trigger(budget: u64) -> u64 {
    budget.saturating_mul(AUTO_COMPACTION_TRIGGER_PERCENT) / 100
}

pub fn compaction_target(budget: u64) -> u64 {
    budget.saturating_mul(AUTO_COMPACTION_TARGET_PERCENT) / 100
}

fn estimate_message(message: &ModelMessage) -> u64 {
    let mut bytes = 8u64;
    match message {
        ModelMessage::System { content }
        | ModelMessage::Developer { content }
        | ModelMessage::User { content } => {
            bytes += text_cost(content);
        }
        ModelMessage::Assistant { items } => {
            for item in items {
                bytes += match item {
                    ModelAssistantItem::Text { content }
                    | ModelAssistantItem::Refusal { content } => text_cost(content),
                    ModelAssistantItem::Reasoning(thinking) => {
                        text_cost(&thinking.summary)
                            + text_cost(&thinking.content)
                            + optional_text_cost(thinking.item_id.as_deref())
                            + optional_text_cost(thinking.encrypted_content.as_deref())
                    }
                    ModelAssistantItem::ToolCall(call) => estimate_tool_call(call),
                };
            }
        }
        ModelMessage::ToolResult {
            tool_call_id,
            tool_name,
            content,
        } => {
            bytes += text_cost(tool_call_id) + text_cost(tool_name) + text_cost(content);
        }
    }
    bytes_to_tokens(bytes)
}

fn estimate_tool(tool: &ToolDefinition) -> u64 {
    let parameters = serde_json::to_string(&tool.parameters).unwrap_or_default();
    let bytes = 16
        + text_cost(&tool.name)
        + optional_text_cost(tool.description.as_deref())
        + text_cost(&parameters);
    bytes_to_tokens(bytes)
}

fn estimate_tool_call(call: &ModelToolCall) -> u64 {
    let bytes = 12
        + optional_text_cost(call.call_id.as_deref())
        + optional_text_cost(call.item_id.as_deref())
        + optional_text_cost(call.name.as_deref())
        + text_cost(&call.arguments);
    bytes_to_tokens(bytes)
}

fn optional_text_cost(value: Option<&str>) -> u64 {
    value.map(text_cost).unwrap_or(0)
}

fn text_cost(value: &str) -> u64 {
    value.len() as u64 + 4
}

fn bytes_to_tokens(bytes: u64) -> u64 {
    // Three UTF-8 bytes per token is intentionally conservative for a
    // provider-neutral estimate, with a small per-message framing cost above.
    bytes.saturating_add(2) / 3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelAssistantItem, ModelToolCall, ToolDefinition};
    use serde_json::json;

    #[test]
    fn estimates_are_deterministic_and_grow_with_content() {
        let estimator = ConservativeTokenEstimator;
        let small = estimator.estimate_messages(&[ModelMessage::user("a")]);
        let large = estimator.estimate_messages(&[ModelMessage::user("a".repeat(100))]);
        assert_eq!(
            small,
            estimator.estimate_messages(&[ModelMessage::user("a")])
        );
        assert!(large > small);
    }

    #[test]
    fn counts_tool_definitions_arguments_results_and_unicode() {
        let estimator = ConservativeTokenEstimator;
        let tools = vec![ToolDefinition {
            name: "read".to_owned(),
            description: Some("read a file".to_owned()),
            parameters: json!({"path": {"type": "string"}}),
        }];
        let base = estimator.estimate_tools(&tools);
        let call = ModelAssistantItem::ToolCall(ModelToolCall {
            index: 0,
            call_id: Some("call-1".to_owned()),
            item_id: Some("item-1".to_owned()),
            name: Some("read".to_owned()),
            arguments: r#"{"path":"世界.txt"}"#.to_owned(),
        });
        let messages = vec![
            ModelMessage::Assistant { items: vec![call] },
            ModelMessage::ToolResult {
                tool_call_id: "call-1".to_owned(),
                tool_name: "read".to_owned(),
                content: "世界".to_owned(),
            },
        ];
        assert!(base > 0);
        assert!(estimator.estimate_messages(&messages) > 0);
        assert!(estimator.estimate_messages(&[ModelMessage::user("世界")]) > 0);
    }

    #[test]
    fn input_budget_reserves_configured_or_bounded_output() {
        assert_eq!(
            input_budget(ModelLimits {
                context_window: Some(200_000),
                max_output_tokens: Some(32_000),
            }),
            Some(168_000)
        );
        assert_eq!(
            input_budget(ModelLimits {
                context_window: Some(200_000),
                max_output_tokens: None,
            }),
            Some(195_904)
        );
        assert_eq!(input_budget(ModelLimits::default()), None);
    }
}
