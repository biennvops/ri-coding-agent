use crate::model::{ModelAssistantItem, ModelMessage, ModelToolCall, ToolDefinition};

use super::super::model::{ModelLimits, ModelRequest};

pub const DEFAULT_COMPACTION_RESERVE_TOKENS: u64 = 16_384;
pub const AUTO_COMPACTION_TARGET_PERCENT: u64 = 50;
pub const CONTEXT_SAFETY_TOKENS: u64 = 4_096;
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

pub fn automatic_compaction_threshold(limits: ModelLimits, reserve_tokens: u64) -> Option<u64> {
    Some(limits.context_window?.saturating_sub(reserve_tokens))
}

pub fn request_input_budget(context_window: Option<u64>, output_tokens: u64) -> Option<u64> {
    Some(
        context_window?
            .saturating_sub(output_tokens)
            .saturating_sub(CONTEXT_SAFETY_TOKENS),
    )
}

pub fn clamp_request_output_tokens(
    limits: ModelLimits,
    estimated_input_tokens: u64,
) -> Option<u64> {
    let maximum = limits.max_output_tokens?;
    Some(maximum.min(request_input_budget(
        limits.context_window,
        estimated_input_tokens,
    )?))
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
    fn automatic_compaction_threshold_uses_configured_reserve_not_model_max_output() {
        for max_output_tokens in [None, Some(8_192), Some(64_000)] {
            let limits = ModelLimits {
                context_window: Some(128_000),
                max_output_tokens,
            };
            assert_eq!(
                automatic_compaction_threshold(limits, 16_384),
                Some(111_616)
            );
            for (reserve, expected) in [(0, 128_000), (128_000, 0), (u64::MAX, 0)] {
                assert_eq!(
                    automatic_compaction_threshold(limits, reserve),
                    Some(expected)
                );
            }
        }
        assert_eq!(
            automatic_compaction_threshold(ModelLimits::default(), 0),
            None
        );
        assert_eq!(
            automatic_compaction_threshold(
                ModelLimits {
                    context_window: Some(0),
                    max_output_tokens: None
                },
                16_384
            ),
            Some(0)
        );
    }

    #[test]
    fn request_output_clamp_uses_remaining_context() {
        let limits = ModelLimits {
            context_window: Some(128_000),
            max_output_tokens: Some(64_000),
        };
        for (input, output) in [
            (40_000, 64_000),
            (60_000, 63_904),
            (100_000, 23_904),
            (u64::MAX, 0),
        ] {
            assert_eq!(clamp_request_output_tokens(limits, input), Some(output));
        }
        assert_eq!(
            clamp_request_output_tokens(
                ModelLimits {
                    context_window: None,
                    ..limits
                },
                0
            ),
            None
        );
        assert_eq!(
            clamp_request_output_tokens(
                ModelLimits {
                    max_output_tokens: None,
                    ..limits
                },
                0
            ),
            None
        );
        assert_eq!(request_input_budget(Some(128_000), 4_096), Some(119_808));
        assert_eq!(request_input_budget(Some(0), u64::MAX), Some(0));
        assert_eq!(request_input_budget(None, 4_096), None);
    }
}
