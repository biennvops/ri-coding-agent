use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::context::{
    request_input_budget, ConservativeTokenEstimator, TokenEstimator, COMPACTION_MAX_OUTPUT_TOKENS,
};
use crate::conversation::{
    segment_history, CompactionSummary, ConversationHistory, HistorySegment,
};
use crate::model::{
    ModelAssistantItem, ModelLimits, ModelMessage, ModelProvider, ModelRequest, ModelResponse,
    ProviderError, StopReason, ToolChoice,
};

use super::{normal_request, request_with_provisional_message, MODEL_EVENT_CHANNEL_CAPACITY};

const COMPACTION_SYSTEM_INSTRUCTION: &str = "Summarize the earlier coding-agent conversation for future continuation.\n\nPreserve concrete technical state:\n- user goals and constraints\n- decisions and rationale that affect future work\n- files inspected or modified and important changes\n- important code architecture and interfaces\n- commands/tests and their significant results\n- important errors and attempted fixes\n- unresolved work and next steps\n- current task state\n- exact identifiers or values when they matter\n\nDo not invent work that did not happen.\nDo not copy large tool outputs verbatim.\nDo not call tools.\nDo not emit function calls or tool calls.\nDo not request external actions.\nReturn only the continuation summary.";
pub(super) const COMPACTION_RETRY_INSTRUCTION: &str = "Retry after an invalid tool call: output plain text only. Never emit a tool/function call or request an external action.";
const COMPACTION_TOOL_CALL_RETRY_LIMIT: usize = 1;

pub(crate) enum CompactionError {
    Cancelled,
    NoHistory,
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompactionReplacement {
    pub summary: CompactionSummary,
    pub retained: Vec<ModelMessage>,
}

pub(crate) async fn compact_native<P>(
    provider: Arc<P>,
    history: &ConversationHistory,
    prefix: Vec<ModelMessage>,
    retained: Vec<ModelMessage>,
    limits: ModelLimits,
    output_limit: u64,
    cancel: CancellationToken,
) -> Result<CompactionReplacement, CompactionError>
where
    P: ModelProvider,
{
    let summary = summarize_compaction_prefix(
        provider,
        history.summary().cloned(),
        prefix,
        limits,
        output_limit,
        cancel,
    )
    .await?;
    Ok(CompactionReplacement { summary, retained })
}

pub(super) fn compaction_output_limit(limits: ModelLimits) -> u64 {
    let output_limit = limits
        .max_output_tokens
        .map(|limit| limit.min(COMPACTION_MAX_OUTPUT_TOKENS))
        .unwrap_or(COMPACTION_MAX_OUTPUT_TOKENS)
        .max(1);
    request_input_budget(limits.context_window, 0).map_or(output_limit, |available| {
        output_limit.min((available / 2).max(1))
    })
}

pub(super) async fn summarize_compaction_prefix<P>(
    provider: Arc<P>,
    mut summary: Option<CompactionSummary>,
    prefix: Vec<ModelMessage>,
    limits: ModelLimits,
    output_limit: u64,
    cancel: CancellationToken,
) -> Result<CompactionSummary, CompactionError>
where
    P: ModelProvider,
{
    if output_limit == 0 {
        return Err(CompactionError::Failed(
            "compaction failed: no safe output capacity remains alongside the retained context"
                .to_owned(),
        ));
    }
    let mut units = compaction_units(prefix);
    let estimator = ConservativeTokenEstimator;
    let initial_overhead =
        estimator.estimate_request(&compaction_request(summary.as_ref(), Vec::new(), 1, true));
    let all_input = initial_overhead.saturating_add(
        units
            .iter()
            .map(|unit| estimator.estimate_messages(unit))
            .sum::<u64>(),
    );
    let mut output_limit = output_limit;
    if request_input_budget(limits.context_window, output_limit)
        .is_some_and(|budget| all_input > budget)
    {
        let carry_overhead = estimator.estimate_request(&compaction_request(
            Some(&CompactionSummary::new("")),
            Vec::new(),
            1,
            true,
        ));
        let largest_unit = units
            .iter()
            .map(|unit| estimator.estimate_messages(unit))
            .max()
            .unwrap_or(0);
        let first_unit = units
            .front()
            .map_or(0, |unit| estimator.estimate_messages(unit));
        // Budget a full carry-forward summary, another safe unit, and the next output.
        let available = request_input_budget(limits.context_window, 0).expect("known input budget");
        output_limit = output_limit
            .min(
                available
                    .saturating_sub(carry_overhead)
                    .saturating_sub(largest_unit)
                    / 2,
            )
            .min(
                available
                    .saturating_sub(initial_overhead)
                    .saturating_sub(first_unit),
            );
        if output_limit == 0 {
            return Err(CompactionError::Failed(
                "compaction failed: no room for a summary alongside a safe history chunk"
                    .to_owned(),
            ));
        }
    }
    let budget = request_input_budget(limits.context_window, output_limit);

    while !units.is_empty() {
        if cancel.is_cancelled() {
            return Err(CompactionError::Cancelled);
        }
        let messages = take_compaction_chunk(&mut units, summary.as_ref(), budget)?;
        let mut summary_content = None;
        for attempt in 0..=COMPACTION_TOOL_CALL_RETRY_LIMIT {
            let retrying = attempt > 0;
            let request =
                compaction_request(summary.as_ref(), messages.clone(), output_limit, retrying);
            let response =
                match stream_private_model(Arc::clone(&provider), request, cancel.clone()).await {
                    Ok(response) => response,
                    Err(ProviderError::Cancelled) => return Err(CompactionError::Cancelled),
                    Err(error) => {
                        return Err(CompactionError::Failed(format!(
                            "compaction failed: {error}"
                        )))
                    }
                };
            let extracted = extract_summary(&response);
            if cancel.is_cancelled() {
                return Err(CompactionError::Cancelled);
            }
            match extracted {
                Ok(content) => {
                    summary_content = Some(content);
                    break;
                }
                Err(SummaryExtractionError::UnexpectedToolCall)
                    if attempt < COMPACTION_TOOL_CALL_RETRY_LIMIT =>
                {
                    tracing::warn!(
                        target: "ri_core::agent",
                        "compaction model returned unexpected tool call; retrying text-only request"
                    );
                }
                Err(SummaryExtractionError::UnexpectedToolCall) => {
                    return Err(CompactionError::Failed(
                        "compaction failed: model violated the text-only compaction contract by returning tool calls on both attempts"
                            .to_owned(),
                    ));
                }
                Err(SummaryExtractionError::Invalid(message)) => {
                    return Err(CompactionError::Failed(message));
                }
            }
        }
        summary = Some(CompactionSummary::new(
            summary_content.expect("compaction attempts produce a summary or return an error"),
        ));
    }

    summary.ok_or_else(|| CompactionError::Failed("compaction selected no history".to_owned()))
}

fn compaction_request(
    summary: Option<&CompactionSummary>,
    messages: Vec<ModelMessage>,
    output_limit: u64,
    retrying: bool,
) -> ModelRequest {
    let system_instruction = if retrying {
        format!("{COMPACTION_SYSTEM_INSTRUCTION}\n\n{COMPACTION_RETRY_INSTRUCTION}")
    } else {
        COMPACTION_SYSTEM_INSTRUCTION.to_owned()
    };
    let mut summary_messages = Vec::with_capacity(messages.len() + 2);
    summary_messages.push(ModelMessage::System {
        content: system_instruction,
    });
    if let Some(summary) = summary {
        summary_messages.push(summary.as_message());
    }
    summary_messages.extend(messages);
    ModelRequest {
        messages: summary_messages,
        tools: Vec::new(),
        tool_choice: Some(ToolChoice::None),
        max_tokens: Some(output_limit),
        reasoning_effort: None,
        sampling_params: Default::default(),
    }
}

fn compaction_units(messages: Vec<ModelMessage>) -> VecDeque<Vec<ModelMessage>> {
    let mut units = VecDeque::new();
    let mut current = Vec::new();
    let mut pending_tool_results = 0usize;

    for message in messages {
        match &message {
            ModelMessage::Assistant { items } => {
                debug_assert_eq!(pending_tool_results, 0);
                pending_tool_results += items
                    .iter()
                    .filter(|item| matches!(item, ModelAssistantItem::ToolCall(_)))
                    .count();
            }
            ModelMessage::ToolResult { .. } => {
                debug_assert!(pending_tool_results > 0);
                pending_tool_results = pending_tool_results.saturating_sub(1);
            }
            ModelMessage::System { .. }
            | ModelMessage::Developer { .. }
            | ModelMessage::User { .. } => {}
        }
        current.push(message);
        if pending_tool_results == 0 {
            units.push_back(std::mem::take(&mut current));
        }
    }

    debug_assert!(current.is_empty());
    if !current.is_empty() {
        units.push_back(current);
    }
    units
}

fn take_compaction_chunk(
    units: &mut VecDeque<Vec<ModelMessage>>,
    summary: Option<&CompactionSummary>,
    budget: Option<u64>,
) -> Result<Vec<ModelMessage>, CompactionError> {
    let Some(budget) = budget else {
        return Ok(units.drain(..).flatten().collect());
    };

    let estimator = ConservativeTokenEstimator;
    let mut estimated_tokens =
        estimator.estimate_request(&compaction_request(summary, Vec::new(), 1, true));
    let mut messages = Vec::new();
    while let Some(unit) = units.front() {
        let next_tokens = estimated_tokens.saturating_add(estimator.estimate_messages(unit));
        if next_tokens > budget {
            if messages.is_empty() {
                return Err(CompactionError::Failed(format!(
                    "compaction failed: the smallest safe history chunk is estimated at {next_tokens} input tokens, exceeding the model budget of {budget}"
                )));
            }
            break;
        }
        estimated_tokens = next_tokens;
        messages.extend(units.pop_front().expect("front unit exists"));
    }
    Ok(messages)
}

pub(super) fn select_compaction_prefix(
    history: &ConversationHistory,
    base_messages: &[ModelMessage],
    tools: &[crate::model::ToolDefinition],
    target: u64,
    force: bool,
    compact_latest: bool,
    provisional_message: Option<&ModelMessage>,
) -> Option<(Vec<ModelMessage>, Vec<ModelMessage>)> {
    let segments = segment_history(history.messages());
    if segments.is_empty() {
        return None;
    }
    let user_segments: Vec<usize> = segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| segment.has_user_message.then_some(index))
        .collect();
    let current_segment = user_segments.last().copied().unwrap_or(segments.len() - 1);
    let eligible_end = if compact_latest {
        segments.len()
    } else if force {
        current_segment
    } else if user_segments.len() > 2 {
        user_segments[user_segments.len() - 2].min(current_segment)
    } else {
        0
    };

    let before_tokens =
        ConservativeTokenEstimator.estimate_request(&request_with_provisional_message(
            normal_request(base_messages, history, tools),
            provisional_message,
        ));
    if !force && before_tokens <= target {
        return None;
    }

    let mut prefix = Vec::new();
    let mut retained_start = 0;
    for (index, segment) in segments.iter().take(eligible_end).enumerate() {
        if !segment.safe_to_compact {
            break;
        }
        prefix.extend(segment.messages.iter().cloned());
        retained_start = index + 1;
        let retained = messages_from_segments(&segments, retained_start);
        if projected_tokens(base_messages, tools, &retained, provisional_message) <= target {
            return Some((prefix, retained));
        }
    }

    if !prefix.is_empty() {
        Some((prefix, messages_from_segments(&segments, retained_start)))
    } else {
        None
    }
}

pub(super) fn projected_tokens(
    base_messages: &[ModelMessage],
    tools: &[crate::model::ToolDefinition],
    retained: &[ModelMessage],
    provisional_message: Option<&ModelMessage>,
) -> u64 {
    let placeholder = ConversationHistory::new(Some(CompactionSummary::new("")), retained.to_vec());
    ConservativeTokenEstimator.estimate_request(&request_with_provisional_message(
        normal_request(base_messages, &placeholder, tools),
        provisional_message,
    ))
}

fn messages_from_segments(segments: &[HistorySegment], start: usize) -> Vec<ModelMessage> {
    segments
        .iter()
        .skip(start)
        .flat_map(|segment| segment.messages.iter().cloned())
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SummaryExtractionError {
    UnexpectedToolCall,
    Invalid(String),
}

pub(super) fn extract_summary(response: &ModelResponse) -> Result<String, SummaryExtractionError> {
    if response.stop_reason == StopReason::ToolCalls
        || response
            .items
            .iter()
            .any(|item| matches!(item, ModelAssistantItem::ToolCall(_)))
    {
        return Err(SummaryExtractionError::UnexpectedToolCall);
    }
    if response.stop_reason != StopReason::Stop {
        return Err(SummaryExtractionError::Invalid(format!(
            "compaction response did not finish successfully: {:?}",
            response.stop_reason
        )));
    }
    let summary: String = response
        .items
        .iter()
        .filter_map(|item| match item {
            ModelAssistantItem::Text { content } => Some(content.as_str()),
            ModelAssistantItem::Reasoning(_)
            | ModelAssistantItem::Refusal { .. }
            | ModelAssistantItem::ToolCall(_) => None,
        })
        .collect();
    if summary.trim().is_empty() {
        return Err(SummaryExtractionError::Invalid(
            "compaction response was empty".to_owned(),
        ));
    }
    Ok(summary)
}

async fn stream_private_model<P>(
    provider: Arc<P>,
    request: ModelRequest,
    cancel: CancellationToken,
) -> Result<ModelResponse, ProviderError>
where
    P: ModelProvider,
{
    let (model_event_tx, mut model_event_rx) = mpsc::channel(MODEL_EVENT_CHANNEL_CAPACITY);
    let provider_cancel = cancel.clone();
    let mut provider_task = tokio::spawn(async move {
        provider
            .stream(request, model_event_tx, provider_cancel)
            .await
    });
    let provider_result = loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                provider_task.abort();
                let _ = provider_task.await;
                return Err(ProviderError::Cancelled);
            }
            _ = model_event_rx.recv() => {}
            result = &mut provider_task => break result,
        }
    };
    while model_event_rx.recv().await.is_some() {}
    match provider_result {
        Ok(result) => result,
        Err(error) => Err(ProviderError::Failed {
            message: format!("provider task failed: {error}"),
        }),
    }
}
