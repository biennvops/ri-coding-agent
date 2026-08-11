use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use futures_util::StreamExt;
use reqwest::header::{HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::{ApiKind, ResolvedModel};

use super::{
    ModelAssistantItem, ModelEvent, ModelLimits, ModelMessage, ModelProvider, ModelRequest,
    ModelResponse, ModelThinking, ModelToolCall, ProviderError, StopReason, ToolDefinition, Usage,
};

const MAX_ERROR_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_ERROR_LOG_BYTES: usize = 2 * 1024;
const ERROR_RESPONSE_TRUNCATED_MARKER: &str = "\n[… provider error body truncated …]";
const REDACTED_DIAGNOSTIC_CONTENT: &str = "<redacted>";

struct ProviderErrorBody {
    content: String,
    truncated: bool,
}

impl ProviderErrorBody {
    fn user_message(&self) -> String {
        let mut message = self.content.clone();
        if self.truncated {
            message.push_str(ERROR_RESPONSE_TRUNCATED_MARKER);
        }
        message
    }
}

#[derive(Clone)]
pub struct OpenAiProvider {
    selected: Arc<RwLock<ResolvedModel>>,
    client: Client,
}

impl OpenAiProvider {
    pub fn new(model: ResolvedModel) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .build()
            .map_err(|error| ProviderError::Failed {
                message: format!("could not create HTTP client: {error}"),
            })?;
        Ok(Self::with_client(model, client))
    }

    pub fn with_client(model: ResolvedModel, client: Client) -> Self {
        Self {
            selected: Arc::new(RwLock::new(model)),
            client,
        }
    }

    pub fn current_model(&self) -> ResolvedModel {
        self.selected
            .read()
            .expect("selected model lock should not be poisoned")
            .clone()
    }

    pub fn set_model(&self, model: ResolvedModel) {
        *self
            .selected
            .write()
            .expect("selected model lock should not be poisoned") = model;
    }

    pub fn limits(&self) -> ModelLimits {
        let model = self.current_model();
        ModelLimits {
            context_window: model.context_window,
            max_output_tokens: model.max_tokens,
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for OpenAiProvider {
    fn limits(&self) -> ModelLimits {
        OpenAiProvider::limits(self)
    }

    async fn stream(
        &self,
        request: ModelRequest,
        events: mpsc::Sender<ModelEvent>,
        cancel: CancellationToken,
    ) -> Result<ModelResponse, ProviderError> {
        if cancel.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }

        let model = self.current_model();
        tracing::debug!(
            target: "ri_core::model",
            provider = %model.model_ref.provider,
            model = %model.model_ref.model,
            api = %model.api,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            reasoning_effort = request.reasoning_effort.as_deref().unwrap_or("off"),
            "provider request started"
        );
        let (endpoint, body) = request_for(&model, &request)?;
        let mut builder = self
            .client
            .post(endpoint)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "text/event-stream")
            .json(&body);

        for (name, value) in &model.headers {
            let name =
                HeaderName::try_from(name.as_str()).map_err(|error| ProviderError::Failed {
                    message: format!("invalid configured header {name:?}: {error}"),
                })?;
            let value = HeaderValue::try_from(value).map_err(|error| ProviderError::Failed {
                message: format!("invalid configured value for header {name}: {error}"),
            })?;
            builder = builder.header(name, value);
        }
        if model.auth_header {
            if let Some(api_key) = &model.api_key {
                builder = builder.header(AUTHORIZATION, format!("Bearer {api_key}"));
            }
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            result = builder.send() => result.map_err(|error| ProviderError::Failed {
                message: error.to_string(),
            })?,
        };

        tracing::debug!(
            target: "ri_core::model",
            provider = %model.model_ref.provider,
            model = %model.model_ref.model,
            api = %model.api,
            status = response.status().as_u16(),
            "provider response received"
        );
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_body = read_error_body(response, cancel.clone()).await?;
            if tracing::enabled!(target: "ri_core::model", tracing::Level::WARN) {
                let diagnostic = provider_error_diagnostic(&error_body, &model, &request);
                tracing::warn!(
                    target: "ri_core::model",
                    provider = %model.model_ref.provider,
                    model = %model.model_ref.model,
                    api = %model.api,
                    status,
                    error_body_bytes = error_body.content.len(),
                    error_body_truncated = error_body.truncated,
                    error_body = %diagnostic,
                    "provider HTTP request failed"
                );
            }
            let context_overflow = is_context_overflow(status, &error_body.content);
            let message = error_body.user_message();
            if context_overflow {
                return Err(ProviderError::ContextOverflow { message });
            }
            return Err(ProviderError::Http { status, message });
        }

        let api = model.api;
        let mut stream = response.bytes_stream();
        let mut parser = SseParser::default();
        let mut collector = ResponseCollector::default();
        let mut done = false;

        while !done {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                chunk = stream.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk.map_err(|error| ProviderError::Failed {
                message: error.to_string(),
            })?;
            done = process_payloads(parser.feed(&chunk)?, api, &events, &cancel, &mut collector)
                .await?;
        }

        if !done {
            process_payloads(parser.finish()?, api, &events, &cancel, &mut collector).await?;
        }

        let result = collector.finish();
        match &result {
            Ok(response) => tracing::debug!(
                target: "ri_core::model",
                provider = %model.model_ref.provider,
                model = %model.model_ref.model,
                stop_reason = ?response.stop_reason,
                "provider request finished"
            ),
            Err(error) => tracing::warn!(
                target: "ri_core::model",
                provider = %model.model_ref.provider,
                model = %model.model_ref.model,
                error_kind = provider_error_kind(error),
                "provider request failed"
            ),
        }
        result
    }
}

fn provider_error_kind(error: &ProviderError) -> &'static str {
    match error {
        ProviderError::Cancelled => "cancelled",
        ProviderError::Failed { .. } => "failed",
        ProviderError::ContextOverflow { .. } => "context_overflow",
        ProviderError::Http { .. } => "http",
        ProviderError::Malformed { .. } => "malformed",
    }
}

async fn process_payloads(
    payloads: Vec<String>,
    api: ApiKind,
    events: &mpsc::Sender<ModelEvent>,
    cancel: &CancellationToken,
    collector: &mut ResponseCollector,
) -> Result<bool, ProviderError> {
    for payload in payloads {
        if payload.trim() == "[DONE]" {
            return Ok(true);
        }

        let parsed = parse_payload(api, &payload)?;
        if parsed.terminal {
            collector.terminal_seen = true;
        }
        if let Some(reason) = parsed.stop_reason {
            collector.stop_reason = Some(reason);
        }
        for event in parsed.events {
            collector.record(&event);
            tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                result = events.send(event) => result.map_err(|_| ProviderError::Failed {
                    message: "model event stream closed".to_owned(),
                })?,
            }
        }
    }
    Ok(false)
}

fn is_context_overflow(status: u16, message: &str) -> bool {
    if status == 413 {
        return true;
    }
    if status != 400 {
        return false;
    }
    let message = message.to_ascii_lowercase();
    [
        "context length",
        "context window",
        "maximum context",
        "too many tokens",
        "max context",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

async fn read_error_body(
    response: reqwest::Response,
    cancel: CancellationToken,
) -> Result<ProviderErrorBody, ProviderError> {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    let mut truncated = false;

    while let Some(chunk) = tokio::select! {
        _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
        chunk = stream.next() => chunk,
    } {
        let chunk = chunk.map_err(|error| ProviderError::Failed {
            message: error.to_string(),
        })?;
        let remaining = MAX_ERROR_RESPONSE_BYTES.saturating_sub(bytes.len());
        if chunk.len() > remaining {
            bytes.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        bytes.extend_from_slice(&chunk);
    }

    Ok(ProviderErrorBody {
        content: String::from_utf8_lossy(&bytes).trim().to_owned(),
        truncated,
    })
}

fn provider_error_diagnostic(
    error_body: &ProviderErrorBody,
    model: &ResolvedModel,
    request: &ModelRequest,
) -> String {
    let mut fragments = request_content_fragments(request);
    fragments.extend(
        model
            .api_key
            .iter()
            .chain(model.headers.values())
            .map(String::as_str),
    );
    fragments.retain(|fragment| !fragment.is_empty());
    let diagnostic = match serde_json::from_str::<Value>(&error_body.content) {
        Ok(mut value) => {
            redact_json_strings(&mut value, &fragments);
            redact_diagnostic_json(&mut value);
            value.to_string()
        }
        Err(_) => escape_diagnostic_text(&redact_fragments(&error_body.content, &fragments)),
    };

    truncate_diagnostic(&diagnostic, error_body.content.len())
}

fn request_content_fragments(request: &ModelRequest) -> Vec<&str> {
    let mut fragments = Vec::new();
    for message in &request.messages {
        match message {
            ModelMessage::System { content }
            | ModelMessage::Developer { content }
            | ModelMessage::User { content }
            | ModelMessage::ToolResult { content, .. } => fragments.push(content.as_str()),
            ModelMessage::Assistant { items } => {
                for item in items {
                    match item {
                        ModelAssistantItem::Text { content }
                        | ModelAssistantItem::Refusal { content } => {
                            fragments.push(content.as_str())
                        }
                        ModelAssistantItem::Reasoning(thinking) => {
                            fragments.push(thinking.summary.as_str());
                            fragments.push(thinking.content.as_str());
                            if let Some(encrypted_content) = thinking.encrypted_content.as_deref() {
                                fragments.push(encrypted_content);
                            }
                        }
                        ModelAssistantItem::ToolCall(tool_call) => {
                            fragments.push(tool_call.arguments.as_str());
                        }
                    }
                }
            }
        }
    }
    for tool in &request.tools {
        if let Some(description) = tool.description.as_deref() {
            fragments.push(description);
        }
    }
    fragments.retain(|fragment| !fragment.is_empty());
    fragments
}

fn redact_json_strings(value: &mut Value, fragments: &[&str]) {
    match value {
        Value::Object(object) => {
            for value in object.values_mut() {
                redact_json_strings(value, fragments);
            }
        }
        Value::Array(values) => {
            for value in values {
                redact_json_strings(value, fragments);
            }
        }
        Value::String(text) => *text = redact_fragments(text, fragments),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn redact_fragments(text: &str, fragments: &[&str]) -> String {
    let mut redacted = String::with_capacity(text.len());
    let mut start = 0;
    while start < text.len() {
        let next = fragments
            .iter()
            .filter_map(|fragment| {
                text[start..]
                    .find(fragment)
                    .map(|offset| (start + offset, fragment.len()))
            })
            .min_by_key(|&(offset, length)| (offset, usize::MAX - length));
        let Some((offset, length)) = next else {
            redacted.push_str(&text[start..]);
            break;
        };
        redacted.push_str(&text[start..offset]);
        redacted.push_str(REDACTED_DIAGNOSTIC_CONTENT);
        start = offset + length;
    }
    redacted
}

fn escape_diagnostic_text(text: &str) -> String {
    text.chars()
        .flat_map(|character| match character {
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect(),
            character if character.is_control() => " ".chars().collect(),
            character => vec![character],
        })
        .collect()
}

fn redact_diagnostic_json(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if is_sensitive_diagnostic_key(key) {
                    *value = Value::String("<redacted>".to_owned());
                } else {
                    redact_diagnostic_json(value);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                redact_diagnostic_json(value);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn is_sensitive_diagnostic_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase().replace('-', "_");
    key.contains("api_key")
        || key.contains("apikey")
        || key.contains("authorization")
        || key.contains("secret")
        || key.contains("password")
        || key.contains("credential")
        || key.contains("access_token")
        || key.contains("refresh_token")
        || matches!(
            key.as_str(),
            "prompt" | "prompts" | "messages" | "input" | "tool_output" | "tool_outputs"
        )
}

fn truncate_diagnostic(diagnostic: &str, original_bytes: usize) -> String {
    if diagnostic.len() <= MAX_ERROR_LOG_BYTES {
        return diagnostic.to_owned();
    }
    let marker = format!("… [diagnostic truncated; response {original_bytes} bytes]");
    let mut end = MAX_ERROR_LOG_BYTES.saturating_sub(marker.len());
    while end > 0 && !diagnostic.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{}", &diagnostic[..end], marker)
}

#[derive(Default)]
struct ResponseCollector {
    items: BTreeMap<usize, ModelAssistantItem>,
    unindexed_items: Vec<ModelAssistantItem>,
    stop_reason: Option<StopReason>,
    terminal_seen: bool,
    usage: Option<Usage>,
}

impl ResponseCollector {
    fn record(&mut self, event: &ModelEvent) {
        match event {
            ModelEvent::ToolCallDelta {
                index,
                call_id,
                item_id,
                name,
                arguments,
                arguments_complete,
            } => {
                let item = self.items.entry(*index).or_insert_with(|| {
                    ModelAssistantItem::ToolCall(ModelToolCall {
                        index: *index,
                        ..ModelToolCall::default()
                    })
                });
                let ModelAssistantItem::ToolCall(tool_call) = item else {
                    return;
                };
                if call_id.is_some() {
                    tool_call.call_id = call_id.clone();
                }
                if item_id.is_some() {
                    tool_call.item_id = item_id.clone();
                }
                if name.is_some() {
                    tool_call.name = name.clone();
                }
                if *arguments_complete {
                    tool_call.arguments = arguments.clone();
                } else {
                    tool_call.arguments.push_str(arguments);
                }
            }
            ModelEvent::AssistantTextDelta { index, text } => {
                if let Some(index) = index {
                    let item =
                        self.items
                            .entry(*index)
                            .or_insert_with(|| ModelAssistantItem::Text {
                                content: String::new(),
                            });
                    if let ModelAssistantItem::Text { content } = item {
                        content.push_str(text);
                    }
                } else {
                    self.append_unindexed_text(text);
                }
            }
            ModelEvent::AssistantTextItem { index, content } => {
                let item = self
                    .items
                    .entry(*index)
                    .or_insert_with(|| ModelAssistantItem::Text {
                        content: String::new(),
                    });
                if let ModelAssistantItem::Text { content: current } = item {
                    if let Some(content) = content {
                        *current = content.clone();
                    }
                }
            }
            ModelEvent::AssistantRefusalDelta { index, text } => {
                if let Some(index) = index {
                    let item =
                        self.items
                            .entry(*index)
                            .or_insert_with(|| ModelAssistantItem::Refusal {
                                content: String::new(),
                            });
                    if matches!(item, ModelAssistantItem::Text { content } if content.is_empty()) {
                        *item = ModelAssistantItem::Refusal {
                            content: String::new(),
                        };
                    }
                    if let ModelAssistantItem::Refusal { content } = item {
                        content.push_str(text);
                    }
                } else {
                    self.append_unindexed_refusal(text);
                }
            }
            ModelEvent::AssistantRefusalItem { index, content } => {
                let item =
                    self.items
                        .entry(*index)
                        .or_insert_with(|| ModelAssistantItem::Refusal {
                            content: String::new(),
                        });
                if matches!(item, ModelAssistantItem::Text { content } if content.is_empty()) {
                    *item = ModelAssistantItem::Refusal {
                        content: String::new(),
                    };
                }
                if let ModelAssistantItem::Refusal { content: current } = item {
                    if let Some(content) = content {
                        *current = content.clone();
                    }
                }
            }
            ModelEvent::AssistantThinkingDelta { item_id, text } => {
                if let Some(index) = self.reasoning_index(item_id.as_deref()) {
                    if let Some(ModelAssistantItem::Reasoning(thinking)) =
                        self.items.get_mut(&index)
                    {
                        thinking.summary.push_str(text);
                    }
                } else {
                    self.append_unindexed_summary(item_id.as_deref(), text);
                }
            }
            ModelEvent::AssistantThinkingContentDelta { item_id, text } => {
                if let Some(index) = self.reasoning_index(item_id.as_deref()) {
                    if let Some(ModelAssistantItem::Reasoning(thinking)) =
                        self.items.get_mut(&index)
                    {
                        thinking.content.push_str(text);
                    }
                } else {
                    self.append_unindexed_content(item_id.as_deref(), text);
                }
            }
            ModelEvent::AssistantThinkingItem {
                index,
                item_id,
                summary,
                content,
                encrypted_content,
            } => {
                let item = self
                    .items
                    .entry(*index)
                    .or_insert_with(|| ModelAssistantItem::Reasoning(ModelThinking::default()));
                let ModelAssistantItem::Reasoning(thinking) = item else {
                    return;
                };
                if item_id.is_some() {
                    thinking.item_id = item_id.clone();
                }
                if summary.is_some() {
                    thinking.summary = summary.clone().unwrap_or_default();
                }
                if content.is_some() {
                    thinking.content = content.clone().unwrap_or_default();
                }
                if encrypted_content.is_some() {
                    thinking.encrypted_content = encrypted_content.clone();
                }
            }
            ModelEvent::UsageUpdated(usage) => self.usage = Some(usage.clone()),
        }
    }

    fn append_unindexed_text(&mut self, text: &str) {
        if let Some(ModelAssistantItem::Text { content }) = self
            .unindexed_items
            .iter_mut()
            .rev()
            .find(|item| matches!(item, ModelAssistantItem::Text { .. }))
        {
            content.push_str(text);
        } else {
            self.unindexed_items.push(ModelAssistantItem::Text {
                content: text.to_owned(),
            });
        }
    }

    fn append_unindexed_refusal(&mut self, text: &str) {
        if let Some(ModelAssistantItem::Refusal { content }) = self
            .unindexed_items
            .iter_mut()
            .rev()
            .find(|item| matches!(item, ModelAssistantItem::Refusal { .. }))
        {
            content.push_str(text);
        } else {
            self.unindexed_items.push(ModelAssistantItem::Refusal {
                content: text.to_owned(),
            });
        }
    }

    fn append_unindexed_summary(&mut self, item_id: Option<&str>, text: &str) {
        let index = self.unindexed_items.iter().rposition(|item| {
            matches!(item, ModelAssistantItem::Reasoning(thinking)
                if item_id.is_none() || thinking.item_id.as_deref() == item_id)
        });
        if let Some(index) = index {
            if let ModelAssistantItem::Reasoning(thinking) = &mut self.unindexed_items[index] {
                thinking.summary.push_str(text);
            }
        } else {
            self.unindexed_items
                .push(ModelAssistantItem::Reasoning(ModelThinking {
                    item_id: item_id.map(str::to_owned),
                    summary: text.to_owned(),
                    ..ModelThinking::default()
                }));
        }
    }

    fn append_unindexed_content(&mut self, item_id: Option<&str>, text: &str) {
        let index = self.unindexed_items.iter().rposition(|item| {
            matches!(item, ModelAssistantItem::Reasoning(thinking)
                if item_id.is_none() || thinking.item_id.as_deref() == item_id)
        });
        if let Some(index) = index {
            if let ModelAssistantItem::Reasoning(thinking) = &mut self.unindexed_items[index] {
                thinking.content.push_str(text);
            }
        } else {
            self.unindexed_items
                .push(ModelAssistantItem::Reasoning(ModelThinking {
                    item_id: item_id.map(str::to_owned),
                    content: text.to_owned(),
                    ..ModelThinking::default()
                }));
        }
    }

    fn reasoning_index(&self, item_id: Option<&str>) -> Option<usize> {
        self.items.iter().rev().find_map(|(index, item)| {
            let ModelAssistantItem::Reasoning(thinking) = item else {
                return None;
            };
            if item_id.is_none() || thinking.item_id.as_deref() == item_id {
                Some(*index)
            } else {
                None
            }
        })
    }

    fn finish(self) -> Result<ModelResponse, ProviderError> {
        if !self.terminal_seen {
            return Err(ProviderError::Failed {
                message: "response stream ended before a terminal event".to_owned(),
            });
        }

        let mut items = self.unindexed_items;
        items.extend(self.items.into_values());
        let has_tool_calls = items
            .iter()
            .any(|item| matches!(item, ModelAssistantItem::ToolCall(_)));
        let stop_reason = match self.stop_reason {
            Some(StopReason::Stop) if has_tool_calls => StopReason::ToolCalls,
            Some(reason) => reason,
            None if has_tool_calls => StopReason::ToolCalls,
            None => StopReason::Stop,
        };
        Ok(ModelResponse {
            items,
            stop_reason,
            usage: self.usage,
        })
    }
}

#[derive(Debug, Default)]
struct ParsedPayload {
    events: Vec<ModelEvent>,
    stop_reason: Option<StopReason>,
    terminal: bool,
}

fn parse_payload(api: ApiKind, payload: &str) -> Result<ParsedPayload, ProviderError> {
    let value: Value = serde_json::from_str(payload).map_err(|error| ProviderError::Malformed {
        message: error.to_string(),
    })?;
    if let Some(message) = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
    {
        return Err(ProviderError::Failed {
            message: message.to_owned(),
        });
    }
    if value.get("type").and_then(Value::as_str) == Some("error") {
        return Err(ProviderError::Failed {
            message: value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Responses stream reported an error")
                .to_owned(),
        });
    }

    match api {
        ApiKind::OpenAiResponses => parse_responses_payload(&value),
        ApiKind::OpenAiCompletions => parse_completions_payload(&value),
    }
}

fn parse_responses_payload(value: &Value) -> Result<ParsedPayload, ProviderError> {
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut parsed = ParsedPayload::default();

    match event_type {
        "response.output_text.delta" => {
            if let Some(text) = value.get("delta").and_then(Value::as_str) {
                parsed.events.push(ModelEvent::AssistantTextDelta {
                    index: value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .map(|index| index as usize),
                    text: text.to_owned(),
                });
            }
        }
        "response.refusal.delta" => {
            if let Some(text) = value.get("delta").and_then(Value::as_str) {
                parsed.events.push(ModelEvent::AssistantRefusalDelta {
                    index: value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .map(|index| index as usize),
                    text: text.to_owned(),
                });
            }
        }
        "response.refusal.done" => {
            parsed.events.push(ModelEvent::AssistantRefusalItem {
                index: value
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                content: value
                    .get("refusal")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
        }
        "response.reasoning_summary_text.delta" | "response.reasoning.delta" => {
            if let Some(text) = value.get("delta").and_then(Value::as_str) {
                parsed.events.push(ModelEvent::AssistantThinkingDelta {
                    item_id: value
                        .get("item_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    text: text.to_owned(),
                });
            }
        }
        "response.reasoning_text.delta" => {
            if let Some(text) = value.get("delta").and_then(Value::as_str) {
                parsed
                    .events
                    .push(ModelEvent::AssistantThinkingContentDelta {
                        item_id: value
                            .get("item_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        text: text.to_owned(),
                    });
            }
        }
        "response.function_call_arguments.delta" => {
            let index = value
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            parsed.events.push(ModelEvent::ToolCallDelta {
                index,
                call_id: value
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                item_id: value
                    .get("item_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                name: value.get("name").and_then(Value::as_str).map(str::to_owned),
                arguments: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments_complete: false,
            });
        }
        "response.output_item.added" => {
            let item = value.get("item").unwrap_or(&Value::Null);
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                parsed.events.push(ModelEvent::ToolCallDelta {
                    index: value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    call_id: item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    item_id: item.get("id").and_then(Value::as_str).map(str::to_owned),
                    name: item.get("name").and_then(Value::as_str).map(str::to_owned),
                    arguments: item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments_complete: false,
                });
            } else if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                parsed.events.push(reasoning_item_event(
                    value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    item,
                ));
            } else if item.get("type").and_then(Value::as_str) == Some("message") {
                parsed.events.push(message_item_event(
                    value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    item,
                ));
            }
        }
        "response.output_item.done" => {
            let item = value.get("item").unwrap_or(&Value::Null);
            if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                parsed.events.push(reasoning_item_event(
                    value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    item,
                ));
            } else if item.get("type").and_then(Value::as_str) == Some("message") {
                parsed.events.push(message_item_event(
                    value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    item,
                ));
            }
        }
        "response.function_call_arguments.done" => {
            let item = value.get("item").unwrap_or(&Value::Null);
            parsed.events.push(ModelEvent::ToolCallDelta {
                index: value
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                call_id: value
                    .get("call_id")
                    .or_else(|| item.get("call_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                item_id: value
                    .get("item_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                name: value
                    .get("name")
                    .or_else(|| item.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                arguments: value
                    .get("arguments")
                    .or_else(|| item.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments_complete: true,
            });
        }
        "response.incomplete" => {
            let response = value.get("response").unwrap_or(value);
            parsed.terminal = true;
            if let Some(usage) = response.get("usage").and_then(parse_usage) {
                parsed.events.push(ModelEvent::UsageUpdated(usage));
            }
            parsed.stop_reason = Some(
                response
                    .get("incomplete_details")
                    .and_then(|details| details.get("reason"))
                    .and_then(Value::as_str)
                    .map(normalize_incomplete_reason)
                    .unwrap_or(StopReason::Incomplete),
            );
        }
        "response.failed" => {
            let response = value.get("response").unwrap_or(value);
            return Err(ProviderError::Failed {
                message: response
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .or_else(|| value.get("message").and_then(Value::as_str))
                    .unwrap_or("Responses stream reported a failure")
                    .to_owned(),
            });
        }
        "response.completed" | "response.done" => {
            let response = value.get("response").unwrap_or(value);
            parsed.terminal = true;
            if let Some(usage) = response.get("usage").and_then(parse_usage) {
                parsed.events.push(ModelEvent::UsageUpdated(usage));
            }
            parsed.stop_reason = response
                .get("status")
                .and_then(Value::as_str)
                .and_then(normalize_stop_reason)
                .or_else(|| {
                    response
                        .get("incomplete_details")
                        .and_then(|details| details.get("reason"))
                        .and_then(Value::as_str)
                        .and_then(normalize_stop_reason)
                });
        }
        _ => {}
    }

    Ok(parsed)
}

fn parse_completions_payload(value: &Value) -> Result<ParsedPayload, ProviderError> {
    let mut parsed = ParsedPayload::default();
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());

    if let Some(delta) = choice.and_then(|choice| choice.get("delta")) {
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            parsed.events.push(ModelEvent::AssistantTextDelta {
                index: None,
                text: text.to_owned(),
            });
        }
        if let Some(text) = delta.get("refusal").and_then(Value::as_str) {
            parsed.events.push(ModelEvent::AssistantRefusalDelta {
                index: None,
                text: text.to_owned(),
            });
        }
        for key in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(text) = delta.get(key).and_then(Value::as_str) {
                parsed.events.push(ModelEvent::AssistantThinkingDelta {
                    item_id: None,
                    text: text.to_owned(),
                });
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let function = tool_call.get("function").unwrap_or(&Value::Null);
                parsed.events.push(ModelEvent::ToolCallDelta {
                    index: tool_call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize,
                    call_id: tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    item_id: None,
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    arguments: function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments_complete: false,
                });
            }
        }
    }

    if let Some(finish_reason) = choice
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
    {
        parsed.terminal = true;
        parsed.stop_reason = normalize_stop_reason(finish_reason);
    }
    if let Some(usage) = value.get("usage").and_then(parse_usage) {
        parsed.events.push(ModelEvent::UsageUpdated(usage));
    }

    Ok(parsed)
}

fn message_item_event(index: usize, item: &Value) -> ModelEvent {
    if let Some(refusal) = refusal_text(item.get("content")) {
        ModelEvent::AssistantRefusalItem {
            index,
            content: Some(refusal),
        }
    } else {
        ModelEvent::AssistantTextItem {
            index,
            content: reasoning_text(item.get("content")),
        }
    }
}

fn refusal_text(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|part| part.get("refusal").and_then(Value::as_str))
                .collect();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn reasoning_item_event(index: usize, item: &Value) -> ModelEvent {
    ModelEvent::AssistantThinkingItem {
        index,
        item_id: item.get("id").and_then(Value::as_str).map(str::to_owned),
        summary: reasoning_text(item.get("summary")),
        content: reasoning_text(item.get("content")),
        encrypted_content: item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

fn reasoning_text(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn parse_usage(value: &Value) -> Option<Usage> {
    let input_tokens = value
        .get("input_tokens")
        .or_else(|| value.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let output_tokens = value
        .get("output_tokens")
        .or_else(|| value.get("completion_tokens"))
        .and_then(Value::as_u64);
    let total_tokens = value.get("total_tokens").and_then(Value::as_u64);
    let cache_read_tokens = value
        .get("input_tokens_details")
        .or_else(|| value.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);

    if input_tokens.is_none()
        && output_tokens.is_none()
        && total_tokens.is_none()
        && cache_read_tokens.is_none()
    {
        return None;
    }

    Some(Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cache_read_tokens,
        cache_write_tokens: None,
    })
}

fn normalize_stop_reason(value: &str) -> Option<StopReason> {
    match value {
        "stop" | "completed" => Some(StopReason::Stop),
        "tool_calls" | "function_call" => Some(StopReason::ToolCalls),
        "length" | "max_output_tokens" | "max_tokens" => Some(StopReason::Length),
        "content_filter" => Some(StopReason::ContentFilter),
        _ => None,
    }
}

fn normalize_incomplete_reason(value: &str) -> StopReason {
    match value {
        "max_output_tokens" | "max_tokens" | "length" => StopReason::Length,
        "content_filter" => StopReason::ContentFilter,
        _ => StopReason::Incomplete,
    }
}

fn request_for(
    model: &ResolvedModel,
    request: &ModelRequest,
) -> Result<(String, Value), ProviderError> {
    let endpoint = format!(
        "{}/{}",
        model.base_url.trim_end_matches('/'),
        match model.api {
            ApiKind::OpenAiResponses => "responses",
            ApiKind::OpenAiCompletions => "chat/completions",
        }
    );
    let body = match model.api {
        ApiKind::OpenAiResponses => responses_body(model, request),
        ApiKind::OpenAiCompletions => completions_body(model, request),
    };
    Ok((endpoint, body))
}

fn responses_body(model: &ResolvedModel, request: &ModelRequest) -> Value {
    let mut body = sampling_body(model, request);
    body.insert(
        "model".to_owned(),
        Value::String(model.model_ref.model.clone()),
    );
    body.insert(
        "input".to_owned(),
        Value::Array(
            request
                .messages
                .iter()
                .flat_map(|message| responses_message_items(message, model))
                .collect(),
        ),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    if let Some(max_tokens) = request.max_tokens.or(model.max_tokens) {
        body.insert("max_output_tokens".to_owned(), json!(max_tokens));
    }
    if let Some(effort) = request.reasoning_effort.as_deref() {
        if model.compatibility.supports_reasoning_effort {
            body.insert("reasoning".to_owned(), json!({"effort": effort}));
        }
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(request.tools.iter().map(responses_tool).collect()),
        );
    }
    Value::Object(body)
}

fn responses_message_items(message: &ModelMessage, model: &ResolvedModel) -> Vec<Value> {
    match message {
        ModelMessage::System { content } => vec![json!({
            "role": "system",
            "content": content,
        })],
        ModelMessage::Developer { content } => vec![json!({
            "role": role_name("developer", model),
            "content": content,
        })],
        ModelMessage::User { content } => vec![json!({
            "role": "user",
            "content": content,
        })],
        ModelMessage::Assistant { items } => items.iter().map(responses_assistant_item).collect(),
        ModelMessage::ToolResult {
            tool_call_id,
            content,
            ..
        } => vec![json!({
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": content,
        })],
    }
}

fn completions_body(model: &ResolvedModel, request: &ModelRequest) -> Value {
    let mut body = sampling_body(model, request);
    body.insert(
        "model".to_owned(),
        Value::String(model.model_ref.model.clone()),
    );
    body.insert(
        "messages".to_owned(),
        Value::Array(
            request
                .messages
                .iter()
                .map(|message| completions_message(message, model))
                .collect(),
        ),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    body.insert("stream_options".to_owned(), json!({"include_usage": true}));
    if let Some(max_tokens) = request.max_tokens.or(model.max_tokens) {
        body.insert("max_tokens".to_owned(), json!(max_tokens));
    }
    if let Some(effort) = request.reasoning_effort.as_deref() {
        if model.compatibility.supports_reasoning_effort {
            body.insert(
                "reasoning_effort".to_owned(),
                Value::String(effort.to_owned()),
            );
        }
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(request.tools.iter().map(completions_tool).collect()),
        );
    }
    Value::Object(body)
}

fn completions_message(message: &ModelMessage, model: &ResolvedModel) -> Value {
    match message {
        ModelMessage::System { content } => json!({
            "role": "system",
            "content": content,
        }),
        ModelMessage::Developer { content } => json!({
            "role": role_name("developer", model),
            "content": content,
        }),
        ModelMessage::User { content } => json!({
            "role": "user",
            "content": content,
        }),
        ModelMessage::Assistant { items } => {
            let content: String = items
                .iter()
                .filter_map(|item| match item {
                    ModelAssistantItem::Text { content } => Some(content.as_str()),
                    ModelAssistantItem::Reasoning(_)
                    | ModelAssistantItem::Refusal { .. }
                    | ModelAssistantItem::ToolCall(_) => None,
                })
                .collect();
            let tool_calls: Vec<&ModelToolCall> = items
                .iter()
                .filter_map(|item| match item {
                    ModelAssistantItem::ToolCall(tool_call) => Some(tool_call),
                    ModelAssistantItem::Text { .. }
                    | ModelAssistantItem::Reasoning(_)
                    | ModelAssistantItem::Refusal { .. } => None,
                })
                .collect();
            let refusals: String = items
                .iter()
                .filter_map(|item| match item {
                    ModelAssistantItem::Refusal { content } => Some(content.as_str()),
                    ModelAssistantItem::Text { .. }
                    | ModelAssistantItem::Reasoning(_)
                    | ModelAssistantItem::ToolCall(_) => None,
                })
                .collect();
            let mut value = Map::new();
            value.insert("role".to_owned(), Value::String("assistant".to_owned()));
            value.insert(
                "content".to_owned(),
                if content.is_empty() && (!tool_calls.is_empty() || !refusals.is_empty()) {
                    Value::Null
                } else {
                    Value::String(content)
                },
            );
            if !refusals.is_empty() {
                value.insert("refusal".to_owned(), Value::String(refusals));
            }
            if !tool_calls.is_empty() {
                value.insert(
                    "tool_calls".to_owned(),
                    Value::Array(tool_calls.into_iter().map(completions_tool_call).collect()),
                );
            }
            Value::Object(value)
        }
        ModelMessage::ToolResult {
            tool_call_id,
            content,
            ..
        } => json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }),
    }
}

fn sampling_body(model: &ResolvedModel, request: &ModelRequest) -> Map<String, Value> {
    let mut body = Map::new();
    body.extend(model.sampling_params.clone());
    body.extend(request.sampling_params.clone());
    body
}

fn role_name(role: &'static str, model: &ResolvedModel) -> &'static str {
    match role {
        "developer" if model.compatibility.supports_developer_role => "developer",
        "developer" => "system",
        _ => role,
    }
}

fn responses_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    })
}

fn responses_assistant_item(item: &ModelAssistantItem) -> Value {
    match item {
        ModelAssistantItem::Text { content } => json!({
            "role": "assistant",
            "content": content,
        }),
        ModelAssistantItem::Reasoning(thinking) => responses_reasoning_item(thinking),
        ModelAssistantItem::Refusal { content } => json!({
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": content}],
        }),
        ModelAssistantItem::ToolCall(tool_call) => responses_tool_call(tool_call),
    }
}

fn responses_reasoning_item(thinking: &ModelThinking) -> Value {
    let mut value = Map::new();
    value.insert("type".to_owned(), Value::String("reasoning".to_owned()));
    if let Some(item_id) = &thinking.item_id {
        value.insert("id".to_owned(), Value::String(item_id.clone()));
    }
    value.insert(
        "summary".to_owned(),
        if thinking.summary.is_empty() {
            json!([])
        } else {
            json!([{"type": "summary_text", "text": thinking.summary}])
        },
    );
    if !thinking.content.is_empty() {
        value.insert(
            "content".to_owned(),
            json!([{"type": "reasoning_text", "text": thinking.content}]),
        );
    }
    if let Some(encrypted_content) = &thinking.encrypted_content {
        value.insert(
            "encrypted_content".to_owned(),
            Value::String(encrypted_content.clone()),
        );
    }
    Value::Object(value)
}

fn responses_tool_call(tool_call: &ModelToolCall) -> Value {
    json!({
        "type": "function_call",
        "id": tool_call.item_id,
        "call_id": tool_call.call_id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    })
}

fn completions_tool_call(tool_call: &ModelToolCall) -> Value {
    json!({
        "id": tool_call.call_id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        }
    })
}

fn completions_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

#[derive(Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseParser {
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>, ProviderError> {
        self.buffer.extend_from_slice(bytes);
        let mut payloads = Vec::new();

        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=newline).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(payload) = self.process_line(&line)? {
                payloads.push(payload);
            }
        }

        Ok(payloads)
    }

    pub fn finish(&mut self) -> Result<Vec<String>, ProviderError> {
        let mut payloads = Vec::new();
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            if let Some(payload) = self.process_line(&line)? {
                payloads.push(payload);
            }
        }
        if !self.data_lines.is_empty() {
            payloads.push(self.data_lines.join("\n"));
            self.data_lines.clear();
        }
        Ok(payloads)
    }

    fn process_line(&mut self, line: &[u8]) -> Result<Option<String>, ProviderError> {
        let line = std::str::from_utf8(line).map_err(|error| ProviderError::Malformed {
            message: error.to_string(),
        })?;
        if line.is_empty() {
            if self.data_lines.is_empty() {
                return Ok(None);
            }
            return Ok(Some(std::mem::take(&mut self.data_lines).join("\n")));
        }
        if let Some(data) = line.strip_prefix("data:") {
            self.data_lines
                .push(data.strip_prefix(' ').unwrap_or(data).to_owned());
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentCommand, AgentEvent, AgentRuntime};
    use crate::config::{Compatibility, CostMetadata, ModelCatalog, ModelRef, ThinkingLevel};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn sse_parser_handles_fragmented_events_and_multiline_data() {
        let mut parser = SseParser::default();
        assert!(parser.feed(b"data: {\"delta\":\"").unwrap().is_empty());
        assert!(parser.feed("hello".as_bytes()).unwrap().is_empty());
        assert_eq!(
            parser.feed(b"\"}\n\ndata: second\ndata: line\n\n").unwrap(),
            [r#"{"delta":"hello"}"#, "second\nline"]
        );
    }

    #[test]
    fn malformed_sse_utf8_is_recoverable_error() {
        let mut parser = SseParser::default();
        let error = parser
            .feed(b"data: \xff\n\n")
            .expect_err("invalid UTF-8 should fail");
        assert!(matches!(error, ProviderError::Malformed { .. }));
    }

    #[test]
    fn completions_payload_assembles_text_tools_and_usage_events() {
        let first = parse_payload(
            ApiKind::OpenAiCompletions,
            r#"{"choices":[{"delta":{"content":"hello","tool_calls":[{"index":0,"id":"call-1","function":{"name":"read","arguments":"{\"path\":"}}]},"finish_reason":null}]}"#,
        )
        .expect("payload should parse");
        let second = parse_payload(
            ApiKind::OpenAiCompletions,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"src/main.rs\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}"#,
        )
        .expect("payload should parse");
        let mut collector = ResponseCollector::default();
        for event in first.events.into_iter().chain(second.events) {
            collector.record(&event);
        }
        collector.stop_reason = second.stop_reason;
        collector.terminal_seen = true;
        let response = collector
            .finish()
            .expect("completed response should finish");

        assert_eq!(response.stop_reason, StopReason::ToolCalls);
        assert_eq!(
            response.items[0],
            ModelAssistantItem::Text {
                content: "hello".to_owned()
            }
        );
        let ModelAssistantItem::ToolCall(tool_call) = &response.items[1] else {
            panic!("expected a tool call item");
        };
        assert_eq!(tool_call.name.as_deref(), Some("read"));
        assert_eq!(tool_call.arguments, "{\"path\":\"src/main.rs\"}");
        assert_eq!(
            response.usage,
            Some(Usage {
                input_tokens: Some(4),
                output_tokens: Some(3),
                total_tokens: Some(7),
                cache_read_tokens: None,
                cache_write_tokens: None,
            })
        );
    }

    #[test]
    fn completions_refusal_deltas_are_retained_and_replayed() {
        let parsed = parse_payload(
            ApiKind::OpenAiCompletions,
            r#"{"choices":[{"delta":{"refusal":"I cannot help with that."},"finish_reason":null}]}"#,
        )
        .expect("payload should parse");
        let finished = parse_payload(
            ApiKind::OpenAiCompletions,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        )
        .expect("finish payload should parse");
        let mut collector = ResponseCollector::default();
        for event in parsed.events.into_iter().chain(finished.events) {
            collector.record(&event);
        }
        collector.stop_reason = finished.stop_reason;
        collector.terminal_seen = true;
        let response = collector
            .finish()
            .expect("completed response should finish");

        assert_eq!(
            response.items,
            [ModelAssistantItem::Refusal {
                content: "I cannot help with that.".to_owned()
            }]
        );

        let model = test_model(
            ApiKind::OpenAiCompletions,
            "https://example.test/v1".to_owned(),
        );
        let request = ModelRequest {
            messages: vec![ModelMessage::Assistant {
                items: response.items,
            }],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, body) = request_for(&model, &request).expect("request should build");
        assert_eq!(
            body["messages"][0],
            json!({
                "role": "assistant",
                "content": null,
                "refusal": "I cannot help with that."
            })
        );
    }

    #[test]
    fn completions_reasoning_deltas_are_retained_in_response_items() {
        let parsed = parse_payload(
            ApiKind::OpenAiCompletions,
            r#"{"choices":[{"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}"#,
        )
        .expect("payload should parse");
        let mut collector = ResponseCollector::default();
        for event in parsed.events {
            collector.record(&event);
        }
        collector.terminal_seen = true;
        let response = collector
            .finish()
            .expect("completed response should finish");

        assert_eq!(
            response.items,
            [ModelAssistantItem::Reasoning(ModelThinking {
                summary: "thinking...".to_owned(),
                ..ModelThinking::default()
            })]
        );
    }

    #[test]
    fn detects_context_overflow_errors() {
        assert!(is_context_overflow(413, "payload too large"));
        assert!(is_context_overflow(400, "maximum context length is 1000"));
        assert!(!is_context_overflow(400, "invalid parameter"));
    }

    #[test]
    fn responses_incomplete_preserves_reason_and_does_not_infer_tool_calls() {
        let added = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_123","call_id":"call_123","name":"read","arguments":""}}"#,
        )
        .expect("output item should parse");
        let incomplete = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#,
        )
        .expect("incomplete response should parse");

        let mut collector = ResponseCollector::default();
        for event in added.events {
            collector.record(&event);
        }
        collector.terminal_seen = incomplete.terminal;
        collector.stop_reason = incomplete.stop_reason;
        let response = collector
            .finish()
            .expect("incomplete response should finish");

        assert_eq!(response.stop_reason, StopReason::Length);
        assert!(matches!(
            response.items.as_slice(),
            [ModelAssistantItem::ToolCall(_)]
        ));
    }

    #[test]
    fn responses_refusal_stream_is_preserved_and_replayed() {
        let payloads = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"response.refusal.delta","output_index":0,"delta":"I cannot help with that."}"#,
            r#"{"type":"response.refusal.done","output_index":0,"refusal":"I cannot help with that."}"#,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I cannot help with that."}]}}"#,
            r#"{"type":"response.completed","response":{"status":"completed"}}"#,
        ];
        let mut collector = ResponseCollector::default();
        for payload in payloads {
            let parsed = parse_payload(ApiKind::OpenAiResponses, payload)
                .expect("Responses refusal event should parse");
            collector.terminal_seen |= parsed.terminal;
            collector.stop_reason = parsed.stop_reason.or(collector.stop_reason);
            for event in parsed.events {
                collector.record(&event);
            }
        }
        let response = collector.finish().expect("refusal response should finish");
        assert_eq!(
            response.items,
            [ModelAssistantItem::Refusal {
                content: "I cannot help with that.".to_owned()
            }]
        );

        let model = test_model(
            ApiKind::OpenAiResponses,
            "https://example.test/v1".to_owned(),
        );
        let request = ModelRequest {
            messages: vec![ModelMessage::Assistant {
                items: response.items,
            }],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, body) = request_for(&model, &request).expect("request should build");
        assert_eq!(
            body["input"][0],
            json!({
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I cannot help with that."}]
            })
        );
    }

    #[test]
    fn responses_failed_and_top_level_error_events_surface_provider_errors() {
        let failed = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"response.failed","response":{"error":{"message":"model overloaded"}}}"#,
        )
        .expect_err("failed response should be an error");
        assert!(
            matches!(failed, ProviderError::Failed { message } if message == "model overloaded")
        );

        let top_level = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"error","message":"stream interrupted"}"#,
        )
        .expect_err("top-level error event should be an error");
        assert!(
            matches!(top_level, ProviderError::Failed { message } if message == "stream interrupted")
        );
    }

    #[test]
    fn responses_payload_emits_text_reasoning_and_tool_events() {
        let text = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"response.output_text.delta","delta":"hello"}"#,
        )
        .expect("text event should parse");
        assert_eq!(
            text.events,
            [ModelEvent::AssistantTextDelta {
                index: None,
                text: "hello".to_owned()
            }]
        );

        let reasoning = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"thinking"}"#,
        )
        .expect("reasoning event should parse");
        assert_eq!(
            reasoning.events,
            [ModelEvent::AssistantThinkingDelta {
                item_id: None,
                text: "thinking".to_owned()
            }]
        );

        let tool = parse_payload(
            ApiKind::OpenAiResponses,
            r#"{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc-2","delta":"{}"}"#,
        )
        .expect("tool event should parse");
        assert_eq!(
            tool.events,
            [ModelEvent::ToolCallDelta {
                index: 1,
                call_id: None,
                item_id: Some("fc-2".to_owned()),
                name: None,
                arguments: "{}".to_owned(),
                arguments_complete: false,
            }]
        );
    }

    #[test]
    fn responses_reasoning_item_includes_empty_summary_and_opaque_state() {
        let value = responses_reasoning_item(&ModelThinking {
            item_id: Some("rs_test".to_owned()),
            summary: String::new(),
            content: String::new(),
            encrypted_content: Some("encrypted".to_owned()),
        });

        assert!(value.get("summary").is_some());
        assert_eq!(value["summary"], json!([]));
        assert_eq!(
            value,
            json!({
                "type": "reasoning",
                "id": "rs_test",
                "summary": [],
                "encrypted_content": "encrypted"
            })
        );
    }

    #[test]
    fn responses_reasoning_item_preserves_non_empty_summary() {
        let value = responses_reasoning_item(&ModelThinking {
            summary: "Inspecting repository state".to_owned(),
            ..ModelThinking::default()
        });

        assert_eq!(
            value["summary"],
            json!([{
                "type": "summary_text",
                "text": "Inspecting repository state"
            }])
        );
    }

    #[test]
    fn responses_reasoning_item_preserves_content_with_empty_summary() {
        let value = responses_reasoning_item(&ModelThinking {
            content: "private reasoning".to_owned(),
            ..ModelThinking::default()
        });

        assert_eq!(
            value,
            json!({
                "type": "reasoning",
                "summary": [],
                "content": [{
                    "type": "reasoning_text",
                    "text": "private reasoning"
                }]
            })
        );
    }

    #[test]
    fn responses_reasoning_item_default_still_includes_summary() {
        assert_eq!(
            responses_reasoning_item(&ModelThinking::default()),
            json!({
                "type": "reasoning",
                "summary": []
            })
        );
    }

    #[test]
    fn responses_reasoning_replays_before_function_call_and_output() {
        let payloads = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_123","summary":[],"content":[],"encrypted_content":"enc_123"}}"#,
            r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_123","delta":"inspect the file"}"#,
            r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_123","call_id":"call_123","name":"read","arguments":"{\"path\":\"src/main.rs\"}"}}"#,
            r#"{"type":"response.completed","response":{"status":"completed"}}"#,
        ];
        let mut collector = ResponseCollector::default();
        for payload in payloads {
            let parsed = parse_payload(ApiKind::OpenAiResponses, payload)
                .expect("Responses event should parse");
            collector.terminal_seen |= parsed.terminal;
            if parsed.stop_reason.is_some() {
                collector.stop_reason = parsed.stop_reason;
            }
            for event in parsed.events {
                collector.record(&event);
            }
        }

        let response = collector
            .finish()
            .expect("completed response should finish");
        assert!(matches!(
            response.items.first(),
            Some(ModelAssistantItem::Reasoning(ModelThinking {
                item_id: Some(item_id),
                summary,
                encrypted_content: Some(encrypted_content),
                ..
            })) if item_id == "rs_123"
                && summary == "inspect the file"
                && encrypted_content == "enc_123"
        ));

        let model = test_model(
            ApiKind::OpenAiResponses,
            "https://example.test/v1".to_owned(),
        );
        let request = ModelRequest {
            messages: vec![
                ModelMessage::user("inspect src/main.rs"),
                ModelMessage::Assistant {
                    items: response.items,
                },
                ModelMessage::ToolResult {
                    tool_call_id: "call_123".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "file contents".to_owned(),
                },
            ],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, body) = request_for(&model, &request).expect("request should build");

        assert_eq!(
            body["input"][1],
            json!({
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{"type": "summary_text", "text": "inspect the file"}],
                "encrypted_content": "enc_123"
            })
        );
        assert_eq!(body["input"][2]["type"], "function_call");
        assert_eq!(body["input"][2]["id"], "fc_123");
        assert_eq!(body["input"][2]["call_id"], "call_123");
        assert_eq!(
            body["input"][3],
            json!({
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "file contents"
            })
        );
    }

    #[test]
    fn responses_replay_preserves_multiple_interleaved_reasoning_and_tool_items() {
        let payloads = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"reasoning A"}]}}"#,
            r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"a\"}"}}"#,
            r#"{"type":"response.output_item.added","output_index":2,"item":{"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"reasoning B"}]}}"#,
            r#"{"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"read","arguments":"{\"path\":\"b\"}"}}"#,
            r#"{"type":"response.completed","response":{"status":"completed"}}"#,
        ];
        let mut collector = ResponseCollector::default();
        for payload in payloads {
            let parsed = parse_payload(ApiKind::OpenAiResponses, payload)
                .expect("Responses event should parse");
            collector.terminal_seen |= parsed.terminal;
            collector.stop_reason = parsed.stop_reason.or(collector.stop_reason);
            for event in parsed.events {
                collector.record(&event);
            }
        }
        let response = collector
            .finish()
            .expect("completed response should finish");

        assert!(matches!(
            response.items.as_slice(),
            [
                ModelAssistantItem::Reasoning(ModelThinking { item_id: Some(first), .. }),
                ModelAssistantItem::ToolCall(ModelToolCall { item_id: Some(first_item), call_id: Some(first_call), .. }),
                ModelAssistantItem::Reasoning(ModelThinking { item_id: Some(second), .. }),
                ModelAssistantItem::ToolCall(ModelToolCall { item_id: Some(second_item), call_id: Some(second_call), .. }),
            ] if first == "rs_1"
                && first_item == "fc_1"
                && first_call == "call_1"
                && second == "rs_2"
                && second_item == "fc_2"
                && second_call == "call_2"
        ));

        let model = test_model(
            ApiKind::OpenAiResponses,
            "https://example.test/v1".to_owned(),
        );
        let request = ModelRequest {
            messages: vec![
                ModelMessage::Assistant {
                    items: response.items,
                },
                ModelMessage::ToolResult {
                    tool_call_id: "call_1".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "result A".to_owned(),
                },
                ModelMessage::ToolResult {
                    tool_call_id: "call_2".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "result B".to_owned(),
                },
            ],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, body) = request_for(&model, &request).expect("request should build");
        assert_eq!(body["input"][0]["id"], "rs_1");
        assert_eq!(body["input"][1]["id"], "fc_1");
        assert_eq!(body["input"][2]["id"], "rs_2");
        assert_eq!(body["input"][3]["id"], "fc_2");
        assert_eq!(
            body["input"][4],
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result A"
            })
        );
        assert_eq!(
            body["input"][5],
            json!({
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "result B"
            })
        );
    }

    #[test]
    fn responses_function_calls_preserve_ids_and_override_completed_status() {
        let payloads = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_123","call_id":"call_123","name":"read","arguments":""}}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_123","delta":"{\"path\":\""}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_123","delta":"src/main.rs\"}"}"#,
            r#"{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_123","arguments":"{\"path\":\"src/main.rs\"}"}"#,
            r#"{"type":"response.completed","response":{"status":"completed"}}"#,
        ];
        let mut collector = ResponseCollector::default();
        for payload in payloads {
            let parsed = parse_payload(ApiKind::OpenAiResponses, payload)
                .expect("Responses event should parse");
            if parsed.terminal {
                collector.terminal_seen = true;
            }
            if let Some(reason) = parsed.stop_reason {
                collector.stop_reason = Some(reason);
            }
            for event in parsed.events {
                collector.record(&event);
            }
        }

        let response = collector
            .finish()
            .expect("completed response should finish");
        assert_eq!(response.stop_reason, StopReason::ToolCalls);
        assert!(matches!(
            response.items.as_slice(),
            [ModelAssistantItem::ToolCall(ModelToolCall {
                call_id: Some(call_id),
                item_id: Some(item_id),
                arguments,
                ..
            })] if call_id == "call_123"
                && item_id == "fc_123"
                && arguments == "{\"path\":\"src/main.rs\"}"
        ));
    }

    #[test]
    fn sampling_params_are_included_in_generated_request() {
        let mut model = test_model(
            ApiKind::OpenAiResponses,
            "https://example.test/v1".to_owned(),
        );
        model
            .sampling_params
            .insert("temperature".to_owned(), json!(0.7));
        model.sampling_params.insert("top_p".to_owned(), json!(0.9));
        let (_, body) =
            request_for(&model, &ModelRequest::single_user("hello")).expect("request should build");

        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
    }

    #[test]
    fn replays_assistant_tool_call_and_result_for_completions() {
        let model = test_model(
            ApiKind::OpenAiCompletions,
            "https://example.test/v1".to_owned(),
        );
        let request = ModelRequest {
            messages: vec![
                ModelMessage::user("inspect src/main.rs"),
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::ToolCall(ModelToolCall {
                        index: 0,
                        call_id: Some("call_123".to_owned()),
                        item_id: None,
                        name: Some("read".to_owned()),
                        arguments: r#"{"path":"src/main.rs"}"#.to_owned(),
                    })],
                },
                ModelMessage::ToolResult {
                    tool_call_id: "call_123".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "file contents".to_owned(),
                },
            ],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, body) = request_for(&model, &request).expect("request should build");

        assert_eq!(
            body["messages"][0],
            json!({
                "role": "user",
                "content": "inspect src/main.rs"
            })
        );
        assert_eq!(body["messages"][1]["content"], Value::Null);
        assert_eq!(
            body["messages"][1]["tool_calls"][0],
            json!({
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": r#"{"path":"src/main.rs"}"#
                }
            })
        );
        assert_eq!(
            body["messages"][2],
            json!({
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "file contents"
            })
        );

        let plain_request = ModelRequest {
            messages: vec![ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "continuing".to_owned(),
                }],
            }],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, plain_body) = request_for(&model, &plain_request).expect("request should build");
        assert_eq!(
            plain_body["messages"][0],
            json!({
                "role": "assistant",
                "content": "continuing"
            })
        );
        assert!(plain_body["messages"][0].get("tool_calls").is_none());
    }

    #[test]
    fn replays_responses_function_call_and_output_with_distinct_ids() {
        let model = test_model(
            ApiKind::OpenAiResponses,
            "https://example.test/v1".to_owned(),
        );
        let request = ModelRequest {
            messages: vec![
                ModelMessage::user("inspect src/main.rs"),
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::ToolCall(ModelToolCall {
                        index: 0,
                        call_id: Some("call_123".to_owned()),
                        item_id: Some("fc_123".to_owned()),
                        name: Some("read".to_owned()),
                        arguments: r#"{"path":"src/main.rs"}"#.to_owned(),
                    })],
                },
                ModelMessage::ToolResult {
                    tool_call_id: "call_123".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "file contents".to_owned(),
                },
            ],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        let (_, body) = request_for(&model, &request).expect("request should build");

        assert_eq!(
            body["input"][0],
            json!({
                "role": "user",
                "content": "inspect src/main.rs"
            })
        );
        assert_eq!(
            body["input"][1],
            json!({
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_123",
                "name": "read",
                "arguments": r#"{"path":"src/main.rs"}"#
            })
        );
        assert_eq!(
            body["input"][2],
            json!({
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "file contents"
            })
        );
        assert_ne!(body["input"][1]["id"], body["input"][1]["call_id"]);
    }

    #[test]
    fn requests_use_selected_api_endpoint_and_compatibility() {
        let model = ResolvedModel {
            model_ref: ModelRef::new("p", "m"),
            name: "m".to_owned(),
            base_url: "https://example.test/v1".to_owned(),
            api: ApiKind::OpenAiCompletions,
            api_key: None,
            headers: BTreeMap::new(),
            auth_header: true,
            compatibility: Compatibility {
                supports_developer_role: false,
                ..Compatibility::default()
            },
            reasoning: false,
            thinking_level_map: BTreeMap::new(),
            input: vec!["text".to_owned()],
            context_window: None,
            max_tokens: Some(123),
            cost: CostMetadata::default(),
            sampling_params: BTreeMap::new(),
        };
        let (endpoint, body) = request_for(
            &model,
            &ModelRequest {
                messages: vec![ModelMessage::Developer {
                    content: "instructions".to_owned(),
                }],
                tools: Vec::new(),
                max_tokens: None,
                reasoning_effort: None,
                sampling_params: BTreeMap::new(),
            },
        )
        .expect("request should build");
        assert_eq!(endpoint, "https://example.test/v1/chat/completions");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["max_tokens"], 123);
    }

    #[test]
    fn provider_limits_follow_the_currently_selected_model() {
        let mut first = test_model(ApiKind::OpenAiResponses, "https://example.test".to_owned());
        first.context_window = Some(200_000);
        first.max_tokens = Some(32_000);
        let mut second = first.clone();
        second.context_window = Some(32_000);
        second.max_tokens = None;
        let provider = OpenAiProvider::with_client(first, Client::new());

        assert_eq!(
            ModelProvider::limits(&provider),
            ModelLimits {
                context_window: Some(200_000),
                max_output_tokens: Some(32_000),
            }
        );
        provider.set_model(second);
        assert_eq!(
            ModelProvider::limits(&provider),
            ModelLimits {
                context_window: Some(32_000),
                max_output_tokens: None,
            }
        );
    }

    #[test]
    fn reasoning_effort_uses_api_shape_native_mapping_and_omits_off() {
        let catalog = ModelCatalog::from_json(
            "models.json",
            r#"{
                "providers": {
                    "p": {
                        "baseUrl": "https://example.test/v1",
                        "api": "openai-responses",
                        "models": [{
                            "id": "m",
                            "reasoning": true,
                            "thinkingLevelMap": {"max": "max"}
                        }]
                    }
                }
            }"#,
        )
        .unwrap();
        let responses_model = catalog.resolve(None, None).unwrap();
        let mut request = ModelRequest::single_user("hello");
        request.reasoning_effort = responses_model.thinking_effort(ThinkingLevel::Max);

        let (_, responses) = request_for(&responses_model, &request).unwrap();
        assert_eq!(responses["reasoning"], json!({"effort": "max"}));
        assert!(responses.get("reasoning_effort").is_none());

        let mut completions_model = responses_model.clone();
        completions_model.api = ApiKind::OpenAiCompletions;
        let (_, completions) = request_for(&completions_model, &request).unwrap();
        assert_eq!(completions["reasoning_effort"], "max");
        assert!(completions.get("reasoning").is_none());

        request.reasoning_effort = responses_model.thinking_effort(ThinkingLevel::Off);
        let (_, responses_off) = request_for(&responses_model, &request).unwrap();
        let (_, completions_off) = request_for(&completions_model, &request).unwrap();
        assert!(responses_off.get("reasoning").is_none());
        assert!(completions_off.get("reasoning_effort").is_none());
    }

    fn test_model(api: ApiKind, base_url: String) -> ResolvedModel {
        ResolvedModel {
            model_ref: ModelRef::new("p", "m"),
            name: "m".to_owned(),
            base_url,
            api,
            api_key: Some("test-key".to_owned()),
            headers: BTreeMap::new(),
            auth_header: true,
            compatibility: Compatibility::default(),
            reasoning: false,
            thinking_level_map: BTreeMap::new(),
            input: vec!["text".to_owned()],
            context_window: None,
            max_tokens: None,
            cost: CostMetadata::default(),
            sampling_params: BTreeMap::new(),
        }
    }

    async fn read_json_request(socket: &mut TcpStream) -> Value {
        let mut request = Vec::new();
        let header_end = loop {
            if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
            let mut chunk = [0_u8; 4096];
            let read = socket.read(&mut chunk).await.expect("request should read");
            assert!(read > 0, "request ended before headers completed");
            request.extend_from_slice(&chunk[..read]);
        };
        let headers =
            std::str::from_utf8(&request[..header_end]).expect("request headers should be UTF-8");
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("valid content length"))
            })
            .expect("request should include content length");
        while request.len() < header_end + content_length {
            let mut chunk = [0_u8; 4096];
            let read = socket
                .read(&mut chunk)
                .await
                .expect("request body should read");
            assert!(read > 0, "request ended before body completed");
            request.extend_from_slice(&chunk[..read]);
        }
        serde_json::from_slice(&request[header_end..header_end + content_length])
            .expect("request body should be JSON")
    }

    async fn write_http_response(
        socket: &mut TcpStream,
        status: &str,
        content_type: &str,
        body: &str,
    ) {
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("response should write");
    }

    async fn spawn_responses_tool_round_server() -> (
        String,
        Arc<std::sync::Mutex<Vec<Value>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");
        let address = listener
            .local_addr()
            .expect("test server should have address");
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            let first_body = concat!(
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_tool\",\"summary\":[],\"content\":[],\"encrypted_content\":\"enc_tool\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_tool\",\"call_id\":\"call_tool\",\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"printf tool-ok\\\"}\"}}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
            );
            let (mut first_socket, _) = listener
                .accept()
                .await
                .expect("first request should arrive");
            let first_request = read_json_request(&mut first_socket).await;
            captured.lock().unwrap().push(first_request);
            write_http_response(&mut first_socket, "200 OK", "text/event-stream", first_body).await;

            let (mut second_socket, _) = listener
                .accept()
                .await
                .expect("second request should arrive");
            let second_request = read_json_request(&mut second_socket).await;
            let has_empty_summary = second_request["input"]
                .as_array()
                .and_then(|items| {
                    items
                        .iter()
                        .find(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
                })
                .and_then(|item| item.get("summary"))
                .is_some_and(|summary| summary == &json!([]));
            captured.lock().unwrap().push(second_request);

            if has_empty_summary {
                let second_body = concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"tool round complete\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                );
                write_http_response(
                    &mut second_socket,
                    "200 OK",
                    "text/event-stream",
                    second_body,
                )
                .await;
            } else {
                write_http_response(
                    &mut second_socket,
                    "400 Bad Request",
                    "application/json",
                    r#"{"error":{"message":"Missing required parameter: 'input[2].summary'.","type":"invalid_request_error"}}"#,
                )
                .await;
            }
        });
        (format!("http://{address}"), requests, task)
    }

    async fn spawn_sse_server(status: &str, body: &str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");
        let address = listener
            .local_addr()
            .expect("test server should have address");
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("request should arrive");
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(response.as_bytes())
                .await
                .expect("response should write");
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn provider_streams_completions_from_mock_http_server() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, server) = spawn_sse_server("200 OK", body).await;
        let provider = OpenAiProvider::new(test_model(ApiKind::OpenAiCompletions, base_url))
            .expect("provider should build");
        let (tx, mut rx) = mpsc::channel(16);
        let response = provider
            .stream(
                ModelRequest::single_user("hello"),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect("provider should stream");

        let mut text = String::new();
        let mut usage = None;
        while let Some(event) = rx.recv().await {
            match event {
                ModelEvent::AssistantTextDelta { text: chunk, .. } => text.push_str(&chunk),
                ModelEvent::UsageUpdated(value) => usage = Some(value),
                _ => {}
            }
        }
        assert_eq!(text, "hello");
        assert_eq!(response.stop_reason, StopReason::Stop);
        assert_eq!(usage.and_then(|value| value.total_tokens), Some(3));
        server.await.expect("test server should finish");
    }

    #[tokio::test]
    async fn responses_agent_replays_empty_reasoning_summary_after_tool_call() {
        let root = std::env::temp_dir().join(format!(
            "ri-responses-reasoning-replay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let (base_url, requests, server) = spawn_responses_tool_round_server().await;
        let provider = OpenAiProvider::new(test_model(ApiKind::OpenAiResponses, base_url))
            .expect("provider should build");
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "run the tool".to_owned(),
            })
            .await
            .unwrap();
        let mut events = Vec::new();
        loop {
            let event = event_rx.recv().await.expect("turn event should arrive");
            let finished = matches!(event, AgentEvent::TurnFinished { .. });
            events.push(event);
            if finished {
                break;
            }
        }
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        server.await.expect("test server should finish");
        std::fs::remove_dir_all(root).unwrap();

        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::ToolExecutionFinished { result, .. }
                    if result.metadata.success && result.model_content.contains("tool-ok")
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::AssistantTextDelta { text, .. } if text == "tool round complete"
            )
        }));
        assert!(events
            .iter()
            .all(|event| !matches!(event, AgentEvent::Error(_))));
        assert_eq!(
            events.last(),
            Some(&AgentEvent::TurnFinished {
                reason: StopReason::Stop
            })
        );

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let input = requests[1]["input"]
            .as_array()
            .expect("second request should contain input items");
        let reasoning_index = input
            .iter()
            .position(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
            .expect("reasoning item should be replayed");
        let function_call_index = input
            .iter()
            .position(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function call should be replayed");
        let tool_result_index = input
            .iter()
            .position(|item| {
                item.get("type").and_then(Value::as_str) == Some("function_call_output")
            })
            .expect("tool result should be included");
        assert_eq!(
            input[reasoning_index],
            json!({
                "type": "reasoning",
                "id": "rs_tool",
                "summary": [],
                "encrypted_content": "enc_tool"
            })
        );
        assert!(reasoning_index < function_call_index);
        assert!(function_call_index < tool_result_index);
        assert_eq!(input[function_call_index]["call_id"], "call_tool");
        assert_eq!(input[tool_result_index]["call_id"], "call_tool");
    }

    #[tokio::test]
    async fn provider_surfaces_http_errors() {
        let (base_url, server) = spawn_sse_server(
            "401 Unauthorized",
            r#"{"error":{"message":"invalid API key"}}"#,
        )
        .await;
        let provider = OpenAiProvider::new(test_model(ApiKind::OpenAiResponses, base_url))
            .expect("provider should build");
        let (tx, _rx) = mpsc::channel(4);
        let error = provider
            .stream(
                ModelRequest::single_user("hello"),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect_err("HTTP error should surface");
        assert!(matches!(error, ProviderError::Http { status: 401, .. }));
        assert!(error.to_string().contains("invalid API key"));
        server.await.expect("test server should finish");
    }

    #[test]
    fn provider_error_diagnostics_redact_credentials_and_prompt_fields() {
        let mut model = test_model(ApiKind::OpenAiResponses, "https://example.test".to_owned());
        model
            .headers
            .insert("X-Custom-Secret".to_owned(), "header-secret".to_owned());
        let message = r#"{"error":{"message":"bad request for test-key and header-secret","api_key":"test-key"},"prompt":"private prompt"}"#;
        let error_body = ProviderErrorBody {
            content: message.to_owned(),
            truncated: false,
        };
        let request = ModelRequest::single_user("unrelated request");

        let diagnostic = provider_error_diagnostic(&error_body, &model, &request);

        assert!(diagnostic.contains("bad request"));
        assert!(!diagnostic.contains("test-key"));
        assert!(!diagnostic.contains("header-secret"));
        assert!(!diagnostic.contains("private prompt"));
        assert!(diagnostic.contains("<redacted>"));
    }

    #[test]
    fn provider_error_diagnostics_are_bounded_with_an_explicit_marker() {
        let model = test_model(ApiKind::OpenAiResponses, "https://example.test".to_owned());
        let message = format!(r#"{{"error":{{"message":"{}"}}}}"#, "x".repeat(8_000));
        let error_body = ProviderErrorBody {
            content: message.clone(),
            truncated: false,
        };
        let request = ModelRequest::single_user("unrelated request");

        let diagnostic = provider_error_diagnostic(&error_body, &model, &request);

        assert!(diagnostic.len() <= MAX_ERROR_LOG_BYTES);
        assert!(diagnostic.contains("diagnostic truncated"));
        assert!(diagnostic.contains(&format!("response {} bytes", message.len())));
    }

    #[test]
    fn provider_error_diagnostics_redact_request_content_in_json_and_plain_text() {
        let model = test_model(ApiKind::OpenAiResponses, "https://example.test".to_owned());
        let user_prompt = "full private user prompt";
        let tool_output = "full private tool output";
        let request = ModelRequest {
            messages: vec![
                ModelMessage::user(user_prompt),
                ModelMessage::ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    tool_name: "read".to_owned(),
                    content: tool_output.to_owned(),
                },
            ],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling_params: BTreeMap::new(),
        };
        for content in [
            format!(
                r#"{{"error":{{"message":"invalid prompt: {user_prompt}; output: {tool_output}"}}}}"#
            ),
            format!("invalid prompt: {user_prompt}; output: {tool_output}"),
        ] {
            let error_body = ProviderErrorBody {
                content,
                truncated: false,
            };
            let diagnostic = provider_error_diagnostic(&error_body, &model, &request);
            assert!(!diagnostic.contains(user_prompt));
            assert!(!diagnostic.contains(tool_output));
            assert!(diagnostic.contains(REDACTED_DIAGNOSTIC_CONTENT));
        }
    }

    #[test]
    fn truncated_provider_error_keeps_metadata_outside_sanitized_json() {
        let model = test_model(ApiKind::OpenAiResponses, "https://example.test".to_owned());
        let error_body = ProviderErrorBody {
            content: r#"{"error":{"message":"bad request"},"prompt":"private prompt"}"#.to_owned(),
            truncated: true,
        };
        let request = ModelRequest::single_user("unrelated request");

        let diagnostic = provider_error_diagnostic(&error_body, &model, &request);

        assert!(diagnostic.contains("bad request"));
        assert!(!diagnostic.contains("private prompt"));
        assert!(error_body
            .user_message()
            .ends_with(ERROR_RESPONSE_TRUNCATED_MARKER));
    }

    #[tokio::test]
    async fn provider_error_response_body_is_bounded_with_an_explicit_marker() {
        let body = "x".repeat(MAX_ERROR_RESPONSE_BYTES + 1_024);
        let (base_url, server) = spawn_sse_server("400 Bad Request", &body).await;
        let provider = OpenAiProvider::new(test_model(ApiKind::OpenAiResponses, base_url))
            .expect("provider should build");
        let (tx, _rx) = mpsc::channel(4);

        let error = provider
            .stream(
                ModelRequest::single_user("hello"),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect_err("HTTP error should surface");

        let ProviderError::Http { message, .. } = error else {
            panic!("expected HTTP error");
        };
        assert!(message.len() <= MAX_ERROR_RESPONSE_BYTES + ERROR_RESPONSE_TRUNCATED_MARKER.len());
        assert!(message.ends_with(ERROR_RESPONSE_TRUNCATED_MARKER));
        server.await.expect("test server should finish");
    }

    #[tokio::test]
    async fn provider_honors_pre_cancelled_request() {
        let (base_url, server) = spawn_sse_server("200 OK", "data: [DONE]\n\n").await;
        let provider = OpenAiProvider::new(test_model(ApiKind::OpenAiResponses, base_url))
            .expect("provider should build");
        let (tx, _rx) = mpsc::channel(4);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert_eq!(
            provider
                .stream(ModelRequest::single_user("hello"), tx, cancel)
                .await,
            Err(ProviderError::Cancelled)
        );
        server.abort();
    }
}
