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
    ModelEvent, ModelMessage, ModelProvider, ModelRequest, ModelResponse, ModelToolCall,
    ProviderError, StopReason, ToolDefinition, Usage,
};

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
}

#[async_trait::async_trait]
impl ModelProvider for OpenAiProvider {
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

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = read_error_body(response, cancel.clone()).await?;
            if is_context_overflow(status, &message) {
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

        collector.finish()
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
) -> Result<String, ProviderError> {
    const MAX_ERROR_BYTES: usize = 64 * 1024;
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();

    while let Some(chunk) = tokio::select! {
        _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
        chunk = stream.next() => chunk,
    } {
        let chunk = chunk.map_err(|error| ProviderError::Failed {
            message: error.to_string(),
        })?;
        let remaining = MAX_ERROR_BYTES.saturating_sub(bytes.len());
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if bytes.len() == MAX_ERROR_BYTES {
            break;
        }
    }

    Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
}

#[derive(Default)]
struct ResponseCollector {
    content: String,
    thinking: String,
    stop_reason: Option<StopReason>,
    terminal_seen: bool,
    tool_calls: BTreeMap<usize, ModelToolCall>,
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
                let tool_call = self
                    .tool_calls
                    .entry(*index)
                    .or_insert_with(|| ModelToolCall {
                        index: *index,
                        ..ModelToolCall::default()
                    });
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
            ModelEvent::UsageUpdated(usage) => self.usage = Some(usage.clone()),
            ModelEvent::AssistantTextDelta { text } => self.content.push_str(text),
            ModelEvent::AssistantThinkingDelta { text } => self.thinking.push_str(text),
        }
    }

    fn finish(self) -> Result<ModelResponse, ProviderError> {
        if !self.terminal_seen {
            return Err(ProviderError::Failed {
                message: "response stream ended before a terminal event".to_owned(),
            });
        }

        let stop_reason = match self.stop_reason {
            Some(StopReason::Stop) if !self.tool_calls.is_empty() => StopReason::ToolCalls,
            Some(reason) => reason,
            None if !self.tool_calls.is_empty() => StopReason::ToolCalls,
            None => StopReason::Stop,
        };
        Ok(ModelResponse {
            content: self.content,
            thinking: (!self.thinking.is_empty()).then_some(self.thinking),
            stop_reason,
            tool_calls: self.tool_calls.into_values().collect(),
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
                    text: text.to_owned(),
                });
            }
        }
        "response.reasoning_summary_text.delta"
        | "response.reasoning_text.delta"
        | "response.reasoning.delta" => {
            if let Some(text) = value.get("delta").and_then(Value::as_str) {
                parsed.events.push(ModelEvent::AssistantThinkingDelta {
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
                text: text.to_owned(),
            });
        }
        for key in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(text) = delta.get(key).and_then(Value::as_str) {
                parsed.events.push(ModelEvent::AssistantThinkingDelta {
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
        ModelMessage::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let mut items = Vec::new();
            if !content.is_empty() || tool_calls.is_empty() {
                items.push(json!({
                    "role": "assistant",
                    "content": content,
                }));
            }
            items.extend(tool_calls.iter().map(responses_tool_call));
            items
        }
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
        ModelMessage::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let content = if content.is_empty() && !tool_calls.is_empty() {
                Value::Null
            } else {
                Value::String(content.clone())
            };
            json!({
                "role": "assistant",
                "content": content,
                "tool_calls": (!tool_calls.is_empty()).then(|| {
                    tool_calls.iter().map(completions_tool_call).collect::<Vec<_>>()
                }),
            })
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
    use crate::config::{Compatibility, CostMetadata, ModelRef};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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

        assert_eq!(response.content, "hello");
        assert_eq!(response.thinking, None);
        assert_eq!(response.stop_reason, StopReason::ToolCalls);
        assert_eq!(response.tool_calls[0].name.as_deref(), Some("read"));
        assert_eq!(
            response.tool_calls[0].arguments,
            "{\"path\":\"src/main.rs\"}"
        );
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
        assert_eq!(response.tool_calls.len(), 1);
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
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].call_id.as_deref(), Some("call_123"));
        assert_eq!(response.tool_calls[0].item_id.as_deref(), Some("fc_123"));
        assert_eq!(
            response.tool_calls[0].arguments,
            "{\"path\":\"src/main.rs\"}"
        );
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
                    content: String::new(),
                    thinking: None,
                    tool_calls: vec![ModelToolCall {
                        index: 0,
                        call_id: Some("call_123".to_owned()),
                        item_id: None,
                        name: Some("read".to_owned()),
                        arguments: r#"{"path":"src/main.rs"}"#.to_owned(),
                    }],
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
                    content: String::new(),
                    thinking: None,
                    tool_calls: vec![ModelToolCall {
                        index: 0,
                        call_id: Some("call_123".to_owned()),
                        item_id: Some("fc_123".to_owned()),
                        name: Some("read".to_owned()),
                        arguments: r#"{"path":"src/main.rs"}"#.to_owned(),
                    }],
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
            input: vec!["text".to_owned()],
            context_window: None,
            max_tokens: None,
            cost: CostMetadata::default(),
            sampling_params: BTreeMap::new(),
        }
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
                ModelEvent::AssistantTextDelta { text: chunk } => text.push_str(&chunk),
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
