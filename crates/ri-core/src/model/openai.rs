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
    MessageRole, ModelEvent, ModelMessage, ModelProvider, ModelRequest, ModelResponse,
    ModelToolCall, ProviderError, StopReason, ToolDefinition, Usage,
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

        Ok(collector.finish())
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
    stop_reason: Option<StopReason>,
    tool_calls: BTreeMap<usize, ModelToolCall>,
    usage: Option<Usage>,
}

impl ResponseCollector {
    fn record(&mut self, event: &ModelEvent) {
        match event {
            ModelEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                let tool_call = self
                    .tool_calls
                    .entry(*index)
                    .or_insert_with(|| ModelToolCall {
                        index: *index,
                        ..ModelToolCall::default()
                    });
                if id.is_some() {
                    tool_call.id = id.clone();
                }
                if name.is_some() {
                    tool_call.name = name.clone();
                }
                tool_call.arguments.push_str(arguments);
            }
            ModelEvent::UsageUpdated(usage) => self.usage = Some(usage.clone()),
            ModelEvent::AssistantTextDelta { .. } | ModelEvent::AssistantThinkingDelta { .. } => {}
        }
    }

    fn finish(self) -> ModelResponse {
        let stop_reason = self.stop_reason.unwrap_or(if self.tool_calls.is_empty() {
            StopReason::Stop
        } else {
            StopReason::ToolCalls
        });
        ModelResponse {
            stop_reason,
            tool_calls: self.tool_calls.into_values().collect(),
            usage: self.usage,
        }
    }
}

#[derive(Default)]
struct ParsedPayload {
    events: Vec<ModelEvent>,
    stop_reason: Option<StopReason>,
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
                id: value
                    .get("call_id")
                    .or_else(|| value.get("item_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                name: value.get("name").and_then(Value::as_str).map(str::to_owned),
                arguments: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
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
                    id: item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    name: item.get("name").and_then(Value::as_str).map(str::to_owned),
                    arguments: item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            }
        }
        "response.completed" | "response.done" => {
            let response = value.get("response").unwrap_or(value);
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
                    id: tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    arguments: function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            }
        }
    }

    if let Some(finish_reason) = choice
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
    {
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
                .map(|message| {
                    json!({
                        "role": role_name(message, model),
                        "content": message.content,
                    })
                })
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
                .map(|message| {
                    let mut value = Map::new();
                    value.insert(
                        "role".to_owned(),
                        Value::String(role_name(message, model).to_owned()),
                    );
                    value.insert("content".to_owned(), Value::String(message.content.clone()));
                    if let Some(name) = &message.name {
                        value.insert("name".to_owned(), Value::String(name.clone()));
                    }
                    if let Some(tool_call_id) = &message.tool_call_id {
                        value.insert(
                            "tool_call_id".to_owned(),
                            Value::String(tool_call_id.clone()),
                        );
                    }
                    Value::Object(value)
                })
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

fn sampling_body(model: &ResolvedModel, request: &ModelRequest) -> Map<String, Value> {
    let mut body = Map::new();
    body.extend(model.sampling.clone());
    body.extend(request.sampling.clone());
    body
}

fn role_name(message: &ModelMessage, model: &ResolvedModel) -> &'static str {
    match message.role {
        MessageRole::System => "system",
        MessageRole::Developer if model.compatibility.supports_developer_role => "developer",
        MessageRole::Developer => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
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
        let response = collector.finish();

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
            r#"{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"call-2","delta":"{}"}"#,
        )
        .expect("tool event should parse");
        assert_eq!(
            tool.events,
            [ModelEvent::ToolCallDelta {
                index: 1,
                id: Some("call-2".to_owned()),
                name: None,
                arguments: "{}".to_owned(),
            }]
        );
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
            compatibility: Compatibility::default(),
            reasoning: false,
            input: vec!["text".to_owned()],
            context_window: None,
            max_tokens: Some(123),
            cost: CostMetadata::default(),
            sampling: BTreeMap::new(),
        };
        let (endpoint, body) = request_for(
            &model,
            &ModelRequest {
                messages: vec![ModelMessage {
                    role: MessageRole::Developer,
                    content: "instructions".to_owned(),
                    name: None,
                    tool_call_id: None,
                }],
                tools: Vec::new(),
                max_tokens: None,
                reasoning_effort: None,
                sampling: BTreeMap::new(),
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
            sampling: BTreeMap::new(),
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
