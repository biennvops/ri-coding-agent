use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::{ModelRef, ResolvedModel};

pub mod openai;

pub use openai::{OpenAiProvider, SseParser};

#[derive(Clone)]
pub enum ConfiguredProvider {
    Mock(MockProvider),
    OpenAi(OpenAiProvider),
}

impl ConfiguredProvider {
    pub fn mock() -> Self {
        Self::Mock(MockProvider::new())
    }

    pub fn openai(model: ResolvedModel) -> Result<Self, ProviderError> {
        Ok(Self::OpenAi(OpenAiProvider::new(model)?))
    }

    pub fn set_model(&self, model: ResolvedModel) -> Result<(), ProviderError> {
        match self {
            Self::Mock(_) => Err(ProviderError::Failed {
                message: "model switching requires a configured models.json provider".to_owned(),
            }),
            Self::OpenAi(provider) => {
                provider.set_model(model);
                Ok(())
            }
        }
    }

    pub fn model_ref(&self) -> ModelRef {
        match self {
            Self::Mock(_) => ModelRef::new("mock", "mock"),
            Self::OpenAi(provider) => provider.current_model().model_ref,
        }
    }
}

#[async_trait]
impl ModelProvider for ConfiguredProvider {
    async fn stream(
        &self,
        request: ModelRequest,
        events: mpsc::Sender<ModelEvent>,
        cancel: CancellationToken,
    ) -> Result<ModelResponse, ProviderError> {
        match self {
            Self::Mock(provider) => provider.stream(request, events, cancel).await,
            Self::OpenAi(provider) => provider.stream(request, events, cancel).await,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ModelMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelRequest {
    pub messages: Vec<ModelMessage>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u64>,
    pub reasoning_effort: Option<String>,
    pub sampling: BTreeMap<String, Value>,
}

impl ModelRequest {
    pub fn single_user(text: impl Into<String>) -> Self {
        Self {
            messages: vec![ModelMessage::user(text)],
            tools: Vec::new(),
            max_tokens: None,
            reasoning_effort: None,
            sampling: BTreeMap::new(),
        }
    }

    pub fn last_user_message(&self) -> &str {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == MessageRole::User)
            .map(|message| message.content.as_str())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelEvent {
    AssistantTextDelta {
        text: String,
    },
    AssistantThinkingDelta {
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        item_id: Option<String>,
        name: Option<String>,
        arguments: String,
        arguments_complete: bool,
    },
    UsageUpdated(Usage),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelToolCall {
    pub index: usize,
    /// The callable tool identifier (`call_...` for Responses API calls).
    pub id: Option<String>,
    /// The output item identifier (`fc_...` for Responses API calls).
    pub item_id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelResponse {
    pub stop_reason: StopReason,
    pub tool_calls: Vec<ModelToolCall>,
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProviderError {
    #[error("model stream cancelled")]
    Cancelled,

    #[error("provider request failed: {message}")]
    Failed { message: String },

    #[error("provider context window exceeded: {message}")]
    ContextOverflow { message: String },

    #[error("provider returned HTTP {status}: {message}")]
    Http { status: u16, message: String },

    #[error("provider returned malformed streaming data: {message}")]
    Malformed { message: String },
}

#[async_trait]
pub trait ModelProvider: Send + Sync + 'static {
    async fn stream(
        &self,
        request: ModelRequest,
        events: mpsc::Sender<ModelEvent>,
        cancel: CancellationToken,
    ) -> Result<ModelResponse, ProviderError>;
}

#[derive(Clone, Debug)]
pub struct MockProvider {
    response: Option<String>,
    chunk_size: usize,
    delay: Duration,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self {
            response: None,
            chunk_size: 8,
            delay: Duration::from_millis(20),
        }
    }
}

impl MockProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_response(response: impl Into<String>) -> Self {
        Self {
            response: Some(response.into()),
            ..Self::default()
        }
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn response_for(&self, request: &ModelRequest) -> String {
        self.response
            .clone()
            .unwrap_or_else(|| format!("Mock response to: {}", request.last_user_message()))
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    async fn stream(
        &self,
        request: ModelRequest,
        events: mpsc::Sender<ModelEvent>,
        cancel: CancellationToken,
    ) -> Result<ModelResponse, ProviderError> {
        let response = self.response_for(&request);
        let characters: Vec<char> = response.chars().collect();

        for chunk in characters.chunks(self.chunk_size) {
            if !self.delay.is_zero() {
                tokio::select! {
                    _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                    _ = tokio::time::sleep(self.delay) => {}
                }
            } else if cancel.is_cancelled() {
                return Err(ProviderError::Cancelled);
            }

            let text: String = chunk.iter().collect();
            tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                result = events.send(ModelEvent::AssistantTextDelta { text }) => {
                    result.map_err(|_| ProviderError::Failed {
                        message: "agent event stream closed".to_owned(),
                    })?;
                }
            }
        }

        if cancel.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }

        Ok(ModelResponse {
            stop_reason: StopReason::Stop,
            tool_calls: Vec::new(),
            usage: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Incomplete,
    Cancelled,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_provider_streams_configured_chunks() {
        let provider = MockProvider::with_response("abcdef")
            .with_chunk_size(2)
            .with_delay(Duration::ZERO);
        let (tx, mut rx) = mpsc::channel(8);

        let response = provider
            .stream(
                ModelRequest::single_user("hello"),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect("mock stream should succeed");

        let mut chunks = Vec::new();
        while let Some(event) = rx.recv().await {
            let ModelEvent::AssistantTextDelta { text } = event else {
                continue;
            };
            chunks.push(text);
        }

        assert_eq!(chunks, ["ab", "cd", "ef"]);
        assert_eq!(response.stop_reason, StopReason::Stop);
        assert!(response.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn mock_provider_honors_cancellation() {
        let provider = MockProvider::with_response("a long response")
            .with_chunk_size(1)
            .with_delay(Duration::from_millis(10));
        let (tx, mut rx) = mpsc::channel(8);
        let cancel = CancellationToken::new();
        let provider_cancel = cancel.clone();

        let task = tokio::spawn(async move {
            provider
                .stream(ModelRequest::single_user("hello"), tx, provider_cancel)
                .await
        });

        let _ = rx.recv().await;
        cancel.cancel();

        assert_eq!(
            task.await.expect("provider task should join"),
            Err(ProviderError::Cancelled)
        );
    }
}
