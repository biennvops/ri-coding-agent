pub mod agent;
pub mod app;
pub mod config;
pub mod context;
pub mod conversation;
mod fs;
pub mod model;
pub mod session;
pub mod tools;

pub use agent::{AgentCommand, AgentError, AgentEvent, AgentRuntime, AgentRuntimeConfig};
pub use app::{
    AppState, MessageRole, StreamingAssistantState, ToolOutputChunk, ToolStatus,
    ToolTranscriptEntry, TranscriptEntries, TranscriptEntry, TranscriptEntryId,
    TranscriptEntryState, TranscriptMessage, TranscriptMessages, UserMessageStatus,
};
pub use config::{
    default_state_path, load_state, persist_recent_model, ApiKind, CompactionSettings,
    Compatibility, ConfigError, ConfigWarning, ContextSettings, CostMetadata, ModelCatalog,
    ModelRef, RecentModel, RecentModelState, ResolvedModel, ResolvedSettings, Settings,
    SettingsError, SettingsLoad, StateError, ThinkingLevel, ThinkingLevelError,
    WorkspaceRecentModel,
};
pub use context::{
    automatic_trigger, compaction_target, input_budget, ConservativeTokenEstimator, ContextUsage,
    GenericTokenEstimator, TokenEstimator, UsageSource, AUTO_COMPACTION_TARGET_PERCENT,
    AUTO_COMPACTION_TRIGGER_PERCENT, COMPACTION_MAX_OUTPUT_TOKENS, DEFAULT_RESERVED_OUTPUT_TOKENS,
};
pub use conversation::{segment_history, CompactionSummary, ConversationHistory, HistorySegment};
pub use model::{
    ConfiguredProvider, MockProvider, ModelAssistantItem, ModelEvent, ModelLimits, ModelMessage,
    ModelProvider, ModelRequest, ModelResponse, ModelThinking, ModelToolCall, ProviderError,
    StopReason, ToolDefinition, Usage,
};
pub use session::{
    read_session, validate_name, workspace_id, MessageId, OpenedSession, SessionAssistantItem,
    SessionError, SessionHandle, SessionHeader, SessionId, SessionInfo, SessionMessage,
    SessionMode, SessionRecord, SessionRepository, SessionSnapshot, SessionSummary, WorkspaceId,
    MAX_SESSION_RECORD_BYTES, SESSION_VERSION,
};
pub use tools::{
    Tool, ToolCallPresentation, ToolContext, ToolError, ToolEvent, ToolEventSender,
    ToolExecutionMetadata, ToolExecutionResult, ToolOutputStream, ToolPreviewKind, ToolPreviewLine,
    ToolRegistry, DEFAULT_BASH_TIMEOUT_MS, MAX_TOOL_OUTPUT_BYTES, MAX_TOOL_PREVIEW_BYTES,
    MAX_TOOL_PREVIEW_LINES,
};
