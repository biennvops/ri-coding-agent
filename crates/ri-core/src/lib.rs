pub mod agent;
pub mod app;
pub mod config;
pub mod context;
pub mod conversation;
mod fs;
pub mod model;
pub mod plugin;
pub mod session;
pub mod tools;

pub use agent::{AgentCommand, AgentError, AgentEvent, AgentRuntime, AgentRuntimeConfig};
pub use app::{
    AppState, MessageRole, StreamingAssistantState, ToolOutputChunk, ToolStatus,
    ToolTranscriptEntry, TranscriptEntries, TranscriptEntry, TranscriptEntryId,
    TranscriptEntryState, TranscriptMessage, TranscriptMessages, UserMessageStatus,
};
pub use config::{
    default_state_path, load_state, persist_recent_model, persist_recent_thinking, ApiKind,
    CompactionSettings, Compatibility, ConfigError, ConfigWarning, ContextSettings, CostMetadata,
    ModelCatalog, ModelRef, PluginSettings, RecentModel, RecentModelState, ResolvedModel,
    ResolvedSettings, Settings, SettingsError, SettingsLoad, StateError, ThinkingLevel,
    ThinkingLevelError, WorkspaceRecentModel,
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
    StopReason, ToolChoice, ToolDefinition, Usage,
};
pub use plugin::{
    builtin_plugins, default_plugins_dir, load_plugin_manifest, resolve_installed_plugins,
    ExternalToolError, ExternalToolSet, InstalledPluginError, LoadedPluginManifest,
    PluginActivationError, PluginCapabilities, PluginDiagnostics, PluginEntrypoint, PluginHost,
    PluginIdentity, PluginManifest, PluginManifestError, PluginProcess, PluginProcessError,
    PluginRegistry, PluginShutdownFailure, PluginToolDefinition, ToolCallParams, ToolCallResult,
    ToolsListResult, MAX_EXTERNAL_TOOLS_PER_PLUGIN, MAX_EXTERNAL_TOOL_DESCRIPTION_BYTES,
    MAX_EXTERNAL_TOOL_NAME_BYTES, MAX_PLUGIN_FRAME_BYTES, MAX_PLUGIN_STDERR_BYTES,
    PLUGIN_MANIFEST_FILENAME, PLUGIN_MANIFEST_VERSION, PLUGIN_PROTOCOL_VERSION,
    PLUGIN_SHUTDOWN_TIMEOUT, PLUGIN_STARTUP_TIMEOUT,
};
pub use session::{
    read_session, validate_name, workspace_id, MessageId, OpenedSession, SessionAssistantItem,
    SessionError, SessionHandle, SessionHeader, SessionId, SessionInfo, SessionMessage,
    SessionMode, SessionRecord, SessionRepository, SessionSnapshot, SessionSummary, WorkspaceId,
    MAX_SESSION_RECORD_BYTES, SESSION_VERSION,
};
pub use tools::{
    builtin_tool_registry, Tool, ToolCallPresentation, ToolContext, ToolError, ToolEvent,
    ToolEventSender, ToolExecutionMetadata, ToolExecutionResult, ToolOutputKind, ToolOutputStream,
    ToolPreviewKind, ToolPreviewLine, ToolRegistry, ToolRegistryError, ToolSummaryKind,
    DEFAULT_BASH_TIMEOUT_MS, MAX_TOOL_OUTPUT_BYTES, MAX_TOOL_PREVIEW_BYTES, MAX_TOOL_PREVIEW_LINES,
};
