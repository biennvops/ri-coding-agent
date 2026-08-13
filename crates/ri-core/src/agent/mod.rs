use std::collections::VecDeque;
use std::future::{poll_fn, Future};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::config::ModelRef;
use crate::context::{
    automatic_trigger, compaction_target, input_budget, ConservativeTokenEstimator, ContextUsage,
    TokenEstimator, COMPACTION_MAX_OUTPUT_TOKENS,
};
use crate::conversation::{
    segment_history, CompactionSummary, ConversationHistory, HistorySegment,
};
use crate::model::{
    ModelAssistantItem, ModelEvent, ModelLimits, ModelMessage, ModelProvider, ModelRequest,
    ModelResponse, ModelToolCall, ProviderError, StopReason, Usage,
};
use crate::session::{SessionInfo, SessionMode};
use crate::tools::{
    ToolContext, ToolError, ToolEvent, ToolExecutionMetadata, ToolExecutionResult,
    ToolOutputStream, ToolRegistry,
};

const MODEL_EVENT_CHANNEL_CAPACITY: usize = 64;
const TOOL_EVENT_CHANNEL_CAPACITY: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentCommand {
    Submit {
        text: String,
    },
    Steer {
        text: String,
    },
    Compact,
    Cancel,
    NewSession {
        session: crate::session::SessionHandle,
    },
    LoadSession {
        session: crate::session::SessionHandle,
        history: Vec<ModelMessage>,
    },
    RenameSession {
        name: String,
    },
    RefreshContext,
    SetReasoningEffort {
        effort: Option<String>,
    },
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentError {
    pub message: String,
}

impl AgentError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvent {
    TurnStarted,
    SteeringMessageDelivered {
        text: String,
    },
    SteeringMessagesRecovered {
        messages: Vec<String>,
    },
    AssistantMessageStarted,
    AssistantTextDelta {
        index: Option<usize>,
        text: String,
    },
    AssistantTextItem {
        index: usize,
        content: Option<String>,
    },
    AssistantRefusalDelta {
        index: Option<usize>,
        text: String,
    },
    AssistantRefusalItem {
        index: usize,
        content: Option<String>,
    },
    AssistantThinkingDelta {
        item_id: Option<String>,
        text: String,
    },
    AssistantThinkingContentDelta {
        item_id: Option<String>,
        text: String,
    },
    AssistantThinkingItem {
        index: usize,
        item_id: Option<String>,
        summary: Option<String>,
        content: Option<String>,
        encrypted_content: Option<String>,
    },
    ToolCallDelta {
        index: usize,
        call_id: Option<String>,
        item_id: Option<String>,
        name: Option<String>,
        arguments: String,
        arguments_complete: bool,
    },
    AssistantMessageFinished {
        items: Vec<ModelAssistantItem>,
    },
    ToolExecutionStarted {
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolExecutionOutput {
        call_id: String,
        stream: ToolOutputStream,
        chunk: String,
    },
    ToolExecutionFinished {
        call_id: String,
        name: String,
        result: ToolExecutionResult,
    },
    UsageUpdated(Usage),
    ContextUsageUpdated(ContextUsage),
    ContextLimitsUpdated(ModelLimits),
    CompactionStarted {
        automatic: bool,
    },
    CompactionFinished {
        automatic: bool,
        before_tokens: u64,
        after_tokens: u64,
    },
    CompactionFailed {
        message: String,
    },
    ModelChanged(ModelRef),
    SessionChanged {
        info: SessionInfo,
    },
    SessionLoaded {
        info: SessionInfo,
        history: Vec<ModelMessage>,
    },
    TurnFinished {
        reason: StopReason,
    },
    Error(AgentError),
}

#[derive(Clone, Debug)]
pub struct AgentRuntimeConfig {
    pub tool_context: ToolContext,
    pub base_messages: Vec<ModelMessage>,
    pub initial_history: Vec<ModelMessage>,
    pub session: SessionMode,
    pub reasoning_effort: Option<String>,
}

impl AgentRuntimeConfig {
    pub fn new(tool_context: ToolContext) -> Self {
        Self {
            tool_context,
            base_messages: Vec::new(),
            initial_history: Vec::new(),
            session: SessionMode::Disabled,
            reasoning_effort: None,
        }
    }
}

#[derive(Default)]
struct SteeringState {
    messages: VecDeque<String>,
    reserved: Option<String>,
    delivering: bool,
    cancel_pending: bool,
    sealed: bool,
}

impl SteeringState {
    fn accept(&mut self, text: String) -> Option<String> {
        if self.sealed {
            Some(text)
        } else {
            self.messages.push_back(text);
            None
        }
    }

    fn reserve(&mut self, seal_when_empty: bool) -> bool {
        self.reserved = self.messages.pop_front();
        if self.reserved.is_none() && seal_when_empty {
            self.sealed = true;
        }
        self.reserved.is_some()
    }

    fn begin_delivery(&mut self) -> Option<String> {
        let text = self.reserved.take()?;
        self.delivering = true;
        Some(text)
    }

    fn cancel(&mut self, cancel: &CancellationToken) {
        if let Some(text) = self.reserved.take() {
            self.messages.push_front(text);
            cancel.cancel();
        } else if self.delivering {
            self.cancel_pending = true;
        } else {
            cancel.cancel();
        }
    }

    fn finish_delivery(&mut self, cancel: &CancellationToken) {
        self.delivering = false;
        if self.cancel_pending {
            self.cancel_pending = false;
            cancel.cancel();
        }
    }

    fn abort_delivery(&mut self, cancel: &CancellationToken) {
        self.delivering = false;
        self.cancel_pending = false;
        cancel.cancel();
    }

    fn seal_and_drain(&mut self) -> Vec<String> {
        self.sealed = true;
        if let Some(text) = self.reserved.take() {
            self.messages.push_front(text);
        }
        self.messages.drain(..).collect()
    }
}

type SteeringQueue = Arc<Mutex<SteeringState>>;

#[cfg(test)]
#[derive(Default)]
struct FinalResponseBarrier {
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[cfg(test)]
#[derive(Default)]
struct SteeringInjectionBarrier {
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

struct ActiveTurn {
    cancel: CancellationToken,
    task: JoinHandle<RuntimeTaskOutcome>,
    compaction: bool,
    steering: Option<SteeringQueue>,
}

impl ActiveTurn {
    fn cancel(&self) {
        cancel_task(&self.cancel, self.steering.as_ref());
    }
}

fn cancel_task(cancel: &CancellationToken, steering: Option<&SteeringQueue>) {
    if let Some(steering) = steering {
        steering
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cancel(cancel);
    } else {
        cancel.cancel();
    }
}

enum RuntimeTaskOutcome {
    Turn(TurnOutcome),
    Compaction(Result<ConversationHistory, String>),
}

struct TurnOutcome {
    history: ConversationHistory,
    reason: StopReason,
}

pub struct AgentRuntime<P> {
    provider: Arc<P>,
    registry: Arc<ToolRegistry>,
    context: ToolContext,
    base_messages: Vec<ModelMessage>,
    conversation: ConversationHistory,
    session: SessionMode,
    reasoning_effort: Option<String>,
    compaction_enabled: bool,
}

impl<P> AgentRuntime<P>
where
    P: ModelProvider,
{
    pub fn new(provider: P) -> Self {
        let context = ToolContext::from_current_dir().unwrap_or_else(|_| ToolContext {
            workspace_root: PathBuf::from("."),
        });
        Self::with_context(provider, context)
    }

    pub fn with_context(provider: P, context: ToolContext) -> Self {
        Self::with_config(provider, AgentRuntimeConfig::new(context))
    }

    pub fn with_config(provider: P, config: AgentRuntimeConfig) -> Self {
        Self::with_config_and_compaction(provider, config, true)
    }

    pub fn with_config_and_compaction(
        provider: P,
        config: AgentRuntimeConfig,
        compaction_enabled: bool,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            registry: Arc::new(ToolRegistry::new()),
            context: config.tool_context,
            base_messages: config.base_messages,
            conversation: ConversationHistory::from_provider_messages(config.initial_history),
            session: config.session,
            reasoning_effort: config.reasoning_effort,
            compaction_enabled,
        }
    }

    pub fn with_workspace_root(
        provider: P,
        workspace_root: impl Into<PathBuf>,
    ) -> Result<Self, ToolError> {
        Ok(Self::with_context(
            provider,
            ToolContext::new(workspace_root)?,
        ))
    }

    pub async fn run(
        mut self,
        mut commands: mpsc::Receiver<AgentCommand>,
        events: mpsc::Sender<AgentEvent>,
    ) {
        emit_context_snapshot(
            self.provider.as_ref(),
            &self.registry,
            &self.base_messages,
            &self.conversation,
            &events,
        )
        .await;
        let mut active: Option<ActiveTurn> = None;

        loop {
            if active.is_some() {
                let mut task_finished = false;
                if let Some(active_task) = active.as_mut() {
                    tokio::select! {
                        result = &mut active_task.task => {
                            task_finished = true;
                            recover_pending_steering(&active_task.steering, &events).await;
                            apply_task_result(
                                result,
                                active_task.compaction,
                                &mut self.conversation,
                                self.provider.as_ref(),
                                &self.registry,
                                &self.base_messages,
                                &events,
                            )
                            .await;
                        }
                        command = commands.recv() => {
                            match command {
                                Some(AgentCommand::Cancel) => active_task.cancel(),
                                Some(AgentCommand::Steer { text }) if !active_task.compaction => {
                                    if !text.trim().is_empty() {
                                        let recovered = active_task
                                            .steering
                                            .as_ref()
                                            .expect("active turns have a steering queue")
                                            .lock()
                                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                                            .accept(text);
                                        if let Some(text) = recovered {
                                            recover_steering(vec![text], &events).await;
                                        }
                                    }
                                }
                                Some(AgentCommand::Submit { .. })
                                | Some(AgentCommand::Steer { .. })
                                | Some(AgentCommand::Compact)
                                | Some(AgentCommand::NewSession { .. })
                                | Some(AgentCommand::LoadSession { .. })
                                | Some(AgentCommand::RenameSession { .. })
                                | Some(AgentCommand::RefreshContext)
                                | Some(AgentCommand::SetReasoningEffort { .. }) => {
                                    let _ = events.send(AgentEvent::Error(AgentError::new(
                                        "a turn or compaction is already active",
                                    ))).await;
                                }
                                Some(AgentCommand::Shutdown) | None => {
                                    active_task.cancel();
                                    let result = (&mut active_task.task).await;
                                    recover_pending_steering(&active_task.steering, &events).await;
                                    apply_task_result(
                                        result,
                                        active_task.compaction,
                                        &mut self.conversation,
                                        self.provider.as_ref(),
                                        &self.registry,
                                        &self.base_messages,
                                        &events,
                                    )
                                    .await;
                                    return;
                                }
                            }
                        }
                    }
                }
                if task_finished {
                    active = None;
                }
            } else {
                match commands.recv().await {
                    Some(AgentCommand::Submit { text }) => {
                        let cancel = CancellationToken::new();
                        let provider = Arc::clone(&self.provider);
                        let registry = Arc::clone(&self.registry);
                        let turn_config = AgentRuntimeConfig {
                            tool_context: self.context.clone(),
                            base_messages: self.base_messages.clone(),
                            initial_history: Vec::new(),
                            session: self.session.clone(),
                            reasoning_effort: self.reasoning_effort.clone(),
                        };
                        let history = self.conversation.clone();
                        let turn_events = events.clone();
                        let turn_cancel = cancel.clone();
                        let compaction_enabled = self.compaction_enabled;
                        let steering = Arc::new(Mutex::new(SteeringState::default()));
                        let turn_steering = Arc::clone(&steering);
                        let task = tokio::spawn(async move {
                            RuntimeTaskOutcome::Turn(
                                run_turn(
                                    provider,
                                    registry,
                                    turn_config,
                                    history,
                                    text,
                                    turn_events,
                                    turn_cancel,
                                    compaction_enabled,
                                    turn_steering,
                                    #[cfg(test)]
                                    None,
                                    #[cfg(test)]
                                    None,
                                )
                                .await,
                            )
                        });
                        active = Some(ActiveTurn {
                            cancel,
                            task,
                            compaction: false,
                            steering: Some(steering),
                        });
                    }
                    Some(AgentCommand::Steer { text }) => {
                        recover_steering(vec![text], &events).await;
                    }
                    Some(AgentCommand::Compact) => {
                        let cancel = CancellationToken::new();
                        let provider = Arc::clone(&self.provider);
                        let registry = Arc::clone(&self.registry);
                        let config = AgentRuntimeConfig {
                            tool_context: self.context.clone(),
                            base_messages: self.base_messages.clone(),
                            initial_history: Vec::new(),
                            session: self.session.clone(),
                            reasoning_effort: None,
                        };
                        let history = self.conversation.clone();
                        let compaction_events = events.clone();
                        let compaction_cancel = cancel.clone();
                        let task = tokio::spawn(async move {
                            RuntimeTaskOutcome::Compaction(
                                compact_conversation(
                                    provider,
                                    registry,
                                    config,
                                    history,
                                    compaction_events,
                                    compaction_cancel,
                                    false,
                                    true,
                                )
                                .await
                                .map_err(compaction_error_message),
                            )
                        });
                        active = Some(ActiveTurn {
                            cancel,
                            task,
                            compaction: true,
                            steering: None,
                        });
                    }
                    Some(AgentCommand::NewSession { session }) => {
                        self.session = SessionMode::Enabled(session);
                        self.conversation.clear();
                        emit_session_loaded(&events, &self.session, &self.conversation).await;
                        emit_context_snapshot(
                            self.provider.as_ref(),
                            &self.registry,
                            &self.base_messages,
                            &self.conversation,
                            &events,
                        )
                        .await;
                    }
                    Some(AgentCommand::LoadSession { session, history }) => {
                        self.session = SessionMode::Enabled(session);
                        self.conversation = ConversationHistory::from_provider_messages(history);
                        emit_session_loaded(&events, &self.session, &self.conversation).await;
                        emit_context_snapshot(
                            self.provider.as_ref(),
                            &self.registry,
                            &self.base_messages,
                            &self.conversation,
                            &events,
                        )
                        .await;
                    }
                    Some(AgentCommand::RenameSession { name }) => match &self.session {
                        SessionMode::Disabled => {
                            let _ = events
                                .send(AgentEvent::Error(AgentError::new(
                                    "sessions are disabled for this run",
                                )))
                                .await;
                        }
                        SessionMode::Enabled(session) => match session.rename(&name) {
                            Ok(info) => {
                                let _ = events.send(AgentEvent::SessionChanged { info }).await;
                            }
                            Err(error) => {
                                let _ = events
                                    .send(AgentEvent::Error(AgentError::new(error.to_string())))
                                    .await;
                            }
                        },
                    },
                    Some(AgentCommand::RefreshContext) => {
                        emit_context_snapshot(
                            self.provider.as_ref(),
                            &self.registry,
                            &self.base_messages,
                            &self.conversation,
                            &events,
                        )
                        .await;
                    }
                    Some(AgentCommand::SetReasoningEffort { effort }) => {
                        self.reasoning_effort = effort;
                    }
                    Some(AgentCommand::Cancel) => {
                        let _ = events
                            .send(AgentEvent::Error(AgentError::new("no turn is active")))
                            .await;
                    }
                    Some(AgentCommand::Shutdown) | None => return,
                }
            }
        }
    }
}

async fn emit_context_snapshot<P>(
    provider: &P,
    registry: &ToolRegistry,
    base_messages: &[ModelMessage],
    conversation: &ConversationHistory,
    events: &mpsc::Sender<AgentEvent>,
) where
    P: ModelProvider,
{
    let limits = provider.limits();
    if limits.context_window.is_none()
        && base_messages.is_empty()
        && conversation.messages().is_empty()
    {
        return;
    }
    let request = normal_request(base_messages, conversation, &registry.definitions());
    let estimate = ConservativeTokenEstimator.estimate_request(&request);
    let _ = events.send(AgentEvent::ContextLimitsUpdated(limits)).await;
    let _ = events
        .send(AgentEvent::ContextUsageUpdated(ContextUsage::estimated(
            estimate, limits,
        )))
        .await;
}

async fn emit_session_loaded(
    events: &mpsc::Sender<AgentEvent>,
    session: &SessionMode,
    conversation: &ConversationHistory,
) {
    match session.info() {
        Ok(Some(info)) => {
            let history = match session.transcript_history() {
                Ok(Some(history)) => history,
                Ok(None) => conversation.messages().to_vec(),
                Err(error) => {
                    let _ = events
                        .send(AgentEvent::Error(AgentError::new(error.to_string())))
                        .await;
                    return;
                }
            };
            let _ = events
                .send(AgentEvent::SessionLoaded { info, history })
                .await;
        }
        Ok(None) => {}
        Err(error) => {
            let _ = events
                .send(AgentEvent::Error(AgentError::new(error.to_string())))
                .await;
        }
    }
}

async fn recover_pending_steering(
    steering: &Option<SteeringQueue>,
    events: &mpsc::Sender<AgentEvent>,
) {
    let Some(steering) = steering else {
        return;
    };
    let messages = steering
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .seal_and_drain();
    recover_steering(messages, events).await;
}

async fn recover_steering(messages: Vec<String>, events: &mpsc::Sender<AgentEvent>) {
    if !messages.is_empty() {
        let _ = events
            .send(AgentEvent::SteeringMessagesRecovered { messages })
            .await;
    }
}

async fn apply_task_result<P>(
    result: Result<RuntimeTaskOutcome, tokio::task::JoinError>,
    compaction: bool,
    conversation: &mut ConversationHistory,
    provider: &P,
    registry: &ToolRegistry,
    base_messages: &[ModelMessage],
    events: &mpsc::Sender<AgentEvent>,
) where
    P: ModelProvider,
{
    match result {
        Ok(RuntimeTaskOutcome::Turn(outcome)) => {
            *conversation = outcome.history;
            emit_context_snapshot(provider, registry, base_messages, conversation, events).await;
            let _ = events
                .send(AgentEvent::TurnFinished {
                    reason: outcome.reason,
                })
                .await;
        }
        Ok(RuntimeTaskOutcome::Compaction(Ok(history))) => {
            *conversation = history;
        }
        Ok(RuntimeTaskOutcome::Compaction(Err(_))) if compaction => {}
        Ok(RuntimeTaskOutcome::Compaction(Err(error))) => {
            let _ = events
                .send(AgentEvent::CompactionFailed { message: error })
                .await;
        }
        Err(error) => {
            let message = format!("agent task failed: {error}");
            let _ = events
                .send(AgentEvent::Error(AgentError::new(message.clone())))
                .await;
            if compaction {
                let _ = events.send(AgentEvent::CompactionFailed { message }).await;
            } else {
                let _ = events
                    .send(AgentEvent::TurnFinished {
                        reason: StopReason::Error,
                    })
                    .await;
            }
        }
    }
}

enum CompactionError {
    Cancelled,
    NoHistory,
    Failed(String),
}

fn compaction_error_message(error: CompactionError) -> String {
    match error {
        CompactionError::Cancelled => "compaction cancelled".to_owned(),
        CompactionError::NoHistory => "nothing to compact".to_owned(),
        CompactionError::Failed(message) => message,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn<P>(
    provider: Arc<P>,
    registry: Arc<ToolRegistry>,
    config: AgentRuntimeConfig,
    mut history: ConversationHistory,
    text: String,
    events: mpsc::Sender<AgentEvent>,
    cancel: CancellationToken,
    compaction_enabled: bool,
    steering: SteeringQueue,
    #[cfg(test)] final_response_barrier: Option<Arc<FinalResponseBarrier>>,
    #[cfg(test)] steering_injection_barrier: Option<Arc<SteeringInjectionBarrier>>,
) -> TurnOutcome
where
    P: ModelProvider,
{
    let user_message = ModelMessage::user(text);
    if let Err(error) = commit_message(&mut history, &config.session, user_message, &events).await {
        fail_turn(&events, error).await;
        return turn_outcome(history, StopReason::Error);
    }
    if events.send(AgentEvent::TurnStarted).await.is_err() {
        return turn_outcome(history, StopReason::Error);
    }

    let mut tool_rounds = 0;
    let mut steering_pending = false;
    loop {
        if cancel.is_cancelled() {
            return turn_outcome(history, StopReason::Cancelled);
        }

        let limits = provider.limits();
        let tools = registry.definitions();
        let mut emergency_compaction = false;
        let mut assistant_started = false;
        let response = loop {
            if cancel.is_cancelled() {
                return turn_outcome(history, StopReason::Cancelled);
            }
            let mut request = normal_request_with_effort(
                &config.base_messages,
                &history,
                &tools,
                config.reasoning_effort.clone(),
            );
            let estimate = ConservativeTokenEstimator.estimate_request(&request);
            let _ = events.send(AgentEvent::ContextLimitsUpdated(limits)).await;
            let _ = events
                .send(AgentEvent::ContextUsageUpdated(ContextUsage::estimated(
                    estimate, limits,
                )))
                .await;

            if compaction_enabled
                && !emergency_compaction
                && input_budget(limits).is_some_and(|budget| estimate > automatic_trigger(budget))
            {
                match compact_conversation(
                    Arc::clone(&provider),
                    Arc::clone(&registry),
                    config.clone(),
                    history.clone(),
                    events.clone(),
                    cancel.clone(),
                    true,
                    false,
                )
                .await
                {
                    Ok(compacted) => {
                        history = compacted;
                        continue;
                    }
                    Err(CompactionError::NoHistory) => {}
                    Err(CompactionError::Cancelled) => {
                        return turn_outcome(history, StopReason::Cancelled);
                    }
                    Err(CompactionError::Failed(error)) => {
                        fail_turn(&events, error).await;
                        return turn_outcome(history, StopReason::Error);
                    }
                }
            }

            if !assistant_started && !steering_pending {
                if events
                    .send(AgentEvent::AssistantMessageStarted)
                    .await
                    .is_err()
                {
                    return turn_outcome(history, StopReason::Error);
                }
                assistant_started = true;
            }
            let steering_for_request = steering_pending.then_some(&steering);
            steering_pending = false;
            match stream_model(
                Arc::clone(&provider),
                &mut request,
                &mut history,
                &config.session,
                &events,
                &mut assistant_started,
                cancel.clone(),
                steering_for_request,
                #[cfg(test)]
                steering_injection_barrier.as_deref(),
            )
            .await
            {
                Ok(response) => break response,
                Err(ProviderError::Cancelled) => {
                    return turn_outcome(history, StopReason::Cancelled);
                }
                Err(ProviderError::ContextOverflow { message })
                    if compaction_enabled && !emergency_compaction =>
                {
                    emergency_compaction = true;
                    match compact_conversation(
                        Arc::clone(&provider),
                        Arc::clone(&registry),
                        config.clone(),
                        history.clone(),
                        events.clone(),
                        cancel.clone(),
                        true,
                        true,
                    )
                    .await
                    {
                        Ok(compacted) => {
                            history = compacted;
                        }
                        Err(CompactionError::Cancelled) => {
                            return turn_outcome(history, StopReason::Cancelled);
                        }
                        Err(CompactionError::NoHistory) => {
                            fail_turn(
                                &events,
                                format!(
                                    "context still exceeds the selected model's limit after compaction: {message}"
                                ),
                            )
                            .await;
                            return turn_outcome(history, StopReason::Error);
                        }
                        Err(CompactionError::Failed(error)) => {
                            fail_turn(&events, error).await;
                            return turn_outcome(history, StopReason::Error);
                        }
                    }
                    continue;
                }
                Err(error) => {
                    fail_turn(&events, error.to_string()).await;
                    return turn_outcome(history, StopReason::Error);
                }
            }
        };

        let calls = match validated_tool_calls(&response) {
            Ok(calls) => calls,
            Err(error) => {
                fail_turn(&events, error).await;
                return turn_outcome(history, StopReason::Error);
            }
        };
        if response.stop_reason == StopReason::ToolCalls && calls.is_empty() {
            fail_turn(
                &events,
                "provider requested tool execution but returned no tool calls",
            )
            .await;
            return turn_outcome(history, StopReason::Error);
        }

        let assistant_message = ModelMessage::Assistant {
            items: response.items.clone(),
        };
        if let Err(error) =
            commit_message(&mut history, &config.session, assistant_message, &events).await
        {
            fail_turn(&events, error).await;
            return turn_outcome(history, StopReason::Error);
        }
        if events
            .send(AgentEvent::AssistantMessageFinished {
                items: response.items.clone(),
            })
            .await
            .is_err()
        {
            return turn_outcome(history, StopReason::Error);
        }

        if calls.is_empty() {
            if response.stop_reason == StopReason::Error {
                fail_turn(&events, "provider returned an error stop reason").await;
                return turn_outcome(history, StopReason::Error);
            }
            #[cfg(test)]
            if let Some(barrier) = &final_response_barrier {
                barrier.reached.notify_one();
                barrier.release.notified().await;
            }
            match reserve_next_steering(&steering, &cancel, true) {
                SteeringInjection::Pending => {
                    steering_pending = true;
                    continue;
                }
                SteeringInjection::Empty => {
                    return turn_outcome(history, response.stop_reason);
                }
                SteeringInjection::Cancelled => {
                    return turn_outcome(history, StopReason::Cancelled);
                }
            }
        }

        tool_rounds += 1;
        tracing::debug!(
            target: "ri_core::agent",
            tool_round = tool_rounds,
            tool_calls = calls.len(),
            "tool round started"
        );

        for (index, call) in calls.iter().enumerate() {
            if cancel.is_cancelled() {
                if let Err(error) = append_synthetic_results(
                    &mut history,
                    &config.session,
                    &calls[index..],
                    &events,
                    "Tool execution cancelled by user.",
                )
                .await
                {
                    fail_turn(&events, error).await;
                    return turn_outcome(history, StopReason::Error);
                }
                return turn_outcome(history, StopReason::Cancelled);
            }

            let result = execute_tool_call(
                Arc::clone(&registry),
                config.tool_context.clone(),
                call,
                &events,
                cancel.clone(),
            )
            .await;
            let tool_call_id = call.call_id.clone().expect("validated call id");
            let tool_name = call.name.clone().expect("validated tool name");
            let tool_message = ModelMessage::ToolResult {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                content: result.model_content.clone(),
            };
            let persistence_error =
                commit_message(&mut history, &config.session, tool_message, &events).await;
            let _ = events
                .send(AgentEvent::ToolExecutionFinished {
                    call_id: tool_call_id,
                    name: tool_name,
                    result: result.clone(),
                })
                .await;
            if let Err(error) = persistence_error {
                fail_turn(&events, error).await;
                return turn_outcome(history, StopReason::Error);
            }

            if cancel.is_cancelled() || result.metadata.cancelled {
                if index + 1 < calls.len() {
                    if let Err(error) = append_synthetic_results(
                        &mut history,
                        &config.session,
                        &calls[index + 1..],
                        &events,
                        "Tool execution cancelled by user.",
                    )
                    .await
                    {
                        fail_turn(&events, error).await;
                        return turn_outcome(history, StopReason::Error);
                    }
                }
                return turn_outcome(history, StopReason::Cancelled);
            }
        }

        match reserve_next_steering(&steering, &cancel, false) {
            SteeringInjection::Pending => steering_pending = true,
            SteeringInjection::Empty => {}
            SteeringInjection::Cancelled => {
                return turn_outcome(history, StopReason::Cancelled);
            }
        }
    }
}

enum SteeringInjection {
    Pending,
    Empty,
    Cancelled,
}

fn reserve_next_steering(
    steering: &SteeringQueue,
    cancel: &CancellationToken,
    seal_when_empty: bool,
) -> SteeringInjection {
    let pending = {
        let mut steering = steering
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancel.is_cancelled() {
            return SteeringInjection::Cancelled;
        }
        steering.reserve(seal_when_empty)
    };
    if !pending {
        return SteeringInjection::Empty;
    }
    SteeringInjection::Pending
}

fn normal_request_with_effort(
    base_messages: &[ModelMessage],
    history: &ConversationHistory,
    tools: &[crate::model::ToolDefinition],
    reasoning_effort: Option<String>,
) -> ModelRequest {
    let mut request = normal_request(base_messages, history, tools);
    request.reasoning_effort = reasoning_effort;
    request
}

const COMPACTION_SYSTEM_INSTRUCTION: &str = "Summarize the earlier coding-agent conversation for future continuation.\n\nPreserve concrete technical state:\n- user goals and constraints\n- decisions and rationale that affect future work\n- files inspected or modified and important changes\n- important code architecture and interfaces\n- commands/tests and their significant results\n- important errors and attempted fixes\n- unresolved work and next steps\n- current task state\n- exact identifiers or values when they matter\n\nDo not invent work that did not happen.\nDo not copy large tool outputs verbatim.\nReturn only the continuation summary.";

fn normal_request(
    base_messages: &[ModelMessage],
    history: &ConversationHistory,
    tools: &[crate::model::ToolDefinition],
) -> ModelRequest {
    let messages = base_messages
        .iter()
        .cloned()
        .chain(history.provider_messages())
        .collect();
    ModelRequest {
        messages,
        tools: tools.to_vec(),
        max_tokens: None,
        reasoning_effort: None,
        sampling_params: Default::default(),
    }
}

#[allow(clippy::too_many_arguments)]
async fn compact_conversation<P>(
    provider: Arc<P>,
    registry: Arc<ToolRegistry>,
    config: AgentRuntimeConfig,
    history: ConversationHistory,
    events: mpsc::Sender<AgentEvent>,
    cancel: CancellationToken,
    automatic: bool,
    force: bool,
) -> Result<ConversationHistory, CompactionError>
where
    P: ModelProvider,
{
    let limits = provider.limits();
    let tools = registry.definitions();
    let estimator = ConservativeTokenEstimator;
    let before_request = normal_request(&config.base_messages, &history, &tools);
    let before_tokens = estimator.estimate_request(&before_request);
    let target = input_budget(limits)
        .map(compaction_target)
        .unwrap_or_else(|| before_tokens.saturating_div(2).max(1));
    let Some((prefix, retained)) =
        select_compaction_prefix(&history, &config.base_messages, &tools, target, force)
    else {
        if !automatic {
            let _ = events
                .send(AgentEvent::CompactionFailed {
                    message: "nothing to compact".to_owned(),
                })
                .await;
        }
        return Err(CompactionError::NoHistory);
    };

    if cancel.is_cancelled() {
        let _ = events
            .send(AgentEvent::CompactionFailed {
                message: "compaction cancelled".to_owned(),
            })
            .await;
        return Err(CompactionError::Cancelled);
    }
    if events
        .send(AgentEvent::CompactionStarted { automatic })
        .await
        .is_err()
    {
        return Err(CompactionError::Failed(
            "agent event stream closed".to_owned(),
        ));
    }

    let mut summary_messages = vec![ModelMessage::System {
        content: COMPACTION_SYSTEM_INSTRUCTION.to_owned(),
    }];
    if let Some(summary) = history.summary() {
        summary_messages.push(summary.as_message());
    }
    summary_messages.extend(prefix);
    let output_limit = limits
        .max_output_tokens
        .map(|limit| limit.min(COMPACTION_MAX_OUTPUT_TOKENS))
        .unwrap_or(COMPACTION_MAX_OUTPUT_TOKENS)
        .max(1);
    let request = ModelRequest {
        messages: summary_messages,
        tools: Vec::new(),
        max_tokens: Some(output_limit),
        reasoning_effort: None,
        sampling_params: Default::default(),
    };
    let response = match stream_private_model(provider, request, cancel.clone()).await {
        Ok(response) => response,
        Err(ProviderError::Cancelled) => {
            let _ = events
                .send(AgentEvent::CompactionFailed {
                    message: "compaction cancelled".to_owned(),
                })
                .await;
            return Err(CompactionError::Cancelled);
        }
        Err(error) => {
            let message = format!("compaction failed: {error}");
            let _ = events
                .send(AgentEvent::CompactionFailed {
                    message: message.clone(),
                })
                .await;
            return Err(CompactionError::Failed(message));
        }
    };
    if cancel.is_cancelled() {
        let _ = events
            .send(AgentEvent::CompactionFailed {
                message: "compaction cancelled".to_owned(),
            })
            .await;
        return Err(CompactionError::Cancelled);
    }
    let summary = match extract_summary(&response) {
        Ok(summary) => CompactionSummary::new(summary),
        Err(message) => {
            let _ = events
                .send(AgentEvent::CompactionFailed {
                    message: message.clone(),
                })
                .await;
            return Err(CompactionError::Failed(message));
        }
    };
    let compacted = ConversationHistory::new(Some(summary.clone()), retained.clone());
    let after_tokens =
        estimator.estimate_request(&normal_request(&config.base_messages, &compacted, &tools));
    if after_tokens >= before_tokens {
        let message = format!(
            "compaction did not reduce context: estimated {before_tokens} tokens before and {after_tokens} after"
        );
        let _ = events
            .send(AgentEvent::CompactionFailed {
                message: message.clone(),
            })
            .await;
        return Err(CompactionError::Failed(message));
    }

    if let Err(error) = config
        .session
        .append_compaction(&summary.content, &retained)
    {
        let message = format!("session persistence failed during compaction: {error}");
        let _ = events
            .send(AgentEvent::CompactionFailed {
                message: message.clone(),
            })
            .await;
        return Err(CompactionError::Failed(message));
    }
    if let Ok(Some(info)) = config.session.info() {
        let _ = events.send(AgentEvent::SessionChanged { info }).await;
    }

    let _ = events
        .send(AgentEvent::ContextUsageUpdated(ContextUsage::estimated(
            after_tokens,
            limits,
        )))
        .await;
    let _ = events
        .send(AgentEvent::CompactionFinished {
            automatic,
            before_tokens,
            after_tokens,
        })
        .await;
    Ok(compacted)
}

fn select_compaction_prefix(
    history: &ConversationHistory,
    base_messages: &[ModelMessage],
    tools: &[crate::model::ToolDefinition],
    target: u64,
    force: bool,
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
    let eligible_end = if force {
        current_segment
    } else if user_segments.len() > 2 {
        user_segments[user_segments.len() - 2].min(current_segment)
    } else {
        0
    };

    let before_tokens =
        ConservativeTokenEstimator.estimate_request(&normal_request(base_messages, history, tools));
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
        if projected_tokens(base_messages, tools, &retained) <= target {
            return Some((prefix, retained));
        }
    }

    if !prefix.is_empty() {
        Some((prefix, messages_from_segments(&segments, retained_start)))
    } else {
        None
    }
}

fn projected_tokens(
    base_messages: &[ModelMessage],
    tools: &[crate::model::ToolDefinition],
    retained: &[ModelMessage],
) -> u64 {
    let placeholder = ConversationHistory::new(
        Some(CompactionSummary::new("[summary of earlier conversation]")),
        retained.to_vec(),
    );
    ConservativeTokenEstimator.estimate_request(&normal_request(base_messages, &placeholder, tools))
}

fn messages_from_segments(segments: &[HistorySegment], start: usize) -> Vec<ModelMessage> {
    segments
        .iter()
        .skip(start)
        .flat_map(|segment| segment.messages.iter().cloned())
        .collect()
}

fn extract_summary(response: &ModelResponse) -> Result<String, String> {
    if response.stop_reason == StopReason::ToolCalls
        || response
            .items
            .iter()
            .any(|item| matches!(item, ModelAssistantItem::ToolCall(_)))
    {
        return Err("compaction response contained a tool call".to_owned());
    }
    if response.stop_reason != StopReason::Stop {
        return Err(format!(
            "compaction response did not finish successfully: {:?}",
            response.stop_reason
        ));
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
        return Err("compaction response was empty".to_owned());
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

#[allow(clippy::too_many_arguments)]
async fn stream_model<P>(
    provider: Arc<P>,
    request: &mut ModelRequest,
    history: &mut ConversationHistory,
    session: &SessionMode,
    events: &mpsc::Sender<AgentEvent>,
    assistant_started: &mut bool,
    cancel: CancellationToken,
    steering: Option<&SteeringQueue>,
    #[cfg(test)] steering_injection_barrier: Option<&SteeringInjectionBarrier>,
) -> Result<ModelResponse, ProviderError>
where
    P: ModelProvider,
{
    let (model_event_tx, mut model_event_rx) = mpsc::channel(MODEL_EVENT_CHANNEL_CAPACITY);

    let mut steering_delivery = false;
    if let Some(steering) = steering {
        #[cfg(test)]
        if let Some(barrier) = steering_injection_barrier {
            barrier.reached.notify_one();
            barrier.release.notified().await;
        }
        let text = {
            let mut state = steering
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cancel.is_cancelled() {
                return Err(ProviderError::Cancelled);
            }
            state
                .begin_delivery()
                .expect("pending steering has a reserved message")
        };
        if let Err(message) =
            commit_message(history, session, ModelMessage::user(text.clone()), events).await
        {
            steering
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .abort_delivery(&cancel);
            return Err(ProviderError::Failed { message });
        }
        request.messages.push(ModelMessage::user(text.clone()));
        if events
            .send(AgentEvent::SteeringMessageDelivered { text })
            .await
            .is_err()
        {
            steering
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .abort_delivery(&cancel);
            return Err(ProviderError::Failed {
                message: "agent event stream closed".to_owned(),
            });
        }
        if !*assistant_started {
            if events
                .send(AgentEvent::AssistantMessageStarted)
                .await
                .is_err()
            {
                steering
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .abort_delivery(&cancel);
                return Err(ProviderError::Failed {
                    message: "agent event stream closed".to_owned(),
                });
            }
            *assistant_started = true;
        }
        steering_delivery = true;
    }

    let provider_events = model_event_tx;
    let provider_request = request.clone();
    let provider_cancel = cancel.child_token();
    let (provider_polled_tx, provider_polled_rx) = oneshot::channel();
    let mut provider_task = tokio::spawn(async move {
        let provider = provider.stream(provider_request, provider_events, provider_cancel);
        tokio::pin!(provider);
        let result = poll_fn(|context| match provider.as_mut().poll(context) {
            std::task::Poll::Ready(result) => std::task::Poll::Ready(Some(result)),
            std::task::Poll::Pending => std::task::Poll::Ready(None),
        })
        .await;
        let _ = provider_polled_tx.send(());
        match result {
            Some(result) => result,
            None => provider.await,
        }
    });
    if provider_polled_rx.await.is_err() {
        if steering_delivery {
            steering
                .expect("delivery has steering state")
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .abort_delivery(&cancel);
        }
        return match provider_task.await {
            Ok(result) => result,
            Err(error) => Err(ProviderError::Failed {
                message: format!("provider task failed: {error}"),
            }),
        };
    }
    if steering_delivery {
        steering
            .expect("delivery has steering state")
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .finish_delivery(&cancel);
    }
    let mut usage_event_seen = false;

    let provider_result = loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                provider_task.abort();
                let _ = provider_task.await;
                return Err(ProviderError::Cancelled);
            }
            model_event = model_event_rx.recv() => {
                let Some(model_event) = model_event else { continue };
                usage_event_seen |= matches!(&model_event, ModelEvent::UsageUpdated(_));
                if let Err(error) = send_model_event(events, model_event, &cancel).await {
                    provider_task.abort();
                    let _ = provider_task.await;
                    return Err(error);
                }
            }
            result = &mut provider_task => {
                break result;
            }
        }
    };

    while let Some(model_event) = model_event_rx.recv().await {
        usage_event_seen |= matches!(&model_event, ModelEvent::UsageUpdated(_));
        send_model_event(events, model_event, &cancel).await?;
    }

    match provider_result {
        Ok(result) => {
            let result = result?;
            if !usage_event_seen {
                if let Some(usage) = result.usage.clone() {
                    send_model_event(events, ModelEvent::UsageUpdated(usage), &cancel).await?;
                }
            }
            Ok(result)
        }
        Err(error) => Err(ProviderError::Failed {
            message: format!("provider task failed: {error}"),
        }),
    }
}

fn validated_tool_calls(response: &ModelResponse) -> Result<Vec<ModelToolCall>, String> {
    let calls: Vec<ModelToolCall> = response
        .items
        .iter()
        .filter_map(|item| match item {
            ModelAssistantItem::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect();
    for call in &calls {
        if call.call_id.as_deref().is_none_or(str::is_empty) {
            return Err("provider returned a tool call without a call_id".to_owned());
        }
        if call.name.as_deref().is_none_or(str::is_empty) {
            return Err("provider returned a tool call without a name".to_owned());
        }
    }
    Ok(calls)
}

async fn execute_tool_call(
    registry: Arc<ToolRegistry>,
    context: ToolContext,
    call: &ModelToolCall,
    events: &mpsc::Sender<AgentEvent>,
    cancel: CancellationToken,
) -> ToolExecutionResult {
    let call_id = call.call_id.clone().expect("validated call id");
    let name = call.name.clone().expect("validated tool name");
    let _ = events
        .send(AgentEvent::ToolExecutionStarted {
            call_id: call_id.clone(),
            name: name.clone(),
            arguments: call.arguments.clone(),
        })
        .await;

    let result = match serde_json::from_str::<Value>(&call.arguments) {
        Ok(arguments) => {
            let (tool_events, mut tool_event_rx) = mpsc::channel(TOOL_EVENT_CHANNEL_CAPACITY);
            let tool_cancel = cancel.clone();
            let tool_name = name.clone();
            let tool_registry = Arc::clone(&registry);
            let tool_context = context.clone();
            let mut task = tokio::spawn(async move {
                tool_registry
                    .execute(
                        &tool_name,
                        arguments,
                        &tool_context,
                        tool_events,
                        tool_cancel,
                    )
                    .await
            });
            let mut tool_events_open = true;
            let result = loop {
                tokio::select! {
                    tool_event = tool_event_rx.recv(), if tool_events_open => {
                        match tool_event {
                            Some(tool_event) => {
                                forward_tool_event(events, &call_id, tool_event, &cancel).await
                            }
                            None => tool_events_open = false,
                        }
                    }
                    joined = &mut task => {
                        break match joined {
                            Ok(Ok(result)) => result,
                            Ok(Err(error)) => tool_error_result(error, cancel.is_cancelled()),
                            Err(error) => ToolExecutionResult::failure(format!(
                                "Tool error: internal tool task failed: {error}"
                            )),
                        };
                    }
                    _ = cancel.cancelled() => {
                        break match task.await {
                            Ok(Ok(result)) => result,
                            Ok(Err(error)) => tool_error_result(error, true),
                            Err(error) => ToolExecutionResult::failure(format!(
                                "Tool error: internal tool task failed: {error}"
                            )),
                        };
                    }
                }
            };
            while let Some(tool_event) = tool_event_rx.recv().await {
                forward_tool_event(events, &call_id, tool_event, &cancel).await;
            }
            result
        }
        Err(error) => {
            ToolExecutionResult::failure(format!("Tool error: invalid JSON arguments: {error}"))
        }
    };

    result
}

async fn forward_tool_event(
    events: &mpsc::Sender<AgentEvent>,
    call_id: &str,
    event: ToolEvent,
    cancel: &CancellationToken,
) {
    match event {
        ToolEvent::Output { stream, chunk } => {
            let _ = tokio::select! {
                _ = cancel.cancelled() => Ok(()),
                result = events.send(AgentEvent::ToolExecutionOutput {
                    call_id: call_id.to_owned(),
                    stream,
                    chunk,
                }) => result,
            };
        }
    }
}

async fn send_model_event(
    events: &mpsc::Sender<AgentEvent>,
    event: ModelEvent,
    cancel: &CancellationToken,
) -> Result<(), ProviderError> {
    tokio::select! {
        _ = cancel.cancelled() => Err(ProviderError::Cancelled),
        result = events.send(agent_event_from_model(event)) => result.map_err(|_| ProviderError::Failed {
            message: "agent event stream closed".to_owned(),
        }),
    }
}

fn tool_error_result(error: ToolError, cancelled: bool) -> ToolExecutionResult {
    if matches!(error, ToolError::Cancelled) || cancelled {
        let mut metadata = ToolExecutionMetadata::failure();
        metadata.cancelled = true;
        ToolExecutionResult {
            model_content: "Tool execution cancelled by user.".to_owned(),
            metadata,
        }
    } else {
        ToolExecutionResult::failure(format!("Tool error: {error}"))
    }
}

async fn commit_message(
    history: &mut ConversationHistory,
    session: &SessionMode,
    message: ModelMessage,
    events: &mpsc::Sender<AgentEvent>,
) -> Result<(), String> {
    let info = session
        .append_message(&message)
        .map_err(|error| format!("session persistence failed: {error}"))?;
    history.push(message);
    if let Some(info) = info {
        events
            .send(AgentEvent::SessionChanged { info })
            .await
            .map_err(|_| "agent event stream closed".to_owned())?;
    }
    Ok(())
}

async fn append_synthetic_results(
    history: &mut ConversationHistory,
    session: &SessionMode,
    calls: &[ModelToolCall],
    events: &mpsc::Sender<AgentEvent>,
    message: &str,
) -> Result<(), String> {
    for call in calls {
        let call_id = call.call_id.clone().expect("validated call id");
        let name = call.name.clone().expect("validated tool name");
        let result = {
            let mut metadata = ToolExecutionMetadata::failure();
            metadata.cancelled = message.contains("cancelled");
            ToolExecutionResult {
                model_content: message.to_owned(),
                metadata,
            }
        };
        events
            .send(AgentEvent::ToolExecutionStarted {
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: call.arguments.clone(),
            })
            .await
            .map_err(|_| "agent event stream closed".to_owned())?;
        let tool_message = ModelMessage::ToolResult {
            tool_call_id: call_id.clone(),
            tool_name: name.clone(),
            content: result.model_content.clone(),
        };
        commit_message(history, session, tool_message, events).await?;
        events
            .send(AgentEvent::ToolExecutionFinished {
                call_id,
                name,
                result,
            })
            .await
            .map_err(|_| "agent event stream closed".to_owned())?;
    }
    Ok(())
}

async fn fail_turn(events: &mpsc::Sender<AgentEvent>, message: impl Into<String>) {
    let _ = events
        .send(AgentEvent::Error(AgentError::new(message)))
        .await;
}

fn turn_outcome(history: ConversationHistory, reason: StopReason) -> TurnOutcome {
    TurnOutcome { history, reason }
}

fn agent_event_from_model(event: ModelEvent) -> AgentEvent {
    match event {
        ModelEvent::AssistantTextDelta { index, text } => {
            AgentEvent::AssistantTextDelta { index, text }
        }
        ModelEvent::AssistantTextItem { index, content } => {
            AgentEvent::AssistantTextItem { index, content }
        }
        ModelEvent::AssistantRefusalDelta { index, text } => {
            AgentEvent::AssistantRefusalDelta { index, text }
        }
        ModelEvent::AssistantRefusalItem { index, content } => {
            AgentEvent::AssistantRefusalItem { index, content }
        }
        ModelEvent::AssistantThinkingDelta { item_id, text } => {
            AgentEvent::AssistantThinkingDelta { item_id, text }
        }
        ModelEvent::AssistantThinkingContentDelta { item_id, text } => {
            AgentEvent::AssistantThinkingContentDelta { item_id, text }
        }
        ModelEvent::AssistantThinkingItem {
            index,
            item_id,
            summary,
            content,
            encrypted_content,
        } => AgentEvent::AssistantThinkingItem {
            index,
            item_id,
            summary,
            content,
            encrypted_content,
        },
        ModelEvent::ToolCallDelta {
            index,
            call_id,
            item_id,
            name,
            arguments,
            arguments_complete,
        } => AgentEvent::ToolCallDelta {
            index,
            call_id,
            item_id,
            name,
            arguments,
            arguments_complete,
        },
        ModelEvent::UsageUpdated(usage) => AgentEvent::UsageUpdated(usage),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;
    use crate::model::MockProvider;

    #[tokio::test]
    async fn runtime_forwards_stream_and_finishes_turn() {
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(32);
        let runtime = AgentRuntime::new(
            MockProvider::with_response("hello")
                .with_chunk_size(1)
                .with_delay(Duration::ZERO),
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "test".to_owned(),
            })
            .await
            .expect("runtime should be listening");

        let mut received = Vec::new();
        loop {
            let event = event_rx.recv().await.expect("runtime should emit events");
            let finished = matches!(event, AgentEvent::TurnFinished { .. });
            received.push(event);
            if finished {
                break;
            }
        }

        assert_eq!(received[0], AgentEvent::TurnStarted);
        assert_eq!(
            received
                .iter()
                .filter_map(|event| match event {
                    AgentEvent::AssistantTextDelta { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>(),
            "hello"
        );
        assert_eq!(
            received.last(),
            Some(&AgentEvent::TurnFinished {
                reason: StopReason::Stop,
            })
        );
        let context_snapshots: Vec<u64> = received
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ContextUsageUpdated(usage) => Some(usage.estimated_input_tokens),
                _ => None,
            })
            .collect();
        assert_eq!(context_snapshots.len(), 2);
        assert!(context_snapshots[1] > context_snapshots[0]);

        command_tx.send(AgentCommand::RefreshContext).await.unwrap();
        assert!(matches!(
            event_rx.recv().await,
            Some(AgentEvent::ContextLimitsUpdated(_))
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(AgentEvent::ContextUsageUpdated(_))
        ));

        command_tx
            .send(AgentCommand::Shutdown)
            .await
            .expect("runtime should shut down");
        runtime_task.await.expect("runtime should join");
    }

    #[tokio::test]
    async fn response_usage_is_forwarded_when_provider_does_not_emit_an_event() {
        let mut step = final_step("done");
        step.response.usage = Some(Usage {
            input_tokens: Some(42),
            output_tokens: Some(7),
            total_tokens: Some(49),
            ..Usage::default()
        });
        let provider = ScriptedProvider::new(vec![step]);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(32);
        let runtime = AgentRuntime::new(provider);
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "usage".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    AgentEvent::UsageUpdated(usage) => Some(usage.input_tokens),
                    _ => None,
                })
                .count(),
            1
        );
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
    }

    #[tokio::test]
    async fn reasoning_effort_updates_next_turn_and_stays_stable_across_tool_rounds() {
        let root = unique_test_dir("reasoning-effort");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "hello\n").unwrap();
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(tool_call(
                        "read-call",
                        "read",
                        r#"{"path":"note.txt"}"#,
                    ))],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("first done"),
            final_step("second done"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(64);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history: Vec::new(),
                session: SessionMode::Disabled,
                reasoning_effort: Some("old-native".to_owned()),
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "read note".to_owned(),
            })
            .await
            .unwrap();
        collect_turn(&mut event_rx).await;
        command_tx
            .send(AgentCommand::SetReasoningEffort {
                effort: Some("new-native".to_owned()),
            })
            .await
            .unwrap();
        command_tx
            .send(AgentCommand::Submit {
                text: "continue".to_owned(),
            })
            .await
            .unwrap();
        collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].reasoning_effort.as_deref(), Some("old-native"));
        assert_eq!(requests[1].reasoning_effort.as_deref(), Some("old-native"));
        assert_eq!(requests[2].reasoning_effort.as_deref(), Some("new-native"));
        drop(requests);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn runtime_cancels_active_turn() {
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(32);
        let runtime = AgentRuntime::new(
            MockProvider::with_response("a long response")
                .with_chunk_size(1)
                .with_delay(Duration::from_millis(10)),
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "test".to_owned(),
            })
            .await
            .expect("runtime should be listening");
        assert_eq!(event_rx.recv().await, Some(AgentEvent::TurnStarted));

        command_tx
            .send(AgentCommand::Cancel)
            .await
            .expect("runtime should accept cancellation");

        let mut finished_reason = None;
        while let Some(event) = event_rx.recv().await {
            if let AgentEvent::TurnFinished { reason } = event {
                finished_reason = Some(reason);
                break;
            }
        }
        assert_eq!(finished_reason, Some(StopReason::Cancelled));

        command_tx
            .send(AgentCommand::Shutdown)
            .await
            .expect("runtime should shut down");
        runtime_task.await.expect("runtime should join");
    }

    #[tokio::test]
    async fn provider_task_panic_becomes_a_recoverable_runtime_error() {
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(64);
        let runtime = AgentRuntime::new(PanickingProvider);
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "panic".to_owned(),
            })
            .await
            .unwrap();

        let events = collect_turn(&mut event_rx).await;
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::Error(error) if error.message.contains("provider task failed"))
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Error
                }
            )
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_during_provider_request_cancels_and_joins_without_a_second_request() {
        let provider = BlockingProvider::default();
        let calls = Arc::clone(&provider.calls);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(64);
        let runtime = AgentRuntime::new(provider);
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "wait".to_owned(),
            })
            .await
            .unwrap();
        wait_for_call(&calls).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();

        let mut saw_cancelled = false;
        while let Some(event) = event_rx.recv().await {
            if matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Cancelled
                }
            ) {
                saw_cancelled = true;
                break;
            }
        }
        assert!(saw_cancelled);
        runtime_task.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn shutdown_during_compaction_cancels_without_starting_a_turn() {
        let provider = BlockingProvider::default();
        let calls = Arc::clone(&provider.calls);
        let root = unique_test_dir("agent-shutdown-compaction");
        std::fs::create_dir_all(&root).unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let session_path = handle.info().unwrap().path.clone();
        let initial_history = vec![
            ModelMessage::user("old one"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "old answer".to_owned(),
                }],
            },
            ModelMessage::user("old two"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "second answer".to_owned(),
                }],
            },
            ModelMessage::user("recent"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "recent answer".to_owned(),
                }],
            },
        ];
        for message in &initial_history {
            handle.append_message(message).unwrap();
        }
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(64);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx.send(AgentCommand::Compact).await.unwrap();

        loop {
            if matches!(
                event_rx.recv().await.unwrap(),
                AgentEvent::CompactionStarted { .. }
            ) {
                break;
            }
        }
        wait_for_call(&calls).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        while let Ok(event) = event_rx.try_recv() {
            assert!(!matches!(event, AgentEvent::TurnStarted));
        }
        drop(handle);
        let snapshot = crate::session::read_session(&session_path).unwrap();
        assert_eq!(snapshot.history.len(), 6);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn shutdown_during_bash_cancels_the_process_tree_before_runtime_exit() {
        let root = unique_test_dir("agent-shutdown-bash");
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("marker");
        let arguments = serde_json::json!({
            "command": format!(
                "while :; do printf x >> \"{}\"; sleep 0.02; done",
                marker.display()
            )
        })
        .to_string();
        let call = tool_call("long", "bash", &arguments);
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(call)],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("should not run"),
        ]);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "run command".to_owned(),
            })
            .await
            .unwrap();
        loop {
            if matches!(
                event_rx.recv().await.unwrap(),
                AgentEvent::ToolExecutionStarted { .. }
            ) {
                break;
            }
        }
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        let mut saw_cancelled = false;
        while let Some(event) = event_rx.recv().await {
            match event {
                AgentEvent::ToolExecutionFinished { result, .. } => {
                    saw_cancelled |= result.metadata.cancelled;
                }
                AgentEvent::TurnFinished {
                    reason: StopReason::Cancelled,
                } => break,
                _ => {}
            }
        }
        runtime_task.await.unwrap();
        assert!(saw_cancelled);
        if marker.exists() {
            let size = std::fs::metadata(&marker).unwrap().len();
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert_eq!(std::fs::metadata(&marker).unwrap().len(), size);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn tool_loop_replays_assistant_and_results_with_tools_every_time() {
        let root = unique_test_dir("agent-loop");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "alpha\nbeta\n").unwrap();
        let first_call = ModelToolCall {
            index: 0,
            call_id: Some("call-read".to_owned()),
            item_id: Some("item-read".to_owned()),
            name: Some("read".to_owned()),
            arguments: r#"{"path":"note.txt"}"#.to_owned(),
        };
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: vec![ModelEvent::AssistantTextDelta {
                    index: None,
                    text: "I will inspect it.".to_owned(),
                }],
                response: ModelResponse {
                    items: vec![
                        ModelAssistantItem::Text {
                            content: "I will inspect it.".to_owned(),
                        },
                        ModelAssistantItem::ToolCall(first_call.clone()),
                    ],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            ScriptedStep {
                events: vec![ModelEvent::AssistantTextDelta {
                    index: None,
                    text: "The file contains alpha and beta.".to_owned(),
                }],
                response: ModelResponse {
                    items: vec![ModelAssistantItem::Text {
                        content: "The file contains alpha and beta.".to_owned(),
                    }],
                    stop_reason: StopReason::Stop,
                    usage: None,
                },
            },
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "inspect note.txt".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::AssistantMessageStarted))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::AssistantMessageFinished { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolExecutionStarted { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::ToolExecutionFinished { result, .. } if result.model_content.contains("1 | alpha"))
        }));

        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].messages,
            vec![ModelMessage::user("inspect note.txt")]
        );
        assert_eq!(requests[0].tools.len(), 4);
        assert_eq!(requests[1].tools.len(), 4);
        assert_eq!(requests[1].messages.len(), 3);
        assert!(matches!(
            &requests[1].messages[1],
            ModelMessage::Assistant { items } if items == &vec![
                ModelAssistantItem::Text { content: "I will inspect it.".to_owned() },
                ModelAssistantItem::ToolCall(first_call),
            ]
        ));
        assert!(matches!(
            &requests[1].messages[2],
            ModelMessage::ToolResult { tool_call_id, tool_name, content }
                if tool_call_id == "call-read" && tool_name == "read" && content.contains("2 | beta")
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn tool_calls_in_one_response_execute_in_order() {
        let root = unique_test_dir("agent-order");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.txt"), "a").unwrap();
        std::fs::write(root.join("b.txt"), "b").unwrap();
        let calls = [
            tool_call("a", "read", r#"{"path":"a.txt"}"#),
            tool_call("b", "read", r#"{"path":"b.txt"}"#),
            tool_call("c", "bash", r#"{"command":"printf c"}"#),
        ];
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: calls
                        .iter()
                        .cloned()
                        .map(ModelAssistantItem::ToolCall)
                        .collect(),
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("done"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "inspect all".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        let order: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionStarted { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(order, ["a", "b", "c"]);
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        let requests = requests.lock().unwrap();
        assert!(matches!(
            &requests[1].messages[1],
            ModelMessage::Assistant { items } if items.len() == 3
        ));
        assert!(
            matches!(&requests[1].messages[2], ModelMessage::ToolResult { tool_call_id, .. } if tool_call_id == "a")
        );
        assert!(
            matches!(&requests[1].messages[3], ModelMessage::ToolResult { tool_call_id, .. } if tool_call_id == "b")
        );
        assert!(
            matches!(&requests[1].messages[4], ModelMessage::ToolResult { tool_call_id, .. } if tool_call_id == "c")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn invalid_arguments_and_unknown_tools_recover_in_later_rounds() {
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(tool_call(
                        "bad-json",
                        "read",
                        "{not json}",
                    ))],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(tool_call(
                        "unknown",
                        "totally_fake_tool",
                        "{}",
                    ))],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("recovered"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::new(provider);
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "recover".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events
            .iter()
            .all(|event| !matches!(event, AgentEvent::Error(_))));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(
            matches!(&requests[1].messages[2], ModelMessage::ToolResult { content, .. } if content.contains("invalid JSON"))
        );
        assert!(
            matches!(&requests[2].messages[4], ModelMessage::ToolResult { content, .. } if content.contains("unknown tool"))
        );
    }

    #[tokio::test]
    async fn cancellation_resolves_remaining_tool_calls_and_preserves_history() {
        let root = unique_test_dir("agent-cancel-batch");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("other.txt"), "other").unwrap();
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![
                        ModelAssistantItem::ToolCall(tool_call(
                            "long",
                            "bash",
                            r#"{"command":"sleep 5"}"#,
                        )),
                        ModelAssistantItem::ToolCall(tool_call(
                            "later",
                            "read",
                            r#"{"path":"other.txt"}"#,
                        )),
                    ],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("cancelled turn was replayable"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "stop the command".to_owned(),
            })
            .await
            .unwrap();
        loop {
            let event = event_rx.recv().await.unwrap();
            if matches!(
                event,
                AgentEvent::ToolExecutionStarted { ref call_id, .. } if call_id == "long"
            ) {
                command_tx.send(AgentCommand::Cancel).await.unwrap();
                break;
            }
        }
        let cancelled_events = collect_turn(&mut event_rx).await;
        assert!(cancelled_events.iter().any(|event| {
            matches!(event, AgentEvent::ToolExecutionFinished { call_id, result, .. } if call_id == "long" && result.metadata.cancelled)
        }));
        assert!(cancelled_events.iter().any(|event| {
            matches!(event, AgentEvent::ToolExecutionFinished { call_id, result, .. } if call_id == "later" && result.metadata.cancelled)
        }));
        assert!(cancelled_events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Cancelled
                }
            )
        }));

        tokio::task::yield_now().await;
        command_tx
            .send(AgentCommand::Submit {
                text: "continue".to_owned(),
            })
            .await
            .unwrap();
        let continued_events = collect_turn(&mut event_rx).await;
        assert!(continued_events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Stop
                }
            )
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages.len(), 5);
        assert!(
            matches!(&requests[1].messages[1], ModelMessage::Assistant { items } if items.len() == 2)
        );
        assert!(
            matches!(&requests[1].messages[2], ModelMessage::ToolResult { tool_call_id, .. } if tool_call_id == "long")
        );
        assert!(
            matches!(&requests[1].messages[3], ModelMessage::ToolResult { tool_call_id, .. } if tool_call_id == "later")
        );
        assert!(
            matches!(&requests[1].messages[4], ModelMessage::User { content } if content == "continue")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn forty_tool_rounds_complete_normally() {
        const TOOL_ROUNDS: usize = 40;

        let root = unique_test_dir("agent-long-tool-loop");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "ok\n").unwrap();
        let mut steps = (1..=TOOL_ROUNDS)
            .map(|round| ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(tool_call(
                        &format!("call-{round}"),
                        "read",
                        r#"{"path":"note.txt"}"#,
                    ))],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            })
            .collect::<Vec<_>>();
        steps.push(final_step("finished after 40 tool rounds"));
        let provider = ScriptedProvider::new(steps);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(512);
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "inspect repeatedly".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolExecutionStarted { .. }))
                .count(),
            TOOL_ROUNDS
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolExecutionFinished { .. }))
                .count(),
            TOOL_ROUNDS
        );
        assert!(events
            .iter()
            .all(|event| !matches!(event, AgentEvent::Error(_))));
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnFinished {
                reason: StopReason::Stop
            })
        ));
        assert_eq!(requests.lock().unwrap().len(), TOOL_ROUNDS + 1);

        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancellation_after_thirty_two_rounds_stops_before_another_tool_call() {
        const CANCEL_ROUND: usize = 40;

        let root = unique_test_dir("agent-cancel-long-tool-loop");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "ok\n").unwrap();
        let steps = (1..=64)
            .map(|round| {
                let (name, arguments) = if round == CANCEL_ROUND {
                    ("bash", r#"{"command":"sleep 5"}"#)
                } else {
                    ("read", r#"{"path":"note.txt"}"#)
                };
                ScriptedStep {
                    events: Vec::new(),
                    response: ModelResponse {
                        items: vec![ModelAssistantItem::ToolCall(tool_call(
                            &format!("call-{round}"),
                            name,
                            arguments,
                        ))],
                        stop_reason: StopReason::ToolCalls,
                        usage: None,
                    },
                }
            })
            .collect();
        let provider = ScriptedProvider::new(steps);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(512);
        let runtime = AgentRuntime::with_workspace_root(provider, &root).unwrap();
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "continue until cancelled".to_owned(),
            })
            .await
            .unwrap();

        let mut started = 0usize;
        loop {
            let event = event_rx.recv().await.unwrap();
            if matches!(event, AgentEvent::ToolExecutionStarted { .. }) {
                started += 1;
                if started == CANCEL_ROUND {
                    command_tx.send(AgentCommand::Cancel).await.unwrap();
                    break;
                }
            }
        }
        let events = collect_turn(&mut event_rx).await;

        assert_eq!(started, CANCEL_ROUND);
        assert!(events.iter().all(|event| {
            !matches!(event, AgentEvent::ToolExecutionStarted { .. })
                && !matches!(event, AgentEvent::Error(error) if error.message.contains("tool loop limit"))
        }));
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::ToolExecutionFinished { result, .. } if result.metadata.cancelled)
        }));
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnFinished {
                reason: StopReason::Cancelled
            })
        ));
        assert_eq!(requests.lock().unwrap().len(), CANCEL_ROUND);

        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn persisted_runtime_history_is_incremental_and_excludes_base_context() {
        let root = unique_test_dir("agent-session");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "alpha\n").unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let path = handle.info().unwrap().path.clone();
        let call = tool_call("read-call", "read", r#"{"path":"note.txt"}"#);
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(call.clone())],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("read complete"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: vec![ModelMessage::System {
                    content: "SECRET BASE CONTEXT".to_owned(),
                }],
                initial_history: Vec::new(),
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "inspect note".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::SessionChanged { info } if info.message_count == 1)
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("SECRET BASE CONTEXT"));
        assert_eq!(raw.lines().count(), 5);
        let snapshot = crate::session::read_session(&path).unwrap();
        assert_eq!(snapshot.history.len(), 4);
        let requests = requests.lock().unwrap();
        assert_eq!(
            requests[0].messages[0],
            ModelMessage::System {
                content: "SECRET BASE CONTEXT".to_owned(),
            }
        );
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[1].messages.len(), 4);
        std::mem::drop(handle);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn session_write_failure_stops_the_turn_instead_of_falling_back() {
        let root = unique_test_dir("agent-session-failure");
        std::fs::create_dir_all(&root).unwrap();
        let sessions_file = root.join("sessions-file");
        std::fs::write(&sessions_file, "not a directory").unwrap();
        let repository =
            crate::session::SessionRepository::new(&sessions_file, &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            MockProvider::with_response("should not run").with_delay(Duration::ZERO),
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history: Vec::new(),
                session: SessionMode::Enabled(handle),
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "cannot save".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::Error(error) if error.message.contains("session persistence failed"))
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Error
                }
            )
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancelled_persisted_turn_keeps_the_valid_user_history() {
        let root = unique_test_dir("agent-session-cancel");
        std::fs::create_dir_all(&root).unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let path = handle.info().unwrap().path.clone();
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            MockProvider::with_response("a long response")
                .with_chunk_size(1)
                .with_delay(Duration::from_millis(10)),
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history: Vec::new(),
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "cancel this".to_owned(),
            })
            .await
            .unwrap();
        loop {
            let event = event_rx.recv().await.unwrap();
            if matches!(event, AgentEvent::TurnStarted) {
                break;
            }
        }
        command_tx.send(AgentCommand::Cancel).await.unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Cancelled
                }
            )
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        let snapshot = crate::session::read_session(&path).unwrap();
        assert_eq!(snapshot.history, vec![ModelMessage::user("cancel this")]);
        drop(handle);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancellation_before_final_steering_injection_recovers_the_message() {
        let root = unique_test_dir("agent-session-cancel-steering");
        std::fs::create_dir_all(&root).unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let path = handle.info().unwrap().path.clone();
        let provider = Arc::new(ScriptedProvider::new(vec![
            final_step("done"),
            final_step("should not run"),
        ]));
        let requests = Arc::clone(&provider.requests);
        let steering = Arc::new(Mutex::new(SteeringState::default()));
        steering
            .lock()
            .unwrap()
            .accept("queued guidance".to_owned());
        let cancel = CancellationToken::new();
        let barrier = Arc::new(FinalResponseBarrier::default());
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let turn = tokio::spawn(run_turn(
            provider,
            Arc::new(ToolRegistry::new()),
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history: Vec::new(),
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
            ConversationHistory::default(),
            "initial".to_owned(),
            event_tx.clone(),
            cancel.clone(),
            false,
            Arc::clone(&steering),
            Some(Arc::clone(&barrier)),
            None,
        ));

        barrier.reached.notified().await;
        cancel_task(&cancel, Some(&steering));
        barrier.release.notify_one();

        let outcome = turn.await.unwrap();
        recover_pending_steering(&Some(steering), &event_tx).await;
        drop(event_tx);
        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            events.push(event);
        }

        assert_eq!(outcome.reason, StopReason::Cancelled);
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::SteeringMessagesRecovered { messages }
                if messages == &["queued guidance".to_owned()]
        )));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::SteeringMessageDelivered { .. })));
        assert!(!outcome
            .history
            .messages()
            .contains(&ModelMessage::user("queued guidance")));
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(!crate::session::read_session(&path)
            .unwrap()
            .history
            .contains(&ModelMessage::user("queued guidance")));

        drop(handle);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancellation_after_steering_reservation_recovers_the_message() {
        let root = unique_test_dir("agent-session-cancel-reserved-steering");
        std::fs::create_dir_all(&root).unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let path = handle.info().unwrap().path.clone();
        let provider = Arc::new(ScriptedProvider::new(vec![
            final_step("done"),
            final_step("should not run"),
        ]));
        let requests = Arc::clone(&provider.requests);
        let steering = Arc::new(Mutex::new(SteeringState::default()));
        steering
            .lock()
            .unwrap()
            .accept("queued guidance".to_owned());
        let cancel = CancellationToken::new();
        let barrier = Arc::new(SteeringInjectionBarrier::default());
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let turn = tokio::spawn(run_turn(
            provider,
            Arc::new(ToolRegistry::new()),
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history: Vec::new(),
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
            ConversationHistory::default(),
            "initial".to_owned(),
            event_tx.clone(),
            cancel.clone(),
            false,
            Arc::clone(&steering),
            None,
            Some(Arc::clone(&barrier)),
        ));

        barrier.reached.notified().await;
        cancel_task(&cancel, Some(&steering));
        barrier.release.notify_one();

        let outcome = turn.await.unwrap();
        recover_pending_steering(&Some(steering), &event_tx).await;
        drop(event_tx);
        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            events.push(event);
        }

        assert_eq!(outcome.reason, StopReason::Cancelled);
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::SteeringMessagesRecovered { messages }
                if messages == &["queued guidance".to_owned()]
        )));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::SteeringMessageDelivered { .. })));
        assert!(!outcome
            .history
            .messages()
            .contains(&ModelMessage::user("queued guidance")));
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(!crate::session::read_session(&path)
            .unwrap()
            .history
            .contains(&ModelMessage::user("queued guidance")));

        drop(handle);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn steering_near_completion_is_delivered_fifo_one_per_continuation() {
        let provider = ScriptedProvider::new(vec![
            final_step("initial answer"),
            final_step("after A"),
            final_step("after B"),
            final_step("after C"),
        ])
        .with_delay(Duration::from_millis(25));
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime_task = tokio::spawn(AgentRuntime::new(provider).run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "initial".to_owned(),
            })
            .await
            .unwrap();
        while !matches!(event_rx.recv().await, Some(AgentEvent::TurnStarted)) {}
        for text in ["A", "B", "C"] {
            command_tx
                .send(AgentCommand::Steer {
                    text: text.to_owned(),
                })
                .await
                .unwrap();
        }
        let events = collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    AgentEvent::SteeringMessageDelivered { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["A", "B", "C"]
        );
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        for (request, expected) in requests.iter().skip(1).zip(["A", "B", "C"]) {
            assert_eq!(request.messages.last(), Some(&ModelMessage::user(expected)));
        }
        for text in ["A", "B", "C"] {
            assert_eq!(
                requests[3]
                    .messages
                    .iter()
                    .filter(|message| **message == ModelMessage::user(text))
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn automatic_compaction_accounts_for_reserved_steering_before_delivery() {
        let initial_history = vec![
            ModelMessage::user("old request ".repeat(120)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "old answer ".repeat(120),
                }],
            },
            ModelMessage::user("recent request ".repeat(40)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "recent answer ".repeat(40),
                }],
            },
        ];
        let steering_text = "steering guidance ".repeat(20);
        let mut history_before_continuation = initial_history.clone();
        history_before_continuation.push(ModelMessage::user("initial"));
        history_before_continuation.push(ModelMessage::Assistant {
            items: vec![ModelAssistantItem::Text {
                content: "initial answer".to_owned(),
            }],
        });
        let registry = ToolRegistry::new();
        let request_without_steering = normal_request(
            &[],
            &ConversationHistory::new(None, history_before_continuation),
            &registry.definitions(),
        );
        let mut request_with_steering = request_without_steering.clone();
        request_with_steering
            .messages
            .push(ModelMessage::user(steering_text.clone()));
        let estimator = ConservativeTokenEstimator;
        let without_steering = estimator.estimate_request(&request_without_steering);
        let with_steering = estimator.estimate_request(&request_with_steering);
        let context_window = (100u64..without_steering.saturating_mul(2).saturating_add(200))
            .find(|context_window| {
                let budget = context_window.saturating_sub(100);
                let trigger = automatic_trigger(budget);
                without_steering <= trigger && with_steering > trigger
            })
            .expect("limits should separate the requests at the automatic trigger");
        let limits = ModelLimits {
            context_window: Some(context_window),
            max_output_tokens: Some(100),
        };
        let trigger = automatic_trigger(input_budget(limits).unwrap());
        assert!(without_steering <= trigger);
        assert!(with_steering > trigger);

        let provider = Arc::new(
            ScriptedProvider::new(vec![
                final_step("initial answer"),
                final_step("compaction summary"),
                final_step("continued"),
            ])
            .with_limits(limits),
        );
        let requests = Arc::clone(&provider.requests);
        let steering = Arc::new(Mutex::new(SteeringState::default()));
        assert!(steering
            .lock()
            .unwrap()
            .accept(steering_text.clone())
            .is_none());
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let turn = tokio::spawn(run_turn(
            provider,
            Arc::new(registry),
            AgentRuntimeConfig {
                tool_context: ToolContext::from_current_dir().unwrap(),
                base_messages: Vec::new(),
                initial_history: Vec::new(),
                session: SessionMode::Disabled,
                reasoning_effort: None,
            },
            ConversationHistory::new(None, initial_history),
            "initial".to_owned(),
            event_tx.clone(),
            CancellationToken::new(),
            true,
            steering,
            None,
            None,
        ));
        let outcome = turn.await.unwrap();
        drop(event_tx);
        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            events.push(event);
        }

        assert_eq!(outcome.reason, StopReason::Stop);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::ContextUsageUpdated(usage)
                    if usage.estimated_input_tokens == with_steering
            )
        }));
        let compaction_started = events
            .iter()
            .position(|event| matches!(event, AgentEvent::CompactionStarted { automatic: true }))
            .expect("reserved steering should trigger automatic compaction");
        let steering_delivered = events
            .iter()
            .position(|event| matches!(event, AgentEvent::SteeringMessageDelivered { .. }))
            .expect("steering should be delivered");
        assert!(compaction_started < steering_delivered);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(matches!(
            requests[1].messages.first(),
            Some(ModelMessage::System { content }) if content.contains("Return only the continuation summary")
        ));
        assert_eq!(
            requests[2]
                .messages
                .iter()
                .filter(|message| **message == ModelMessage::user(steering_text.clone()))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn steering_after_the_final_response_boundary_is_recovered() {
        let provider = ScriptedProvider::new(vec![final_step("done")]).with_limits(ModelLimits {
            context_window: Some(1_000),
            max_output_tokens: Some(100),
        });
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        // A single slot makes each context snapshot a deterministic barrier. The
        // final snapshot is emitted only after run_turn has returned.
        let (event_tx, mut event_rx) = mpsc::channel(1);
        let runtime_task = tokio::spawn(AgentRuntime::new(provider).run(command_rx, event_tx));

        assert!(matches!(
            event_rx.recv().await,
            Some(AgentEvent::ContextLimitsUpdated(_))
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(AgentEvent::ContextUsageUpdated(_))
        ));
        command_tx
            .send(AgentCommand::Submit {
                text: "initial".to_owned(),
            })
            .await
            .unwrap();

        let mut response_finished = false;
        loop {
            let event = event_rx.recv().await.expect("turn event");
            response_finished |= matches!(event, AgentEvent::AssistantMessageFinished { .. });
            if response_finished && matches!(event, AgentEvent::ContextLimitsUpdated(_)) {
                break;
            }
        }
        command_tx
            .send(AgentCommand::Steer {
                text: "too late".to_owned(),
            })
            .await
            .unwrap();

        let mut tail = Vec::new();
        while !tail
            .iter()
            .any(|event| matches!(event, AgentEvent::SteeringMessagesRecovered { .. }))
        {
            tail.push(event_rx.recv().await.expect("completion event"));
        }
        assert!(tail.iter().any(|event| matches!(
            event,
            AgentEvent::TurnFinished {
                reason: StopReason::Stop
            }
        )));
        assert!(tail.iter().any(|event| matches!(
            event,
            AgentEvent::SteeringMessagesRecovered { messages }
                if messages == &["too late".to_owned()]
        )));
        assert!(!tail
            .iter()
            .any(|event| matches!(event, AgentEvent::Error(_))));
        assert_eq!(requests.lock().unwrap().len(), 1);

        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
    }

    #[tokio::test]
    async fn steering_waits_for_an_entire_tool_batch() {
        let root = unique_test_dir("steering-tool-batch");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "content").unwrap();
        let calls = ["call-a", "call-b", "call-c"]
            .into_iter()
            .map(|id| tool_call(id, "read", r#"{"path":"note.txt"}"#))
            .collect::<Vec<_>>();
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: calls
                        .iter()
                        .cloned()
                        .map(ModelAssistantItem::ToolCall)
                        .collect(),
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("finished"),
        ])
        .with_delay(Duration::from_millis(25));
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_context(provider, ToolContext::new(&root).unwrap());
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "inspect".to_owned(),
            })
            .await
            .unwrap();
        while !matches!(event_rx.recv().await, Some(AgentEvent::TurnStarted)) {}
        command_tx
            .send(AgentCommand::Steer {
                text: "also add tests".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        let continuation = &requests[1].messages;
        let steering_index = continuation
            .iter()
            .position(|message| *message == ModelMessage::user("also add tests"))
            .unwrap();
        let tool_result_indices = continuation
            .iter()
            .enumerate()
            .filter_map(|(index, message)| {
                matches!(message, ModelMessage::ToolResult { .. }).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(tool_result_indices.len(), 3);
        assert!(tool_result_indices
            .iter()
            .all(|index| *index < steering_index));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancellation_recovers_pending_steering_messages() {
        let provider = BlockingProvider::default();
        let calls = Arc::clone(&provider.calls);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime_task = tokio::spawn(AgentRuntime::new(provider).run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "initial".to_owned(),
            })
            .await
            .unwrap();
        while !matches!(event_rx.recv().await, Some(AgentEvent::TurnStarted)) {}
        wait_for_call(&calls).await;
        command_tx
            .send(AgentCommand::Steer {
                text: "do not lose this".to_owned(),
            })
            .await
            .unwrap();
        command_tx.send(AgentCommand::Cancel).await.unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::SteeringMessagesRecovered { messages }
                    if messages == &["do not lose this".to_owned()]
            )
        }));

        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
    }

    #[tokio::test]
    async fn conversation_history_survives_the_next_user_turn() {
        let provider = ScriptedProvider::new(vec![final_step("alpha"), final_step("remembered")]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::new(provider);
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "remember alpha".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        tokio::task::yield_now().await;
        command_tx
            .send(AgentCommand::Submit {
                text: "what did I say?".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1].messages,
            vec![
                ModelMessage::user("remember alpha"),
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::Text {
                        content: "alpha".to_owned(),
                    }],
                },
                ModelMessage::user("what did I say?"),
            ]
        );
    }

    #[tokio::test]
    async fn base_messages_are_replayed_once_for_tool_continuations_and_later_turns() {
        let root = unique_test_dir("agent-base-context");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "context").unwrap();
        let call = tool_call("read-call", "read", r#"{"path":"note.txt"}"#);
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(call.clone())],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("read complete"),
            final_step("remembered"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let base = ModelMessage::System {
            content: "base instructions".to_owned(),
        };
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: vec![base.clone()],
                initial_history: Vec::new(),
                session: SessionMode::Disabled,
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx
            .send(AgentCommand::Submit {
                text: "inspect note".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        command_tx
            .send(AgentCommand::Submit {
                text: "what happened?".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].messages[0], base);
        assert_eq!(requests[0].messages[1], ModelMessage::user("inspect note"));
        assert_eq!(requests[1].messages[0], base);
        assert!(matches!(
            &requests[1].messages[1],
            ModelMessage::User { content } if content == "inspect note"
        ));
        assert!(matches!(
            &requests[1].messages[2],
            ModelMessage::Assistant { items } if items == &vec![ModelAssistantItem::ToolCall(call)]
        ));
        assert!(matches!(
            &requests[1].messages[3],
            ModelMessage::ToolResult { tool_call_id, .. } if tool_call_id == "read-call"
        ));
        assert_eq!(requests[2].messages[0], base);
        assert_eq!(
            requests[2].messages.last(),
            Some(&ModelMessage::user("what happened?"))
        );
        for request in requests.iter() {
            assert_eq!(
                request
                    .messages
                    .iter()
                    .filter(|message| matches!(message, ModelMessage::System { .. }))
                    .count(),
                1
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compaction_never_summarizes_work_after_a_retained_user_request() {
        let history = ConversationHistory::new(
            None,
            vec![
                ModelMessage::user("old"),
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::Text {
                        content: "old answer".to_owned(),
                    }],
                },
                ModelMessage::user("recent"),
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::Text {
                        content: "recent answer".to_owned(),
                    }],
                },
                ModelMessage::user("current request"),
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::ToolCall(tool_call("a", "read", "{}"))],
                },
                ModelMessage::ToolResult {
                    tool_call_id: "a".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "first result".to_owned(),
                },
                ModelMessage::Assistant {
                    items: vec![ModelAssistantItem::ToolCall(tool_call("b", "read", "{}"))],
                },
                ModelMessage::ToolResult {
                    tool_call_id: "b".to_owned(),
                    tool_name: "read".to_owned(),
                    content: "latest result".to_owned(),
                },
            ],
        );
        let current_turn = history.messages()[4..].to_vec();
        let (prefix, retained) = select_compaction_prefix(&history, &[], &[], 1, true)
            .expect("a completed prior turn should be compactable");

        assert!(prefix.iter().all(|message| !current_turn.contains(message)));
        assert!(retained.ends_with(&current_turn));
    }

    #[test]
    fn compaction_rejects_length_limited_summary() {
        let response = ModelResponse {
            items: vec![ModelAssistantItem::Text {
                content: "partial summary".to_owned(),
            }],
            stop_reason: StopReason::Length,
            usage: None,
        };

        assert!(extract_summary(&response).is_err());
    }

    #[test]
    fn compaction_rejects_incomplete_summary() {
        let response = ModelResponse {
            items: vec![ModelAssistantItem::Text {
                content: "partial summary".to_owned(),
            }],
            stop_reason: StopReason::Incomplete,
            usage: None,
        };

        assert!(extract_summary(&response).is_err());
    }

    #[tokio::test]
    async fn automatic_compaction_keeps_summary_developer_and_current_turn() {
        let mut initial_history = Vec::new();
        for index in 0..3 {
            initial_history.push(ModelMessage::user(format!(
                "old request {index} {}",
                "x".repeat(300)
            )));
            initial_history.push(ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: format!("old answer {index} {}", "y".repeat(300)),
                }],
            });
        }
        let provider = ScriptedProvider::new(vec![final_step("summary"), final_step("done")])
            .with_limits(ModelLimits {
                context_window: Some(1_350),
                max_output_tokens: Some(100),
            });
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::from_current_dir().unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Disabled,
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "fix foo.rs exactly".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::CompactionStarted { automatic: true })));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::CompactionFinished {
                automatic: true,
                ..
            }
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::TurnFinished { .. }))
                .count(),
            1
        );
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].tools.is_empty());
        assert!(matches!(
            requests[0].messages.first(),
            Some(ModelMessage::System { content }) if content.contains("Return only the continuation summary")
        ));
        assert!(requests[1].tools.len() >= 4);
        assert!(matches!(
            requests[1].messages.first(),
            Some(ModelMessage::Developer { content }) if content.contains("summary")
        ));
        assert!(requests[1].messages.iter().any(|message| {
            matches!(message, ModelMessage::User { content } if content == "fix foo.rs exactly")
        }));
        assert!(!requests[1].messages.iter().any(
            |message| matches!(message, ModelMessage::User { content } if content == "summary")
        ));
    }

    #[tokio::test]
    async fn persistent_compaction_appends_checkpoint_without_deleting_messages() {
        let root = unique_test_dir("agent-compaction-session");
        std::fs::create_dir_all(&root).unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let mut initial_history = Vec::new();
        for index in 0..3 {
            initial_history.push(ModelMessage::user(format!(
                "old request {index} {}",
                "x".repeat(300)
            )));
            initial_history.push(ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: format!("old answer {index} {}", "y".repeat(300)),
                }],
            });
        }
        for message in &initial_history {
            handle.append_message(message).unwrap();
        }
        let path = handle.info().unwrap().path.clone();
        let provider = ScriptedProvider::new(vec![final_step("summary"), final_step("done")])
            .with_limits(ModelLimits {
                context_window: Some(1_350),
                max_output_tokens: Some(100),
            });
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history: initial_history.clone(),
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "current request".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::CompactionFinished { .. })));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        drop(handle);

        let snapshot = crate::session::read_session(&path).unwrap();
        assert_eq!(snapshot.transcript.len(), initial_history.len() + 2);
        assert!(snapshot
            .transcript
            .iter()
            .any(|message| matches!(message, ModelMessage::User { content } if content == "current request")));
        assert!(snapshot
            .history
            .iter()
            .any(|message| matches!(message, ModelMessage::Developer { .. })));
        assert!(snapshot
            .transcript
            .iter()
            .any(|message| matches!(message, ModelMessage::User { content } if content.starts_with("old request 0"))));
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"type\":\"compaction\""));
        assert!(raw.contains("old request 0"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn non_reducing_compaction_is_not_persisted() {
        let root = unique_test_dir("agent-non-reducing-compaction");
        std::fs::create_dir_all(&root).unwrap();
        let repository =
            crate::session::SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let initial_history = vec![
            ModelMessage::user("first request ".repeat(20)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "first answer ".repeat(20),
                }],
            },
            ModelMessage::user("second request ".repeat(20)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "second answer ".repeat(20),
                }],
            },
            ModelMessage::user("recent request ".repeat(20)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "recent answer ".repeat(20),
                }],
            },
        ];
        for message in &initial_history {
            handle.append_message(message).unwrap();
        }
        let path = handle.info().unwrap().path.clone();
        let provider = ScriptedProvider::new(vec![final_step(&"verbose summary ".repeat(500))]);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(32);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::new(&root).unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Enabled(handle.clone()),
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx.send(AgentCommand::Compact).await.unwrap();
        let failure = loop {
            if let AgentEvent::CompactionFailed { message } = event_rx.recv().await.unwrap() {
                break message;
            }
        };
        assert!(failure.contains("did not reduce context"));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        drop(handle);

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("\"type\":\"compaction\""));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn manual_compaction_does_not_create_a_turn() {
        let initial_history = vec![
            ModelMessage::user("old one ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "answer one ".repeat(100),
                }],
            },
            ModelMessage::user("old two ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "answer two ".repeat(100),
                }],
            },
            ModelMessage::user("recent ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "answer recent ".repeat(100),
                }],
            },
        ];
        let provider = ScriptedProvider::new(vec![final_step("manual summary")]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::from_current_dir().unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Disabled,
                reasoning_effort: Some("high-native".to_owned()),
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx.send(AgentCommand::Compact).await.unwrap();
        let mut received = Vec::new();
        loop {
            let event = event_rx.recv().await.unwrap();
            let finished = matches!(event, AgentEvent::CompactionFinished { .. });
            received.push(event);
            if finished {
                break;
            }
        }
        assert!(received.iter().all(|event| !matches!(
            event,
            AgentEvent::TurnStarted | AgentEvent::TurnFinished { .. }
        )));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].tools.is_empty());
        assert_eq!(requests[0].reasoning_effort, None);
    }

    #[tokio::test]
    async fn repeated_compaction_replaces_summary_after_reincluding_previous_state() {
        let initial_history = vec![
            ModelMessage::user("first ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "first answer ".repeat(100),
                }],
            },
            ModelMessage::user("second ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "second answer ".repeat(100),
                }],
            },
            ModelMessage::user("third ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "third answer ".repeat(100),
                }],
            },
        ];
        let provider = ScriptedProvider::new(vec![
            final_step("summary A"),
            final_step("continued"),
            final_step("summary B"),
            final_step("finished"),
        ]);
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::from_current_dir().unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Disabled,
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

        command_tx.send(AgentCommand::Compact).await.unwrap();
        wait_for_compaction_finished(&mut event_rx).await;
        command_tx
            .send(AgentCommand::Submit {
                text: "new work".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Compact).await.unwrap();
        wait_for_compaction_finished(&mut event_rx).await;
        command_tx
            .send(AgentCommand::Submit {
                text: "final work".to_owned(),
            })
            .await
            .unwrap();
        let _ = collect_turn(&mut event_rx).await;
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests[2].messages.iter().any(|message| {
            matches!(message, ModelMessage::Developer { content } if content.contains("summary A"))
        }));
        let summaries: Vec<&str> = requests[3]
            .messages
            .iter()
            .filter_map(|message| match message {
                ModelMessage::Developer { content } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].contains("summary B"));
        assert!(!summaries[0].contains("summary A"));
    }

    #[tokio::test]
    async fn emergency_compaction_recovers_after_a_large_current_turn_tool_result() {
        let initial_history = vec![
            ModelMessage::user("previous request ".repeat(100)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "previous answer ".repeat(100),
                }],
            },
        ];
        let provider = ScriptedProvider::new(vec![
            ScriptedStep {
                events: Vec::new(),
                response: ModelResponse {
                    items: vec![ModelAssistantItem::ToolCall(tool_call(
                        "large",
                        "bash",
                        r#"{"command":"printf '%*s' 20000 '' | tr ' ' x"}"#,
                    ))],
                    stop_reason: StopReason::ToolCalls,
                    usage: None,
                },
            },
            final_step("emergency summary"),
            final_step("done after recovery"),
        ])
        .with_limits(ModelLimits {
            context_window: Some(1_000),
            max_output_tokens: Some(100),
        })
        .with_overflow_after_tool_result_once();
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::from_current_dir().unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Disabled,
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "inspect a large result".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events
            .iter()
            .any(|event| { matches!(event, AgentEvent::CompactionStarted { automatic: true }) }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Stop
                }
            )
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests[1].messages.iter().any(|message| {
            matches!(message, ModelMessage::ToolResult { content, .. } if content.len() > 1_000)
        }));
        assert!(matches!(
            requests[2].messages.first(),
            Some(ModelMessage::System { content }) if content.contains("Return only the continuation summary")
        ));
        assert!(matches!(
            requests[3].messages.first(),
            Some(ModelMessage::Developer { content }) if content.contains("emergency summary")
        ));
    }

    #[tokio::test]
    async fn context_overflow_is_retried_once_after_emergency_compaction() {
        let initial_history = vec![
            ModelMessage::user("old one ".repeat(20)),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "answer one ".repeat(20),
                }],
            },
            ModelMessage::user("old two"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "answer two".to_owned(),
                }],
            },
            ModelMessage::user("recent"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "answer recent".to_owned(),
                }],
            },
        ];
        let provider =
            ScriptedProvider::new(vec![final_step("emergency summary"), final_step("done")])
                .with_limits(ModelLimits {
                    context_window: Some(1_000),
                    max_output_tokens: Some(100),
                })
                .with_overflow_once();
        let requests = Arc::clone(&provider.requests);
        let (command_tx, command_rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(128);
        let runtime = AgentRuntime::with_config(
            provider,
            AgentRuntimeConfig {
                tool_context: ToolContext::from_current_dir().unwrap(),
                base_messages: Vec::new(),
                initial_history,
                session: SessionMode::Disabled,
                reasoning_effort: None,
            },
        );
        let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
        command_tx
            .send(AgentCommand::Submit {
                text: "continue".to_owned(),
            })
            .await
            .unwrap();
        let events = collect_turn(&mut event_rx).await;
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::CompactionStarted { .. })));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::TurnFinished {
                    reason: StopReason::Stop
                }
            )
        }));
        command_tx.send(AgentCommand::Shutdown).await.unwrap();
        runtime_task.await.unwrap();
        assert_eq!(requests.lock().unwrap().len(), 3);
    }

    struct PanickingProvider;

    #[async_trait::async_trait]
    impl ModelProvider for PanickingProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _events: mpsc::Sender<ModelEvent>,
            _cancel: CancellationToken,
        ) -> Result<ModelResponse, ProviderError> {
            panic!("intentional provider task panic");
        }
    }

    #[derive(Clone, Default)]
    struct BlockingProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for BlockingProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _events: mpsc::Sender<ModelEvent>,
            cancel: CancellationToken,
        ) -> Result<ModelResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            cancel.cancelled().await;
            Err(ProviderError::Cancelled)
        }
    }

    async fn wait_for_call(calls: &AtomicUsize) {
        for _ in 0..100 {
            if calls.load(Ordering::SeqCst) > 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("provider request did not start");
    }

    #[derive(Clone)]
    struct ScriptedProvider {
        steps: Arc<Mutex<VecDeque<ScriptedStep>>>,
        requests: Arc<Mutex<Vec<ModelRequest>>>,
        limits: ModelLimits,
        delay: Duration,
        overflow_once: Arc<Mutex<bool>>,
        overflow_after_tool_result_once: Arc<Mutex<bool>>,
    }

    struct ScriptedStep {
        events: Vec<ModelEvent>,
        response: ModelResponse,
    }

    impl ScriptedProvider {
        fn new(steps: Vec<ScriptedStep>) -> Self {
            Self {
                steps: Arc::new(Mutex::new(steps.into_iter().collect())),
                requests: Arc::new(Mutex::new(Vec::new())),
                limits: ModelLimits::default(),
                delay: Duration::ZERO,
                overflow_once: Arc::new(Mutex::new(false)),
                overflow_after_tool_result_once: Arc::new(Mutex::new(false)),
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }

        fn with_limits(mut self, limits: ModelLimits) -> Self {
            self.limits = limits;
            self
        }

        fn with_overflow_once(self) -> Self {
            *self.overflow_once.lock().unwrap() = true;
            self
        }

        fn with_overflow_after_tool_result_once(self) -> Self {
            *self.overflow_after_tool_result_once.lock().unwrap() = true;
            self
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for ScriptedProvider {
        fn limits(&self) -> ModelLimits {
            self.limits
        }

        async fn stream(
            &self,
            request: ModelRequest,
            events: mpsc::Sender<ModelEvent>,
            cancel: CancellationToken,
        ) -> Result<ModelResponse, ProviderError> {
            let has_tools = !request.tools.is_empty();
            let has_tool_result = request
                .messages
                .iter()
                .any(|message| matches!(message, ModelMessage::ToolResult { .. }));
            self.requests.lock().unwrap().push(request);
            if !self.delay.is_zero() {
                tokio::select! {
                    _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                    _ = tokio::time::sleep(self.delay) => {}
                }
            }
            let overflow = (has_tools && *self.overflow_once.lock().unwrap())
                || (has_tool_result && *self.overflow_after_tool_result_once.lock().unwrap());
            if overflow {
                if has_tools && *self.overflow_once.lock().unwrap() {
                    *self.overflow_once.lock().unwrap() = false;
                }
                if has_tool_result && *self.overflow_after_tool_result_once.lock().unwrap() {
                    *self.overflow_after_tool_result_once.lock().unwrap() = false;
                }
                return Err(ProviderError::ContextOverflow {
                    message: "scripted context overflow".to_owned(),
                });
            }
            let step = self
                .steps
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted step");
            for event in step.events {
                tokio::select! {
                    _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                    result = events.send(event) => result.map_err(|_| ProviderError::Failed {
                        message: "scripted event receiver closed".to_owned(),
                    })?,
                }
            }
            Ok(step.response)
        }
    }

    fn tool_call(call_id: &str, name: &str, arguments: &str) -> ModelToolCall {
        ModelToolCall {
            index: 0,
            call_id: Some(call_id.to_owned()),
            item_id: None,
            name: Some(name.to_owned()),
            arguments: arguments.to_owned(),
        }
    }

    fn final_step(text: &str) -> ScriptedStep {
        ScriptedStep {
            events: vec![ModelEvent::AssistantTextDelta {
                index: None,
                text: text.to_owned(),
            }],
            response: ModelResponse {
                items: vec![ModelAssistantItem::Text {
                    content: text.to_owned(),
                }],
                stop_reason: StopReason::Stop,
                usage: None,
            },
        }
    }

    async fn wait_for_compaction_finished(event_rx: &mut mpsc::Receiver<AgentEvent>) {
        loop {
            if matches!(
                event_rx.recv().await.expect("compaction event"),
                AgentEvent::CompactionFinished { .. }
            ) {
                return;
            }
        }
    }

    async fn collect_turn(event_rx: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        loop {
            let event = event_rx.recv().await.expect("turn event");
            let finished = matches!(event, AgentEvent::TurnFinished { .. });
            events.push(event);
            if finished {
                return events;
            }
        }
    }

    fn unique_test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
