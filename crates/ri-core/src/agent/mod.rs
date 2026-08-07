use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::config::ModelRef;
use crate::model::{ModelEvent, ModelProvider, ModelRequest, ProviderError, StopReason, Usage};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentCommand {
    Submit { text: String },
    Cancel,
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
    AssistantTextDelta {
        text: String,
    },
    AssistantThinkingDelta {
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    UsageUpdated(Usage),
    ModelChanged(ModelRef),
    TurnFinished {
        reason: StopReason,
    },
    Error(AgentError),
}

struct ActiveTurn {
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

pub struct AgentRuntime<P> {
    provider: Arc<P>,
}

impl<P> AgentRuntime<P>
where
    P: ModelProvider,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }

    pub async fn run(
        self,
        mut commands: mpsc::Receiver<AgentCommand>,
        events: mpsc::Sender<AgentEvent>,
    ) {
        let mut active: Option<ActiveTurn> = None;

        loop {
            if active.is_some() {
                let mut task_finished = false;
                if let Some(active_turn) = active.as_mut() {
                    tokio::select! {
                        result = &mut active_turn.task => {
                            task_finished = true;
                            if let Err(error) = result {
                                let _ = events.send(AgentEvent::Error(AgentError::new(
                                    format!("agent turn task failed: {error}"),
                                ))).await;
                            }
                        }
                        command = commands.recv() => {
                            match command {
                                Some(AgentCommand::Cancel) => active_turn.cancel.cancel(),
                                Some(AgentCommand::Submit { .. }) => {
                                    let _ = events.send(AgentEvent::Error(AgentError::new(
                                        "a turn is already active",
                                    ))).await;
                                }
                                Some(AgentCommand::Shutdown) | None => {
                                    active_turn.cancel.cancel();
                                    let _ = (&mut active_turn.task).await;
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
                        let turn_events = events.clone();
                        let turn_cancel = cancel.clone();
                        let task = tokio::spawn(async move {
                            run_turn(provider, text, turn_events, turn_cancel).await;
                        });
                        active = Some(ActiveTurn { cancel, task });
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

async fn run_turn<P>(
    provider: Arc<P>,
    text: String,
    events: mpsc::Sender<AgentEvent>,
    cancel: CancellationToken,
) where
    P: ModelProvider,
{
    if events.send(AgentEvent::TurnStarted).await.is_err() {
        return;
    }

    let (model_event_tx, mut model_event_rx) = mpsc::channel(64);
    let request = ModelRequest::single_user(text);
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
                let _ = events.send(AgentEvent::TurnFinished {
                    reason: StopReason::Cancelled,
                }).await;
                return;
            }
            model_event = model_event_rx.recv() => {
                let Some(model_event) = model_event else { continue };
                let event = agent_event_from_model(model_event);
                if events.send(event).await.is_err() {
                    provider_task.abort();
                    return;
                }
            }
            result = &mut provider_task => {
                break result;
            }
        }
    };

    while let Some(model_event) = model_event_rx.recv().await {
        if events
            .send(agent_event_from_model(model_event))
            .await
            .is_err()
        {
            return;
        }
    }

    match provider_result {
        Ok(Ok(response)) => {
            let _ = events
                .send(AgentEvent::TurnFinished {
                    reason: response.stop_reason,
                })
                .await;
        }
        Ok(Err(ProviderError::Cancelled)) => {
            let _ = events
                .send(AgentEvent::TurnFinished {
                    reason: StopReason::Cancelled,
                })
                .await;
        }
        Ok(Err(error)) => {
            let _ = events
                .send(AgentEvent::Error(AgentError::new(error.to_string())))
                .await;
            let _ = events
                .send(AgentEvent::TurnFinished {
                    reason: StopReason::Error,
                })
                .await;
        }
        Err(error) => {
            let _ = events
                .send(AgentEvent::Error(AgentError::new(format!(
                    "provider task failed: {error}"
                ))))
                .await;
            let _ = events
                .send(AgentEvent::TurnFinished {
                    reason: StopReason::Error,
                })
                .await;
        }
    }
}

fn agent_event_from_model(event: ModelEvent) -> AgentEvent {
    match event {
        ModelEvent::AssistantTextDelta { text } => AgentEvent::AssistantTextDelta { text },
        ModelEvent::AssistantThinkingDelta { text } => AgentEvent::AssistantThinkingDelta { text },
        ModelEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        } => AgentEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        },
        ModelEvent::UsageUpdated(usage) => AgentEvent::UsageUpdated(usage),
    }
}

#[cfg(test)]
mod tests {
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
                    AgentEvent::AssistantTextDelta { text } => Some(text.as_str()),
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

        command_tx
            .send(AgentCommand::Shutdown)
            .await
            .expect("runtime should shut down");
        runtime_task.await.expect("runtime should join");
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
}
