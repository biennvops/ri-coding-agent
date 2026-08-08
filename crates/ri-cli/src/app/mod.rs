use std::io::{self, Write};

use anyhow::{anyhow, bail, Context, Result};
use crossterm::event::{Event, EventStream, KeyEventKind};
use futures_util::StreamExt;
use ri_core::{
    config::load_default_models, AgentCommand, AgentEvent, AgentRuntime, AppState,
    ConfiguredProvider, ModelCatalog, ModelRef, ResolvedModel, StopReason,
};
use tokio::sync::mpsc;

use crate::input::{self, Action, VisualLayout};
use crate::render;
use crate::terminal::TerminalGuard;

const COMMAND_CHANNEL_CAPACITY: usize = 16;
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub print_prompt: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub show_help: bool,
}

impl Options {
    pub fn parse<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut options = Self::default();
        let mut args = args.into_iter();

        while let Some(argument) = args.next() {
            match argument.as_str() {
                "-p" | "--print" => {
                    let prompt = args
                        .next()
                        .ok_or_else(|| anyhow!("{argument} requires a prompt"))?;
                    options.print_prompt = Some(prompt);
                }
                "--provider" => {
                    options.provider = Some(
                        args.next()
                            .ok_or_else(|| anyhow!("--provider requires a provider id"))?,
                    );
                }
                "--model" => {
                    options.model = Some(
                        args.next()
                            .ok_or_else(|| anyhow!("--model requires a model id"))?,
                    );
                }
                "-h" | "--help" => options.show_help = true,
                unknown => bail!("unknown argument: {unknown}"),
            }
        }

        Ok(options)
    }

    pub fn print_help() {
        println!(
            "ri — a small Rust coding agent\n\n\
             Usage:\n  ri                              start the interactive TUI\n  ri -p <prompt>                  run one prompt without the TUI\n  ri --provider <id> --model <id> select a configured model\n  ri --help                       show this help"
        );
    }
}

pub async fn run(options: Options) -> Result<()> {
    let setup = ModelSetup::load(&options)?;
    if let Some(prompt) = options.print_prompt {
        run_print(prompt, setup).await
    } else {
        run_tui(setup).await
    }
}

struct ModelSetup {
    provider: ConfiguredProvider,
    catalog: Option<ModelCatalog>,
    selected: Option<ResolvedModel>,
}

impl ModelSetup {
    fn load(options: &Options) -> Result<Self> {
        let catalog = load_default_models().context("could not load models.json")?;
        if let Some(catalog) = catalog {
            for warning in catalog.warnings() {
                eprintln!("ri: warning: {}: {}", warning.path, warning.message);
            }
            let selected = catalog
                .resolve(options.provider.as_deref(), options.model.as_deref())
                .context("could not select configured model")?;
            let provider = ConfiguredProvider::openai(selected.clone())
                .map_err(|error| anyhow!(error.to_string()))?;
            return Ok(Self {
                provider,
                catalog: Some(catalog),
                selected: Some(selected),
            });
        }

        if options.provider.is_some() || options.model.is_some() {
            bail!("no models.json found; create ~/.ri/agent/models.json or set RI_MODELS_FILE");
        }

        Ok(Self {
            provider: ConfiguredProvider::mock(),
            catalog: None,
            selected: None,
        })
    }

    fn model_ref(&self) -> ModelRef {
        self.selected
            .as_ref()
            .map(|model| model.model_ref.clone())
            .unwrap_or_else(|| self.provider.model_ref())
    }
}

async fn run_print(prompt: String, setup: ModelSetup) -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::new(setup.provider.clone());
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    command_tx
        .send(AgentCommand::Submit { text: prompt })
        .await
        .context("could not start the mock agent")?;

    let mut state = AppState::new();
    state.reduce(AgentEvent::ModelChanged(setup.model_ref()));
    let mut turn_reason = None;
    let mut output = io::stdout();
    let mut error_message = None;

    while let Some(event) = event_rx.recv().await {
        match &event {
            AgentEvent::AssistantTextDelta { text, .. } => {
                print_and_flush(&mut output, text)?;
            }
            AgentEvent::Error(error) => {
                error_message = Some(error.message.clone());
                eprintln!("ri: {}", error.message);
            }
            AgentEvent::TurnFinished { reason } => turn_reason = Some(reason.clone()),
            AgentEvent::TurnStarted
            | AgentEvent::AssistantTextItem { .. }
            | AgentEvent::AssistantThinkingDelta { .. }
            | AgentEvent::AssistantThinkingContentDelta { .. }
            | AgentEvent::AssistantThinkingItem { .. }
            | AgentEvent::ToolCallDelta { .. }
            | AgentEvent::UsageUpdated(_)
            | AgentEvent::ModelChanged(_) => {}
        }
        let finished = matches!(event, AgentEvent::TurnFinished { .. });
        state.reduce(event);
        if finished {
            break;
        }
    }

    if turn_reason.is_none() {
        let _ = command_tx.send(AgentCommand::Shutdown).await;
        let _ = runtime_task.await;
        bail!("agent stopped before finishing the turn");
    }

    writeln!(output)?;
    command_tx
        .send(AgentCommand::Shutdown)
        .await
        .context("could not stop the mock agent")?;
    runtime_task
        .await
        .map_err(|error| anyhow!("agent task failed: {error}"))?;

    if let Some(error) = error_message {
        bail!(error);
    }
    if turn_reason == Some(StopReason::Error) {
        bail!("agent turn failed");
    }

    Ok(())
}

async fn run_tui(mut setup: ModelSetup) -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::new(setup.provider.clone());
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    let mut terminal = TerminalGuard::new().context("could not initialize terminal")?;
    let mut state = AppState::new();
    state.reduce(AgentEvent::ModelChanged(setup.model_ref()));
    let tui_result = run_tui_loop(
        &mut terminal,
        &mut state,
        &command_tx,
        &mut event_rx,
        &mut setup,
    )
    .await;
    drop(terminal);

    let shutdown_result = command_tx.send(AgentCommand::Shutdown).await;
    let runtime_result = runtime_task
        .await
        .map_err(|error| anyhow!("agent task failed: {error}"));

    tui_result?;
    shutdown_result.context("could not stop the agent")?;
    runtime_result?;
    Ok(())
}

async fn run_tui_loop(
    terminal: &mut TerminalGuard,
    state: &mut AppState,
    command_tx: &mpsc::Sender<AgentCommand>,
    event_rx: &mut mpsc::Receiver<AgentEvent>,
    setup: &mut ModelSetup,
) -> Result<()> {
    let mut dirty = true;
    let mut scroll_from_bottom = 0usize;
    let mut editor_width = terminal
        .terminal_mut()
        .size()
        .context("could not determine terminal size")?
        .width
        .saturating_sub(2)
        .max(1) as usize;
    let mut preferred_column = None;
    let mut terminal_events = EventStream::new();
    let mut exit = false;

    while !exit {
        if dirty {
            render::draw(terminal, state, scroll_from_bottom)
                .context("could not render terminal")?;
            dirty = false;
        }

        tokio::select! {
            terminal_event = terminal_events.next() => {
                let terminal_event = terminal_event
                    .ok_or_else(|| anyhow!("terminal event stream disconnected"))?
                    .context("could not read terminal event")?;
                match terminal_event {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if let Some(action) = input::action_for(key) {
                            dirty = true;
                            if !matches!(action, Action::Up | Action::Down) {
                                preferred_column = None;
                            }
                            match action {
                                Action::Submit => {
                                    if let Some(argument) = model_command(state.input()) {
                                        state.take_input();
                                        handle_model_command(state, setup, argument.as_deref());
                                        scroll_from_bottom = 0;
                                    } else if let Some(text) = state.submit_input() {
                                        command_tx
                                            .try_send(AgentCommand::Submit { text })
                                            .context("could not send prompt to the agent")?;
                                        scroll_from_bottom = 0;
                                    }
                                }
                                Action::Newline => state.insert_newline(),
                                Action::Escape => {
                                    if state.is_turn_active() {
                                        command_tx
                                            .try_send(AgentCommand::Cancel)
                                            .context("could not cancel the active turn")?;
                                    }
                                }
                                Action::CtrlC => {
                                    if state.is_turn_active() {
                                        command_tx
                                            .try_send(AgentCommand::Cancel)
                                            .context("could not cancel the active turn")?;
                                    } else {
                                        exit = true;
                                    }
                                }
                                Action::Insert(character) => state.insert_text(&character.to_string()),
                                Action::Backspace => state.backspace(),
                                Action::Delete => state.delete(),
                                Action::Left => state.move_left(),
                                Action::Right => state.move_right(),
                                Action::Up | Action::Down => {
                                    let direction = if matches!(action, Action::Up) { -1 } else { 1 };
                                    let layout = VisualLayout::new(state.input(), editor_width);
                                    if let Some((cursor, desired_column)) = layout.move_vertical(
                                        state.cursor(),
                                        direction,
                                        preferred_column,
                                    ) {
                                        state.set_cursor(cursor);
                                        preferred_column = Some(desired_column);
                                    }
                                }
                                Action::Home => state.move_home(),
                                Action::End => state.move_end(),
                                Action::PageUp => {
                                    scroll_from_bottom = scroll_from_bottom.saturating_add(10)
                                }
                                Action::PageDown => {
                                    scroll_from_bottom = scroll_from_bottom.saturating_sub(10)
                                }
                            }
                        }
                    }
                    Event::Resize(width, _) => {
                        editor_width = width.saturating_sub(2).max(1) as usize;
                        preferred_column = None;
                        dirty = true;
                    }
                    _ => {}
                }
            }
            agent_event = event_rx.recv() => {
                let event = agent_event
                    .ok_or_else(|| anyhow!("agent event stream disconnected"))?;
                if matches!(event, AgentEvent::TurnFinished { .. }) {
                    scroll_from_bottom = 0;
                }
                state.reduce(event);
                dirty = true;
            }
        }
    }

    Ok(())
}

fn model_command(input: &str) -> Option<Option<String>> {
    let input = input.trim();
    if input == "/model" {
        Some(None)
    } else {
        input
            .strip_prefix("/model ")
            .map(str::trim)
            .filter(|argument| !argument.is_empty())
            .map(|argument| Some(argument.to_owned()))
    }
}

fn handle_model_command(state: &mut AppState, setup: &mut ModelSetup, argument: Option<&str>) {
    let Some(catalog) = setup.catalog.as_ref() else {
        state.add_system_message(
            "No configured models are available. Create ~/.ri/agent/models.json first.",
        );
        return;
    };

    let selected = if let Some(argument) = argument {
        catalog.resolve(None, Some(argument))
    } else if catalog.models().is_empty() {
        Err(ri_core::ConfigError::Invalid(
            "models.json contains no selectable model".to_owned(),
        ))
    } else {
        let current = setup.selected.as_ref().map(|model| &model.model_ref);
        let current_index = current
            .and_then(|current| {
                catalog
                    .models()
                    .iter()
                    .position(|model| model.model_ref == *current)
            })
            .unwrap_or(usize::MAX);
        let next_index = if current_index == usize::MAX {
            0
        } else {
            (current_index + 1) % catalog.models().len()
        };
        Ok(catalog.models()[next_index].clone())
    };

    match selected {
        Ok(selected) => match setup.provider.set_model(selected.clone()) {
            Ok(()) => {
                let name = selected.model_ref.display_name();
                setup.selected = Some(selected.clone());
                state.reduce(AgentEvent::ModelChanged(selected.model_ref));
                state.add_system_message(format!("active model: {name}"));
            }
            Err(error) => state.add_system_message(error.to_string()),
        },
        Err(error) => state.add_system_message(error.to_string()),
    }
}

fn print_and_flush(output: &mut impl Write, text: &str) -> Result<()> {
    write!(output, "{text}")?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_print_prompt_and_model_flags() {
        assert_eq!(
            Options::parse([
                "-p".to_owned(),
                "hello".to_owned(),
                "--provider".to_owned(),
                "custom".to_owned(),
                "--model".to_owned(),
                "coding".to_owned(),
            ])
            .unwrap(),
            Options {
                print_prompt: Some("hello".to_owned()),
                provider: Some("custom".to_owned()),
                model: Some("coding".to_owned()),
                show_help: false,
            }
        );
    }

    #[test]
    fn recognizes_direct_and_cycling_model_commands() {
        assert_eq!(model_command("/model"), Some(None));
        assert_eq!(
            model_command("  /model custom/coding  "),
            Some(Some("custom/coding".to_owned()))
        );
        assert_eq!(model_command("/modelish"), None);
    }
}
