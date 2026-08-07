use std::io::{self, Write};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use crossterm::event::{self, Event, KeyEventKind};
use ri_core::{AgentCommand, AgentEvent, AgentRuntime, AppState, MockProvider, StopReason};
use tokio::sync::mpsc;

use crate::input::{self, Action};
use crate::render;
use crate::terminal::TerminalGuard;

const COMMAND_CHANNEL_CAPACITY: usize = 16;
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub print_prompt: Option<String>,
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
                "-h" | "--help" => options.show_help = true,
                unknown => bail!("unknown argument: {unknown}"),
            }
        }

        Ok(options)
    }

    pub fn print_help() {
        println!(
            "ri — a small Rust coding agent\n\n\
             Usage:\n  ri                 start the interactive TUI\n  ri -p <prompt>     run one prompt without the TUI\n  ri --help          show this help"
        );
    }
}

pub async fn run(options: Options) -> Result<()> {
    if let Some(prompt) = options.print_prompt {
        run_print(prompt).await
    } else {
        run_tui().await
    }
}

async fn run_print(prompt: String) -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::new(MockProvider::new());
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    command_tx
        .send(AgentCommand::Submit { text: prompt })
        .await
        .context("could not start the mock agent")?;

    let mut state = AppState::new();
    let mut turn_reason = None;
    let mut output = io::stdout();
    let mut error_message = None;

    while let Some(event) = event_rx.recv().await {
        match &event {
            AgentEvent::AssistantTextDelta { text } => {
                print_and_flush(&mut output, text)?;
            }
            AgentEvent::Error(error) => {
                error_message = Some(error.message.clone());
                eprintln!("ri: {}", error.message);
            }
            AgentEvent::TurnFinished { reason } => turn_reason = Some(reason.clone()),
            AgentEvent::TurnStarted | AgentEvent::AssistantThinkingDelta { .. } => {}
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

async fn run_tui() -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::new(MockProvider::new());
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    let mut terminal = TerminalGuard::new().context("could not initialize terminal")?;
    let mut state = AppState::new();
    let tui_result = run_tui_loop(&mut terminal, &mut state, &command_tx, &mut event_rx);
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

fn run_tui_loop(
    terminal: &mut TerminalGuard,
    state: &mut AppState,
    command_tx: &mpsc::Sender<AgentCommand>,
    event_rx: &mut mpsc::Receiver<AgentEvent>,
) -> Result<()> {
    let mut dirty = true;
    let mut scroll_from_bottom = 0usize;
    let mut exit = false;

    while !exit {
        if dirty {
            render::draw(terminal, state, scroll_from_bottom)
                .context("could not render terminal")?;
            dirty = false;
        }

        if event::poll(Duration::from_millis(50)).context("could not read terminal input")? {
            if let Event::Key(key) = event::read().context("could not read terminal event")? {
                if key.kind == KeyEventKind::Press {
                    let Some(action) = input::action_for(key) else {
                        continue;
                    };
                    dirty = true;
                    match action {
                        Action::Submit => {
                            if let Some(text) = state.submit_input() {
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
                        Action::Up => state.move_up(),
                        Action::Down => state.move_down(),
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
        }

        loop {
            match event_rx.try_recv() {
                Ok(event) => {
                    if matches!(event, AgentEvent::TurnFinished { .. }) {
                        scroll_from_bottom = 0;
                    }
                    state.reduce(event);
                    dirty = true;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    return Err(anyhow!("agent event stream disconnected"));
                }
            }
        }
    }

    Ok(())
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
    fn parses_print_prompt() {
        assert_eq!(
            Options::parse(["-p".to_owned(), "hello".to_owned()]).unwrap(),
            Options {
                print_prompt: Some("hello".to_owned()),
                show_help: false,
            }
        );
    }
}
