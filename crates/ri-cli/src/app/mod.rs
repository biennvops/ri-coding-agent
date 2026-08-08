use std::io::{self, Write};

use anyhow::{anyhow, bail, Context, Result};
use crossterm::event::{Event, EventStream, KeyEventKind};
use futures_util::StreamExt;
use ri_core::{
    config::{load_default_models, load_default_settings},
    context::{build_system_prompt, discover_project, load_context, ContextBundle},
    AgentCommand, AgentEvent, AgentRuntime, AgentRuntimeConfig, AppState, ConfiguredProvider,
    ModelCatalog, ModelMessage, ModelRef, OpenedSession, ResolvedModel, ResolvedSettings,
    SessionHandle, SessionInfo, SessionMode, SessionRepository, StopReason, ToolContext,
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
    pub no_context: bool,
    pub show_help: bool,
    pub continue_session: bool,
    pub resume_session: bool,
    pub session: Option<String>,
    pub no_session: bool,
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
                    if prompt == "--no-context" {
                        options.no_context = true;
                        options.print_prompt = Some(
                            args.next()
                                .ok_or_else(|| anyhow!("{argument} requires a prompt"))?,
                        );
                    } else {
                        options.print_prompt = Some(prompt);
                    }
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
                "--no-context" => options.no_context = true,
                "-c" | "--continue" => options.continue_session = true,
                "-r" | "--resume" => options.resume_session = true,
                "--session" => {
                    options.session = Some(
                        args.next()
                            .ok_or_else(|| anyhow!("--session requires an id or path"))?,
                    );
                }
                "--no-session" => options.no_session = true,
                "-h" | "--help" => options.show_help = true,
                unknown => bail!("unknown argument: {unknown}"),
            }
        }

        let selection_count = options.continue_session as u8
            + options.resume_session as u8
            + options.session.is_some() as u8;
        if options.no_session && selection_count > 0 {
            bail!("--no-session cannot be combined with --continue, --resume, or --session");
        }
        if selection_count > 1 {
            bail!("--continue, --resume, and --session are mutually exclusive");
        }
        if options.resume_session && options.print_prompt.is_some() {
            bail!("--resume requires interactive mode; use --continue or --session with --print");
        }

        Ok(options)
    }

    pub fn print_help() {
        println!(
            "ri — a small Rust coding agent\n\n\
             Usage:\n  ri                              start the interactive TUI\n  ri -p <prompt>                  run one prompt without the TUI\n  ri --provider <id> --model <id> select a configured model\n  ri -c                            continue the newest saved session\n  ri -r                            choose a saved session interactively\n  ri --session <id-or-path>        resume one saved session\n  ri --no-session                  disable session persistence\n  ri --no-context                 disable AGENTS context loading\n  ri --help                       show this help"
        );
    }
}

pub async fn run(options: Options) -> Result<()> {
    let mut setup = AppSetup::load(&options)?;
    if setup.resume_requested {
        let repository = setup
            .repository
            .as_ref()
            .ok_or_else(|| anyhow!("sessions are disabled for this run"))?;
        let opened = crate::session_picker::pick(repository)?
            .ok_or_else(|| anyhow!("session picker cancelled"))?;
        setup.apply_opened(opened);
    }
    if let Some(prompt) = options.print_prompt {
        run_print(prompt, setup).await
    } else {
        run_tui(setup).await
    }
}

struct AppSetup {
    provider: ConfiguredProvider,
    catalog: Option<ModelCatalog>,
    selected: Option<ResolvedModel>,
    tool_context: ToolContext,
    context: ContextBundle,
    system_prompt: String,
    repository: Option<SessionRepository>,
    session: Option<SessionHandle>,
    initial_history: Vec<ModelMessage>,
    resume_requested: bool,
}

impl AppSetup {
    fn load(options: &Options) -> Result<Self> {
        let launch_cwd = std::env::current_dir().context("could not determine launch cwd")?;
        let project = discover_project(&launch_cwd).context("could not discover project root")?;
        let settings =
            load_default_settings(&project.project_root).context("could not load settings.json")?;
        for warning in &settings.warnings {
            eprintln!("ri: warning: {}: {}", warning.path, warning.message);
        }

        let (requested_provider, requested_model) = model_selection(options, &settings.settings);
        let catalog = load_default_models().context("could not load models.json")?;
        let (provider, catalog, selected) = if let Some(catalog) = catalog {
            for warning in catalog.warnings() {
                eprintln!("ri: warning: {}: {}", warning.path, warning.message);
            }
            let selected = catalog
                .resolve(requested_provider, requested_model)
                .context("could not select configured model")?;
            let provider = ConfiguredProvider::openai(selected.clone())
                .map_err(|error| anyhow!(error.to_string()))?;
            (provider, Some(catalog), Some(selected))
        } else if requested_provider.is_some() || requested_model.is_some() {
            if settings.settings.default_provider.is_some()
                || settings.settings.default_model.is_some()
            {
                bail!(
                    "settings select {}, but no models.json is available; create ~/.ri/agent/models.json or set RI_MODELS_FILE",
                    settings_selection_description(&settings.settings)
                );
            }
            bail!("no models.json found; create ~/.ri/agent/models.json or set RI_MODELS_FILE");
        } else {
            (ConfiguredProvider::mock(), None, None)
        };

        let context = if settings.settings.context.enabled && !options.no_context {
            load_context(&project.launch_cwd, &project.project_root)
                .context("could not load AGENTS context")?
        } else {
            ContextBundle::disabled(project.launch_cwd.clone(), project.project_root.clone())
        };
        let system_prompt = build_system_prompt(&context);
        let tool_context =
            ToolContext::new(&project.launch_cwd).map_err(|error| anyhow!(error.to_string()))?;

        let (repository, session, initial_history, resume_requested) = if options.no_session {
            (None, None, Vec::new(), false)
        } else {
            let repository =
                SessionRepository::for_workspace(&project.launch_cwd, &project.project_root)
                    .map_err(|error| anyhow!(error.to_string()))?;
            if options.resume_session {
                (Some(repository), None, Vec::new(), true)
            } else if options.continue_session {
                let summary = repository
                    .latest()
                    .map_err(|error| anyhow!(error.to_string()))?
                    .ok_or_else(|| {
                        anyhow!(
                            "no saved sessions found for {}",
                            project.launch_cwd.display()
                        )
                    })?;
                let opened = repository
                    .open_path(&summary.path)
                    .map_err(|error| anyhow!(error.to_string()))?;
                for warning in &opened.warnings {
                    eprintln!("{warning}");
                }
                (Some(repository), Some(opened.handle), opened.history, false)
            } else if let Some(selector) = options.session.as_deref() {
                let opened = repository
                    .open_selector(selector)
                    .map_err(|error| anyhow!(error.to_string()))?;
                for warning in &opened.warnings {
                    eprintln!("{warning}");
                }
                (Some(repository), Some(opened.handle), opened.history, false)
            } else {
                let session = repository
                    .create()
                    .map_err(|error| anyhow!(error.to_string()))?;
                (Some(repository), Some(session), Vec::new(), false)
            }
        };

        Ok(Self {
            provider,
            catalog,
            selected,
            tool_context,
            context,
            system_prompt,
            repository,
            session,
            initial_history,
            resume_requested,
        })
    }

    fn model_ref(&self) -> ModelRef {
        self.selected
            .as_ref()
            .map(|model| model.model_ref.clone())
            .unwrap_or_else(|| self.provider.model_ref())
    }

    fn runtime_config(&self) -> AgentRuntimeConfig {
        AgentRuntimeConfig {
            tool_context: self.tool_context.clone(),
            base_messages: vec![ModelMessage::System {
                content: self.system_prompt.clone(),
            }],
            initial_history: self.initial_history.clone(),
            session: self
                .session
                .as_ref()
                .map(|session| SessionMode::Enabled(session.clone()))
                .unwrap_or(SessionMode::Disabled),
        }
    }

    fn apply_opened(&mut self, opened: OpenedSession) {
        self.session = Some(opened.handle);
        self.initial_history = opened.history;
        self.resume_requested = false;
        for warning in opened.warnings {
            eprintln!("{warning}");
        }
    }

    fn session_info(&self) -> Result<Option<SessionInfo>> {
        self.session
            .as_ref()
            .map(SessionHandle::info)
            .transpose()
            .map_err(|error| anyhow!(error.to_string()))
    }
}

async fn run_print(prompt: String, setup: AppSetup) -> Result<()> {
    eprintln!("{}", setup.context.diagnostic());
    if let Some(info) = setup.session_info()? {
        eprintln!("session: {} ({})", info.display_name(), info.id);
    } else {
        eprintln!("session: ephemeral");
    }
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::with_config(setup.provider.clone(), setup.runtime_config());
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    command_tx
        .send(AgentCommand::Submit { text: prompt })
        .await
        .context("could not start the mock agent")?;

    let mut state = AppState::new();
    state.replace_history(&setup.initial_history);
    state.set_session_info(setup.session_info()?);
    state.reduce(AgentEvent::ModelChanged(setup.model_ref()));
    let mut turn_reason = None;
    let mut output = io::stdout();
    let mut error_message = None;

    while let Some(event) = event_rx.recv().await {
        match &event {
            AgentEvent::AssistantTextDelta { text, .. } => {
                print_and_flush(&mut output, text)?;
            }
            AgentEvent::AssistantRefusalDelta { text, .. } => {
                print_and_flush(&mut output, text)?;
            }
            AgentEvent::Error(error) => {
                error_message = Some(error.message.clone());
                eprintln!("ri: {}", error.message);
            }
            AgentEvent::TurnFinished { reason } => turn_reason = Some(reason.clone()),
            AgentEvent::TurnStarted
            | AgentEvent::AssistantMessageStarted
            | AgentEvent::AssistantMessageFinished { .. }
            | AgentEvent::AssistantTextItem { .. }
            | AgentEvent::AssistantRefusalItem { .. }
            | AgentEvent::AssistantThinkingDelta { .. }
            | AgentEvent::AssistantThinkingContentDelta { .. }
            | AgentEvent::AssistantThinkingItem { .. }
            | AgentEvent::ToolCallDelta { .. }
            | AgentEvent::ToolExecutionStarted { .. }
            | AgentEvent::ToolExecutionOutput { .. }
            | AgentEvent::ToolExecutionFinished { .. }
            | AgentEvent::UsageUpdated(_)
            | AgentEvent::ModelChanged(_)
            | AgentEvent::SessionChanged { .. }
            | AgentEvent::SessionLoaded { .. } => {}
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

async fn run_tui(mut setup: AppSetup) -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::with_config(setup.provider.clone(), setup.runtime_config());
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    let mut terminal = TerminalGuard::new().context("could not initialize terminal")?;
    let mut state = AppState::new();
    state.replace_history(&setup.initial_history);
    state.add_system_message(setup.context.diagnostic());
    state.set_session_info(setup.session_info()?);
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
    setup: &mut AppSetup,
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
                                    if let Some(command) = slash_command(state.input()) {
                                        if state.is_turn_active() {
                                            state.add_system_message("a turn is already active");
                                        } else {
                                            state.take_input();
                                            handle_slash_command(
                                                command,
                                                terminal,
                                                state,
                                                command_tx,
                                                setup,
                                            )
                                            .await?;
                                            scroll_from_bottom = 0;
                                        }
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
                let session_loaded = matches!(event, AgentEvent::SessionLoaded { .. });
                state.reduce(event);
                if session_loaded {
                    state.add_system_message(setup.context.diagnostic());
                }
                dirty = true;
            }
        }
    }

    Ok(())
}

fn model_selection<'a>(
    options: &'a Options,
    settings: &'a ResolvedSettings,
) -> (Option<&'a str>, Option<&'a str>) {
    let model = options
        .model
        .as_deref()
        .or(settings.default_model.as_deref());
    let provider = if let Some(provider) = options.provider.as_deref() {
        Some(provider)
    } else if options
        .model
        .as_deref()
        .is_some_and(|model| model.contains('/'))
    {
        // An explicitly qualified CLI model overrides an inherited provider.
        None
    } else {
        settings.default_provider.as_deref()
    };
    (provider, model)
}

fn settings_selection_description(settings: &ResolvedSettings) -> String {
    match (
        settings.default_provider.as_deref(),
        settings.default_model.as_deref(),
    ) {
        (Some(provider), Some(model)) => {
            format!("provider {provider:?} and model {model:?}")
        }
        (Some(provider), None) => format!("provider {provider:?}"),
        (None, Some(model)) => format!("model {model:?}"),
        (None, None) => "a configured model".to_owned(),
    }
}

enum SlashCommand {
    Model(Option<String>),
    New,
    Resume,
    Name(Option<String>),
    Session,
}

fn slash_command(input: &str) -> Option<SlashCommand> {
    let input = input.trim();
    if input == "/model" {
        return Some(SlashCommand::Model(None));
    }
    if let Some(argument) = input.strip_prefix("/model ") {
        return argument
            .trim()
            .is_empty()
            .then_some(SlashCommand::Model(None))
            .or_else(|| Some(SlashCommand::Model(Some(argument.trim().to_owned()))));
    }
    if input == "/new" {
        Some(SlashCommand::New)
    } else if input == "/resume" {
        Some(SlashCommand::Resume)
    } else if input == "/session" {
        Some(SlashCommand::Session)
    } else if input == "/name" {
        Some(SlashCommand::Name(None))
    } else {
        input
            .strip_prefix("/name ")
            .map(|argument| SlashCommand::Name(Some(argument.trim().to_owned())))
    }
}

async fn handle_slash_command(
    command: SlashCommand,
    terminal: &mut TerminalGuard,
    state: &mut AppState,
    command_tx: &mpsc::Sender<AgentCommand>,
    setup: &mut AppSetup,
) -> Result<()> {
    match command {
        SlashCommand::Model(argument) => handle_model_command(state, setup, argument.as_deref()),
        SlashCommand::Session => state.add_system_message(session_diagnostic(setup)?),
        SlashCommand::Name(None) => state.add_system_message(session_diagnostic(setup)?),
        SlashCommand::Name(Some(name)) => {
            if setup.session.is_none() {
                state.add_system_message("sessions are disabled for this run");
            } else {
                command_tx
                    .send(AgentCommand::RenameSession { name })
                    .await
                    .context("could not rename the session")?;
            }
        }
        SlashCommand::New => {
            let Some(repository) = setup.repository.as_ref() else {
                state.add_system_message("sessions are disabled for this run");
                return Ok(());
            };
            let session = repository
                .create()
                .map_err(|error| anyhow!(error.to_string()))?;
            let handle = session.clone();
            setup.session = Some(session);
            setup.initial_history.clear();
            command_tx
                .send(AgentCommand::NewSession { session: handle })
                .await
                .context("could not create a new session")?;
            state.replace_history(&[]);
            state.set_session_info(setup.session_info()?);
            state.add_system_message(setup.context.diagnostic());
        }
        SlashCommand::Resume => {
            let Some(repository) = setup.repository.as_ref() else {
                state.add_system_message("sessions are disabled for this run");
                return Ok(());
            };
            let path = match crate::session_picker::pick_path_in_terminal(terminal, repository) {
                Ok(Some(path)) => path,
                Ok(None) => return Ok(()),
                Err(error) => {
                    state.add_system_message(format!("ri: {error}"));
                    return Ok(());
                }
            };
            if setup.session_info()?.is_some_and(|info| info.path == path) {
                state.add_system_message("that session is already active");
                return Ok(());
            }
            let opened = repository
                .open_path(path)
                .map_err(|error| anyhow!(error.to_string()))?;
            let handle = opened.handle.clone();
            let history = opened.history.clone();
            setup.apply_opened(opened);
            command_tx
                .send(AgentCommand::LoadSession {
                    session: handle,
                    history,
                })
                .await
                .context("could not resume the session")?;
            state.replace_history(&setup.initial_history);
            state.set_session_info(setup.session_info()?);
            state.add_system_message(setup.context.diagnostic());
        }
    }
    Ok(())
}

fn session_diagnostic(setup: &AppSetup) -> Result<String> {
    let Some(info) = setup.session_info()? else {
        return Ok("session: ephemeral".to_owned());
    };
    let file = if info.materialized {
        info.path.display().to_string()
    } else {
        "not created yet".to_owned()
    };
    Ok(format!(
        "session: {}\nid: {}\nfile: {}\nmessages: {}",
        info.name.as_deref().unwrap_or(info.id.as_str()),
        info.id,
        file,
        info.message_count
    ))
}

#[cfg(test)]
fn model_command(input: &str) -> Option<Option<String>> {
    match slash_command(input) {
        Some(SlashCommand::Model(argument)) => Some(argument),
        _ => None,
    }
}

fn handle_model_command(state: &mut AppState, setup: &mut AppSetup, argument: Option<&str>) {
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
                no_context: false,
                show_help: false,
                continue_session: false,
                resume_session: false,
                session: None,
                no_session: false,
            }
        );
    }

    #[test]
    fn rejects_conflicting_session_flags_and_picker_print_mode() {
        let conflicts = [
            vec!["-c", "-r"],
            vec!["-c", "--session", "abc"],
            vec!["-r", "--session", "abc"],
            vec!["--no-session", "-c"],
            vec!["--no-session", "-r"],
            vec!["--no-session", "--session", "abc"],
            vec!["-r", "-p", "hello"],
        ];
        for arguments in conflicts {
            let arguments = arguments.into_iter().map(str::to_owned).collect::<Vec<_>>();
            assert!(Options::parse(arguments).is_err());
        }
    }

    #[test]
    fn parses_continue_explicit_session_and_ephemeral_flags() {
        assert!(Options::parse(["-c".to_owned()]).unwrap().continue_session);
        assert!(Options::parse(["-r".to_owned()]).unwrap().resume_session);
        assert_eq!(
            Options::parse(["--session".to_owned(), "0198ab".to_owned()])
                .unwrap()
                .session
                .as_deref(),
            Some("0198ab")
        );
        assert!(
            Options::parse(["--no-session".to_owned()])
                .unwrap()
                .no_session
        );
    }

    #[test]
    fn parses_no_context_before_print_prompt() {
        let options = Options::parse([
            "-p".to_owned(),
            "--no-context".to_owned(),
            "hello".to_owned(),
        ])
        .unwrap();

        assert_eq!(options.print_prompt.as_deref(), Some("hello"));
        assert!(options.no_context);
    }

    #[test]
    fn settings_defaults_feed_model_catalog_and_cli_overrides_win() {
        let settings = ResolvedSettings {
            default_provider: Some("provider-a".to_owned()),
            default_model: Some("model-a".to_owned()),
            ..ResolvedSettings::default()
        };
        let options = Options::default();
        let (provider, model) = model_selection(&options, &settings);
        let catalog = ModelCatalog::from_json(
            "models.json",
            r#"{
                "providers": {
                    "provider-a": {
                        "baseUrl": "https://example.test",
                        "api": "openai-responses",
                        "models": [{"id": "model-a"}]
                    },
                    "provider-b": {
                        "baseUrl": "https://example.test",
                        "api": "openai-responses",
                        "models": [{"id": "model-b"}]
                    }
                }
            }"#,
        )
        .unwrap();
        assert_eq!(
            catalog
                .resolve(provider, model)
                .unwrap()
                .model_ref
                .display_name(),
            "provider-a/model-a"
        );

        let options = Options {
            provider: Some("provider-b".to_owned()),
            model: Some("model-b".to_owned()),
            ..Options::default()
        };
        let (provider, model) = model_selection(&options, &settings);
        assert_eq!(
            catalog
                .resolve(provider, model)
                .unwrap()
                .model_ref
                .display_name(),
            "provider-b/model-b"
        );
    }

    #[test]
    fn settings_provider_preserves_slash_containing_model_id() {
        let settings = ResolvedSettings {
            default_provider: Some("openrouter".to_owned()),
            default_model: Some("anthropic/claude-sonnet-4".to_owned()),
            ..ResolvedSettings::default()
        };
        let catalog = ModelCatalog::from_json(
            "models.json",
            r#"{
                "providers": {
                    "openrouter": {
                        "baseUrl": "https://example.test",
                        "api": "openai-responses",
                        "models": [{"id": "anthropic/claude-sonnet-4"}]
                    }
                }
            }"#,
        )
        .unwrap();

        let options = Options::default();
        let (provider, model) = model_selection(&options, &settings);
        let selected = catalog.resolve(provider, model).unwrap();

        assert_eq!(selected.model_ref.provider, "openrouter");
        assert_eq!(selected.model_ref.model, "anthropic/claude-sonnet-4");
    }

    #[test]
    fn qualified_cli_model_can_override_a_settings_provider() {
        let settings = ResolvedSettings {
            default_provider: Some("provider-a".to_owned()),
            ..ResolvedSettings::default()
        };
        let options = Options {
            model: Some("provider-b/model-b".to_owned()),
            ..Options::default()
        };

        assert_eq!(
            model_selection(&options, &settings),
            (None, Some("provider-b/model-b"))
        );
    }

    #[test]
    fn settings_selection_diagnostic_includes_requested_values() {
        let settings = ResolvedSettings {
            default_provider: Some("foo".to_owned()),
            default_model: Some("coding".to_owned()),
            ..ResolvedSettings::default()
        };

        assert_eq!(
            settings_selection_description(&settings),
            "provider \"foo\" and model \"coding\""
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
