use std::io::{self, Write};

use anyhow::{anyhow, bail, Context, Result};
use crossterm::event::{Event, EventStream, KeyEventKind};
use futures_util::StreamExt;
use ri_core::{
    config::{
        default_state_path, load_default_models, load_default_settings, load_state,
        persist_recent_model,
    },
    context::{build_system_prompt, discover_project, load_context, ContextBundle},
    workspace_id, AgentCommand, AgentEvent, AgentRuntime, AgentRuntimeConfig, AppState,
    ConfiguredProvider, ModelCatalog, ModelMessage, ModelRef, OpenedSession, ResolvedModel,
    SessionHandle, SessionInfo, SessionMode, SessionRepository, StopReason, ToolContext,
};
use tokio::sync::mpsc;

use crate::input::{self, Action, VisualLayout};
use crate::json_output::{JsonEmitter, RunStartedData};
use crate::model_selection::resolve_model;
use crate::render;
use crate::terminal::TerminalGuard;

const COMMAND_CHANNEL_CAPACITY: usize = 16;
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub print_prompt: Option<String>,
    pub json: bool,
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
                "--json" => options.json = true,
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

        if options.json && options.print_prompt.is_none() && !options.show_help {
            bail!("--json requires --print <prompt>");
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
             Usage:\n  ri                              start the interactive TUI\n  ri -p, --print <prompt>         run one prompt without the TUI\n  ri --json -p <prompt>           emit versioned NDJSON events\n\n\
             Model:\n  --provider <id>                 select a configured provider\n  --model <id>                   select a configured model\n\n\
             Sessions:\n  -c, --continue                 continue the newest saved session\n  -r, --resume                   choose a saved session interactively\n  --session <id-or-path>         resume one saved session\n  --no-session                   disable session persistence\n\n\
             Context and help:\n  --no-context                   disable AGENTS context loading\n  -h, --help                    show this help\n\n\
             Interactive commands:\n  /model                         open the model picker\n  /model <provider/model>        select a model directly\n  /new                           create a new session\n  /resume                        choose a saved session\n  /name [name]                   show or set the session name\n  /session                       show session details\n  /compact                       compact the current context\n  /quit                          exit the TUI\n\n\
             Environment:\n  RI_LOG=error|warn|info|debug|trace  write private diagnostic logs"
        );
    }
}

#[derive(Debug)]
pub enum RunError {
    Setup(anyhow::Error),
    Runtime(anyhow::Error),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Setup(error) | Self::Runtime(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Setup(error) | Self::Runtime(error) => error.source(),
        }
    }
}

pub async fn run(options: Options) -> Result<(), RunError> {
    let mut setup = AppSetup::load(&options).map_err(RunError::Setup)?;
    if setup.resume_requested {
        let repository = setup
            .repository
            .as_ref()
            .ok_or_else(|| RunError::Setup(anyhow!("sessions are disabled for this run")))?;
        let opened = crate::session_picker::pick(repository)
            .map_err(RunError::Setup)?
            .ok_or_else(|| RunError::Setup(anyhow!("session picker cancelled")))?;
        setup.apply_opened(opened);
    }
    if let Some(prompt) = options.print_prompt {
        if options.json {
            run_json(prompt, setup).await.map_err(RunError::Runtime)
        } else {
            run_print(prompt, setup).await.map_err(RunError::Runtime)
        }
    } else {
        run_tui(setup).await.map_err(RunError::Runtime)
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
    initial_transcript: Vec<ModelMessage>,
    compaction_enabled: bool,
    resume_requested: bool,
    state_path: Option<std::path::PathBuf>,
    workspace_id: String,
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

        let workspace_id = workspace_id(&project.launch_cwd)
            .map_err(|error| anyhow!(error.to_string()))?
            .to_string();
        let state_path = default_state_path();
        let recent_state = match state_path.as_ref() {
            Some(path) => match load_state(path) {
                Ok(state) => state,
                Err(error) => {
                    eprintln!("ri: warning: {error}; ignoring recent model state");
                    None
                }
            },
            None => None,
        };

        let catalog = load_default_models()
            .context("could not load models.json")?
            .ok_or_else(|| {
                anyhow!(
                    "no models.json found; create ~/.ri/agent/models.json or set RI_MODELS_FILE"
                )
            })?;
        for warning in catalog.warnings() {
            eprintln!("ri: warning: {}: {}", warning.path, warning.message);
        }
        let selection = resolve_model(
            &catalog,
            options.provider.as_deref(),
            options.model.as_deref(),
            &settings.settings,
            recent_state.as_ref(),
            &workspace_id,
        )
        .context("could not select configured model")?;
        for warning in &selection.warnings {
            eprintln!("ri: warning: {warning}");
        }
        let selection_source = selection.source;
        let selected = selection.model;
        tracing::info!(
            target: "ri",
            provider = %selected.model_ref.provider,
            model = %selected.model_ref.model,
            source = ?selection_source,
            workspace = %workspace_id,
            "selected model"
        );
        let provider = ConfiguredProvider::openai(selected.clone())
            .map_err(|error| anyhow!(error.to_string()))?;
        if let Some(path) = state_path.as_ref() {
            if let Err(error) = persist_recent_model(path, &workspace_id, &selected.model_ref) {
                eprintln!("ri: warning: could not persist recent model selection: {error}");
            }
        }
        let (provider, catalog, selected) = (provider, Some(catalog), Some(selected));

        let context = if settings.settings.context.enabled && !options.no_context {
            load_context(&project.launch_cwd, &project.project_root)
                .context("could not load AGENTS context")?
        } else {
            ContextBundle::disabled(project.launch_cwd.clone(), project.project_root.clone())
        };
        for file in &context.files {
            tracing::debug!(target: "ri", context_path = %file.path.display(), "loaded context file");
        }
        let system_prompt = build_system_prompt(&context);
        let tool_context =
            ToolContext::new(&project.launch_cwd).map_err(|error| anyhow!(error.to_string()))?;

        let (repository, session, initial_history, initial_transcript, resume_requested) =
            if options.no_session {
                (None, None, Vec::new(), Vec::new(), false)
            } else {
                let repository =
                    SessionRepository::for_workspace(&project.launch_cwd, &project.project_root)
                        .map_err(|error| anyhow!(error.to_string()))?;
                if options.resume_session {
                    (Some(repository), None, Vec::new(), Vec::new(), true)
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
                    (
                        Some(repository),
                        Some(opened.handle),
                        opened.history,
                        opened.transcript,
                        false,
                    )
                } else if let Some(selector) = options.session.as_deref() {
                    let opened = repository
                        .open_selector(selector)
                        .map_err(|error| anyhow!(error.to_string()))?;
                    for warning in &opened.warnings {
                        eprintln!("{warning}");
                    }
                    (
                        Some(repository),
                        Some(opened.handle),
                        opened.history,
                        opened.transcript,
                        false,
                    )
                } else {
                    let session = repository
                        .create()
                        .map_err(|error| anyhow!(error.to_string()))?;
                    (
                        Some(repository),
                        Some(session),
                        Vec::new(),
                        Vec::new(),
                        false,
                    )
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
            initial_transcript,
            compaction_enabled: settings.settings.compaction.enabled,
            resume_requested,
            state_path,
            workspace_id,
        })
    }

    fn model_ref(&self) -> ModelRef {
        self.selected
            .as_ref()
            .map(|model| model.model_ref.clone())
            .unwrap_or_else(|| self.provider.model_ref())
    }

    fn remember_model(&self, model: &ModelRef) -> Result<(), String> {
        let Some(path) = self.state_path.as_ref() else {
            return Ok(());
        };
        persist_recent_model(path, &self.workspace_id, model).map_err(|error| error.to_string())
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
        self.initial_transcript = opened.transcript;
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
    tracing::info!(target: "ri", mode = "print", prompt_bytes = prompt.len(), "run started");
    eprintln!("{}", setup.context.diagnostic());
    if let Some(info) = setup.session_info()? {
        eprintln!("session: {} ({})", info.display_name(), info.id);
    } else {
        eprintln!("session: ephemeral");
    }
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::with_config_and_compaction(
        setup.provider.clone(),
        setup.runtime_config(),
        setup.compaction_enabled,
    );
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    command_tx
        .send(AgentCommand::Submit { text: prompt })
        .await
        .context("could not start the agent")?;

    let mut state = AppState::new();
    state.replace_history(&setup.initial_transcript);
    state.set_session_info(setup.session_info()?);
    state.reduce(AgentEvent::ModelChanged(setup.model_ref()));
    state.reduce(AgentEvent::ContextLimitsUpdated(setup.provider.limits()));
    let mut turn_reason = None;
    let mut output = io::stdout();
    let mut error_message = None;

    while let Some(event) = event_rx.recv().await {
        log_agent_event(&event);
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
            AgentEvent::CompactionFinished {
                before_tokens,
                after_tokens,
                ..
            } => eprintln!("ri: context compacted · ~{before_tokens} → ~{after_tokens} tokens"),
            AgentEvent::CompactionFailed { message } => eprintln!("ri: {message}"),
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
            | AgentEvent::ContextUsageUpdated(_)
            | AgentEvent::ContextLimitsUpdated(_)
            | AgentEvent::CompactionStarted { .. }
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
        .context("could not stop the agent")?;
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

async fn run_json(prompt: String, setup: AppSetup) -> Result<()> {
    tracing::info!(target: "ri", mode = "json", prompt_bytes = prompt.len(), "run started");
    eprintln!("{}", setup.context.diagnostic());
    let session = setup.session_info()?;
    if let Some(info) = session.as_ref() {
        eprintln!("session: {} ({})", info.display_name(), info.id);
    } else {
        eprintln!("session: ephemeral");
    }

    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::with_config_and_compaction(
        setup.provider.clone(),
        setup.runtime_config(),
        setup.compaction_enabled,
    );
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));
    let mut output = JsonEmitter::new(io::stdout());
    output.emit(
        "run_started",
        RunStartedData::new(
            &setup.model_ref(),
            setup.tool_context.workspace_root.display().to_string(),
            session.as_ref(),
        ),
    )?;

    if let Err(error) = command_tx.send(AgentCommand::Submit { text: prompt }).await {
        let message = format!("could not start the agent: {error}");
        output.emit(
            "error",
            serde_json::json!({"message": message, "fatal": true}),
        )?;
        output.emit("run_finished", serde_json::json!({"success": false}))?;
        let _ = command_tx.send(AgentCommand::Shutdown).await;
        let _ = runtime_task.await;
        bail!(message);
    }

    let mut turn_reason = None;
    let mut saw_error = false;
    while let Some(event) = event_rx.recv().await {
        log_agent_event(&event);
        if matches!(&event, AgentEvent::Error(_)) {
            saw_error = true;
        }
        let finished = matches!(&event, AgentEvent::TurnFinished { .. });
        output.emit_agent_event(&event)?;
        if let AgentEvent::TurnFinished { reason } = event {
            turn_reason = Some(reason);
        }
        if finished {
            break;
        }
    }

    let Some(reason) = turn_reason else {
        let message = "agent stopped before finishing the turn";
        output.emit(
            "error",
            serde_json::json!({"message": message, "fatal": true}),
        )?;
        let _ = command_tx.send(AgentCommand::Shutdown).await;
        let _ = runtime_task.await;
        output.emit("run_finished", serde_json::json!({"success": false}))?;
        bail!(message);
    };

    if let Err(error) = command_tx.send(AgentCommand::Shutdown).await {
        let message = format!("could not stop the agent: {error}");
        output.emit(
            "error",
            serde_json::json!({"message": message, "fatal": true}),
        )?;
        let _ = runtime_task.await;
        output.emit("run_finished", serde_json::json!({"success": false}))?;
        bail!(message);
    }
    if let Err(error) = runtime_task
        .await
        .map_err(|error| anyhow!("agent task failed: {error}"))
    {
        let message = error.to_string();
        output.emit(
            "error",
            serde_json::json!({"message": message, "fatal": true}),
        )?;
        output.emit("run_finished", serde_json::json!({"success": false}))?;
        return Err(error);
    }

    let success = !saw_error && !matches!(&reason, StopReason::Error | StopReason::Cancelled);
    if !success && !saw_error {
        let message = format!(
            "agent turn finished with {}",
            crate::json_output::stop_reason_name(&reason)
        );
        output.emit(
            "error",
            serde_json::json!({"message": message, "fatal": true}),
        )?;
    }
    output.emit("run_finished", serde_json::json!({"success": success}))?;
    if success {
        Ok(())
    } else if matches!(&reason, StopReason::Cancelled) {
        bail!("agent turn cancelled")
    } else {
        bail!("agent turn failed")
    }
}

async fn run_tui(mut setup: AppSetup) -> Result<()> {
    tracing::info!(target: "ri", mode = "tui", "run started");
    let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
    let (event_tx, mut event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let runtime = AgentRuntime::with_config_and_compaction(
        setup.provider.clone(),
        setup.runtime_config(),
        setup.compaction_enabled,
    );
    let runtime_task = tokio::spawn(runtime.run(command_rx, event_tx));

    let mut terminal = TerminalGuard::new().context("could not initialize terminal")?;
    let mut state = AppState::new();
    state.replace_history(&setup.initial_transcript);
    state.add_system_message(setup.context.diagnostic());
    if let Some(path) = crate::logging::path() {
        state.add_system_message(format!("logging: {}", path.display()));
    }
    state.set_session_info(setup.session_info()?);
    state.reduce(AgentEvent::ModelChanged(setup.model_ref()));
    state.reduce(AgentEvent::ContextLimitsUpdated(setup.provider.limits()));
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
                                    if is_slash_input(state.input()) {
                                        if state.is_busy() {
                                            state.add_system_message("a turn or compaction is already active");
                                        } else if let Some(command) = slash_command(state.input()) {
                                            state.take_input();
                                            let outcome = handle_slash_command(
                                                command,
                                                terminal,
                                                state,
                                                command_tx,
                                                setup,
                                            )
                                            .await?;
                                            if matches!(outcome, SlashCommandOutcome::Quit) {
                                                exit = true;
                                            }
                                            scroll_from_bottom = 0;
                                        } else {
                                            let command = unknown_command_name(state.input()).to_owned();
                                            state.take_input();
                                            state.add_system_message(format!("unknown command: {command}"));
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
                                    if state.is_busy() {
                                        command_tx
                                            .try_send(AgentCommand::Cancel)
                                            .context("could not cancel the active operation")?;
                                    }
                                }
                                Action::CtrlC => {
                                    if state.is_busy() {
                                        command_tx
                                            .try_send(AgentCommand::Cancel)
                                            .context("could not cancel the active operation")?;
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
                log_agent_event(&event);
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

#[cfg(test)]
fn model_selection<'a>(
    options: &'a Options,
    settings: &'a ri_core::ResolvedSettings,
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

#[cfg(test)]
fn settings_selection_description(settings: &ri_core::ResolvedSettings) -> String {
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
    Quit,
    Compact,
    New,
    Resume,
    Name(Option<String>),
    Session,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlashCommandOutcome {
    Continue,
    Quit,
}

fn is_slash_input(input: &str) -> bool {
    input.trim_start().starts_with('/')
}

fn unknown_command_name(input: &str) -> &str {
    input.split_whitespace().next().unwrap_or(input.trim())
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
    if input == "/quit" {
        Some(SlashCommand::Quit)
    } else if input == "/compact" {
        Some(SlashCommand::Compact)
    } else if input == "/new" {
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
) -> Result<SlashCommandOutcome> {
    match command {
        SlashCommand::Model(argument) => {
            handle_model_command(terminal, state, setup, command_tx, argument.as_deref()).await?;
            Ok(SlashCommandOutcome::Continue)
        }
        SlashCommand::Quit => Ok(SlashCommandOutcome::Quit),
        SlashCommand::Compact => {
            state.set_compaction_active(true);
            command_tx
                .send(AgentCommand::Compact)
                .await
                .context("could not start compaction")?;
            Ok(SlashCommandOutcome::Continue)
        }
        SlashCommand::Session => {
            state.add_system_message(session_diagnostic(setup)?);
            Ok(SlashCommandOutcome::Continue)
        }
        SlashCommand::Name(None) => {
            state.add_system_message(session_diagnostic(setup)?);
            Ok(SlashCommandOutcome::Continue)
        }
        SlashCommand::Name(Some(name)) => {
            if setup.session.is_none() {
                state.add_system_message("sessions are disabled for this run");
            } else {
                command_tx
                    .send(AgentCommand::RenameSession { name })
                    .await
                    .context("could not rename the session")?;
            }
            Ok(SlashCommandOutcome::Continue)
        }
        SlashCommand::New => {
            let Some(repository) = setup.repository.as_ref() else {
                state.add_system_message("sessions are disabled for this run");
                return Ok(SlashCommandOutcome::Continue);
            };
            let session = repository
                .create()
                .map_err(|error| anyhow!(error.to_string()))?;
            let handle = session.clone();
            setup.session = Some(session);
            setup.initial_history.clear();
            setup.initial_transcript.clear();
            command_tx
                .send(AgentCommand::NewSession { session: handle })
                .await
                .context("could not create a new session")?;
            state.replace_history(&[]);
            state.set_session_info(setup.session_info()?);
            state.add_system_message(setup.context.diagnostic());
            Ok(SlashCommandOutcome::Continue)
        }
        SlashCommand::Resume => {
            let Some(repository) = setup.repository.as_ref() else {
                state.add_system_message("sessions are disabled for this run");
                return Ok(SlashCommandOutcome::Continue);
            };
            let path = match crate::session_picker::pick_path_in_terminal(terminal, repository) {
                Ok(Some(path)) => path,
                Ok(None) => return Ok(SlashCommandOutcome::Continue),
                Err(error) => {
                    state.add_system_message(format!("ri: {error}"));
                    return Ok(SlashCommandOutcome::Continue);
                }
            };
            if setup.session_info()?.is_some_and(|info| info.path == path) {
                state.add_system_message("that session is already active");
                return Ok(SlashCommandOutcome::Continue);
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
            state.replace_history(&setup.initial_transcript);
            state.set_session_info(setup.session_info()?);
            state.add_system_message(setup.context.diagnostic());
            Ok(SlashCommandOutcome::Continue)
        }
    }
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

async fn handle_model_command(
    terminal: &mut TerminalGuard,
    state: &mut AppState,
    setup: &mut AppSetup,
    command_tx: &mpsc::Sender<AgentCommand>,
    argument: Option<&str>,
) -> Result<()> {
    let Some(catalog) = setup.catalog.as_ref() else {
        state.add_system_message(
            "No configured models are available. Create ~/.ri/agent/models.json first.",
        );
        return Ok(());
    };

    let selected = if let Some(argument) = argument {
        catalog.resolve(None, Some(argument))
    } else {
        if catalog.models().is_empty() {
            state.add_system_message("models.json contains no selectable model");
            return Ok(());
        }
        let current = setup.selected.as_ref().map(|model| &model.model_ref);
        match crate::model_picker::pick_model_in_terminal(terminal, catalog, current)? {
            Some(selected) => Ok(selected),
            None => return Ok(()),
        }
    };

    match selected {
        Ok(selected) => match setup.provider.set_model(selected.clone()) {
            Ok(()) => {
                let name = selected.model_ref.display_name();
                let model_ref = selected.model_ref.clone();
                setup.selected = Some(selected);
                if let Err(error) = setup.remember_model(&model_ref) {
                    eprintln!("ri: warning: could not persist recent model selection: {error}");
                    state.add_system_message(format!(
                        "could not persist recent model selection: {error}"
                    ));
                }
                state.reduce(AgentEvent::ModelChanged(model_ref));
                state.reduce(AgentEvent::ContextLimitsUpdated(setup.provider.limits()));
                command_tx
                    .send(AgentCommand::RefreshContext)
                    .await
                    .context("could not refresh context for the selected model")?;
                state.add_system_message(format!("active model: {name}"));
            }
            Err(error) => state.add_system_message(error.to_string()),
        },
        Err(error) => state.add_system_message(error.to_string()),
    }
    Ok(())
}

fn print_and_flush(output: &mut impl Write, text: &str) -> Result<()> {
    write!(output, "{text}")?;
    output.flush()?;
    Ok(())
}

fn log_agent_event(event: &AgentEvent) {
    match event {
        AgentEvent::TurnStarted => tracing::info!(target: "ri", "turn started"),
        AgentEvent::AssistantMessageStarted => {
            tracing::debug!(target: "ri", "assistant message started")
        }
        AgentEvent::ToolExecutionStarted {
            call_id,
            name,
            arguments,
        } => tracing::debug!(
            target: "ri",
            call_id = %call_id,
            tool = %name,
            arguments_bytes = arguments.len(),
            "tool started"
        ),
        AgentEvent::ToolExecutionOutput {
            call_id,
            stream,
            chunk,
        } => tracing::trace!(
            target: "ri",
            call_id = %call_id,
            stream = ?stream,
            chunk_bytes = chunk.len(),
            "tool output"
        ),
        AgentEvent::ToolExecutionFinished {
            call_id,
            name,
            result,
        } => tracing::debug!(
            target: "ri",
            call_id = %call_id,
            tool = %name,
            success = result.metadata.success,
            exit_code = ?result.metadata.exit_code,
            timed_out = result.metadata.timed_out,
            cancelled = result.metadata.cancelled,
            truncated = result.metadata.truncated,
            duration_ms = result.metadata.duration.as_millis() as u64,
            "tool finished"
        ),
        AgentEvent::UsageUpdated(usage) => tracing::debug!(
            target: "ri",
            input_tokens = ?usage.input_tokens,
            output_tokens = ?usage.output_tokens,
            total_tokens = ?usage.total_tokens,
            "provider usage"
        ),
        AgentEvent::ContextUsageUpdated(usage) => tracing::debug!(
            target: "ri",
            input_tokens = usage.current_tokens(),
            estimated = matches!(usage.source, ri_core::UsageSource::Estimated),
            context_window = ?usage.context_window,
            "context usage"
        ),
        AgentEvent::CompactionStarted { automatic } => {
            tracing::info!(target: "ri", automatic, "compaction started")
        }
        AgentEvent::CompactionFinished {
            automatic,
            before_tokens,
            after_tokens,
        } => tracing::info!(
            target: "ri",
            automatic,
            before_tokens,
            after_tokens,
            "compaction finished"
        ),
        AgentEvent::CompactionFailed { message } => tracing::warn!(
            target: "ri",
            message_bytes = message.len(),
            "compaction failed"
        ),
        AgentEvent::ModelChanged(model) => tracing::info!(
            target: "ri",
            provider = %model.provider,
            model = %model.model,
            "model changed"
        ),
        AgentEvent::SessionChanged { info } => tracing::debug!(
            target: "ri",
            session_id = %info.id,
            message_count = info.message_count,
            "session changed"
        ),
        AgentEvent::TurnFinished { reason } => tracing::info!(
            target: "ri",
            reason = crate::json_output::stop_reason_name(reason),
            "turn finished"
        ),
        AgentEvent::Error(error) => tracing::error!(
            target: "ri",
            message_bytes = error.message.len(),
            "agent error"
        ),
        AgentEvent::AssistantTextDelta { .. }
        | AgentEvent::AssistantTextItem { .. }
        | AgentEvent::AssistantRefusalDelta { .. }
        | AgentEvent::AssistantRefusalItem { .. }
        | AgentEvent::AssistantThinkingDelta { .. }
        | AgentEvent::AssistantThinkingContentDelta { .. }
        | AgentEvent::AssistantThinkingItem { .. }
        | AgentEvent::ToolCallDelta { .. }
        | AgentEvent::AssistantMessageFinished { .. }
        | AgentEvent::ContextLimitsUpdated(_)
        | AgentEvent::SessionLoaded { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ri_core::ResolvedSettings;

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
                json: false,
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
    fn parses_json_prompt_and_rejects_interactive_json() {
        let options = Options::parse(["--json".to_owned(), "-p".to_owned(), "hello".to_owned()])
            .expect("json print mode should parse");
        assert!(options.json);
        assert_eq!(options.print_prompt.as_deref(), Some("hello"));
        assert!(Options::parse(["--json".to_owned()]).is_err());
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
    fn recognizes_direct_model_quit_and_unknown_commands() {
        assert!(matches!(
            slash_command("/compact"),
            Some(SlashCommand::Compact)
        ));
        assert!(matches!(slash_command("/quit"), Some(SlashCommand::Quit)));
        assert_eq!(model_command("/model"), Some(None));
        assert_eq!(
            model_command("  /model custom/coding  "),
            Some(Some("custom/coding".to_owned()))
        );
        assert_eq!(model_command("/modelish"), None);
        assert!(is_slash_input(" /compcat"));
        assert_eq!(unknown_command_name(" /compcat extra"), "/compcat");
    }
}
