# ri

`ri` is a standalone Rust coding agent for working in a local repository. It can stream responses from OpenAI-compatible providers, inspect and edit workspace files, run bounded shell commands, load repository instructions, and persist workspace-scoped sessions.

The current baseline is intended for dogfooding. It is not a hosted service and does not provide a built-in model or provider account.

## Features

- OpenAI Responses and Chat Completions-compatible APIs
- `read`, `write`, `edit`, and `bash` tools
- workspace-safe file access and bounded tool output
- hierarchical `AGENTS.md` context
- persistent JSONL sessions, resume, crash repair, and compaction
- interactive model picker and recent-model state
- anchored transcript scrollback and slash-command suggestions
- plain print mode and versioned JSON streaming mode
- private file logging through `RI_LOG`

## Install from source

Stable Rust is required.

```bash
cargo install --path crates/ri-cli --locked
ri --version
ri --help
```

The supported build and test targets are Linux, macOS, and Windows.

## models.json

Create `~/.ri/agent/models.json`, or point `RI_MODELS_FILE` at another file. A minimal OpenAI-compatible configuration is:

```json
{
  "providers": {
    "example": {
      "baseUrl": "https://example.invalid/v1",
      "api": "openai-responses",
      "apiKey": "$EXAMPLE_API_KEY",
      "models": [
        {
          "id": "example-model",
          "contextWindow": 128000,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

Set the secret in the environment before running `ri`; do not commit a plaintext API key. The supported API values are `openai-responses` and `openai-completions`.

Model selection precedence is:

1. CLI model/provider selection (`--model`, `--provider`)
2. the settings default
3. the workspace's recent model
4. the global recent model
5. the first configured model

## settings.json

Built-in settings are overridden by the global settings file and then the project settings file:

```text
built-in settings → ~/.ri/agent/settings.json → .ri/settings.json
```

The project file is relative to the discovered project root. CLI model selection overrides settings where applicable. Supported settings currently include `defaultProvider`, `defaultModel`, `context.enabled`, and `compaction.enabled`.

## AGENTS.md

`ri` loads `AGENTS.md` files from the applicable global, project, nested, and launch-directory locations. `AGENTS.override.md` replaces the normal file in the same directory. Use `--no-context` to disable context loading for a run. Invalid, unreadable, or oversized context is reported before the TUI starts.

## Interactive usage

```bash
ri
```

Useful commands include `/model`, `/new`, `/resume`, `/name [name]`, `/session`, `/compact`, and `/quit`. Type `/` while the agent is idle to show command suggestions, use Up/Down to select one, Enter to execute it, Tab to complete it for further editing, or Esc to dismiss the suggestions without clearing the input.

Use PgUp/PgDn or Ctrl+U/Ctrl+D to move through transcript scrollback; mouse-wheel and trackpad scrolling are also supported. The footer shows the distance from the latest output while scrolled upward. Streaming output follows the bottom only when the viewport is already at the bottom, so new output and turn completion do not interrupt reading older content. Submitting a new prompt resumes following the latest output.

`Esc` cancels an active operation when command suggestions are not visible. `Ctrl+C` cancels a busy turn and exits when the TUI is idle.

## Sessions

Sessions are workspace-scoped and stored below `~/.ri/agent/sessions`. The session history is append-only JSONL; interrupted tool calls are repaired when possible.

```bash
ri                         # start a new session
ri -c                      # continue the newest saved session
ri -r                      # choose a saved session
ri --session <id-or-path>  # open one session
ri --no-session             # use ephemeral persistence
```

The interactive equivalents are `/new` and `/resume`. Session metadata and compaction checkpoints are persisted alongside the transcript. A session writer lock is advisory and is released by the operating system when the owning process exits.

## Compaction

Context files and conversation history are projected for the selected model. Automatic compaction is enabled by default when the context budget requires it; `/compact` requests it manually. Set `compaction.enabled` to `false` in settings to disable automatic compaction, or use the runtime's normal error reporting when a selected model still cannot fit the request.

## Print mode

```bash
ri -p "Inspect the failing test and explain the fix"
ri -c -p "Continue the previous task"
```

Print mode writes only assistant text to stdout. Diagnostics, session information, and failures go to stderr. It exits with status 0 on success, 1 for runtime/provider/agent failure, and 2 for setup or command-line errors.

## JSON mode

```bash
ri --json -p "Inspect the repository"
ri -c --json -p "Continue the previous task"
```

JSON mode emits versioned NDJSON records on stdout. Every stdout line is JSON; diagnostics remain on stderr. It uses the same runtime, tools, sessions, cancellation, and exit-code contract as print mode.

## Logging

Enable diagnostic logging before starting the run:

```bash
RI_LOG=debug ri
RI_LOG=trace ri
```

Logs are written under `~/.ri/agent/logs/`. They are not generated retroactively, so a run started without `RI_LOG` cannot be diagnosed from a later log. Target filters and the `error`, `warn`, and `info` levels are also supported. Logs redact configured credentials and do not contain complete prompts or tool output; provider error diagnostics are sanitized and bounded. Logging failures are diagnostic warnings and do not change agent semantics.

## Configuration paths

- `RI_MODELS_FILE`, or `~/.ri/agent/models.json`
- `~/.ri/agent/settings.json`
- `<project>/.ri/settings.json`
- `~/.ri/agent/state.json` and its advisory `.lock` target
- `~/.ri/agent/sessions/<workspace-id>/`
- `~/.ri/agent/logs/`

If neither `HOME` nor `USERPROFILE` is available, setup-free commands such as `ri --help` and `ri --version` still work. Persistent operations that need a global path fail with an actionable error.

## Development

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked --no-fail-fast
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo build --release --locked -p ri
cargo bench -p ri --bench tui_render
```

The benchmark is a manual performance check, not a timing-sensitive CI gate. CI validates formatting, compilation, tests, Clippy, release builds, and source-install smoke tests on Linux, macOS, and Windows. Provider tests use mocks or local scripted HTTP servers; CI does not require model credentials.

## Dogfood smoke checklist

Run these checks with a real configured provider after installation:

- Fresh task: inspect a real repository, use `read`, `bash`, `edit`, and `write`, then run the relevant tests.
- Cancel: start a deliberately long safe command, press `Esc` or `Ctrl+C`, verify the prompt remains usable, then quit and resume.
- Resume: use `ri -c` and confirm the transcript, current context, and tools still work.
- Model switch: use `/model`, switch models, and verify footer limits, compaction, and recent-model persistence.
- Machine modes: pipe `ri -p`, `ri --json -p`, and their `-c` variants into another program; stdout must remain within its documented contract.
- Scrollback: generate more than one screen of output, then verify PgUp/PgDn, Ctrl+U/Ctrl+D, mouse/trackpad scrolling, the footer indicator, and stable anchoring while new output streams.
- Command suggestions: type `/` and `/mo`, then verify Up/Down, Tab, Esc, exact-command Enter behavior, and `/model` and `/name` arguments.
- Forced failures: start with `RI_LOG=debug ri`, try a bad command, a missing file, and an intentionally invalid temporary credential, then verify the full provider error is in the transcript and the sanitized status/body diagnostic is in `~/.ri/agent/logs/`.

A live provider smoke is deliberately manual. It is not part of CI and must be reported as skipped when no usable credentials or endpoint are configured.

## Current non-goals

Plugins, web search, Codex integration, MCP, skills, themes, Markdown rendering, session branching, new provider protocols, OAuth, remote execution, sandboxing, permission prompts, and public release automation are outside this baseline.
