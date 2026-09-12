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
- streaming assistant Markdown: emphasis, lists, quotes, links, and code blocks
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

Assistant responses render CommonMark plus strikethrough and task markers while streaming. Recognized code fences receive foreground syntax highlighting, including Rust, TypeScript/TSX, TOML, Dockerfile, and diff. To keep streaming responsive, an active fence over 16 KiB temporarily renders as plain code until the turn completes; finalized messages are highlighted without that limit. Fence labels and code prefixes remain visible; unknown, unlabeled, indented, and explicit plaintext blocks remain plain. Inline code is not syntax-highlighted; links show their destination. HTML and images have text-only fallbacks. Thinking, user/system messages, and tool output remain literal. Session/model text and print/JSON modes retain raw Markdown.

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

Markdown benchmarks cover mixed prose/list/Rust/TypeScript/TOML history, cached redraw/scroll, resize, active responses growing from 1 to 64 KiB, and a separate prose/list workload. Code-heavy benchmarks cover completed Rust fences at 8/32/64 KiB and streaming Rust growing through 1/8/32/64 KiB against cached history; the streaming labels identify the deliberate plain-code fallback above 16 KiB. Assertions protect cache behavior: only the active answer is reparsed on a content delta, unchanged frames and scrolling reuse rows, and resize reflows once. Full active-message parsing is deliberately O(active message), not an incremental Markdown parser.

The benchmark is a manual performance check, not a timing-sensitive CI gate. CI validates formatting, compilation, tests, Clippy, release builds, and source-install smoke tests on Linux, macOS, and Windows. Provider tests use mocks or local scripted HTTP servers; CI does not require model credentials.

### Syntax-highlighting measurements

Local run on `aarch64-apple-darwin`, Rust 1.98.0 (Homebrew), using the repository's unchanged release profile (thin LTO, one codegen unit, stripped). These are observations, not CI thresholds:

| Workload | Time per draw |
| --- | ---: |
| First Rust highlight, 1 KiB (includes lazy setup) | 24.49 ms |
| Completed Rust, 8 / 32 / 64 KiB | 14.00 / 56.36 / 107.65 ms |
| Streaming Rust, 1 / 8 / 32 / 64 KiB (highlighted / highlighted / plain / plain) | ~1.9 / 13.6 / 3.3 / 6.6 ms |
| Cached Rust redraw / scroll, 8–64 KiB history | 0.12 ms |
| Mixed Markdown history, 100 entries, cold layout | 80.22 ms |
| Mixed history cached redraw / scroll | 0.11 ms |
| Mixed history resize (80 and 100 columns, two layouts) | 71.64 ms |
| Mixed active Markdown, 64 KiB | 51.43 ms |
| Prose/list Markdown, ~64 KiB | 5.17 ms |

Large active fences are capped at 16 KiB for syntax highlighting: in this run, 1 and 8 KiB fences were highlighted, while 32 and 64 KiB fences used the plain-code fallback until turn completion. That keeps large active fences around 3.3 and 6.6 ms instead of paying the completed-highlight costs of about 56.36 and 107.65 ms; finalized messages and historical entries remain highlighted/cached. No incremental syntax state across deltas is maintained. The first mixed-history measurement also encounters languages not warmed by the Rust workload.

`cargo build --release -p ri` produced **4,967,184 bytes before** and **7,474,320 bytes after** highlighting: **+2,507,136 bytes (50.5%, ~2.39 MiB)** on the same toolchain/profile. The bat-curated `two-face` bundle deliberately buys broad coding-agent language coverage, including TypeScript/TOML/Dockerfile, rather than stock syntect alone. Syntax assets and ri's foreground-only palette initialize lazily once. Feature inspection (`cargo tree -p ri -e features`) confirms only `two-face`'s `syntect-fancy` backend, with syntect parsing/dump support and pure-Rust regex support; no Oniguruma or separate theme crate is enabled. The binary increase includes the highlighting/regex implementation, not just embedded grammar data.

Interactive provider dogfooding and cross-platform execution were not performed for these measurements; automated TestBackend, streaming-prefix, Unicode, multiline-state, terminal-safety, and cache-lifecycle tests cover the renderer locally.

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

- Highlighted Markdown: request Rust, TypeScript/TSX, Python, Bash, JSON, TOML, YAML, Dockerfile, SQL, diff, unknown and unlabeled fences. Include multiline comments/strings, 100+ source lines, long lines, and wide Unicode. Resize narrowly during an unfinished streaming fence and scroll through completed responses; check prefixes, colors, alignment, flicker, and streaming CPU/latency.

A live provider smoke is deliberately manual. It is not part of CI and must be reported as skipped when no usable credentials or endpoint are configured.

## Internal capability architecture

`ri-core` separates internal capability composition from an explicitly invoked external-plugin protocol/lifecycle API. Explicitly loaded external tools adapt to the same `Tool` and `ToolRegistry` interfaces as built-ins.

```text
Application bootstrap → PluginRegistry → AgentRuntimeConfig → AgentRuntime
                            └─ ToolRegistry: read, write, edit, bash
```

- A `Tool` is one model-callable capability. `ToolRegistry` provides ordered definitions, presentation, lookup, and execution.
- `ToolRegistry::new()` (and `Default`) creates an empty registry. `register(Arc<dyn Tool>)` derives the canonical name from the tool definition and rejects duplicate names with `ToolRegistryError::DuplicateTool`, without replacing the existing tool. Definitions retain registration order.
- `builtin_tool_registry()` registers the statically compiled `read`, `write`, `edit`, and `bash` tools through that same boundary, with unchanged schemas and behavior. `builtin_plugins()` wraps them in the host-side `PluginRegistry` capability container.
- The application supplies `AgentRuntimeConfig.plugins`. Convenience runtime constructors and `AgentRuntimeConfig::new()` use the built-in bootstrap; `PluginRegistry::default()` is empty. Custom callers can register tools, wrap the registry with `PluginRegistry::new(Arc::new(tools))`, and inject it through configuration.

Capability composition lives outside `AgentRuntime` so the agent loop only consumes the prepared tool registry, rather than deciding which tools exist or how they are supplied. `PluginRegistry` currently contains only tools; no speculative provider, command, context, or hook interfaces are defined.

### External plugin protocol foundation

```text
plugin.json → validated manifest → PluginProcess
                                       │
                              JSON-RPC 2.0 / NDJSON
                                       │
                                       ▼
                               external executable
```

External plugins are **not discovered or loaded during normal `ri` startup**. Project-local files never automatically execute plugin code. Default tools remain exactly `read`, `write`, `edit`, and `bash`. Explicit host callers may load and register external tools through the API below; this is not a Rust dynamic-library ABI.

An explicit host caller uses `load_plugin_manifest(path)`, then `PluginProcess::start(loaded).await`. Loading a manifest only parses and validates it; `start` executes its command directly with a separate argument vector, without host-added shell wrapping. The child runs in the canonical manifest directory and inherits the host environment. Relative executable paths such as `./ri-plugin-echo` resolve against that directory; bare commands use executable lookup. This is not a sandbox: only explicitly start trusted executables.

Manifest version 1 uses strict camelCase fields (unknown fields are errors):

```json
{
  "manifestVersion": 1,
  "id": "dev.example.echo",
  "name": "Echo",
  "version": "0.1.0",
  "protocolVersion": "ri.plugin.v1",
  "entrypoint": {
    "command": "./ri-plugin-echo",
    "args": []
  }
}
```

`args` defaults to `[]`. IDs contain at most 128 bytes of lowercase ASCII letters, digits, `.`, `-`, or `_`, starting with a letter or digit. Name, version, and command must be non-empty after trimming. The release version is opaque, not necessarily SemVer. Manifests contain no configuration or secrets.

Transport and lifecycle:

- stdin/stdout carry UTF-8 JSON-RPC 2.0, one JSON object per line (NDJSON), at most 1 MiB per frame excluding its newline. Stdout is protocol-only; malformed or oversized output fails the connection.
- stderr is diagnostics-only. The host continuously drains it, retaining the first 64 KiB and a truncation flag. Bounded diagnostics accompany startup/shutdown failures.
- Startup sends `initialize` and allows at most 5 seconds for initialization. The negotiated protocol must be `ri.plugin.v1`; returned plugin ID, name, and version must exactly match the manifest. Failed initialization terminates and reaps the child.
- Request IDs start at 1 and increase monotonically. Responses are matched by ID, including out-of-order responses. Each response has exactly one `result` (including `null`) or JSON-RPC `error`. Generic `request` callers choose their own post-startup deadlines; cancellation removes pending request state.
- `recv_notification` exposes uninterpreted notifications. The notification queue and pending-request count are bounded to 64 each. Notification overflow fails the connection rather than blocking response routing. Plugin-to-host requests are not supported.
- `capabilities()` exposes advertised capabilities; unknown capability keys are preserved. Advertising `tools` does not register or enable tools.
- `shutdown().await` sends `shutdown` with `{}` params. After the response, stdin closes and the host waits for exit. The entire graceful shutdown has a 2-second deadline; failure or timeout triggers termination and reaping. Kill-on-drop is an emergency fallback, not the normal shutdown path.

Initialization exchange (each object is transmitted on a single line; the host version is the compiled package version):

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"ri.plugin.v1","host":{"name":"ri","version":"0.1.0"}}}
```

```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"ri.plugin.v1","plugin":{"id":"dev.example.echo","name":"Echo","version":"0.1.0"},"capabilities":{"tools":false}}}
```

Shutdown exchange, assuming no intervening requests:

```json
{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}
{"jsonrpc":"2.0","id":2,"result":null}
```

### External tool capability

Tool support is an additive extension of `ri.plugin.v1`. Advertising `capabilities.tools = true` alone does not execute `tools/list` or register tools. An explicit host composes the capability:

```rust,no_run
use ri_core::{builtin_tool_registry, load_plugin_manifest, ExternalToolSet, PluginProcess};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let process = PluginProcess::start(load_plugin_manifest("plugin.json")?).await?;
let tools = ExternalToolSet::load(&process).await?;
let mut registry = builtin_tool_registry();
tools.register_into(&mut registry)?;
// Inject the registry through PluginRegistry and AgentRuntimeConfig.
// Keep the process alive while its tools are in use, then shut it down explicitly.
process.shutdown().await?;
# Ok(())
# }
```

- With `capabilities.tools = false`, `ExternalToolSet::load` returns an empty set without a `tools/list` request.
- With `true`, each explicit load sends exactly one `tools/list` request with `{}` params, validates the complete list, and retains an immutable snapshot in advertised order. There is no polling or dynamic refresh.
- A plugin may expose at most 128 tools. Names are 1–64 bytes, contain only ASCII letters, digits, `_` or `-`, and start with a letter or digit. Names retain their case and are not prefixed or rewritten.
- Descriptions may be omitted, but supplied descriptions must be nonblank and at most 16 KiB of UTF-8. `inputSchema` must be a JSON object; its contents are preserved, without requiring `additionalProperties: false`. The existing 1 MiB frame bound still applies.
- External tool names share the global namespace with built-ins and other registered tools. Duplicate descriptors are invalid. Registration checks the entire set before mutation: a plugin cannot override `read`, `write`, `edit`, `bash`, or any other existing entry. Collisions are errors, not silent renames or replacements.

`tools/list` exchange (shown pretty-printed; each frame is one line on the wire):

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "echo",
        "description": "Echo input text",
        "inputSchema": {
          "type": "object",
          "properties": {"text": {"type": "string"}},
          "required": ["text"],
          "additionalProperties": false
        }
      }
    ]
  }
}
```

`tools/call` forwards arguments unchanged:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {"name": "echo", "arguments": {"text": "hello"}}
}
```

Successful tool result:

```json
{"jsonrpc":"2.0","id":3,"result":{"content":"hello","isError":false}}
```

Expected tool-level failure:

```json
{"jsonrpc":"2.0","id":3,"result":{"content":"file does not exist","isError":true}}
```

`content` is a required string; `isError` defaults to `false`. Capability messages tolerate future additive fields. `isError: false` becomes normal tool success; `isError: true` becomes a normal tool-level failure visible to the model. JSON-RPC errors (unsupported method, malformed request, internal plugin RPC failure), malformed results, and transport failures instead become host `ToolError`s identifying the plugin and tool. Execution duration is recorded; other metadata retains standard defaults.

External tools use normal tool transcript events, tool-result history, and fallback presentation (name and bounded JSON argument preview). No plugin-specific agent execution path or synchronous presentation RPC exists.

Cancelling an ri turn stops waiting for an external tool call and drops its pending response; late responses are ignored. `ri.plugin.v1` does not yet send cooperative cancellation to the plugin: remote computation may continue until it returns or the process is shut down. Generic request callers still choose their own deadlines.

There is no streamed tool output, plugin-specific presentation, workspace context in `tools/call`, dynamic `tools/list` refresh, or automatic external plugin loading. Plugin discovery/install, activation/settings, MCP, and web search remain unimplemented.

## Current non-goals

Plugin discovery/install, plugin activation/settings, WASM, provider plugins, command plugins, context plugins, web search, Codex integration, MCP, skills, user-selectable themes, semantic/LSP highlighting, session branching, new provider protocols, OAuth, remote execution, sandboxing, permission prompts, and public release automation are outside this baseline.
