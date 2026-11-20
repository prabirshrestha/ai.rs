# Harness

Run coding agents that have their own loop, such as Claude Code, Codex,
Copilot, Cursor, OpenCode, fx, Pi or ai.rs's own agent, behind one Rust API.
Each agent works in a sandbox with its own tools. You get the same
`AgentEvent`s and `Message`s that the ai.rs [`Agent`](../../README.md#agent-core)
gives you.

This is a Rust port of the AI SDK harness (`vercel/ai`, `packages/harness`
and its adapter packages), built on ai.rs types. Nothing runs in JavaScript.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Harness agent](#harness-agent)
  - [Settings](#settings)
  - [Changing one call with `prepare_call`](#changing-one-call-with-prepare_call)
  - [Tool filtering](#tool-filtering)
- [Sessions](#sessions)
  - [Turns and events](#turns-and-events)
  - [Steering](#steering)
  - [Resume, detach, stop and destroy](#resume-detach-stop-and-destroy)
  - [Probing a harness](#probing-a-harness)
- [Sandboxes](#sandboxes)
- [Host tools and approvals](#host-tools-and-approvals)
- [Adapters](#adapters)
  - [Claude Code](#claude-code)
  - [Codex](#codex)
  - [ACP agents: Copilot, Cursor, fx and others](#acp-agents-copilot-cursor-fx-and-others)
  - [OpenCode](#opencode)
  - [fx](#fx)
  - [Pi and native](#pi-and-native)
- [Authentication](#authentication)
- [Trying it on your machine](#trying-it-on-your-machine)
- [Differences from the AI SDK harness](#differences-from-the-ai-sdk-harness)

## Features

Everything is under the `harness` feature. Each adapter has a feature of its
own, and all of them except fx are on by default:

| Feature | Adapter | On by default |
| --- | --- | --- |
| `harness` | The core API, `HostSandbox`, `HarnessAgent` | yes |
| `harness-native` | ai.rs's own agent loop on an ai.rs model | yes |
| `harness-pi` | Pi on the ai.rs agent loop | yes |
| `harness-claude-code` | Claude Code over `stream-json` | yes |
| `harness-codex` | Codex over `codex app-server` | yes |
| `harness-acp` | Any Agent Client Protocol agent | yes |
| `harness-github-copilot` | GitHub Copilot CLI (ACP) | yes |
| `harness-cursor` | Cursor CLI (ACP) | yes |
| `harness-opencode` | OpenCode over `opencode serve` | yes |
| `harness-fx` | fx, with its core on wasmtime | no |

```toml
[dependencies]
ai = { version = "*", default-features = false, features = ["harness-claude-code", "harness-codex"] }
```

## Quick start

```rust
use std::sync::Arc;

use ai::Message;
use ai::harness::adapters::claude_code::{ClaudeCodeHarness, ClaudeCodeHarnessSettings};
use ai::harness::{
    Harness, HarnessAgent, HarnessAgentSessionOptions, HarnessAgentSettings, HarnessSandbox,
    HostSandbox,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Run the `claude` you have installed and signed in to.
    let harness: Arc<dyn Harness> = Arc::new(ClaudeCodeHarness::new(ClaudeCodeHarnessSettings {
        executable: Some("claude".into()),
        ..Default::default()
    }));

    let agent = HarnessAgent::new(
        harness,
        HarnessAgentSettings {
            id: Some("refactor-agent".into()),
            model: Some("sonnet".into()),
            instructions: Some("You are a careful refactoring assistant. Prefer minimal diffs.".into()),
            ..Default::default()
        },
    )?;

    let work_dir = "/path/to/project";
    let sandbox = HarnessSandbox::Basic(Arc::new(HostSandbox::new(work_dir)));
    let session = agent
        .create_session(HarnessAgentSessionOptions::new(sandbox, work_dir))
        .await?;

    let turn = agent
        .prompt(session.as_ref(), Message::user_text("Rename `foo` to `bar`."), Default::default())
        .await?;
    let (events, messages) = turn.collect().await?;
    println!("{} events, {} new messages", events.len(), messages.len());

    session.destroy().await?;
    Ok(())
}
```

## Harness agent

`HarnessAgent` is the AI SDK's `HarnessAgent`. It holds the settings every
turn uses, so you set the model and instructions once. Each turn still sends
them to the harness, because harnesses take them per turn: Codex on each
`turn/start`, ACP agents on the session before a prompt, and Claude Code on
each run. A turn can also pick up a different model without starting a new
session.

### Settings

| Field | What it does |
| --- | --- |
| `id` | Returned by `agent.id()` |
| `model` | The model id every turn uses, in the harness's own ids. `None` runs the CLI's default. |
| `thinking_level` | The effort every turn uses |
| `instructions` | Added to the agent's system prompt |
| `skills` | Skills written into the sandbox for the agent |
| `tools` | Host tools the agent may call, which your process runs |
| `headers` | Extra HTTP headers for model requests. The harness's own headers (`authorization`, `x-api-key`, `user-agent`, `x-client-app`) are refused. |
| `permission_mode` | Which builtin tools run without asking. Defaults to `AllowAll`. |
| `active_tools` / `inactive_tools` | Turn builtin and host tools on or off by name |
| `response_format` | Text, or JSON with an optional schema |
| `before_tool_call` / `after_tool_call` | Approve, block or observe tool calls |
| `prepare_call` | Change one call's prompt and settings |

`HarnessAgent::new` checks the settings against the harness, as the TS
constructor does. It refuses:

- both `active_tools` and `inactive_tools`
- a tool name the harness and your tools don't have
- tool filtering on a harness that can't filter or ask
- a permission mode other than `AllowAll` on a harness that can't ask before
  its builtin tools run
- a managed header
- a host tool named `askUserQuestions` when the harness reserves that name

### Changing one call with `prepare_call`

`prepare_call` gets the call already filled with the agent's settings,
together with the `options` value you pass to `prompt`. It returns the call
to run. The next call starts from the agent's settings again.

```rust
use std::sync::Arc;

use ai::harness::{HarnessAgentCallOptions, HarnessCall, PrepareCallFn};
use serde_json::json;

let prepare: PrepareCallFn = Arc::new(|mut call: HarnessCall| {
    Box::pin(async move {
        if call.options.as_ref().and_then(|o| o["quick"].as_bool()) == Some(true) {
            call.model = Some("haiku".into());
        }
        Ok(call)
    })
});

// HarnessAgentSettings { prepare_call: Some(prepare), .. }
agent
    .prompt(
        session.as_ref(),
        Message::user_text("Summarize the diff."),
        HarnessAgentCallOptions {
            options: Some(json!({ "quick": true })),
            ..Default::default()
        },
    )
    .await?;
```

### Tool filtering

`active_tools` and `inactive_tools` name builtin tools (as the harness lists
them in `builtin_tools()`) and host tools together. Host tools that are
filtered out are never offered to the agent. Builtins that are filtered out
become a `BuiltinToolFilter` on the session: the harness turns them off
natively where it can, and otherwise denies the call with "Tool '…' is
inactive due to the HarnessAgent tool filtering policy."

## Sessions

`HarnessAgent::create_session` starts a `HarnessSession`. You can also drive
a `Harness` directly with `harness.start(HarnessStartOptions)` and pass a
`HarnessTurnOptions` to each `session.prompt(...)`. That is the lower-level
contract every adapter implements, the AI SDK's `HarnessV1`.

### Turns and events

`prompt` returns a `HarnessTurn`. It is a stream of `HarnessEvent`s, then the
turn's new messages:

- `HarnessEvent::Agent(AgentEvent)`: the same events `ai::Agent` emits.
  Builtin tools appear as `ToolExecutionStart` / `ToolExecutionEnd` under
  their harness names.
- `HarnessEvent::Compaction { .. }`: the agent compacted its context.
- `HarnessEvent::FileChange { kind, path }`: the agent changed a file.
- `HarnessEvent::Notice { level, message }`: a warning or diagnostic.

```rust
use futures::StreamExt as _;

let mut turn = agent.prompt(session.as_ref(), Message::user_text("hi"), Default::default()).await?;
while let Some(event) = turn.next().await {
    println!("{event:?}");
}
let messages = turn.result().await?;
```

A cancelled `cancellation_token` interrupts the turn. The turn then ends with
`StopReason::Aborted`.

### Steering

Where the harness supports it (Codex, ACP agents), add a user message to the
running turn:

```rust
if turn.supports_steer() {
    turn.steer(Message::user_text("Also update the tests.")).await?;
}
```

### Resume, detach, stop and destroy

- `detach()` leaves the agent running and returns a `HarnessResumeState`.
- `stop()` stops the agent and returns a `HarnessResumeState`.
- `destroy()` stops the agent and deletes its state.

To pick a session up later, pass the state as `resume` with the original
`session_id`:

```rust
let state = session.stop().await?;
let mut options = HarnessAgentSessionOptions::new(sandbox, work_dir);
options.session_id = Some(session.session_id().to_string());
options.resume = Some(state);
let session = agent.create_session(options).await?;
```

`HarnessResumeState` is serde-serializable, so you can store it anywhere.

### Probing a harness

`session.probe()` reads, from a short-lived process, whether the CLI is
signed in, the account, its models with the thinking levels each takes (and
which one is the default), and its slash commands. Use it to fill a model
picker before any turn runs. Claude Code, Codex and the ACP agents support
it.

```rust
let probe = session.probe().await?;
for model in &probe.models {
    println!("{} {:?} default={}", model.id, model.thinking_levels, model.is_default);
}
```

## Sandboxes

A harness runs its agent through a `SandboxSession`. That is a trait for
spawning processes, running commands and reading or writing files, so the
agent can run on this machine, in a container, or anywhere you implement it.

- `HostSandbox` runs commands with `bash -c` on this machine, as this user,
  and reads and writes files in place. Use it to drive the CLIs a person has
  installed and signed in to. It gives no isolation.
- `HarnessSandbox::Network(...)` wraps a sandbox that can also rewrite
  outgoing requests. With it, adapters keep credentials out of the sandbox
  (credential brokering). Without it, they forward the credentials as
  environment variables and print a warning.

## Host tools and approvals

Host tools are ordinary ai.rs `AgentTool`s (`AgentToolBuilder`). The adapter
serves them to the agent over MCP, and your process runs them.
`before_tool_call` is asked before each host tool, and before each builtin
tool that `permission_mode` does not allow, and can block it.
`after_tool_call` sees each result.

Questions the agent asks the user (Claude Code's `AskUserQuestion`, Codex's
`requestUserInput`, ACP elicitations) arrive as the `askUserQuestions` tool,
which `before_tool_call` can answer.

## Adapters

Every adapter has a settings struct with `Default`. Settings marked
`[rust]` in the docs are additions with no TS counterpart.

### Claude Code

`ClaudeCodeHarness::new(ClaudeCodeHarnessSettings { .. })` runs
`claude --input-format stream-json --output-format stream-json`.

- `executable`: run your own `claude` instead of installing the pinned CLI
- `env`, `extra_args` and `additional_directories` for the process
- `effort` and `thinking`, `max_turns`, `max_budget_usd`, `mcp_servers`,
  `question_timeout_ms`

### Codex

`CodexHarness::new(CodexHarnessSettings { .. })` runs `codex app-server`.

- `executable`: run your own `codex`
- `env`, and `extra_args` (placed before `app-server`)
- `codex_config`: Codex config keys, passed through as-is
- `reasoning_effort`, `web_search`, `mcp_servers`
- `shared_app_server`: run every session of the harness on one
  `codex app-server` (one per sandbox, command and environment) instead of one
  process each. Each session gets only its own thread's messages, the
  process is initialized once, and the last session to stop ends it.

A turn that names no model runs Codex's configured default. The one
exception is when the adapter installs its own provider (AI Gateway or an API
key), where `gpt-5.5` is the default.

### ACP agents: Copilot, Cursor, fx and others

`AcpHarness::new(AcpHarnessSettings::new(..))` drives any Agent Client
Protocol agent over stdio. Presets:

- `create_github_copilot(GitHubCopilotHarnessSettings { .. })` runs `copilot --acp --stdio`
- `create_cursor(CursorHarnessSettings { .. })` runs `agent acp`

Both presets take `executable_override`, `extra_args` and `env`. For another
agent, such as `fx acp` or `opencode acp`, fill in `AcpHarnessSettings`:

```rust
use std::sync::Arc;

use ai::harness::adapters::acp::{
    AcpClientApp, AcpHarness, AcpHarnessSettings, AcpModelMapping, AcpSource,
};

let mut settings = AcpHarnessSettings::new(
    "fx",
    AcpClientApp { name: "my-app".into(), version: "1".into() },
    AcpSource::InstallCommand { command: "true".into() },
    "fx",
    AcpModelMapping::SessionConfigOption { path: "model".into() },
    Arc::new(|_| None),
);
settings.args = vec!["acp".into()];
settings.executable_override = Some("fx".into());
let harness = AcpHarness::new(settings)?;
```

### OpenCode

`OpenCodeHarness::new(OpenCodeHarnessSettings { .. })` runs
`opencode serve` and talks to it over HTTP. It takes `opencode_config`,
`provider`, `reasoning_variant`, `mcp_servers` and `executable_override`.

### fx

`FxHarness::new(FxHarnessSettings { .. })`, behind `harness-fx`, runs fx's
core (`libfx`, a WebAssembly module) in-process on wasmtime. Its sandbox
tools, MCP servers, skills and checkpoints all run in Rust. It signs in with
AI Gateway, or with a Codex or Grok subscription found in `~/.fx`.

### Pi and native

- `PiHarness::new(PiHarnessSettings { .. })` runs Pi in-process on the ai.rs
  agent loop, with Pi's tools, system prompt, skills and session log.
- `NativeHarness::new(model)` runs ai.rs's own agent loop with the same file
  and shell tools, on any ai.rs model.

Both run in your process, so there is no CLI to sign in to. They use the
ai.rs model you give them.

## Authentication

Adapters that run a CLI take `auth`, a `HarnessAuthentication`:

- `Auto` (the default) picks a route from the environment: AI Gateway first,
  then a direct API key, then the CLI's own subscription login, which is read
  and forwarded to the sandbox.
- `AiGateway` routes through AI Gateway.
- `Choice(Direct)` uses a direct API key.
- `Environment(map)` uses only `map`. An empty map leaves the CLI on its own
  login, provider and default model, as configured on this machine. Use this
  with `HostSandbox` when the person has already signed in to the CLI.

## Trying it on your machine

The `harness_live` example probes a CLI you have installed and signed in to,
then runs a one-word turn on it:

```sh
cargo run -p ai --example harness_live -- claude
cargo run -p ai --example harness_live -- codex --shared
cargo run -p ai --example harness_live -- copilot
cargo run -p ai --example harness_live -- cursor
cargo run -p ai --example harness_live -- acp opencode acp --model github-copilot/claude-sonnet-5.5
cargo run -p ai --example harness_live -- acp fx acp
```

`--no-turn` only probes, `--prompt <text>` changes the prompt, and
`--model <id>` sets the agent's model.

## Differences from the AI SDK harness

The port follows `vercel/ai` (commit `04a8f38`) with ai.rs types in place of
AI SDK ones:

- **Results.** A turn is a `HarnessTurn` of ai.rs events and messages, not
  an AI SDK `GenerateTextResult` / `StreamTextResult`. There is no
  `stopWhen`, output parsing or lifecycle callbacks (`onStart`, `onStepEnd`
  and the rest), because the event stream carries the same information.
- **Context.** `toolsContext` and `runtimeContext` are whatever your tool
  closures and `prepare_call` capture.
- **Tool approval.** The per-tool `toolApproval` map is `before_tool_call`.
- **Turn suspend and continue.** `suspendTurn` / `continueFrom` are not
  ported.
- **Not ported:** durable Pi, Cline, Grok Build, Deep Agents,
  `workflow-harness`, and deprecated settings (`sandbox`, `onSandboxSession`,
  `mintBridgeToken` and similar).
- **Added in Rust:**
  - `HostSandbox`
  - `HarnessSession::probe`
  - Codex `shared_app_server`
  - launch passthroughs: `executable`, `extra_args` and `env`
  - `thinking_level` on the agent
