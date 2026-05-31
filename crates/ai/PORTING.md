# ai package porting notes

This crate ports the focused `@earendil-works/pi-ai` and `@earendil-works/pi-agent`
surfaces into Rust. The comparison target for this checkpoint is
`earendil-works/pi` commit `3911d6f5`.

## Scope

Included:

- OpenAI Responses provider behavior.
- OpenAI Chat Completions provider behavior.
- Anthropic Messages provider behavior.
- GitHub Copilot dynamic headers and OAuth flow.
- Anthropic OAuth flow.
- Agent and agent-loop runtime, exposed from this `ai` crate.

Excluded by current scope:

- The TypeScript agent harness.
- A separate Rust agent crate.
- Broad provider parity outside the focused provider set, except where shared
  model catalogs, registries, or handoff behavior require support code.
- Built-in stream providers for Mistral, Google, Azure OpenAI Responses,
  OpenAI Codex Responses, Bedrock, or other non-focused provider APIs.

## Port Parity

The Rust implementation keeps the TypeScript provider contracts for streaming
events, assistant message shape, usage accounting, cache/session metadata,
reasoning replay, tool-call conversion, tool-result image routing, retry
defaults, and provider compatibility flags.

The agent loop preserves the TypeScript lifecycle model:

- `agent_start` / `agent_end`
- `turn_start` / `turn_end`
- message start/update/end events
- tool execution start/update/end events
- queued steering and follow-up messages
- continuation validation
- sequential and parallel tool execution
- `before_tool_call`, `after_tool_call`, `should_stop_after_turn`, and
  `prepare_next_turn` hooks

The OAuth registry is intentionally focused on the current provider set:
`anthropic` and `github-copilot`.

The built-in stream registry is intentionally focused on:
`anthropic-messages`, `openai-completions`, and `openai-responses`.

## Rust Adaptations

The port is Rust-shaped where that improves API safety without changing the
provider contract:

- Errors use typed Rust enums (`Error`, `AgentError`) instead of untyped thrown
  JavaScript errors.
- Streaming is exposed through Rust streams and channel-backed event streams.
- Cancellation uses `tokio_util::sync::CancellationToken`.
- Tool implementations use an async trait rather than JavaScript object
  callbacks.
- Agent state mutations are async and mutex-protected for shared ownership.
- `before_tool_call` supports explicit replacement args and a mutable args
  handle so Rust tools can safely adjust arguments before execution.
- Provider HTTP uses `reqwest` and Tokio.

## Verification

Current verification gates:

- `cargo fmt -p ai --check`
- `cargo clippy -p ai --all-targets -- -D warnings`
- `cargo test -p ai --quiet`
- `PI_REQUIRE_LOCAL_4141=1 cargo test -p ai --test local_4141 --quiet`
- Filtered catalog parity against `/tmp/pi/packages/ai` generated catalogs for
  models using the focused API routes.
- `git diff --check`

The local `4141` integration tests currently cover:

- OpenAI Responses text streaming with `gpt-5.5` low effort.
- OpenAI Responses agent prompt with `gpt-5.5` low effort.
- OpenAI Responses agent tool execution with `gpt-5.5` low effort.
- OpenAI Chat Completions text streaming.
- OpenAI Chat Completions agent prompt.
- OpenAI Chat Completions agent tool execution.

`gpt-5.5` is not accepted by the local `/chat/completions` endpoint at this
checkpoint, so Chat Completions local tests default to `gpt-5.2`.
