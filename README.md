# ai.rs

Rust port of the focused [`pi`](https://github.com/earendil-works/pi) AI and
core agent runtime surfaces.

This repository tracks the upstream `pi` package layout where the scoped Rust
surface exists:

- [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai):
  shared LLM types, model lookup, API registry, streaming helpers, provider
  adapters, OAuth helpers, faux provider, and utility helpers.
- [`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent):
  core agent state, direct agent loop, queueing, lifecycle events, tool
  execution, and loop hooks.

The goal is 1:1 behavior with upstream `pi` for the Rust APIs in scope before
Rust-specific API polish.

## All Crates

| Crate | Upstream package | Description |
| --- | --- | --- |
| [`ai`](crates/ai) | [`packages/ai`](https://github.com/earendil-works/pi/tree/main/packages/ai) + core [`packages/agent`](https://github.com/earendil-works/pi/tree/main/packages/agent) | Unified LLM API plus the core agent loop runtime for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and GitHub Copilot-compatible routing. |

For API usage, model lookup, tool calling, streaming, OAuth, agent loop APIs,
provider scope, and the detailed upstream file mapping, see
[`crates/ai/README.md`](crates/ai/README.md).

## Port Mapping

| Upstream `pi` area | Rust location | Status |
| --- | --- | --- |
| `packages/ai/src/types.ts` | [`crates/ai/src/types.rs`](crates/ai/src/types.rs) | Shared message, context, tool, model, usage, event, and provider compatibility types. Image input/tool-result image content is retained. Image generation types are out of scope. |
| `packages/ai/src/stream.ts` | [`crates/ai/src/stream.rs`](crates/ai/src/stream.rs) | `stream`, `complete`, `stream_simple`, and `complete_simple`. |
| `packages/ai/src/api-registry.ts` | [`crates/ai/src/api_registry.rs`](crates/ai/src/api_registry.rs) | API provider registry with built-in dispatch. |
| `packages/ai/src/models.ts` | [`crates/ai/src/models.rs`](crates/ai/src/models.rs) | Generated model catalog filtered to the active provider scope. |
| `packages/ai/src/providers/openai-completions.ts` | [`crates/ai/src/providers/openai_completions.rs`](crates/ai/src/providers/openai_completions.rs) | OpenAI Chat Completions-compatible streaming. |
| `packages/ai/src/providers/openai-responses.ts` | [`crates/ai/src/providers/openai_responses.rs`](crates/ai/src/providers/openai_responses.rs) | OpenAI Responses-compatible streaming. |
| `packages/ai/src/providers/anthropic.ts` | [`crates/ai/src/providers/anthropic.rs`](crates/ai/src/providers/anthropic.rs) | Anthropic Messages-compatible streaming. |
| `packages/ai/src/providers/faux.ts` | [`crates/ai/src/providers/faux.rs`](crates/ai/src/providers/faux.rs) | Deterministic test provider. |
| `packages/ai/src/providers/register-builtins.ts` | [`crates/ai/src/providers/register_builtins.rs`](crates/ai/src/providers/register_builtins.rs) | Registers only the active built-in stream APIs. |
| `packages/ai/src/providers/transform-messages.ts` | [`crates/ai/src/providers/transform_messages.rs`](crates/ai/src/providers/transform_messages.rs) | Cross-provider message normalization. |
| `packages/agent/src/types.ts` | [`crates/ai/src/agent_types.rs`](crates/ai/src/agent_types.rs) | Core agent types and event shapes. |
| `packages/agent/src/agent-loop.ts` | [`crates/ai/src/agent_loop.rs`](crates/ai/src/agent_loop.rs) | Direct agent loop. |
| `packages/agent/src/agent.ts` | [`crates/ai/src/agent.rs`](crates/ai/src/agent.rs) | Stateful agent wrapper. |

## Supported Provider Scope

The active built-in provider API surface is intentionally narrow:

- OpenAI Chat Completions: `openai-completions`
- OpenAI Responses: `openai-responses`
- Anthropic Messages: `anthropic-messages`
- GitHub Copilot OAuth and dynamic headers over compatible routes

Cloudflare, Bedrock, Google, Mistral, Azure OpenAI Responses, OpenAI Codex
Responses, image generation, and other broad provider-specific APIs are not
part of the active built-in provider surface in this port. PRs to add support
for additional providers are welcome.

The TypeScript coding-agent harness, CLI, and TUI from
[`pi`](https://github.com/earendil-works/pi) are not included. The core agent
loop is implemented directly inside the `ai` crate.

## Development

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ai
cargo test --workspace
```

Useful narrower test commands:

```bash
cargo test -p ai --lib       # unit tests
cargo test -p ai --doc       # doc tests
cargo test -p ai --tests     # integration tests, if present
```

## License

MIT
