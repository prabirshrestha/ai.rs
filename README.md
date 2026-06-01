# ai.rs

This is the Rust port of the focused [`pi`](https://github.com/earendil-works/pi)
AI and core agent runtime surfaces.

- [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai):
  shared LLM types, model lookup, API registry, streaming helpers, provider
  adapters, OAuth helpers, faux provider, and utility helpers.
- [`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent):
  core agent state, direct agent loop, queueing, lifecycle events, tool
  execution, and loop hooks.

The goal is 1:1 behavior with upstream `pi` for the Rust APIs in scope before
Rust-specific API polish.

To compare this Rust port with `pi`:

- [Visit pi.dev](https://pi.dev)
- [Read the upstream repository](https://github.com/earendil-works/pi)
- [Read upstream `packages/ai`](https://github.com/earendil-works/pi/tree/main/packages/ai)
- [Read upstream `packages/agent`](https://github.com/earendil-works/pi/tree/main/packages/agent)

## Share your OSS coding agent sessions

This Rust workspace does not include the TypeScript coding-agent harness,
session publishing workflow, CLI, or TUI from upstream `pi`.

## Upstream Pi Mapping

| Upstream `pi` README section | ai.rs mapping |
| --- | --- |
| Project introduction | This repository is `ai.rs`, a Rust port of the scoped AI and core agent runtime packages from [`pi`](https://github.com/earendil-works/pi). |
| Share your OSS coding agent sessions | Not part of this Rust crate. Upstream session publishing belongs to the TypeScript coding-agent workflow. |
| All Packages | See [All Crates](#all-crates). The Rust workspace currently has one crate that combines the scoped AI package and core agent runtime. |
| Contributing | See [Contributing](#contributing). PRs are welcome, especially for additional provider support that is intentionally out of scope today. |
| Development | See [Development](#development) for the Rust equivalents of build, check, format, clippy, and test commands. |
| Supply-chain hardening | See [Supply-chain hardening](#supply-chain-hardening) for the Rust equivalent. |
| License | MIT. |

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
| `packages/ai/src/providers/faux.ts` | [`crates/ai/src/providers/faux.rs`](crates/ai/src/providers/faux.rs) | Deterministic test provider, matching upstream's opt-in test provider. |
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

## Contributing

The current priority is parity with upstream `pi` for the scoped Rust API.
When adding behavior, prefer a direct upstream mapping first, then Rust-specific
API polish after the behavior is covered.

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

## Supply-chain hardening

Rust dependency changes are reviewed as code changes. `Cargo.lock` is the
workspace dependency ground truth, and CI-style local validation should include
formatting, `cargo check`, clippy, and tests before merging.

## License

MIT
