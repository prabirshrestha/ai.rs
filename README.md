# ai.rs

This repository is a Rust port of the focused [`pi`](https://github.com/earendil-works/pi)
AI and agent runtime surfaces.

The workspace currently contains:

| Crate | Description |
| --- | --- |
| [`ai`](crates/ai) | Unified LLM API plus the agent loop runtime for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and GitHub Copilot-compatible routing. |

The porting target is 1:1 behavior with the relevant upstream `pi` packages
where the scoped Rust surface exists:

- [`packages/ai`](https://github.com/earendil-works/pi/tree/main/packages/ai):
  chat/responses streaming, model registry behavior, OAuth helpers, faux
  provider tests, and shared message/context types.
- [`packages/agent`](https://github.com/earendil-works/pi/tree/main/packages/agent):
  the core agent state, queueing APIs, event lifecycle, tool execution, and
  direct agent loop.

The active provider API surface is intentionally narrow:

- OpenAI Chat Completions: `openai-completions`
- OpenAI Responses: `openai-responses`
- Anthropic Messages: `anthropic-messages`
- GitHub Copilot OAuth and dynamic headers over compatible routes

Cloudflare, Bedrock, Google, Mistral, Azure OpenAI Responses, OpenAI Codex
Responses, and other broad provider-specific APIs are not part of the active
built-in provider surface in this port. PRs to add support for additional
providers are welcome.

The TypeScript coding-agent harness, CLI, and TUI from
[`pi`](https://github.com/earendil-works/pi) are not included. The agent loop is
implemented directly inside the `ai` crate.

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

## Crates

See [`crates/ai/README.md`](crates/ai/README.md) for API usage, model lookup,
tool calling, streaming, OAuth, agent loop APIs, and provider scope.

## License

MIT
