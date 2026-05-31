# ai.rs

This repository is a Rust port of the focused [`pi`](https://github.com/earendil-works/pi)
AI and agent runtime surfaces.

The workspace currently contains:

| Crate | Description |
| --- | --- |
| [`ai`](crates/ai) | Unified LLM API plus the agent loop runtime for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and GitHub Copilot-compatible routing. |

The active provider API surface is intentionally narrow:

- OpenAI Chat Completions: `openai-completions`
- OpenAI Responses: `openai-responses`
- Anthropic Messages: `anthropic-messages`
- GitHub Copilot OAuth and dynamic headers over compatible routes

Cloudflare support is not included in this Rust port.

The TypeScript coding-agent harness, CLI, and TUI from
[`pi`](https://github.com/earendil-works/pi) are not included. The agent loop is
implemented directly inside the `ai` crate.

## Development

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ai
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
