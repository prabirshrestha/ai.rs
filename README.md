# ai.rs

This is the home of `ai.rs`, the Rust port of the focused
[`pi`](https://github.com/earendil-works/pi) AI and core agent runtime
surfaces.

- [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai):
  unified LLM API, model lookup, API registry, streaming helpers, provider
  adapters, OAuth helpers, faux provider, and utility helpers.
- [`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent):
  core agent state, direct agent loop, queueing, lifecycle events, tool
  execution, and loop hooks.

The goal is a 1:1 behavior mapping with upstream `pi` for the Rust APIs in
scope, mapped to Rust naming, ownership, async, and error-handling conventions.

To learn more about `pi`:

- [Visit pi.dev](https://pi.dev), the upstream project website with demos
- [Read the upstream repository](https://github.com/earendil-works/pi)
- [Read upstream `packages/ai`](https://github.com/earendil-works/pi/tree/main/packages/ai)
- [Read upstream `packages/agent`](https://github.com/earendil-works/pi/tree/main/packages/agent)

## Share your OSS coding agent sessions

This Rust workspace does not include the TypeScript coding-agent harness,
session publishing workflow, CLI, or TUI from upstream `pi`. It ports the AI
and core agent runtime surfaces only.

## All Crates

| Crate | Upstream mapping | Description |
| --- | --- | --- |
| [`ai`](crates/ai) | [`packages/ai`](https://github.com/earendil-works/pi/tree/main/packages/ai) + core [`packages/agent`](https://github.com/earendil-works/pi/tree/main/packages/agent) | Unified LLM API plus the core agent loop runtime for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and GitHub Copilot-compatible routing. |

Upstream `pi` packages not ported as Rust crates in this workspace:

| Upstream package | Rust status |
| --- | --- |
| [`packages/coding-agent`](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) | Not included. |
| [`packages/tui`](https://github.com/earendil-works/pi/tree/main/packages/tui) | Not included. |

For API usage, model lookup, tool calling, streaming, OAuth, agent loop APIs,
provider scope, image-input support, out-of-scope Pi surfaces, and the detailed
upstream file mapping, see
[`crates/ai/README.md`](crates/ai/README.md).

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
