# AGENTS.md

Guidance for agents working in this repository.

## Project Shape

This is a Rust workspace for the `ai` crate in `crates/ai`.

The crate provides:

- LLM streaming and one-shot completion APIs.
- Tool calling and JSON Schema based tool definitions.
- Model lookup and custom model configuration.
- OAuth helpers for Anthropic and GitHub Copilot.
- A lightweight agent loop with events, tool execution, steering, and follow-up queues.

The root `README.md` is intentionally short. The detailed crate documentation
lives in `crates/ai/README.md`.

## Commands

Prefer the mise tasks:

```bash
mise run fmt
mise run check
mise run clippy
mise run test-ai
mise run test
mise run all
```

Equivalent cargo commands:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ai
cargo test --workspace
```

## API Guidance

Use `stream_simple` for streaming responses and `complete_simple` for one-shot
responses unless the lower-level `StreamOptions` shape is needed. Use `stream`
or `complete` for direct provider-option forwarding or lower-level request
control.

The active built-in provider scope is OpenAI, Anthropic, and GitHub Copilot.
Azure Foundry and other compatible endpoints should be documented and tested as
custom `Model` values with explicit `base_url`, headers, and `compat` settings.
Do not add broad provider autodetection by provider name or base URL unless that
provider is intentionally in scope.

## Development Notes

- Use semantic/conventional commit messages, such as `feat: add provider`,
  `fix: handle stream errors`, `docs: update README`, or
  `chore: update lockfile`.
- Keep public behavior aligned with the existing Rust API shape before adding
  new abstractions.
- Add or update tests for provider payload changes, stream event ordering,
  tool-call behavior, abort behavior, and agent loop state changes.
- The tests currently live mostly as module-level unit tests under
  `crates/ai/src`; there is no `crates/ai/tests` directory at the moment.
- Avoid unrelated README policy sections, logos, or copied upstream text.
