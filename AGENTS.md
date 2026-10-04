# AGENTS.md

Guidance for agents working in this repository.

## Project Shape

This is a Rust workspace with the `ai` crate in `crates/ai` and example
packages under `examples/` (`examples/simple-coding-agent`).

`ai` is a 1:1 port of Pi's `@earendil-works/pi-ai` and
`@earendil-works/pi-agent-core` **1.0.2**, tracking Pi commit
`200387122ca450d6387f033949423114a270b96c`. Pi is the source of truth; the
ai.rs-specific API (provider handles, `stream_simple`/`complete_simple`
returning `Result`, embeddings) sits on top of the ported core.

Module layout of `crates/ai/src` (mirrors Pi's package folders):

- Root modules: `types` (Pi `types.ts`), `models` (the `Models` registry,
  `Provider`, `create_provider`), `models_store`, `model_catalog`, `compat`
  (the global API registry and `stream`/`complete`/`stream_simple`/
  `complete_simple`), `images`, `image_models`, `images_api_registry`,
  `env_api_keys`, `error`, and `embeddings` (ai.rs extra, not in Pi).
- `src/api/`: API implementations (`anthropic_messages`, `openai_responses`
  + `openai_responses_shared`, `openai_completions`, `openai_prompt_cache`,
  `openai_client`, `transform_messages`, `simple_options`,
  `constrained_sampling`, `github_copilot_headers`, `openrouter_images`,
  `openai_images` and `openai_embeddings` (ai.rs extras)).
- `src/providers/`: providers and the pre-1.0 handles (`openai`, `anthropic`,
  `github_copilot`, `openrouter` (image models only), `faux`, `all`,
  `catalog`, `model_builder`); catalog JSON in `providers/data/`.
- `src/auth/`: auth types, credential stores, resolution with locked
  refresh; `src/auth/oauth/`: Anthropic and GitHub Copilot OAuth flows,
  device code, callback server, PKCE.
- `src/utils/`: ports of Pi's utils (transcript, text, event stream, JSON
  parse, overflow, retry, validation, SSE, ...).
- `src/agent/`: `pi-agent-core` (`agent`, `agent_loop`, `types`, `proxy`,
  `stream_fn`).
- `src/chord/` and `src/durable/`: the chord subset used by Pi Durable and
  Pi Durable itself, behind the `durable` cargo feature (on by default).
  `durable-local-env` (default) adds the local environment and JSONL
  adapter, `durable-sqlite` the `rusqlite` storage, `durable-testing` the
  storage conformance suite. The Pi test suites are ported as unit tests
  next to the code (`durable/harness/tests/`, `durable/tools/tests.rs`).

Scope: chat through OpenAI (Responses and Chat Completions), Anthropic
(Messages) and GitHub Copilot, plus the faux provider for tests; image
generation through OpenAI-compatible `/images/generations` and OpenRouter;
embeddings as an ai.rs extra. Other Pi providers and classifiers are not
ported.

The root `README.md` is intentionally short. The detailed crate documentation
lives in `crates/ai/README.md`, which is also the crate-level rustdoc
(`#![doc = include_str!("../README.md")]`); its snippets and those of the root
README run as doctests, so keep them compiling. Breaking changes go in
`crates/ai/CHANGELOG.md`.

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
or `complete` for API-specific `provider_options` (under Pi's names) or
lower-level request control, and the `Models` registry for credential stores
and OAuth.

The system prompt and tool set are transcript messages (`Message::System`),
as in Pi 1.0; there is no separate system-prompt setter on the agent.

Azure Foundry and other compatible language endpoints should be documented and
tested as configured provider handles, such as `providers::openai::builder()`
plus `provider.model(...).base_url(...).headers(...).compat(...)`.
Do not add broad provider autodetection by provider name or base URL unless that
provider is intentionally in scope.

## Development Notes

- Use semantic/conventional commit messages, such as `feat: add provider`,
  `fix: handle stream errors`, `docs: update README`, or
  `chore: update lockfile`.
- When porting behavior from the original Pi TypeScript implementation to Rust,
  treat Pi as the source of truth and keep the port as close to 1:1 as Rust
  permits. Preserve Pi's behavior, control flow, data model, helper boundaries,
  and naming where possible; make only mechanical adaptations required by the
  language or this crate's existing public API. Do not add downstream consumer-
  specific behavior to the port. Document any unavoidable semantic divergence
  from Pi explicitly.
- Use the same semantic/conventional style for PR titles, such as
  `feat: add provider`, `fix(example): limit bash tool execution`, or
  `ci: run clippy in workflow`. PR bodies should include concise `Summary` and
  `Verification` sections.
- Keep public behavior aligned with the existing Rust API shape before adding
  new abstractions.
- Add or update tests for provider payload changes, stream event ordering,
  tool-call behavior, abort behavior, and agent loop state changes.
- Port Pi's tests alongside the code. Tests live as module-level unit tests
  under `crates/ai/src` (some in `*_tests.rs` siblings); there is no
  `crates/ai/tests` directory. Tests run offline (faux provider, local mock
  HTTP servers); tests touching the global API registry hold
  `compat::REGISTRY_TEST_LOCK`.
- Document every divergence from Pi on the module or item involved, and
  summarize notable ones in the "Differences from Pi" section of
  `crates/ai/README.md`.
- Avoid unrelated README policy sections, logos, or copied upstream text.
