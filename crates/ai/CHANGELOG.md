# Changelog

## 0.8.0

The crate was rewritten from scratch as a 1:1 port of Pi's `pi-ai` and
`pi-agent-core` 1.0.2 (commit `200387122ca450d6387f033949423114a270b96c`).
The provider handles and `Agent` keep their 0.7 shape, but most types
changed, and requests go through the `Models` registry.

### Breaking changes

Streams and messages:

- Stream events are no longer `Result`s: `AssistantMessageEventStream` yields
  `AssistantMessageEvent`, and failures arrive as an `Error` event and a final
  message with `stop_reason` `Error`/`Aborted`.
- The global `stream()`/`complete()`/`stream_simple()`/`complete_simple()`
  functions are removed. Pi 1.0 keeps them only in its temporary `compat`
  module, which it deletes once its coding agent has migrated; ai.rs follows
  Pi's direction instead. Call `Models::stream`/`complete`/`stream_simple`/
  `complete_simple` (`models.complete_simple(&model, &context, options)`)
  on a provider handle's `handle.models()` or on `create_models` /
  `builtin_models`. They return the stream or message directly; failures,
  including an unknown provider or missing auth, arrive as error events.
  The api-registry (`register_api_provider`, `get_api_provider`, ...),
  `register_faux_provider`/`FauxProviderRegistration`, the image
  `generate_images()` function and its images api-registry, and the
  `get_model`/`get_image_model` catalog aliases are not ported either:
  use `faux_provider` with `Models::set_provider`,
  `Models::generate_images` and `providers::all::get_builtin_*`.
- The system prompt and tools are transcript messages: `Message::System`
  (`SystemMessage { content, sections, tools_added, tools_removed }`).
  `Context::system_prompt` and `Context::tools` (now `Option<Vec<Tool>>`)
  are shorthand for the leading one. The old deferred-tools design
  (`added_tool_names`, tool references, deferred tool modes) is removed.
  `Message::Custom` is removed.
- `reasoning` is a `ThinkingLevel`; `signal: Option<CancellationToken>`
  replaces `cancellation_token`. API-specific options are
  `StreamOptions::provider_options` entries under Pi's names.
- `ModelCompat` is one flat struct with Pi's compat fields.

Providers and models:

- `LanguageModelApi`, `ImageModelApi`, `EmbeddingModelApi`,
  `ProviderCapabilities` and the old `Provider` trait are gone. The new
  `Provider` trait and the `Models` registry (`create_models`,
  `create_provider`, credential stores, `get_auth`, `login`) follow Pi.
- `OpenAiApi` drops `Embeddings`/`Images`: use `OpenAi::image_model` for
  images and the embedding model type for embeddings. The `.images()` builder flag is gone.
- `from_env()` errors are `Error::Models` (`ModelsErrorCode::Auth`); HTTP
  errors are `Error::ProviderHttp` and read `"<status> <body>"`.
- `AssistantImages.usage` is an `Option`, and image outputs are `UserContent`.
- Embeddings are a `Models` model type (ai.rs extra, designed like Pi's image
  models). The free `embed`/`embed_many` functions, `OpenAi::embedding_model`/
  `GitHubCopilot::embedding_model`, `EmbeddingModelBuilder`, `EmbeddingOptions`,
  `Embedding`, `EmbeddingBatch`, `EmbeddingUsage`, `EmbeddingEncodingFormat`
  and the `ai::embeddings` module are removed. Look the model up with
  `models.get_model_of_type(ModelType::Embedding, "openai",
  "text-embedding-3-small")` (or build an `EmbeddingModel`) and call
  `Models::embed(&model, &EmbeddingsContext { input }, EmbeddingsOptions)`,
  which returns an `EmbeddingsResult` with errors in-band and `usage` priced
  from the catalog. `encoding_format` and `user` move to
  `EmbeddingsOptions::provider_options` (`encodingFormat`, `user`). Without a
  key, `openai-embeddings` fails like `openai-images` unless the provider
  allows keyless requests (the OpenAI handle with a custom base URL does).
- OAuth moved to `ai::auth::oauth` (re-exported at the root): `OAuthProvider`,
  `OAuthProviderInterface`, `poll_oauth_device_code_flow`,
  `get_oauth_providers`, `refresh_oauth_token` and the provider structs are
  replaced by Pi's `OAuthAuth` flows (`anthropic_oauth()`,
  `github_copilot_oauth()`, `login_anthropic`, `login_github_copilot`).
  `github_copilot::get_oauth_api_key` is kept.
- Env var constants (`*_ENV_VAR`, `KnownProvider`) are replaced by
  `ai::env_api_keys` (`get_env_api_key(provider, env)`, `find_env_keys`).
- `session_resources` is removed. Event-stream types moved to
  `ai::utils::event_stream` (re-exported).
- Faux: `FauxAssistantContent`/`FauxAssistantMessageOptions` are
  `FauxContent`/`FauxMessageOptions`; factories return `Result` and receive
  a state snapshot; `faux_provider()` builds a provider for `Models`.

Agent:

- Removed: `set_system_prompt` (push a `SystemMessage`),
  `AgentState::builder`/`AgentStateBuilder`, `AgentContext::system_prompt`
  and `llm_context`, `should_stop_after_turn`, `added_tool_names`,
  `clear_tools`/`clear_messages`, `AgentError::ToolNotFound`.
- `AgentOptions` takes Pi's fields (hooks, `stream_fn`, `session_id`,
  `thinking_budgets`, `transport`, ...) instead of `options:
  SimpleStreamOptions`. Pass `stream_fn(stream_simple_fn(models))` (a
  `StreamFn` over `Models::stream_simple`) or call `set_default_stream_fn`;
  there is no implicit default.
- State and queue methods are synchronous (`state()`, `set_model`,
  `set_tools`, `steer`, `follow_up`, ...); `reset()` returns `Result`.
  Error texts follow Pi.
- `AgentLoopConfig` holds the stream options in `options`; `agent_loop`
  takes an optional `StreamFn`.

### Added

- `Models` registry, credential stores, Anthropic and GitHub Copilot OAuth
  with locked refresh, `get_available`, deferred responses.
- Transcript utilities (`get_current_tools`, `get_tool_state_changes`, ...),
  `retry`, `uuidv7`, diagnostics.
- Agent hooks (`before_tool_call`, `after_tool_call`, `prepare_request`,
  `prepare_next_turn`, `finish_turn`, ...), `run_tool_call`, `stream_proxy`,
  `peek_queued_messages`. The hooks return `AgentResult`; an `Err` fails the
  run like a thrown error in Pi. `ai::agent::ThinkingLevel` is Pi's agent
  thinking level (with `off`).
- OpenRouter image models in the builtin providers.
- Classifier models, ported from Pi: `ModelType::Classifier`,
  `ClassifierModel`, `AnyModel::Classifier`, `ClassifierContext`/
  `ClassifierQuestion`/`ClassifierAnswer`/`ClassifierResult`,
  `ClassifierOptions` (with `temperature`), `ProviderClassifier`,
  `CreateProviderOptions::classifiers`, `Provider::classify`,
  `Models::classify` and `get_builtin_classifier_model(s)`. APIs:
  `typesafe_system_one_api()`, `cloudflare_workers_ai_system_one_api()` and
  `llama_cpp_classify_api()`.
- Providers `typesafe` (System One classifiers) and `cloudflare-workers-ai`
  (chat models over Chat Completions plus System One classifiers, with
  `cloudflare_workers_ai_auth()` and the `cloudflare_streams`/
  `cloudflare_classifier` endpoint wrappers) in the builtin providers.
  OpenRouter adds its TypeSafe classifier models.
- Embedding models (ai.rs extra): `ModelType::Embedding`, `EmbeddingModel`,
  `AnyModel::Embedding`, `ProviderEmbeddings`,
  `CreateProviderOptions::embeddings`, `Provider::embed`, `Models::embed`,
  `openai_embeddings_api()`, and catalog entries for OpenAI
  (`text-embedding-3-small`, `text-embedding-3-large`,
  `text-embedding-ada-002`) and GitHub Copilot (`text-embedding-3-small`).
- `ai::durable` (feature `durable`, on by default): Pi Durable 1.0.2, with
  `ai::chord`. Durable conversations, tasks, submissions and documents over
  memory, JSONL (`durable-local-env`) or SQLite (`durable-sqlite`) storage;
  the `Harness` with extensions, generation, tool and compaction tasks,
  views and agent events; the `read`, `write`, `edit` and `bash` coding
  tools (`CODING_TOOLS`) over an `ExecutionEnv`; and a storage conformance
  suite (`durable-testing`).

### Not ported yet

- Pi's `pi-mcp` and `pi-codemode` packages are planned for a future release.
- The Cloudflare AI Gateway provider and `cloudflare-ai-binding` (Workers
  runtime only) are not ported.
