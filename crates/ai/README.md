# ai

LLM library for Rust: streaming, tool calling, a models registry with auth and
OAuth, image generation, embeddings and an agent loop.

`ai` is a 1:1 Rust port of Pi's [`@earendil-works/pi-ai`] and
[`@earendil-works/pi-agent-core`] **1.0.2** (Pi commit
[`200387122ca450d6387f033949423114a270b96c`]). Pi is the source of truth: the
data model, event order, provider payloads and error texts follow it. Rust
adaptations and the few intentional differences are listed in
[Differences from Pi](#differences-from-pi). Breaking changes from 0.7 are in
[CHANGELOG.md](CHANGELOG.md).

[`@earendil-works/pi-ai`]: https://github.com/earendil-works/pi/tree/main/packages/ai
[`@earendil-works/pi-agent-core`]: https://github.com/earendil-works/pi/tree/main/packages/agent
[`200387122ca450d6387f033949423114a270b96c`]: https://github.com/earendil-works/pi/tree/200387122ca450d6387f033949423114a270b96c

## Contents

- [Scope](#scope)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Streaming](#streaming)
- [Choosing an entry point](#choosing-an-entry-point)
- [Providers](#providers)
- [Tools](#tools)
- [System messages and mid-conversation tool changes](#system-messages-and-mid-conversation-tool-changes)
- [Thinking and reasoning](#thinking-and-reasoning)
- [Image input](#image-input)
- [Aborting, errors and debugging](#aborting-errors-and-debugging)
- [Models registry](#models-registry)
- [Faux provider for tests](#faux-provider-for-tests)
- [Agent](#agent)
- [Image generation](#image-generation)
- [Embeddings (ai.rs extra)](#embeddings-ai-rs-extra)
- [Durable (feature `durable`, on by default)](#durable-feature-durable-on-by-default)
- [Differences from Pi](#differences-from-pi)
- [License](#license)

## Scope

| Area | Providers / APIs |
| --- | --- |
| Chat | OpenAI (`openai-responses`, `openai-completions`), Anthropic (`anthropic-messages`), GitHub Copilot (all three, routed per model) |
| OpenAI-compatible servers | llama.cpp, Ollama, vLLM, MLX, LM Studio, Azure Foundry, ... through the OpenAI provider handle with a custom `base_url` |
| Auth | API keys from the environment or explicit, credential stores, OAuth for Anthropic (Claude Pro/Max) and GitHub Copilot (device code) |
| Tests | the faux provider (scripted responses, no network) |
| Image generation | OpenAI-compatible `/images/generations` (`openai-images`, ai.rs extra) and OpenRouter (`openrouter-images`) |
| Embeddings | OpenAI-compatible `/embeddings` through the OpenAI and GitHub Copilot handles (ai.rs extra, not in Pi) |

Other Pi providers (Google, Bedrock, Mistral, xAI, OpenRouter chat, Codex,
...) and classifier models are not ported.

Crate features:

- `durable` (default): Pi Durable (`ai::durable`) and the subset of chord it
  uses (`ai::chord`), with memory and portable JSONL/SQLite storage cores and
  the coding tools.
- `durable-local-env` (default): `LocalExecutionEnv` (local files and
  processes) and the local JSONL storage adapter.
- `durable-sqlite`: `SqliteStorage` over a bundled SQLite (`rusqlite`).
- `durable-testing`: the storage conformance suite (`ai::durable::testing`)
  for custom `Storage` backends.

## Installation

```bash
cargo add ai
cargo add tokio --features macros,rt-multi-thread
cargo add futures serde_json
```

The crate runs on Tokio. Cancellation uses `tokio_util`'s `CancellationToken`
(`cargo add tokio-util` if you abort requests). Examples below use
`#[tokio::main]` and `futures::StreamExt`.

## Quick start

```rust,no_run
use ai::{Context, Message, Result, SimpleStreamOptions, content_text, providers::openai};

#[tokio::main]
async fn main() -> Result<()> {
    // Reads OPENAI_API_KEY.
    let openai = openai::from_env()?;
    let models = openai.models();
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder()
        .system_prompt("You are a concise assistant.")
        .message(Message::user_text("Write a haiku about Rust."))
        .build();

    let message = models
        .complete_simple(&model, &context, SimpleStreamOptions::default())
        .await;
    println!("{}", content_text(&message.content));
    Ok(())
}
```

Requests go through a [`Models`](#models-registry) registry, as in Pi 1.0.
A provider handle owns one with its provider registered (`handle.models()`).
`complete_simple` returns the final `AssistantMessage`. Failures never come
back as `Err`: they are reported in-band with
`stop_reason == StopReason::Error` (or `Aborted`) and `error_message`, like in
Pi, including an unknown provider or missing auth.

## Streaming

```rust,no_run
use ai::{
    AssistantMessageEvent, Context, Message, Result, SimpleStreamOptions, providers::anthropic,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Reads ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN / ANTHROPIC_OAUTH_TOKEN).
    let anthropic = anthropic::from_env()?;
    let model = anthropic.model("claude-sonnet-4-5").build()?;
    let context = Context::builder()
        .message(Message::user_text("Explain ownership in one paragraph."))
        .build();

    let mut events = anthropic
        .models()
        .stream_simple(&model, &context, SimpleStreamOptions::default());
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::TextDelta { delta, .. } => print!("{delta}"),
            AssistantMessageEvent::Done { message, .. } => {
                println!("\n{} tokens", message.usage.total_tokens)
            }
            AssistantMessageEvent::Error { error, .. } => {
                eprintln!("\nerror: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    // The stream also resolves to the final message.
    let _final_message = events.result().await;
    Ok(())
}
```

`AssistantMessageEventStream` is a `futures::Stream` of
`AssistantMessageEvent`s. Every event carries the `partial` message built so
far. The sequence is:

| Event | Meaning |
| --- | --- |
| `Start { partial }` | the response started |
| `TextStart` / `TextDelta { delta }` / `TextEnd { content }` | a text block |
| `ThinkingStart` / `ThinkingDelta` / `ThinkingEnd` | a reasoning block |
| `ToolCallStart` / `ToolCallDelta { delta }` / `ToolCallEnd { tool_call }` | a tool call; `delta` is raw argument JSON |
| `Done { reason, message }` | success; `reason` is `Stop`, `Length`, `ToolUse` or `Deferred` |
| `Error { reason, error }` | failure; `reason` is `Error` or `Aborted` |

Each block event has a `content_index` into `partial.content`. While a tool
call streams, `parse_streaming_json(Some(&partial_json))` turns incomplete
argument JSON into the best-effort value so far.

## Choosing an entry point

| Function | Options | Use it for |
| --- | --- | --- |
| `Models::stream_simple` / `Models::complete_simple` | `SimpleStreamOptions` | most code: one `reasoning` level, `tool_choice`, thinking budgets, plus everything in `StreamOptions` |
| `Models::stream` / `Models::complete` | `StreamOptions` | lower-level control; API-specific options go in `provider_options` under Pi's names |
| `api::*::stream_*` (e.g. `stream_openai_responses`) | typed per-API options | calling one API implementation directly, with explicit auth |

The `Models` methods also take `ModelsOptions { options, transform_headers }`
to transform the assembled request headers.

`SimpleStreamOptions` derefs to `StreamOptions`, so both kinds of field are
set the same way:

```rust,no_run
use ai::{
    CacheRetention, Context, Message, Result, SimpleStreamOptions, StreamOptions, ThinkingLevel,
    providers::openai,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let models = openai.models();
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder().message(Message::user_text("Hi")).build();

    let simple = SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::Low),
        stream: StreamOptions {
            max_tokens: Some(1024),
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-1".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    models.complete_simple(&model, &context, simple).await;

    // Lower level: API-specific options under Pi's names.
    let mut options = StreamOptions::default();
    options
        .provider_options
        .insert("reasoningEffort".into(), json!("high"));
    options
        .provider_options
        .insert("reasoningSummary".into(), json!("detailed"));
    models.complete(&model, &context, options).await;
    Ok(())
}
```

When `api_key` is not set, the `Models` methods resolve the provider's auth:
a stored credential, OAuth, or the provider's environment variable
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COPILOT_GITHUB_TOKEN`, ...).
`get_env_api_key(provider, None)` and `find_env_keys` expose the environment
lookup.

## Providers

Provider handles (`providers::openai`, `providers::anthropic`,
`providers::github_copilot`, `providers::openrouter`) are the ai.rs entry point
kept from 0.7. A handle owns a private [`Models`](#models-registry) collection
with one provider in it, configured with the handle's key, base URL and HTTP
client; `handle.models()` returns it. `handle.model("id")` starts from the
catalog entry (or a default shape for unknown ids) and returns a
`ModelBuilder` for a plain `Model`. Send requests for it through
`handle.models()` (or any `Models` that has the handle's provider).

`ModelBuilder` can override `name`, `base_url`, `reasoning`,
`thinking_level_map`, `input`, `cost`, `context_window`, `max_tokens`,
`compat` and `headers`/`header`.

### OpenAI: Responses and Chat Completions

```rust,no_run
use ai::{Result, providers::openai};

fn main() -> Result<()> {
    // OPENAI_API_KEY; models use the Responses API.
    let openai = openai::from_env()?;
    let _gpt = openai.model("gpt-5.5").build()?;

    // Explicit key, Chat Completions API, custom HTTP client.
    let chat = openai::builder()
        .api_key(Some("sk-..."))
        .chat_completions()
        .http_client(reqwest::Client::new())
        .build()?;
    let _gpt4o = chat.model("gpt-4o").build()?;
    Ok(())
}
```

### OpenAI-compatible servers: llama.cpp, Ollama, Azure Foundry

Point the OpenAI handle at the server and pick the API it speaks. A handle
with a custom `base_url` and no key sends no `Authorization` header.

```rust,no_run
use ai::{ModelCompat, ModelInput, Result, providers::openai};

fn main() -> Result<()> {
    // Ollama (or llama.cpp: http://localhost:8080/v1, vLLM, MLX, LM Studio).
    let ollama = openai::builder()
        .provider_id("ollama")
        .base_url("http://localhost:11434/v1")
        .chat_completions()
        .build()?;
    let _gemma = ollama
        .model("gemma4:12b")
        .context_window(128_000)
        .max_tokens(8192)
        .input([ModelInput::Text])
        // Servers that do not understand the `developer` role or
        // `reasoning_effort`.
        .compat(ModelCompat {
            supports_developer_role: Some(false),
            supports_reasoning_effort: Some(false),
            ..Default::default()
        })
        .build()?;

    // Azure AI Foundry (OpenAI v1 endpoint), Responses API.
    let foundry = openai::builder()
        .provider_id("azure-foundry")
        .api_key(Some("..."))
        .base_url("https://example.services.ai.azure.com/openai/v1")
        .responses()
        .build()?;
    let _deployment = foundry
        .model("gpt-5.5")
        .header("x-ms-client-request-id", "my-app")?
        .build()?;
    Ok(())
}
```

`ModelCompat` mirrors Pi's per-API compat records (`OpenAICompletionsCompat`,
`OpenAIResponsesCompat`, `AnthropicMessagesCompat`) as one flat struct:
`max_tokens_field`, `thinking_format`, `chat_template_kwargs`,
`supports_store`, `supports_strict_mode`, `cache_control_format`,
`session_affinity_format`, `supports_mid_convo_system_messages`, and so on.
Unset fields fall back to Pi's detection by provider and base URL.

### Anthropic

```rust,no_run
use ai::{Result, providers::anthropic};

fn main() -> Result<()> {
    let _from_env = anthropic::from_env()?;
    let with_key = anthropic::builder().api_key("sk-ant-...").build()?;
    // An OAuth access token (Claude Pro/Max) or a gateway bearer token.
    let _with_token = anthropic::builder().auth_token("sk-ant-oat...").build()?;

    let _sonnet = with_key.model("claude-sonnet-4-5").build()?;
    Ok(())
}
```

Typed Messages options (`AnthropicOptions`, `AnthropicEffort`, ...) and
`stream_anthropic` live in `ai::api::anthropic_messages`. Through
`Models::stream`/`Models::complete` they are `provider_options` entries named as in Pi
(`thinkingEnabled`, `effort`, `toolChoice`, ...).

### GitHub Copilot

Copilot routes each model to the API its catalog entry names: Claude models to
Anthropic Messages, GPT-5/Grok/MAI models to Responses, the rest to Chat
Completions. `.anthropic_messages()`, `.responses()` and `.chat_completions()`
on the builder force one API.

```rust,no_run
use ai::{OAuthLoginCallbacks, Result, login_github_copilot, providers::github_copilot};

#[tokio::main]
async fn main() -> Result<()> {
    // Device-code login. Persist `credential` (it serializes) and reuse it.
    let callbacks = OAuthLoginCallbacks::builder()
        // Asked for a GitHub Enterprise domain; empty means github.com.
        .on_prompt(|_prompt| async { Ok(String::new()) })
        .on_device_code(|info| {
            println!("Open {} and enter {}", info.verification_uri, info.user_code)
        })
        .on_progress(|message| println!("{message}"))
        .build();
    let credential = login_github_copilot(callbacks).await?;

    // Refreshes an expired credential and returns the request token.
    let token = github_copilot::get_oauth_api_key(&credential).await?;
    let copilot = github_copilot::builder()
        .api_key(token.api_key)
        .base_url(github_copilot::base_url_for_credentials(&token.new_credentials))
        .build()?;
    let _claude = copilot.model("claude-sonnet-4.6").build()?;
    let _gpt = copilot.model("gpt-5.5").build()?;

    // Or: COPILOT_GITHUB_TOKEN.
    let _from_env = github_copilot::from_env()?;
    Ok(())
}
```

With a [models registry](#models-registry) and a credential store, Copilot
OAuth credentials are refreshed automatically under a lock instead
(`Models::login` / `Models::get_auth`). `login_anthropic` is the equivalent
legacy entry point for Claude Pro/Max.

## Tools

Tools are JSON Schema declarations. Validate arguments with
`validate_tool_call`, answer with a `ToolResultMessage` and call the model
again until it stops asking for tools:

```rust,no_run
use ai::{
    AssistantContent, Context, Message, Result, SimpleStreamOptions, StopReason, Tool,
    ToolResultMessage, UserContent, providers::openai, validate_tool_call,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let models = openai.models();
    let model = openai.model("gpt-5.5").build()?;
    let weather = Tool::builder("get_weather")
        .description("Current weather for a city")
        .parameters(json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }))
        .build()?;
    let tools = vec![weather.clone()];
    let mut context = Context::builder()
        .tool(weather)
        .message(Message::user_text("Weather in Paris?"))
        .build();

    loop {
        let message = models
            .complete_simple(&model, &context, SimpleStreamOptions::default())
            .await;
        context.messages.push(message.clone().into());
        if message.stop_reason != StopReason::ToolUse {
            break;
        }
        for content in &message.content {
            let AssistantContent::ToolCall(call) = content else {
                continue;
            };
            let (text, is_error) = match validate_tool_call(&tools, call) {
                Ok(args) => (format!("Sunny in {}", args["city"]), false),
                Err(error) => (error.to_string(), true),
            };
            context.messages.push(Message::ToolResult(ToolResultMessage {
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                content: vec![UserContent::text(text)],
                details: None,
                usage: None,
                nested_calls: None,
                is_error,
                timestamp: 0,
            }));
        }
    }
    Ok(())
}
```

`Tool::builder(..).constrained_sampling(..)` opts a tool into strict JSON
Schema or grammar-constrained sampling where the API supports it (Pi's
`constrainedSampling`). Tool results can contain images
(`UserContent::Image`).

## System messages and mid-conversation tool changes

In Pi 1.0 the system prompt and the tool set live in the transcript.
`Context::system_prompt` and `Context::tools` are shorthand for a leading
`SystemMessage`. Later system messages change things from that point on,
without rewriting earlier turns (so prompt caches stay valid):

- `content`: extra instructions from here on;
- `sections`: named prompt sections, replaced by name (`None` removes one);
- `tools_added` / `tools_removed`: the tool set changes.

```rust,no_run
use ai::{
    Context, Message, Result, SimpleStreamOptions, SystemMessage, Tool, ToolReference,
    providers::anthropic,
};
use indexmap::IndexMap;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let anthropic = anthropic::from_env()?;
    let models = anthropic.models();
    let model = anthropic.model("claude-sonnet-4-5").build()?;
    let search = Tool::builder("search")
        .description("Search the docs")
        .parameters(json!({ "type": "object", "properties": { "q": { "type": "string" } } }))
        .build()?;
    let mut context = Context::builder()
        .system_prompt("You are a support agent.")
        .message(Message::user_text("Hi"))
        .build();
    let reply = models
        .complete_simple(&model, &context, SimpleStreamOptions::default())
        .await;
    context.messages.push(reply.into());

    // Later: new instructions, a named section and a new tool.
    context.messages.push(Message::System(SystemMessage {
        content: "The user is on the Pro plan.".into(),
        sections: Some(IndexMap::from([(
            "tone".to_string(),
            Some("Answer in one sentence.".to_string()),
        )])),
        tools_added: Some(vec![search]),
        ..Default::default()
    }));
    context.messages.push(Message::user_text("How do I export data?"));
    let reply = models
        .complete_simple(&model, &context, SimpleStreamOptions::default())
        .await;
    context.messages.push(reply.into());

    // Remove the tool again.
    context.messages.push(Message::System(SystemMessage {
        tools_removed: Some(vec![ToolReference { name: "search".into() }]),
        ..Default::default()
    }));
    Ok(())
}
```

Each API sends changes natively where it can (Anthropic tool additions and
removals, mid-conversation system/developer messages) and otherwise folds
them into the request the way Pi does. `get_current_system_prompt`,
`get_current_tools` and `get_tool_state_changes` replay a transcript.

## Thinking and reasoning

`SimpleStreamOptions::reasoning` takes a `ThinkingLevel` (`Minimal`, `Low`,
`Medium`, `High`, `Xhigh`, `Max`); each API maps it to its own setting
(effort, budget tokens, `reasoning_effort`, ...). `None` turns reasoning off.
`get_supported_thinking_levels(&model)` and `clamp_thinking_level` report
what a model accepts, and `thinking_budgets` overrides the token budgets of
budget-based APIs. Reasoning streams as `Thinking*` events and ends up as
`AssistantContent::Thinking` blocks, which are replayed to the same model on
later turns.

## Image input

```rust,no_run
use ai::{
    Context, ImageContent, Message, Result, SimpleStreamOptions, UserContent, UserMessage,
    UserMessageContent, providers::openai,
};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let png_base64 = String::from("iVBORw0KGgo...");
    let context = Context::builder()
        .message(UserMessage {
            content: UserMessageContent::Parts(vec![
                UserContent::text("What is in this image?"),
                UserContent::Image(ImageContent {
                    data: png_base64,
                    mime_type: "image/png".into(),
                }),
            ]),
            timestamp: 0,
        })
        .build();
    openai
        .models()
        .complete_simple(&model, &context, SimpleStreamOptions::default())
        .await;
    Ok(())
}
```

Models without `ModelInput::Image` get a text placeholder instead of the
image.

## Aborting, errors and debugging

```rust,no_run
use std::sync::Arc;

use ai::{Context, Message, Result, StopReason, StreamOptions, providers::openai};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder().message(Message::user_text("Count to 1000")).build();

    let signal = CancellationToken::new();
    let options = StreamOptions {
        signal: Some(signal.clone()),
        timeout_ms: Some(60_000),
        max_retries: Some(2),
        // Inspect (or replace, by returning Some) the provider payload.
        on_payload: Some(Arc::new(|payload, _model| {
            eprintln!("payload: {payload}");
            Box::pin(async { Ok(None) })
        })),
        ..Default::default()
    };
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        signal.cancel();
    });

    let message = openai.models().complete(&model, &context, options).await;
    match message.stop_reason {
        StopReason::Aborted => println!("aborted; partial content is kept"),
        StopReason::Error => println!("failed: {:?}", message.error_message),
        _ => println!("done"),
    }
    Ok(())
}
```

An aborted message can stay in the transcript; continue by appending a user
message. `is_context_overflow(&message, Some(model.context_window))` detects
context-window errors across providers. `on_response` sees status and headers,
and `on_provider_stream_event` sees raw provider events.

## Models registry

`Models` is Pi's runtime registry: providers, their catalogs, auth resolution
(explicit keys, environment, credential stores, OAuth with locked refresh) and
the stream operations. `builtin_models` registers the built-in providers
(`anthropic`, `github-copilot`, `openai`, and `openrouter` for images); it
lives in `ai::providers::all`.

```rust,no_run
use std::sync::Arc;

use ai::{
    AuthOperationOptions, Context, CreateModelsOptions, InMemoryCredentialStore, Message, Result,
    SimpleStreamOptions, providers::all::builtin_models,
};

#[tokio::main]
async fn main() -> Result<()> {
    let models = builtin_models(CreateModelsOptions {
        credentials: Some(Arc::new(InMemoryCredentialStore::new())),
        ..Default::default()
    });

    for provider in models.get_providers() {
        println!("{}: {} models", provider.id(), provider.get_models()?.len());
    }
    // Models whose provider has usable auth (env, stored key or OAuth).
    let available = models
        .get_available(None, AuthOperationOptions::default())
        .await?;
    println!("{} available", available.len());

    let model = models.get_model("openai", "gpt-5.5").expect("in the catalog");
    let context = Context::builder().message(Message::user_text("Hi")).build();
    let message = models
        .complete_simple(&model, &context, SimpleStreamOptions::default())
        .await;
    println!("{:?}", message.stop_reason);
    Ok(())
}
```

`Models::login(provider, AuthType::OAuth, interaction, LoginOptions::default())`
runs a provider's login flow with an `AuthInteraction` (prompts plus
`AuthEvent` notifications such as `DeviceCode`) and stores the credential;
`logout` removes it. `create_provider(CreateProviderOptions { .. })` builds a
custom provider from models and `ProviderStreams` implementations, and
`set_provider` registers it. Static catalog lookups without a registry:
`ai::providers::all::{get_builtin_model, get_builtin_models,
get_builtin_providers, get_builtin_image_model, get_builtin_image_models}`.
`calculate_cost`, `models_are_equal` and
`get_model_type` are the remaining model helpers.

## Faux provider for tests

The faux provider replays scripted assistant messages through the real event
pipeline, without network. Register `faux.provider` in a `Models` registry:

```rust
use ai::{
    Context, FauxMessageOptions, Message, RegisterFauxProviderOptions, SimpleStreamOptions,
    content_text, create_models, faux_assistant_message, faux_provider,
};

#[tokio::main]
async fn main() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([faux_assistant_message("Hello!", FauxMessageOptions::default()).into()]);
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());

    let context = Context::builder().message(Message::user_text("Hi")).build();
    let message = models
        .complete_simple(&faux.get_model(), &context, SimpleStreamOptions::default())
        .await;
    assert_eq!(content_text(&message.content), "Hello!");
}
```

Responses can be built with
`faux_text`, `faux_thinking` and `faux_tool_call`, or computed per request
with `FauxResponseStep::factory`. `faux.state().call_count` counts requests,
and `RegisterFauxProviderOptions` sets models, token pacing and deferred
behavior.

## Agent

`Agent` (Pi's `pi-agent-core`) keeps the transcript, streams assistant turns,
runs tools, and offers steering and follow-up queues. Pass the stream
function explicitly, as Pi's coding agent does with `models.streamSimple`:
`stream_simple_fn(models)` wraps `Models::stream_simple`. Or install one for
agents that omit it with `set_default_stream_fn`.

```rust
use ai::{
    Agent, AgentEvent, AgentOptions, AgentToolBuilder, AgentToolResult, AssistantMessageEvent,
    FauxMessageOptions, RegisterFauxProviderOptions, create_models, faux_assistant_message,
    faux_provider, faux_tool_call, stream_simple_fn,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Swap in e.g. `let openai = openai::from_env()?;`, `openai.models()` and
    // `openai.model("gpt-5.5").build()?`.
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());
    faux.set_responses([
        faux_assistant_message(
            faux_tool_call("add", json!({ "a": 2, "b": 3 }), None),
            FauxMessageOptions::default(),
        )
        .into(),
        faux_assistant_message("2 + 3 = 5", FauxMessageOptions::default()).into(),
    ]);

    let add = AgentToolBuilder::new("add")
        .description("Add two numbers")
        .parameters(json!({
            "type": "object",
            "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
            "required": ["a", "b"]
        }))
        .execute(|args| async move {
            let sum = args["a"].as_f64().unwrap_or(0.0) + args["b"].as_f64().unwrap_or(0.0);
            Ok(AgentToolResult::text(sum.to_string()))
        })
        .build()?;

    let agent = Agent::new(
        AgentOptions::builder(faux.get_model())
            .system_prompt("You are a calculator.")
            .tool(add)
            .stream_fn(stream_simple_fn(models))
            .build(),
    );

    // Listeners are awaited in order; keep the subscription alive.
    let _subscription = agent.subscribe(|event, _signal| async move {
        match event {
            AgentEvent::MessageUpdate {
                assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
                ..
            } => print!("{delta}"),
            AgentEvent::ToolExecutionStart { tool_name, args, .. } => {
                println!("{tool_name}({args})")
            }
            _ => {}
        }
        Ok(())
    });

    agent.prompt_text("What is 2 + 3?", Vec::new()).await?;
    // system, user, assistant (tool call), tool result, assistant
    assert_eq!(agent.messages().len(), 5);
    Ok(())
}
```

Events per run: `AgentStart`, then per turn `TurnStart`, `MessageStart` /
`MessageUpdate` / `MessageEnd` for each message, `ToolExecutionStart` /
`ToolExecutionUpdate` / `ToolExecutionEnd` per tool call, `TurnEnd`, and
finally `AgentEnd { messages }`.

Main methods:

- `prompt_text(text, images)`, `prompt_message`, `prompt_messages`,
  `continue_run()` (from a user or tool-result message);
- `steer(message)` (delivered after the current tool batch) and
  `follow_up(message)` (after the run would stop), with
  `QueueMode::{OneAtATime, All}`, `clear_*_queue`, `peek_queued_messages`;
- `abort()`, `wait_for_idle()`, `reset()` (keeps the system prompt and tool
  baseline), `state()`, `messages()`;
- `set_model`, `set_thinking_level`, `set_tools` (tool changes are announced
  to the model with a system message), `push_message` (e.g. a
  `SystemMessage` that changes the prompt), `set_messages`;
- hooks on `AgentOptions` (and setters on `Agent`): `before_tool_call`,
  `after_tool_call`, `transform_context`, `convert_to_llm`,
  `prepare_request`, `prepare_next_turn`, `finish_turn`, `get_api_key`,
  `on_payload`, ...

Tools implement `AgentTool` (or use `AgentToolBuilder`). A tool returns
`Err` or an `AgentToolResult` with `is_error: true` on failure; `terminate:
true` on every result of a batch ends the run. `execute_with_context` gives
the tool call id, the abort signal and an update callback for
`ToolExecutionUpdate` events. `ToolExecutionMode::{Parallel, Sequential}`
controls batches.

The low-level loop is available as `agent_loop(prompts, AgentContext,
AgentLoopConfig, signal, stream_fn)` (a stream of `AgentEvent`s whose
`result()` is the new messages), `agent_loop_continue`, and the
callback-based `run_agent_loop`. `stream_proxy` streams through a Pi-style
proxy server, and `run_tool_call` runs a single tool call with the hooks.

## Image generation

```rust,no_run
use ai::{ImagesContext, ImagesOptions, ImagesStopReason, Result, UserContent, providers};

#[tokio::main]
async fn main() -> Result<()> {
    // OpenRouter (OPENROUTER_API_KEY), Pi's `openrouter-images` API.
    let openrouter = providers::openrouter::from_env()?;
    let model = openrouter.model("google/gemini-3-pro-image").build_image()?;
    let context = ImagesContext::builder()
        .text("A small watercolor robot reading a book.")
        .build();
    let images = openrouter
        .models()
        .generate_images(&model, &context, ImagesOptions::default())
        .await;
    if images.stop_reason == ImagesStopReason::Error {
        eprintln!("{:?}", images.error_message);
    }
    for output in images.output {
        if let UserContent::Image(image) = output {
            println!("{} ({} base64 bytes)", image.mime_type, image.data.len());
        }
    }

    // ai.rs extra: OpenAI-compatible /images/generations (`openai-images`),
    // also for local servers through a custom base URL.
    let openai = providers::openai::from_env()?;
    let model = openai.image_model("gpt-image-2").build_image()?;
    let context = ImagesContext::builder().text("A robot.").build();
    openai
        .models()
        .generate_images(&model, &context, ImagesOptions::default())
        .await;
    Ok(())
}
```

`ImagesContext::builder().image(..)` adds input images for editing models.
`builtin_models(..)` registers OpenRouter's image catalog too
(`Models::get_model_of_type(ModelType::Image, ..)`), and a custom provider
adds image APIs through `CreateProviderOptions::images`.

## Embeddings (ai.rs extra)

Not in Pi. OpenAI-compatible `/embeddings` through the OpenAI and GitHub
Copilot handles; auth resolves like chat requests.

```rust,no_run
use ai::{EmbeddingVector, Result, embed, embed_many, providers::openai};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai
        .embedding_model("text-embedding-3-small")
        .build_embedding()?;

    let one = embed(model.clone(), "hello", None).await?;
    if let EmbeddingVector::Float(vector) = &one.embedding {
        println!("{} dimensions", vector.len());
    }
    let batch = embed_many(model, ["first", "second"], None).await?;
    println!("{} embeddings", batch.embeddings.len());
    Ok(())
}
```

## Durable (feature `durable`, on by default)

`ai::durable` is Pi Durable: conversations whose transcript, tasks,
submissions and documents live in a `Storage`, so a run survives a crash or
restart and resumes where it stopped. A `Harness` opens one storage and runs
the built-in generation, tool and compaction tasks; extensions add tools,
prompt sections, hooks and tasks through a `Registry`. `ai::chord` holds the
contexts, JSON deltas and replicated state it builds on.

The example below opens a Harness over memory storage, plays the model with
the faux provider, offers the `read`, `write`, `edit` and `bash` coding tools
(`CODING_TOOLS`), and submits input to the root conversation:

```rust
use std::sync::Arc;

use ai::{
    FauxMessageOptions, Message, StopReason, content_text, create_models, faux_assistant_message,
    faux_provider, faux_tool_call,
};
# #[cfg(feature = "durable-local-env")]
use ai::{
    chord::background_context,
    durable::{
        ASSISTANT_ENTRY, MemoryStorage, SubmissionStatus,
        env::{ExecutionEnv, local::LocalExecutionEnv},
        harness::{
            AgentChange, CreateOptions, Harness, HarnessOptions, ModelRef, SubmissionDraft,
            create_registry,
        },
        tools::CODING_TOOLS,
    },
};

# #[cfg(not(feature = "durable-local-env"))]
# fn main() {}
# #[cfg(feature = "durable-local-env")]
#[tokio::main]
async fn main() -> ai::durable::Result<()> {
    let context = background_context();

    // The faux provider plays the model: one `bash` call, then an answer.
    let faux = faux_provider(Default::default());
    faux.set_responses([
        faux_assistant_message(
            faux_tool_call("bash", serde_json::json!({ "command": "echo hello" }), Some("call-1")),
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("It printed hello.", FauxMessageOptions::default()).into(),
    ]);
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());

    let registry = create_registry();
    registry.install(CODING_TOOLS.clone())?;
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    // Tools reach files and processes only through the environment the Harness builds for each
    // call, here a local one in the conversation's directory.
    options.env = Some(Arc::new(|target, _| {
        let cwd = target.cwd.unwrap_or_else(|| ".".into());
        let env: Arc<dyn ExecutionEnv> = Arc::new(LocalExecutionEnv::at(cwd));
        Box::pin(async move { Ok(Some(env)) })
    }));

    // Memory storage keeps nothing across processes; `open_local_jsonl_storage` or
    // `open_rusqlite_storage` (feature `durable-sqlite`) persist the session.
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context).await?;
    // The root conversation remembers its model and directory.
    let agent = AgentChange::default()
        .model(ModelRef::new("faux", "faux-1"))
        .cwd(std::env::temp_dir().to_string_lossy());
    let root = harness
        .root(&context, CreateOptions { agent: Some(agent), init: None })
        .await?;

    // The input runs as durable tasks: a generation, a `bash` tool task, then a second generation.
    let submission = root.submit(SubmissionDraft::input("Say hello."), &context).await?;
    let settled = submission.wait(&context).await?;
    assert_eq!(settled.status, SubmissionStatus::Done);

    let answer = settled.answer.expect("a done input has an answer");
    let entry = root
        .commit(move |tx| async move { tx.entry_of(&ASSISTANT_ENTRY, answer).await }, &context)
        .await?
        .expect("the answer entry");
    if let Some(Message::Assistant(message)) = entry.model.as_deref().and_then(<[_]>::first) {
        assert_eq!(content_text(&message.content), "It printed hello.");
    }
    harness.close(&context).await
}
```

Beyond this:

- **Storage.** `MemoryStorage`, the portable `JsonlStorage` and
  `SqliteStorage` cores over any `FileSystem`/database, the local adapters
  (`open_local_jsonl_storage`, `open_rusqlite_storage`), and a conformance
  suite for custom backends (`durable-testing`).
- **Conversations.** `submit` with `when_busy` steering and follow-ups,
  `configure` (model, thinking level, extensions, tools, instructions,
  `cwd`), owned child conversations for subagents, `abort`, `fork`,
  `reset`, `compact`, `context`/`view_state` reads, and `watch_events` for
  Pi's agent events.
- **Tasks.** `define_task` state machines with phases, waits
  (`all_settled`/`fail_fast`), owned work, abort handlers and migrations;
  `Harness::inspect` and `task_graph` show live work.
- **Extensions.** `define_extension` with tools (`define_tool`), prompt
  sections, hooks on the generation and tool tasks, wraps, and documents
  (`define_doc`) for extension state.
- **Coding tools.** `create_read_tool`, `create_write_tool`,
  `create_edit_tool` and `create_bash_tool` (output limits, spill files,
  a command prefix and a `prepare` hook) over an `ExecutionEnv`.

## Differences from Pi

The port keeps Pi's behavior; these are the deliberate or Rust-forced
differences. Each is also documented on the module or item involved.

- **API shape.** `Models` is the only request entry point, as Pi intends
  once its temporary `compat` module is gone: the global api-registry,
  `stream`/`complete`/`stream_simple`/`complete_simple`, `generateImages`,
  the images api-registry, `registerFauxProvider` and the `getModel`/
  `getImageModel` aliases are not ported. Rust keeps the 0.7 provider
  handles (`openai::builder()`, `anthropic::from_env()`,
  `handle.model(id).build()`), each owning a `Models` (`handle.models()`);
  a handle also serves Chat Completions and keyless custom base URLs.
- **Types.** `ModelCompat` is one flat struct. `AgentMessage = Message`
  (no custom message roles; `Message::Custom` is gone). Pi's open records
  become `provider_options` maps; optional provider methods become `Option`
  returns or `supports_*()` probes. Unknown model types are dropped when a
  catalog is deserialized.
- **Runtime.** Abort signals are `CancellationToken`s, producers run on
  `tokio::spawn`, and abandoned operations are dropped. A synchronous throw
  becomes `Err` or an error stream. `AgentEventStream::result()` returns `Err`
  instead of hanging, and a stream that ends without a terminal event gives
  `AgentError::StreamClosed`. The default stream function is resolved at run
  time; `stream_simple_fn(models)` wraps `Models::stream_simple` as a
  `StreamFn`.
- **HTTP.** No vendor SDKs: requests are built by hand (reqwest + SSE) the way
  the SDKs send them, minus `X-Stainless` headers. HTTP errors read
  `"<status> <body>"`. `fetch` and SDK `client` options become `http_client`.
- **Auth.** OAuth flows take an injectable `OAuthFetch`, deadlines use the
  Tokio clock, `PI_OAUTH_CALLBACK_HOST` is read per login, and a failed lazy
  OAuth load is retried. GitHub Copilot keeps the `ghr_` refresh-token grant.
  Anthropic workload identity federation is not supported yet.
- **Agent.** `before_tool_call` may mutate arguments
  (`Arc<Mutex<Value>>`), tool updates after a tool settles are dropped, and
  the proxy pads sparse content indices.
- **Faux.** Text is chunked by `char`; factories return `Result` and get a
  state snapshot.
- **chord.** No JS Proxy: change drafts are owned values diffed at prepare
  time; diffs use deep equality.
- **Durable.** Records serialize to Pi's JSON shapes, so stores stay
  compatible with the TS backends. Documents are typed tokens with one
  address argument, transaction drafts are `DocDraft` handles
  (`get`/`edit`), and Pi's eager promises are spawned Tokio tasks.
  Callbacks are `Arc` closures returning `Result`; a panicking phase handler
  faults its task like a throw. `AgentChange` fields are `Option<Option<_>>`
  (keep or clear), and settings are a closure read at each resolution. The
  local environment is POSIX only, and SQLite runs on a dedicated thread
  through `rusqlite`. The coding tools' edit diff uses `similar`'s Myers
  diff, whose hunks can differ from jsdiff's on ties; `BashPrepare` takes
  and returns the execution by value. `CompactionPolicy.reserve_tokens` is
  an `i64` like Pi's `number`; a negative reserve sends `max_tokens` 0 for
  the summary request. `ToolExecutionApi::detached` (feature
  `durable-testing`) runs a tool outside a Harness for tests.
- **Not ported.** Providers other than OpenAI, Anthropic, GitHub Copilot and
  OpenRouter images; OpenAI ChatGPT/Codex OAuth; Azure OpenAI Responses;
  classifiers; telemetry contexts.
- **ai.rs extras.** `embeddings`, the `openai-images` API, the provider
  handles, `github_copilot::get_oauth_api_key`, and `AgentToolBuilder` /
  `AgentOptions::builder` conveniences.

## License

MIT
