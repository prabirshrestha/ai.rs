# ai

Simple to use LLM library for Rust with streaming, tool calling, OAuth helpers,
and a lightweight agent loop, inspired by [`pi`](https://github.com/earendil-works/pi).

## Table of Contents

- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tools](#tools)
  - [Defining Tools](#defining-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Streaming Tool Calls with Partial JSON](#streaming-tool-calls-with-partial-json)
  - [Validating Tool Arguments](#validating-tool-arguments)
  - [Complete Event Reference](#complete-event-reference)
- [Image Input](#image-input)
- [Image Generation](#image-generation)
  - [Basic Image Generation](#basic-image-generation)
  - [Notes and Limitations](#notes-and-limitations)
- [Thinking/Reasoning](#thinkingreasoning)
  - [Unified Interface](#unified-interface-streamsimplecompletesimple)
  - [Provider-Specific Options](#provider-specific-options-streamcomplete)
  - [Streaming Thinking Content](#streaming-thinking-content)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
  - [Aborting Requests](#aborting-requests)
  - [Continuing After Abort](#continuing-after-abort)
  - [Debugging Provider Payloads](#debugging-provider-payloads)
- [APIs, Models, and Providers](#apis-models-and-providers)
  - [Faux provider for tests](#faux-provider-for-tests)
  - [Providers and Models](#providers-and-models)
  - [Querying Providers and Models](#querying-providers-and-models)
  - [Custom Models](#custom-models)
  - [OpenAI Compatibility Settings](#openai-compatibility-settings)
  - [Thread Safety](#thread-safety)
  - [Type Safety](#type-safety)
- [Cross-Provider Handoffs](#cross-provider-handoffs)
  - [How It Works](#how-it-works)
  - [Example: Multi-Provider Conversation](#example-multi-provider-conversation)
  - [Provider Compatibility](#provider-compatibility)
- [Context Serialization](#context-serialization)
- [Browser Usage](#browser-usage)
  - [Browser Compatibility Notes](#browser-compatibility-notes)
  - [Environment Variables](#environment-variables)
  - [Checking Environment Variables](#checking-environment-variables)
- [OAuth Providers](#oauth-providers)
  - [CLI Login](#cli-login)
  - [Programmatic OAuth](#programmatic-oauth)
  - [Login Flow Example](#login-flow-example)
  - [Using OAuth Tokens](#using-oauth-tokens)
  - [Provider Notes](#provider-notes)
- [Agent Core](#agent-core)
  - [Installation](#agent-installation)
  - [Quick Start](#agent-quick-start)
  - [Core Concepts](#core-concepts)
  - [Event Flow](#event-flow)
  - [prompt_text() Event Sequence](#prompt_text-event-sequence)
  - [With Tool Calls](#with-tool-calls)
  - [continue_run() Event Sequence](#continue_run-event-sequence)
  - [Event Types](#event-types)
  - [Agent Options](#agent-options)
  - [Agent State](#agent-state)
  - [Methods](#methods)
  - [Session and Thinking Budgets](#session-and-thinking-budgets)
  - [Steering and Follow-up](#steering-and-follow-up)
  - [Custom Message Types](#custom-message-types)
  - [Tools](#agent-tools)
  - [Tool Error Handling](#agent-tool-error-handling)
  - [Proxy Usage](#proxy-usage)
  - [Low-Level API](#low-level-api)
- [Development](#development)
  - [Adding a New Provider](#adding-a-new-provider)
- [License](#license)

## Supported Providers

- **OpenAI** via Chat Completions, Responses, and Images
- **Anthropic** via Messages
- **GitHub Copilot** through OAuth-backed OpenAI/Anthropic-compatible routes
- **OpenRouter** for image generation
- **Azure Foundry and other compatible endpoints** through provider handles with
  explicit `base_url`, headers, and compatibility settings

The active built-in stream APIs are:

- `openai-completions`
- `openai-responses`
- `anthropic-messages`

The active built-in image generation APIs are:

- `openai-images`
- `openrouter-images`

The active built-in provider handles are focused on `openai`, `anthropic`, and
`github_copilot` for chat, plus `openai` and `openrouter` for image generation. Azure
Foundry, Ollama, vLLM, and other compatible endpoints can use configured
provider handles with explicit `base_url`, HTTP headers, and compatibility
settings.

Broad native provider-specific APIs outside OpenAI, Anthropic, GitHub Copilot,
and custom compatible routing are not part of the active built-in provider
surface. PRs to add support for additional providers are welcome.

Image generation is exposed through OpenAI-compatible image models and
OpenRouter image models. Chat image input and image blocks in tool results are
still supported by the regular chat APIs.

## Installation

```bash
cargo add ai
cargo add tokio --features macros,rt-multi-thread
cargo add futures serde_json
```

This crate uses Tokio-compatible async APIs. The examples use
`#[tokio::main]`, which requires Tokio's `macros` and `rt-multi-thread`
features. The examples also use `futures::StreamExt` for stream iteration and
`serde_json::json` for JSON Schema values.

## Quick Start

```rust
use ai::{complete_simple, providers::openai, Context, Message, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;

    let context = Context::builder()
        .system_prompt("You are a helpful assistant.")
        .message(Message::user_text("What is the capital of France?"))
        .build();

    let message = complete_simple(model, context, None).await?;
    println!("{message:?}");
    Ok(())
}
```

### Streaming

Use `stream_simple` when the UI should update as tokens arrive. The
`futures::StreamExt` import is only needed for `.next().await`.

```rust
use futures::StreamExt;

use ai::{providers::openai, stream_simple, AssistantMessageEvent, Context, Message, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder()
        .message(Message::user_text("Write a haiku about Rust."))
        .build();

    let mut events = stream_simple(model, context, None)?;
    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::TextDelta { delta, .. } = event? {
            print!("{delta}");
        }
    }

    Ok(())
}
```

### Provider Handles

Use `openai::from_env()` for OpenAI Responses. Use `openai::builder()` when
selecting Chat Completions or an OpenAI-compatible endpoint.

#### OpenAI Responses

```rust
use ai::providers::openai;

let openai_responses_from_env = openai::from_env()?;

let openai_responses_with_key = openai::builder()
    .api_key(Some("sk-..."))
    .responses()
    .build()?;
```

#### OpenAI Chat Completions

```rust
use ai::providers::openai;

let openai_chat_with_key = openai::builder()
    .api_key(Some("sk-..."))
    .chat_completions()
    .build()?;

let ollama_chat = openai::builder()
    .base_url("http://localhost:11434/v1")
    .chat_completions()
    .build()?;
```

#### Anthropic

```rust
use ai::providers::anthropic;

let anthropic_from_env = anthropic::from_env()?;

let anthropic_with_key = anthropic::builder()
    .api_key("sk-ant-...")
    .build()?;
```

### Dynamic Provider Choice

Provider handles are trait objects when the application wants to choose a
backend at runtime.

```rust
use ai::{complete_simple, providers::openai, Context, Message, Provider, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let (provider, model_id): (Box<dyn Provider>, &str) =
        if std::env::var("OPENAI_API_KEY").is_ok() {
            (Box::new(openai::from_env()?), "gpt-5.5")
        } else {
            let local = openai::builder()
                .provider_id("ollama")
                .base_url("http://localhost:11434/v1")
                .chat_completions()
                .build()?;
            (Box::new(local), "gemma3")
        };

    let model = provider.model(model_id).build()?;
    let context = Context::builder()
        .message(Message::user_text("Summarize Rust ownership."))
        .build();

    let message = complete_simple(model, context, None).await?;
    println!("{message:?}");
    Ok(())
}
```

## Tools

Tools enable LLMs to interact with external systems. This crate uses JSON
Schema values for tool definitions and provides validation helpers for tool
calls.

### Defining Tools

```rust
use ai::Tool;
use serde_json::json;

let weather_tool = Tool::builder("get_weather")
    .description("Get current weather for a location.")
    .parameters(json!({
        "type": "object",
        "properties": {
            "location": { "type": "string" },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    }))
    .build()?;
```

### Handling Tool Calls

Tool results use content blocks and can include both text and images.

```rust
use ai::{AssistantContent, Message, ToolResultContent, ToolResultMessage};

if let AssistantContent::ToolCall(call) = block {
    let result = run_weather_lookup(&call.arguments).await;
    context.messages.push(Message::ToolResult(ToolResultMessage {
        tool_call_id: call.id,
        tool_name: call.name,
        content: vec![ToolResultContent::text(result)],
        details: None,
        is_error: false,
        timestamp: 0,
    }));
}
```

### Streaming Tool Calls with Partial JSON

During streaming, tool call arguments are progressively parsed as they arrive.
This enables real-time UI updates before the complete arguments are available.

```rust
use ai::AssistantMessageEvent;

match event {
    AssistantMessageEvent::ToolCallDelta {
        content_index,
        delta,
        partial,
    } => {
        println!("tool block {content_index} delta: {delta}");
        println!("partial message: {partial:?}");
    }
    AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
        println!("tool completed: {} {:?}", tool_call.name, tool_call.arguments);
    }
    _ => {}
}
```

Important notes about partial tool arguments:

- During `ToolCallDelta`, arguments may be incomplete.
- Fields may be missing or partially parsed.
- String values may be truncated mid-word.
- Arrays and nested objects may be incomplete.
- Always validate final tool arguments before executing external effects.
- Use `content_index` to associate events with the right assistant content block.

### Validating Tool Arguments

When using the agent loop, tool arguments are validated before execution. When
implementing your own loop with `stream` or `complete`, use
`validate_tool_call` or `validate_tool_arguments`.

```rust
use ai::{validate_tool_call, Tool, ToolCall};

let validated = validate_tool_call(&tools, &tool_call)?;
```

### Complete Event Reference

All streaming events emitted during assistant message generation:

| Event | Description | Key Properties |
| --- | --- | --- |
| `Start` | Stream begins | `partial`: initial assistant message structure |
| `TextStart` | Text block starts | `content_index`: position in content array |
| `TextDelta` | Text chunk received | `delta`, `content_index` |
| `TextEnd` | Text block complete | `content`, `content_index` |
| `ThinkingStart` | Thinking block starts | `content_index` |
| `ThinkingDelta` | Thinking chunk received | `delta`, `content_index` |
| `ThinkingEnd` | Thinking block complete | `content`, `content_index` |
| `ToolCallStart` | Tool call begins | `content_index` |
| `ToolCallDelta` | Tool arguments stream | `delta`, `partial` |
| `ToolCallEnd` | Tool call complete | `tool_call` |
| `Done` | Stream complete | `reason`, `message` |
| `Error` | Error occurred | `reason`, `error` |

Streaming events for different content blocks are not guaranteed to be
contiguous. Consumers should use `content_index` to associate deltas and end
events with their blocks.

## Image Input

Models with vision capabilities can process images. Check `model.input` for
`ModelInput::Image`. If you pass images to a non-vision model, the message
transform layer downgrades unsupported image content to text placeholders.

```rust
use ai::{
    providers::openai, Context, ImageContent, Message, ModelInput, UserContent,
    UserMessage, UserMessageContent,
};

let openai = openai::from_env()?;
let model = openai.model("gpt-5.5").build()?;
if model.input.contains(&ModelInput::Image) {
    println!("model supports vision");
}

let context = Context {
    messages: vec![Message::User(UserMessage {
        content: UserMessageContent::Parts(vec![
            UserContent::text("What is in this image?"),
            UserContent::Image(ImageContent {
                data: "...base64...".to_string(),
                mime_type: "image/png".to_string(),
            }),
        ]),
        timestamp: 0,
    })],
    ..Default::default()
};
```

## Image Generation

Use `generate_images` with an OpenAI-compatible or OpenRouter image model. The
returned `AssistantImages` can contain text and image output blocks, matching
the selected model's capabilities.

### Basic Image Generation

```rust
use ai::{
    generate_images, providers::openai, ImageOutput, ImagesContext, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.image_model("gpt-image-2").build_image()?;

    let context = ImagesContext::builder()
        .text("Generate a small watercolor robot reading a book.")
        .build();

    let images = generate_images(model, context, None).await?;
    for output in images.output {
        match output {
            ImageOutput::Text(text) => println!("{}", text.text),
            ImageOutput::Image(image) => {
                println!("{} bytes of {}", image.data.len(), image.mime_type);
            }
        }
    }

    Ok(())
}
```

For Ollama's OpenAI-compatible image endpoint, use the OpenAI provider with an
Ollama base URL:

```rust
use ai::{generate_images, providers::openai, ImagesContext};

let ollama = openai::builder()
    .provider_id("ollama")
    .base_url("http://localhost:11434/v1")
    .images()
    .build()?;

let model = ollama.model("x/z-image-turbo").build_image()?;
let context = ImagesContext::builder()
    .text("Generate a small watercolor robot reading a book.")
    .build();

let images = generate_images(model, context, None).await?;
```

Set `OPENROUTER_API_KEY` for `openrouter::from_env()`, or pass a key through
`providers::openrouter::builder().api_key(Some("..."))`.

OpenRouter image models remain available through the `openrouter` provider:

```rust
use ai::{generate_images, providers::openrouter, ImagesContext};

let openrouter = openrouter::from_env()?;
let model = openrouter
    .model("google/gemini-3.1-flash-image-preview")
    .build_image()?;
let context = ImagesContext::builder().text("Generate a logo.").build();

let images = generate_images(model, context, None).await?;
```

### Notes and Limitations

The active Rust image-generation surface covers OpenAI-compatible
`/images/generations` models through the `openai-images` API and OpenRouter's
chat-completions-style image models through the `openrouter-images` API. The
OpenAI-compatible generations path supports text input; image edits are not
implemented yet. Provider errors are returned as `AssistantImages` with
`stop_reason: ImagesStopReason::Error`; cancelled requests use
`ImagesStopReason::Aborted`.

## Thinking/Reasoning

Many models support thinking or reasoning content. Check `model.reasoning` and
use `get_supported_thinking_levels` to inspect supported levels.

### Unified Interface (streamSimple/completeSimple)

Rust exports these as `stream_simple` and `complete_simple`.

```rust
use ai::{
    complete_simple, providers::anthropic, Context, Message, ModelThinkingLevel,
    SimpleStreamOptions,
};

let anthropic = anthropic::from_env()?;
let model = anthropic.model("claude-sonnet-4-5").build()?;
let options = SimpleStreamOptions {
    reasoning: Some(ModelThinkingLevel::Medium),
    ..Default::default()
};

let response = complete_simple(
    model,
    Context {
        messages: vec![Message::user_text("Solve: 2x + 5 = 13")],
        ..Default::default()
    },
    Some(options),
)
.await?;
```

### Provider-Specific Options (stream/complete)

`stream_simple` and `complete_simple` are the preferred app-level APIs,
They take `SimpleStreamOptions`, resolve the model's API, and map common
options such as reasoning, cache retention, API key, cancellation, payload
hooks, retry settings, and provider options onto the selected provider.

`stream` and `complete` are the lower-level APIs. Use them when you need the
non-simple `StreamOptions` shape or direct provider-option forwarding. For
provider-specific escape hatches, place fields in
`StreamOptions::provider_options` using provider option names such as
`toolChoice`, `serviceTier`, or `thinkingDisplay`.

The crate root also exports scoped direct provider stream functions:

- `stream_openai_completions` / `stream_simple_openai_completions`
- `stream_openai_responses` / `stream_simple_openai_responses`
- `stream_anthropic` / `stream_simple_anthropic`

Provider modules expose typed provider options for direct provider calls:

- `providers::openai_completions::OpenAICompletionsOptions`
- `providers::openai_responses::OpenAIResponsesOptions`
- `providers::anthropic::AnthropicOptions`

### Streaming Thinking Content

Thinking content streams through `ThinkingStart`, `ThinkingDelta`, and
`ThinkingEnd` events. Completed messages store thinking blocks as
`AssistantContent::Thinking`.

## Stop Reasons

Every `AssistantMessage` includes a `stop_reason` field that indicates how the
generation ended:

- `Stop` - Normal completion
- `Length` - Output hit the maximum token limit
- `ToolUse` - Model is calling tools and expects tool results
- `Error` - An error occurred during generation
- `Aborted` - Request was cancelled

`AssistantMessage` may also include `response_id`, a provider-specific response
or message identifier when the underlying API exposes one.

## Error Handling

Setup failures before a stream exists are returned as `Error` values. Once a
provider stream exists, provider-declared failures and cancellation are
surfaced as terminal `AssistantMessageEvent::Error` events carrying the final
assistant message. The `complete_*` helpers return that final assistant message;
check `message.stop_reason` for `StopReason::Error` or `StopReason::Aborted`.
Transport or decoder failures that cannot be represented as provider messages
still return `Err`.

### Aborting Requests

Use a Tokio cancellation token to abort in-flight requests. Prefer direct
struct initialization over mutating default options:

```rust
use ai::{SimpleStreamOptions, StreamOptions};
use tokio_util::sync::CancellationToken;

let token = CancellationToken::new();
let options = SimpleStreamOptions {
    stream: StreamOptions {
        cancellation_token: Some(token.clone()),
        ..Default::default()
    },
    ..Default::default()
};

token.cancel();
```

### Continuing After Abort

Abort produces an assistant message with `StopReason::Aborted`. The transform
layer drops aborted assistant turns before follow-up messages so conversations
can continue cleanly.

### Debugging Provider Payloads

Use `StreamOptions::on_payload` and `StreamOptions::on_response` hooks to
inspect or override provider payloads and observe raw provider responses. The
hooks are supported by `stream`, `complete`, `stream_simple`, and
`complete_simple`.

## APIs, Models, and Providers

Provider handles build executable models. Built-in language model APIs include:

- **`anthropic-messages`**: Anthropic Messages API
- **`openai-completions`**: OpenAI Chat Completions API
- **`openai-responses`**: OpenAI Responses API

`register_faux_provider()` is legacy support for the crate's own unit tests.
New application code should prefer provider handles and custom provider
implementations instead of registering global providers.

### Faux provider for tests

`register_faux_provider()` registers a temporary in-memory provider for tests.
It is opt-in and not part of the built-in provider set.

### Providers and Models

A provider offers models through a specific API. In this crate:

- **Anthropic** models use `anthropic-messages`.
- **OpenAI** models use `openai-completions` or `openai-responses`.
- **GitHub Copilot** models use OAuth-backed OpenAI/Anthropic-compatible
  routes.
- **Azure Foundry** and other compatible endpoints are configured as custom
  models by choosing an active API, setting `base_url`, and filling
  `ModelCompat` where the endpoint differs from the default request shape.

Built-in provider handles create executable model values directly from string
IDs. There is no built-in model catalog; applications that need one should keep
it in application state and build models through configured provider handles.

### Querying Providers and Models

```rust
use ai::{providers::{github_copilot, openai}, Provider};

let provider = openai::from_env()?;
let capabilities = provider.capabilities();
let model = provider.model("gpt-5.5").build()?;

let copilot = github_copilot::builder()
    .api_key("...")
    .anthropic_messages()
    .build()?;
let claude = copilot.model("claude-opus-4.5").build()?;
```

### Custom Models

You can create provider-bound models for local inference servers or custom
endpoints:

```rust
use ai::{
    providers::openai, stream_simple, Context, Message, SimpleStreamOptions,
};

let provider = openai::builder()
    .provider_id("azure-foundry")
    .api_key(Some("..."))
    .base_url("https://example.services.ai.azure.com/openai/v1")
    .chat_completions()
    .build()?;
let model = provider.model("gpt-5.5").build()?;

let stream = stream_simple(
    model,
    Context {
        messages: vec![Message::user_text("What is the capital of France?")],
        ..Default::default()
    },
    Some(SimpleStreamOptions::default()),
)?;
```

The same pattern works for local inference servers such as Ollama, vLLM, and LM
Studio when they expose an OpenAI-compatible chat endpoint.

Some OpenAI-compatible servers do not understand the `developer` role used for
reasoning-capable models. For those endpoints, build the model with compat
metadata so the system prompt is sent as a `system` message instead. If the
server also does not support `reasoning_effort`, disable that compat flag too.

### OpenAI Compatibility Settings

The `openai-completions` API is implemented by many providers with minor
differences. `ModelCompat` stores compatibility metadata for explicit custom
models, but the active built-in surface does not infer broad provider-specific
behavior from provider names or base URLs.

Set model-builder compat metadata when the target OpenAI-compatible endpoint
needs payload differences such as non-standard reasoning, cache-control,
max-token, or tool-result behavior.

### Thread Safety

Provider handles are regular cloneable values. Build them during application
startup, pass them where needed, and create executable model values with
`provider.model(id).build()?`.

### Type Safety

Public types are serializable with `serde` where they represent portable
context or message state.

## Cross-Provider Handoffs

The library supports handoffs between OpenAI, Anthropic, and GitHub
Copilot-compatible models within the same conversation.

### How It Works

When messages from one provider are sent to a different provider, the crate
transforms them for compatibility:

- User and tool-result messages are passed through.
- Assistant messages from the same provider/API are preserved as-is.
- Assistant messages from different providers have thinking blocks converted to
  text with `<thinking>` tags where needed.
- Tool calls and regular text are preserved.

### Example: Multi-Provider Conversation

```rust
use ai::{complete_simple, providers::{anthropic, openai}, Context, Message, SimpleStreamOptions};

let mut context = Context {
    messages: vec![Message::user_text("What is 25 * 18?")],
    ..Default::default()
};

let anthropic = anthropic::from_env()?;
let claude = anthropic.model("claude-sonnet-4-5").build()?;
let claude_response =
    complete_simple(claude, context.clone(), Some(SimpleStreamOptions::default())).await?;
context.messages.push(Message::Assistant(claude_response));

let openai = openai::from_env()?;
let gpt = openai.model("gpt-5.5").build()?;
context.messages.push(Message::user_text("Is that calculation correct?"));
let gpt_response =
    complete_simple(gpt, context.clone(), Some(SimpleStreamOptions::default())).await?;
context.messages.push(Message::Assistant(gpt_response));
```

### Provider Compatibility

All active providers can handle shared text, tool calls, tool results including
images, thinking/reasoning blocks after transformation, and aborted messages
with partial content.

## Context Serialization

`Context`, `Message`, assistant content, tool calls, and tool results implement
`Serialize` and `Deserialize`, so context can be persisted or handed to another
process.

```rust
use ai::Context;

let serialized = serde_json::to_string(&context)?;
let restored: Context = serde_json::from_str(&serialized)?;
```

If the context contains images encoded as base64, those are serialized too.
Serialized user, assistant, and tool-result messages include stable `role`
fields, including assistant messages nested in stream events.

## Browser Usage

This Rust crate is server/native focused and does not provide browser-specific
packaging. Pass API keys explicitly through options or use environment
variables on the server.

### Browser Compatibility Notes

Not applicable to this Rust crate.

### Environment Variables

| Provider | Environment variables |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_OAUTH_TOKEN`, then `ANTHROPIC_API_KEY` |
| `github-copilot` | `COPILOT_GITHUB_TOKEN` |

Explicit API keys in `StreamOptions` take precedence over environment lookup.

### Checking Environment Variables

```rust
use ai::get_env_api_key;

let key = get_env_api_key("openai");
```

## OAuth Providers

The OAuth registry includes:

- **Anthropic** (Claude Pro/Max subscription)
- **GitHub Copilot** (Copilot subscription)

### CLI Login

This crate exposes login primitives for applications that want to provide their
own login UI.

### Programmatic OAuth

```rust
use ai::{get_oauth_provider, OAuthCredentials};

let provider = get_oauth_provider("github-copilot").expect("provider");
```

### Login Flow Example

Use `login_anthropic` or `login_github_copilot` with `OAuthLoginCallbacks` to
drive the login UI from your application.

```rust
use ai::{providers::github_copilot, OAuthLoginCallbacks, Result};

async fn login() -> Result<()> {
    let callbacks = OAuthLoginCallbacks::builder()
        .on_device_code(|info| {
            eprintln!("Open {} and enter {}", info.verification_uri, info.user_code);
        })
        .on_prompt(|_| async { Ok(String::new()) })
        .build();

    let credentials = github_copilot::oauth().login(callbacks).await?;
    // Persist credentials here.

    Ok(())
}
```

### Using OAuth Tokens

Use provider-specific helpers to turn stored credentials into the API key used
by provider builders or stream options. For GitHub Copilot, the helper refreshes
expired credentials and returns `new_credentials`; persist those back to your
auth store.

```rust
use ai::{providers::github_copilot, OAuthCredentials, Result};

async fn provider_from_stored_credentials(
    credentials: OAuthCredentials,
) -> Result<github_copilot::GitHubCopilot> {
    let oauth_key = github_copilot::get_oauth_api_key(&credentials).await?;

    // Persist oauth_key.new_credentials here.

    github_copilot::builder()
        .api_key(oauth_key.api_key)
        .base_url(github_copilot::base_url_for_credentials(
            &oauth_key.new_credentials,
        ))
        .responses()
        .build()
}
```

### Provider Notes

**GitHub Copilot**: OAuth helpers and dynamic request headers are included.
Some Copilot model ids use vendor names, but they are routed through the active
OpenAI/Anthropic-compatible APIs in this crate. Other native provider APIs are
not registered.

**Anthropic**: OAuth follows the Claude Pro/Max OAuth flow.

## Agent Core

Stateful agent support is part of this crate. It runs model turns, executes
registered tools, appends tool results, and continues until the assistant
stops, an error occurs, or a hook asks the loop to stop.

### Agent Installation

No separate crate is required. The agent core lives in `ai`.

```bash
cargo add ai
```

### Agent Quick Start

```rust
use ai::{providers::anthropic, Agent, AgentEvent, AgentOptions, AssistantMessageEvent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let anthropic = anthropic::from_env()?;
    let model = anthropic.model("claude-sonnet-4-5").build()?;
    let agent = Agent::new(AgentOptions::new(model));

    agent.set_system_prompt("You are a helpful assistant.").await;

    let subscription = agent.subscribe(async |event, _cancellation_token| {
        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = event
        {
            print!("{delta}");
        }

        Ok(())
    });

    agent.prompt_text("Hello!", Vec::new()).await?;
    subscription.unsubscribe();

    Ok(())
}
```

### Core Concepts

#### AgentMessage vs LLM Message

`AgentMessage` is the same portable `Message` enum used by the shared LLM API.
It can contain standard LLM messages (`User`, `Assistant`, `ToolResult`) and
`Custom` app-owned messages.

LLMs only understand user, assistant, and tool-result messages. The
`convert_to_llm` function bridges this gap by filtering or transforming custom
messages before each provider call.

#### Message Flow

```text
AgentMessage[] -> transform_context() -> AgentMessage[] -> convert_to_llm() -> Message[] -> LLM
                    (optional)                           (required)
```

`transform_context` is intended for pruning, compaction, or external context
injection. `convert_to_llm` filters or converts app-owned messages.

### Event Flow

The agent emits events for UI updates. Understanding the event sequence helps
build responsive interfaces.

#### prompt_text() Event Sequence

When you call `prompt_text("Hello")`, the wrapper emits this core sequence:

```text
prompt_text("Hello")
|- agent_start
|- turn_start
|- message_start   user
|- message_end     user
|- message_start   assistant
|- message_update  assistant delta
|- message_end     assistant
|- turn_end
`- agent_end
```

#### With Tool Calls

If the assistant calls tools, the loop emits `tool_execution_start`, optional
`tool_execution_update`, `tool_execution_end`, then a tool-result
`message_start` / `message_end`. If the batch does not terminate, the next turn
starts and the model receives the tool results.

Tool execution mode is configurable:

- `Parallel` is the default. Preflight runs sequentially, allowed tools execute
  concurrently, completion events emit as each tool finalizes, and persisted
  tool-result messages remain in assistant source order.
- `Sequential` executes tool calls one by one.

`before_tool_call` runs after `tool_execution_start` and validated argument
parsing. `after_tool_call` runs after tool execution and before final tool
events. Tool results can set `terminate = true`; the loop stops early only when
every finalized result in the batch terminates.

Low-level loop callers can set `should_stop_after_turn` to stop gracefully after
the current turn completes. It runs after `turn_end`, before steering/follow-up
queues are polled, and before another model request starts.

#### continue_run() Event Sequence

`continue_run()` resumes from existing context without adding a new message.
Use it for retries after errors. The last message in context must be a user or
tool-result message, not an assistant message.

#### Event Types

| Event | Description |
| --- | --- |
| `AgentStart` | Agent begins processing |
| `AgentEnd` | Final event for the run. Awaited subscribers for this event still count toward settlement |
| `TurnStart` | New turn begins: one LLM call plus tool executions |
| `TurnEnd` | Turn completes with assistant message and tool results |
| `MessageStart` | Any message begins: user, assistant, or tool result |
| `MessageUpdate` | Assistant-only update containing the underlying assistant stream event |
| `MessageEnd` | Message completes |
| `ToolExecutionStart` | Tool begins |
| `ToolExecutionUpdate` | Tool streams progress |
| `ToolExecutionEnd` | Tool completes |

`Agent::subscribe` listeners are awaited in registration order. `agent_end`
means no more loop events will be emitted, but `wait_for_idle` and
`prompt_text` settle only after awaited final listeners finish.

### Agent Options

`AgentOptions` contains:

- `initial_state`: system prompt, model, thinking level, tools, and messages.
- `convert_to_llm`: converts agent messages to LLM messages.
- `transform_context`: prunes, compacts, or injects context before conversion.
- `steering_mode` and `follow_up_mode`: queue handling behavior.
- `stream_fn`: custom stream function for proxy backends.
- `session_id`: forwarded through `SimpleStreamOptions`.
- `tool_execution`: parallel or sequential tool execution.
- `before_tool_call` and `after_tool_call`: preflight and postprocess hooks.
- `prepare_next_turn`: updates context, model, or thinking level before another
  turn starts.
- `options`: transport, retry, cancellation, payload hooks, provider options,
  thinking budgets, and API key defaults.

```rust
use ai::{Agent, AgentOptions, AgentState, ModelThinkingLevel, ToolExecutionMode};

let initial_state = AgentState::builder(model.clone())
    .system_prompt("You are a helpful assistant.")
    .thinking_level(ModelThinkingLevel::Medium)
    .tools(tools)
    .messages(messages)
    .build();

let agent = Agent::new(
    AgentOptions::builder(model)
        .initial_state(initial_state)
        .tool_execution(ToolExecutionMode::Parallel)
        .session_id("session-123")
        .build(),
);
```

### Agent State

`AgentState` contains the system prompt, active model, thinking level, tools,
message history, streaming status, pending tool call IDs, and the latest error
message.

```rust
let state = AgentState::builder(model)
    .system_prompt("You are a helpful assistant.")
    .thinking_level(ModelThinkingLevel::Medium)
    .message(Message::user_text("Hello"))
    .build();
```

During streaming, `streaming_message` contains the current partial assistant
message. `is_streaming` remains true until the run fully settles, including
awaited `agent_end` subscribers.

### Methods

#### Prompting

```rust
agent.prompt_text("Hello", Vec::new()).await?;
agent.prompt_messages(vec![Message::user_text("Hello")]).await?;
agent.continue_run().await?;
```

`continue_run` resumes from current context. The last message must be a user or
tool-result message.

#### State Management

Use the state and option mutation helpers to update system prompt, model,
thinking level, tools, messages, session ID, queues, hooks, and tool execution
mode. `reset` returns the agent to its initial state.

```rust
agent.set_system_prompt("New prompt").await;
agent.set_model(model).await;
agent.set_thinking_level(ModelThinkingLevel::Medium).await;
agent.set_tools(tools).await;
agent.set_tool_execution(ToolExecutionMode::Sequential).await;
agent.set_messages(new_messages).await;
agent.push_message(message).await;
agent.reset().await;
```

#### Session and Thinking Budgets

`AgentOptions::session_id` is forwarded to providers that support prompt-cache
or session affinity behavior. `AgentOptions::options.thinking_budgets` is
applied by the simple-stream option builder before each model call.

```rust
agent.set_session_id(Some("session-123".to_string())).await;
agent.set_thinking_budgets(Some(ThinkingBudgets {
    minimal: Some(128),
    low: Some(512),
    medium: Some(1024),
    high: Some(2048),
}));
```

#### Control

```rust
agent.abort().await;
agent.wait_for_idle().await;
```

#### Events

```rust
let subscription = agent.subscribe(|event, cancellation_token| async move {
    if matches!(event, AgentEvent::AgentEnd { .. }) {
        flush_session_state(cancellation_token).await?;
    }

    Ok(())
});

subscription.unsubscribe();
```

Keep the subscription handle alive while the listener remains registered.
Dropping the handle also unsubscribes.

### Steering and Follow-up

Steering messages let you interrupt the agent while it is running. Follow-up
messages let you queue work after the agent would otherwise stop.

When steering messages are detected after a turn completes:

1. All tool calls from the current assistant message have already finished.
2. Steering messages are injected.
3. The LLM responds on the next turn.

Follow-up messages are checked only when there are no more tool calls and no
steering messages.

### Custom Message Types

Use `Message::Custom` for app-specific agent transcript entries. Custom
messages are retained in agent state, then filtered or converted by
`convert_to_llm` before provider calls.

### Agent Tools

Agent tools implement the `AgentTool` trait. `definition()` returns the shared
`Tool` schema, `label()` provides UI text, `execution_mode()` can force a whole
batch to run sequentially, `prepare_arguments()` can reshape model arguments
before validation, and `execute()` performs the tool work.

```rust
use ai::{
    providers::openai, Agent, AgentOptions, AgentToolBuilder, AgentToolResult, Result,
};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<()> {
    let weather_tool = AgentToolBuilder::new("get_weather")
        .description("Get current weather for a city.")
        .parameters(json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }))
        .label("Weather")
        .execute(|args| async move {
            let city = args
                .get("city")
                .and_then(Value::as_str)
                .unwrap_or("unknown city");

            Ok(AgentToolResult::text(format!(
                "The weather in {city} is 68F and clear."
            )))
        })
        .build()?;

    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;

    let agent = Agent::new(
        AgentOptions::builder(model)
            .tool(weather_tool)
            .build(),
    );

    agent
        .prompt_text("What is the weather in Seattle?", Vec::new())
        .await?;

    Ok(())
}
```

Implement `AgentTool` directly when a tool needs state, custom argument
preparation, an execution mode override, cancellation handling, or streaming
updates.

```rust
use ai::{AgentTool, AgentToolResult, AgentToolUpdateCallback, Tool};
use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

struct WeatherTool;

#[async_trait]
impl AgentTool for WeatherTool {
    fn definition(&self) -> Tool {
        Tool::builder("get_weather")
            .description("Get current weather for a city.")
            .build()
            .expect("valid weather tool")
    }

    fn label(&self) -> &str {
        "Weather"
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        _cancellation_token: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> ai::AgentResult<AgentToolResult> {
        let city = args
            .get("city")
            .and_then(Value::as_str)
            .unwrap_or("unknown city");

        Ok(AgentToolResult::text(format!(
            "The weather in {city} is 68F and clear."
        )))
    }
}
```

#### Agent Tool Error Handling

Tool failures should return an error from `execute()`. The loop catches that
error and reports a tool-result message with `is_error = true`.

Return `terminate = true` from `execute()` or `after_tool_call` to hint that
the agent should stop after the current tool batch. This only takes effect when
every finalized tool result in the batch is terminating.

### Proxy Usage

For proxy backends, pass a custom `StreamFn` through `AgentOptions::stream_fn`
or directly to `agent_loop`. The function receives the selected `Model`, the
converted `Context`, and `SimpleStreamOptions`.

### Low-Level API

Use `agent_loop` or `agent_loop_continue` when you want an event stream, and
`run_agent_loop` or `run_agent_loop_continue` when you want to await the whole
loop directly.

```rust
use futures::StreamExt;

use ai::{
    agent_loop, agent_loop_continue, providers::openai, AgentContext, AgentLoopConfig, Message,
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;

    let context = AgentContext::builder()
        .system_prompt("You are helpful.")
        .build();
    let config = AgentLoopConfig::new(model);
    let user_message = Message::user_text("Hello");

    let mut events = agent_loop(
        vec![user_message.clone()],
        context.clone(),
        config.clone(),
        None,
        None,
    );
    while let Some(event) = events.next().await {
        println!("{event:?}");
    }

    let mut context = context;
    context.messages.push(user_message);
    let mut events = agent_loop_continue(context, config, None, None)?;
    while let Some(event) = events.next().await {
        println!("{event:?}");
    }

    Ok(())
}
```

Low-level streams are observational. They preserve event order, but they do not
wait for async event handling to settle before later producer phases continue.
Use `Agent` when message processing must be a barrier before tool preflight.

## Development

```bash
cargo fmt --all --check
cargo check -p ai --all-targets
cargo clippy -p ai --all-targets -- -D warnings
cargo test -p ai
cargo test -p ai --lib
cargo test -p ai --doc
cargo test -p ai --tests
cargo test --workspace
```

This crate currently keeps Rust test coverage in module-level unit tests under
`src`; there is no `crates/ai/tests` integration-test directory at the moment.

### Adding a New Provider

Adding a new LLM provider generally requires changes across multiple files:

#### 1. Core Types (`src/types.rs`)

- Add the API identifier if the provider needs a new transport shape.
- Create provider-specific options where direct provider calls need typed
  options.
- Add or extend compatibility metadata only when the payload behavior differs.

#### 2. Provider Implementation (`src/providers/`)

Create a provider module that exports:

- `stream_<provider>()`
- `stream_simple_<provider>()`
- Provider-specific options
- Message conversion from `Context` to provider payload
- Tool conversion if the provider supports tools
- Response parsing into standardized assistant events

#### 3. Provider Factory

- Implement `Provider` for the configured provider handle.
- Return model builders from `model(id)` and future capability builders.
- Add root-level exports in `src/lib.rs` when the provider should be public.

#### 4. Runtime API

- Implement the capability runtime trait carried by the built model, such as
  `LanguageModelApi`.
- Map provider capability, cost, input, context-window, and reasoning metadata
  onto the shared `Model` type.

#### 5. Tests

Create or update tests for streaming, tool use, token usage, abort behavior,
context overflow, empty messages, Unicode handling, tool-result edge cases,
image input/tool-result images if applicable, and cross-provider handoff.

#### 6. Agent Integration

If the provider needs agent-specific behavior, update this crate's agent tests
and examples directly.

#### 7. Documentation

Update this README with provider scope, authentication, provider-specific
options, and environment variables.

## License

MIT
