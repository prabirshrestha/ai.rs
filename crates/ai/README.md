# ai

Unified LLM API with automatic model discovery, provider configuration, token
and cost tracking, streaming events, tool calling, context persistence, and
handoff to other models mid-session.

This crate is a Rust port of the focused
[`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai)
and
[`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent)
surfaces. Unlike the TypeScript repo, the agent loop lives in this `ai` crate;
there is no separate harness crate here.

**Note**: Like upstream `pi-ai`, this crate focuses on models that support tool
calling, because tool calling is essential for agentic workflows.

## Table of Contents

- [Supported Providers](#supported-providers)
- [Port Scope](#port-scope)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agent Loop](#agent-loop)
- [Tools](#tools)
  - [Defining Tools](#defining-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Streaming Tool Calls with Partial JSON](#streaming-tool-calls-with-partial-json)
  - [Validating Tool Arguments](#validating-tool-arguments)
  - [Complete Event Reference](#complete-event-reference)
- [Image Input](#image-input)
- [Thinking/Reasoning](#thinkingreasoning)
  - [Unified Interface](#unified-interface-stream_simplecomplete_simple)
  - [Provider-Specific Options](#provider-specific-options-streamcomplete)
  - [Streaming Thinking Content](#streaming-thinking-content)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
  - [Aborting Requests](#aborting-requests)
  - [Continuing After Abort](#continuing-after-abort)
  - [Debugging Provider Payloads](#debugging-provider-payloads)
- [APIs, Models, and Providers](#apis-models-and-providers)
  - [Faux Provider for Tests](#faux-provider-for-tests)
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
- [Development](#development)
  - [Adding a New Provider](#adding-a-new-provider)
- [License](#license)

## Supported Providers

- **OpenAI** via Chat Completions and Responses
- **Anthropic** via Messages
- **GitHub Copilot** via OAuth-backed OpenAI/Anthropic-compatible routes

The active built-in stream APIs are:

- `openai-completions`
- `openai-responses`
- `anthropic-messages`

Cloudflare, Bedrock, Google, Mistral, Azure OpenAI Responses, OpenAI Codex
Responses, and other broad provider-specific APIs are not part of the active
built-in provider surface in this port. PRs to add support for additional
providers are welcome.

The TypeScript coding-agent harness, CLI, and TUI are not included.

## Port Scope

This crate tracks upstream `pi` behavior for the Rust APIs that correspond to
the active scope:

| Upstream `pi` area | Rust status |
| --- | --- |
| `packages/ai/src/types.ts`, `stream.ts`, `models.ts`, `api-registry.ts` | Ported as shared message/context types, stream helpers, generated model metadata, and API registry. |
| `packages/ai/src/providers/openai-completions.ts` | Ported for Chat Completions-compatible streaming and simple options. |
| `packages/ai/src/providers/openai-responses.ts` | Ported for OpenAI Responses streaming and simple options. |
| `packages/ai/src/providers/anthropic.ts` | Ported for Anthropic Messages streaming and simple options. |
| `packages/ai/src/providers/faux.ts` | Ported for deterministic provider and agent-loop tests. |
| `packages/agent/src/agent.ts`, `agent-loop.ts`, `types.ts` | Ported into this crate for core agent state, queueing, lifecycle events, tool execution, hooks, and continuation. |
| `packages/agent/src/harness/**`, coding-agent CLI, and TUI | Not included in this Rust crate. |
| Image generation, Cloudflare, Bedrock, Google, Mistral, Azure OpenAI Responses, OpenAI Codex Responses | Not part of the active built-in provider surface. PRs to add support are welcome. |

The goal is behavioral parity before Rust-specific API polish. Where upstream
uses TypeScript casts for provider-specific escape hatches, this crate exposes
the equivalent through typed options where possible and
`StreamOptions::provider_options` where the upstream behavior is intentionally
loose.

## Installation

```bash
cargo add ai
cargo add tokio --features macros,rt-multi-thread
```

This crate uses Tokio streams and `reqwest` internally. Applications normally
use Tokio as their async runtime. The examples use `#[tokio::main]`, which
requires the `macros` and `rt-multi-thread` Tokio features.

## Quick Start

```rust
use futures::StreamExt;

use ai::{
    complete_simple, get_model, stream_simple, AssistantContent, AssistantMessageEvent, Context,
    Message, Result, SimpleStreamOptions, Tool, ToolResultContent, ToolResultMessage,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let model = get_model("openai", "gpt-4o-mini").expect("model");

    let weather_tool = Tool {
        name: "get_weather".to_string(),
        description: "Get current weather for a location.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" }
            },
            "required": ["location"]
        }),
    };

    let mut context = Context {
        system_prompt: Some("You are a helpful assistant.".to_string()),
        messages: vec![Message::user_text("What is the capital of France?")],
        tools: vec![weather_tool],
    };

    let mut events =
        stream_simple(model.clone(), context.clone(), Some(SimpleStreamOptions::default()))?;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::TextDelta { delta, .. } => print!("{delta}"),
            AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                println!("tool: {}({})", tool_call.name, tool_call.arguments);
            }
            AssistantMessageEvent::Done { reason, .. } => println!("\nfinished: {reason:?}"),
            AssistantMessageEvent::Error { error, .. } => eprintln!("{error:?}"),
            _ => {}
        }
    }

    let response = complete_simple(model, context.clone(), Some(SimpleStreamOptions::default())).await?;
    context.messages.push(Message::Assistant(response.clone()));

    for block in response.content {
        match block {
            AssistantContent::Text(text) => println!("{}", text.text),
            AssistantContent::ToolCall(call) => {
                context.messages.push(Message::ToolResult(ToolResultMessage {
                    tool_call_id: call.id,
                    tool_name: call.name,
                    content: vec![ToolResultContent::text("Paris")],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                }));
            }
            AssistantContent::Thinking(_) => {}
        }
    }

    Ok(())
}
```

## Agent Loop

The reusable agent loop is part of this crate. It runs model turns, executes
registered tools, appends tool results, and continues until the assistant stops,
an error occurs, or a hook asks the loop to stop. The TypeScript harness, CLI,
and TUI are not included in this Rust port.

For application-owned state, use `Agent`. For direct loop control, use
`agent_loop` or `run_agent_loop`.

```rust
use futures::StreamExt;

use ai::{
    agent_loop, get_model, AgentContext, AgentEvent, AgentLoopConfig, Message, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = get_model("openai", "gpt-4o-mini").expect("model");
    let context = AgentContext {
        system_prompt: Some("You are a concise assistant.".to_string()),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    let config = AgentLoopConfig::new(model);
    let prompts = vec![Message::user_text("What is the capital of France?")];

    let mut events = agent_loop(prompts, context, config, None, None);
    while let Some(event) = events.next().await {
        match event {
            AgentEvent::MessageEnd { message } => println!("{message:?}"),
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                println!("running tool: {tool_name}");
            }
            _ => {}
        }
    }

    let _new_messages = events.result().await?;
    Ok(())
}
```

Core loop hooks track upstream `pi` semantics:

- `transform_context` runs before `convert_to_llm`.
- `before_tool_call` receives already-validated arguments. Its `args` field is
  shared mutable state so hooks can mutate arguments before execution, matching
  upstream JavaScript object-reference behavior; the loop does not revalidate
  after that mutation.
- `after_tool_call` can replace content/details/error state and can set
  `terminate`; a tool batch terminates only when every finalized result has
  `terminate = true`.
- On `AgentLoopConfig`, `prepare_next_turn` can replace the next context,
  model, or reasoning level before steering/follow-up polling starts another
  provider request. On `AgentOptions`, `prepare_next_turn` mirrors upstream
  `Agent.prepareNextTurn`: it receives only the active cancellation token and
  can return the same turn update.

## Tools

Tools enable LLMs to interact with external systems. This crate uses JSON Schema
values for tool definitions and provides validation helpers for tool calls.

### Defining Tools

```rust
use ai::Tool;
use serde_json::json;

let weather_tool = Tool {
    name: "get_weather".to_string(),
    description: "Get current weather for a location.".to_string(),
    parameters: json!({
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
    }),
};
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
- Always validate final tool arguments before executing external effects.
- Use `content_index` to associate events with the right assistant content block.

### Validating Tool Arguments

When using the agent loop, tool arguments are validated before execution. When
implementing your own loop, use `validate_tool_call` or
`validate_tool_arguments`.

```rust
use ai::{validate_tool_call, Tool, ToolCall};

let validated = validate_tool_call(&tools, &tool_call)?;
```

### Complete Event Reference

| Event | Description |
| --- | --- |
| `Start` | Stream begins with the initial partial assistant message. |
| `TextStart` | Text block starts. |
| `TextDelta` | Text chunk received. |
| `TextEnd` | Text block complete. |
| `ThinkingStart` | Thinking block starts. |
| `ThinkingDelta` | Thinking chunk received. |
| `ThinkingEnd` | Thinking block complete. |
| `ToolCallStart` | Tool call block starts. |
| `ToolCallDelta` | Tool arguments stream as partial JSON. |
| `ToolCallEnd` | Tool call is complete. |
| `Done` | Stream completed with a final assistant message. |
| `Error` | Provider or cancellation error with partial assistant message. |

Streaming events for different content blocks are not guaranteed to be
contiguous. Consumers should use `content_index` to associate deltas and end
events with their blocks.

## Image Input

Models with vision capabilities can process images. Check `model.input` for
`ModelInput::Image`. If you pass images to a non-vision model, the message
transform layer downgrades unsupported image content to text placeholders.

```rust
use ai::{
    get_model, Context, ImageContent, Message, ModelInput, UserContent, UserMessage,
    UserMessageContent,
};

let model = get_model("openai", "gpt-4o-mini").expect("model");
if model.input.contains(&ModelInput::Image) {
    println!("model supports image input");
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

## Thinking/Reasoning

Many models support thinking or reasoning content. Check `model.reasoning` and
use `get_supported_thinking_levels` to inspect supported levels.

### Unified Interface (stream_simple/complete_simple)

```rust
use ai::{complete_simple, get_model, Context, Message, ModelThinkingLevel, SimpleStreamOptions};

let model = get_model("anthropic", "claude-sonnet-4-5").expect("model");
let options = SimpleStreamOptions {
    reasoning: Some(ModelThinkingLevel::Medium),
    ..SimpleStreamOptions::default()
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

Use `stream_simple` and `complete_simple` for normal application code. They take
`SimpleStreamOptions`, resolve the model's API, and map common options such as
reasoning, cache retention, service tier, API key, cancellation, payload hooks,
and retry settings onto the selected provider.

Use `stream` and `complete` when you need provider-specific options beyond the
common `SimpleStreamOptions` shape:

- `OpenAICompletionsOptions`
- `OpenAIResponsesOptions`
- `AnthropicOptions`

For parity with upstream's cast-based escape hatch, `stream_simple` also checks
`StreamOptions::provider_options` for provider-specific fields that upstream
accepts through casts, such as OpenAI Chat Completions `toolChoice`.

### Streaming Thinking Content

Thinking content streams through `ThinkingStart`, `ThinkingDelta`, and
`ThinkingEnd` events. Completed messages store thinking blocks as
`AssistantContent::Thinking`.

## Stop Reasons

`StopReason` variants:

- `Stop`
- `Length`
- `ToolUse`
- `Error`
- `Aborted`

## Error Handling

Provider errors are surfaced as `AssistantMessageEvent::Error` while streaming
and as `Error` values from the `complete_*` helpers.

### Aborting Requests

Use the cancellation token in `StreamOptions` to abort in-flight requests.

```rust
use ai::{SimpleStreamOptions, StreamOptions};
use tokio_util::sync::CancellationToken;

let token = CancellationToken::new();
let options = SimpleStreamOptions {
    stream: StreamOptions {
        cancellation_token: Some(token.clone()),
        ..StreamOptions::default()
    },
    ..SimpleStreamOptions::default()
};
token.cancel();
```

### Continuing After Abort

Abort produces an assistant message with `StopReason::Aborted`. The transform
layer drops aborted assistant turns before follow-up messages so conversations
can continue cleanly.

### Debugging Provider Payloads

Use `StreamOptions::on_payload` and `StreamOptions::on_response` hooks to
inspect or override provider payloads and observe raw provider responses.

## APIs, Models, and Providers

The stream registry is API-based. A model has an `api` field that selects the
transport shape and a `provider` field that identifies the model provider.

### Faux Provider for Tests

The faux provider is a queued in-process provider for tests. It is useful for
agent loop tests and deterministic stream behavior.

```rust
use ai::{faux_assistant_message, register_faux_provider};

let registration = register_faux_provider(None);
registration.set_responses([faux_assistant_message("hello", None)]);
let model = registration.get_model();
registration.unregister();
```

### Providers and Models

Built-in model metadata is loaded from the generated upstream model catalog and
filtered to the active provider scope: `openai`, `anthropic`, and
`github-copilot`. Built-in models use only the active stream APIs:
`openai-completions`, `openai-responses`, and `anthropic-messages`.

Cloudflare, Bedrock, Google, Mistral, Azure OpenAI Responses, OpenAI Codex
Responses, and other broad provider-specific APIs are not part of the active
built-in provider surface in this port. PRs to add support for additional
providers are welcome.

### Querying Providers and Models

```rust
use ai::{get_model, get_models, get_providers};

let providers = get_providers();
let openai_models = get_models("openai");
let model = get_model("anthropic", "claude-sonnet-4-5");
```

### Custom Models

```rust
use ai::{register_model, Model, ModelCost, ModelInput};

let model = Model {
    id: "local-model".to_string(),
    name: "Local Model".to_string(),
    api: "openai-completions".to_string(),
    provider: "local".to_string(),
    base_url: "http://localhost:11434/v1".to_string(),
    input: vec![ModelInput::Text],
    cost: ModelCost::default(),
    ..Default::default()
};

register_model("local", model);
```

### OpenAI Compatibility Settings

OpenAI-compatible providers can require small payload differences. The Rust port
keeps the compatibility metadata on `ModelCompat` and resolves it through
`get_openai_completions_compat` and `get_openai_responses_compat`.

### Thread Safety

The model, API, and OAuth registries are global registries backed by
`OnceLock` plus `RwLock`. Lookup functions such as `get_model` take a read lock
and return cloned values, and registration functions such as `register_model`
take a write lock. They are safe to call from multiple threads, but registration
is global mutable state, so applications should register custom models during
startup rather than concurrently with request dispatch.

### Type Safety

Rust types replace the TypeScript type-level provider/model inference. Public
types are serializable with `serde` where they represent portable context or
message state.

## Cross-Provider Handoffs

Messages use a provider-neutral shape so a conversation can move between
OpenAI, Anthropic, and GitHub Copilot-compatible models.

### How It Works

Before a request is sent, the provider converts the shared `Context` into the
target API payload. During conversion it normalizes tool call IDs, thinking
blocks, image content, cache metadata, and provider-specific signatures.

### Example: Multi-Provider Conversation

```rust
use ai::{complete_simple, get_model, Context, Message, SimpleStreamOptions};

let mut context = Context {
    messages: vec![Message::user_text("Plan a migration.")],
    ..Default::default()
};

let openai = get_model("openai", "gpt-4o-mini").expect("openai model");
let first = complete_simple(openai, context.clone(), Some(SimpleStreamOptions::default())).await?;
context.messages.push(Message::Assistant(first));

let anthropic = get_model("anthropic", "claude-sonnet-4-5").expect("anthropic model");
let second = complete_simple(anthropic, context, Some(SimpleStreamOptions::default())).await?;
```

### Provider Compatibility

Provider compatibility is handled at conversion time. Unsupported content is
downgraded or omitted where the target API cannot accept it.

## Context Serialization

`Context`, `Message`, assistant content, tool calls, and tool results implement
`Serialize` and `Deserialize`, so context can be persisted or handed to another
process.

```rust
use ai::Context;

let json = serde_json::to_string(&context)?;
let restored: Context = serde_json::from_str(&json)?;
```

## Browser Usage

The upstream TypeScript package documents browser bundling. This Rust crate is
server/native focused and does not provide browser-specific packaging.

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
use ai::{find_env_keys, get_env_api_key};

let key = get_env_api_key("openai");
let keys = find_env_keys("openai");
```

## OAuth Providers

The OAuth registry includes:

- `anthropic`
- `github-copilot`

### CLI Login

This crate exposes login primitives. It does not include the TypeScript CLI
harness.

### Programmatic OAuth

```rust
use ai::{get_oauth_provider, OAuthCredentials};

let provider = get_oauth_provider("github-copilot").expect("provider");
```

### Login Flow Example

Use `login_anthropic` or `login_github_copilot` with `OAuthLoginCallbacks` to
drive the login UI from your application.

### Using OAuth Tokens

Use `get_oauth_api_key` or the provider's `get_api_key` method to turn stored
credentials into the API key used by stream options.

### Provider Notes

GitHub Copilot support includes OAuth helpers and dynamic request headers.
Anthropic OAuth follows the Claude Pro/Max OAuth flow.

## Development

```bash
cargo fmt --all --check
cargo check -p ai --all-targets
cargo clippy -p ai --all-targets -- -D warnings
cargo test -p ai
cargo test --workspace
```

Useful narrower test commands:

```bash
cargo test -p ai --lib       # unit tests
cargo test -p ai --doc       # doc tests
cargo test -p ai --tests     # integration tests, if present
```

This crate currently keeps its Rust test coverage in module-level unit tests
under `src`. The prior Rust-only integration test files under `crates/ai/tests`
were removed because they were not a 1:1 port of the upstream `pi` test layout.

### Adding a New Provider

Provider additions should generally include:

1. Core type or compatibility metadata updates in `src/types.rs`.
2. Provider implementation under `src/providers/`.
3. API registry integration in `src/providers/register_builtins.rs`.
4. Model metadata updates.
5. Unit tests for payload conversion, streaming, errors, and provider-specific
   compatibility.
6. Documentation updates in this README.

## License

MIT
