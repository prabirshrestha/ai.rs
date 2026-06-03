# ai.rs

Simple to use LLM library for Rust with streaming, tool calling, OAuth helpers,
and a lightweight agent loop, inspired by [`pi`](https://github.com/earendil-works/pi).

## Using the Library

```bash
cargo add ai
cargo add tokio --features macros,rt-multi-thread
cargo add futures
```

See [crates/ai/README.md](crates/ai/README.md) for the full API reference.

## Choosing an API

Most applications should start with `stream_simple` for streaming responses and
`complete_simple` for one-shot responses. They take `SimpleStreamOptions` and
map common settings like reasoning, cache retention, API keys, retries,
cancellation, and provider options onto the selected provider. Use `stream` or
`complete` when you need the lower-level `StreamOptions` shape or direct
provider-option forwarding.

## Examples

Provider handles are available for OpenAI, Anthropic, and GitHub Copilot. Use
`providers::openai::builder()` for OpenAI-compatible endpoints such as Ollama,
vLLM, and Azure Foundry.

### Complete

```rust
use ai::{complete_simple, providers::openai, Context, Message, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder()
        .message(Message::user_text("Write a haiku about Rust."))
        .build();

    let message = complete_simple(model, context, None).await?;
    println!("{message:?}");
    Ok(())
}
```

### Streaming

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

```rust
use ai::providers::{anthropic, openai};

let openai_responses = openai::from_env()?;

let openai_chat = openai::builder()
    .base_url("http://localhost:11434/v1")
    .chat_completions()
    .build()?;

let anthropic = anthropic::from_env()?;
```

### Agent Loop

```rust
use futures::StreamExt;

use ai::{
    agent_loop, providers::anthropic, AgentContext, AgentEvent, AgentLoopConfig,
    AssistantMessageEvent, Message, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let anthropic = anthropic::from_env()?;
    let model = anthropic.model("claude-sonnet-4-5").build()?;
    let context = AgentContext::builder()
        .system_prompt("You are a concise coding assistant.")
        .build();

    let mut events = agent_loop(
        vec![Message::user_text("Explain ownership in one paragraph.")],
        context,
        AgentLoopConfig::new(model),
        None,
        None,
    );

    while let Some(event) = events.next().await {
        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = event
        {
            print!("{delta}");
        }
    }

    Ok(())
}
```

## Development

```bash
mise run fmt
mise run check
mise run clippy
mise run test-ai
mise run test
mise run ci
mise run all
```

## License

MIT
