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

### Streaming

```rust
use futures::StreamExt;

use ai::{get_model, stream_simple, AssistantMessageEvent, Context, Message, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let model = get_model("openai", "gpt-4o-mini").expect("model");
    let context = Context {
        messages: vec![Message::user_text("Write a haiku about Rust.")],
        ..Default::default()
    };

    let mut events = stream_simple(model, context, None)?;

    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::TextDelta { delta, .. } = event {
            print!("{delta}");
        }
    }

    let message = events.result().await?;
    println!("\n\nused {} output tokens", message.usage.output);

    Ok(())
}
```

### Agent Loop

```rust
use futures::StreamExt;

use ai::{
    agent_loop, get_model, AgentContext, AgentEvent, AgentLoopConfig,
    AssistantMessageEvent, Message, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = get_model("anthropic", "claude-sonnet-4-5").expect("model");
    let context = AgentContext {
        system_prompt: "You are a concise coding assistant.".to_string(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

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
