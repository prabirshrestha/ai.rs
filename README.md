# ai.rs

LLM library for Rust with streaming, tool calling, a models registry with
OAuth, image generation, embeddings, and an agent loop. It is a 1:1 port of
[`pi`](https://github.com/earendil-works/pi)'s `pi-ai` and `pi-agent-core`
1.0.2.

## Using the Library

```bash
cargo add ai
cargo add tokio --features macros,rt-multi-thread
cargo add futures serde_json
```

See [crates/ai/README.md](crates/ai/README.md) for the full API reference.

## Choosing an API

Most applications should start with `stream_simple` for streaming responses and
`complete_simple` for one-shot responses. They take `SimpleStreamOptions`,
which map one reasoning level, tool choice, cache retention, API keys,
retries and cancellation onto the selected provider. Use `stream` or
`complete` for the lower-level `StreamOptions` shape and API-specific
`provider_options`, and the `Models` registry for credential stores and
OAuth.

## Examples

Provider handles are available for OpenAI, Anthropic, GitHub Copilot, and
OpenRouter image generation. Use `providers::openai::builder()` for
OpenAI-compatible endpoints such as llama.cpp, MLX, Ollama, vLLM, and Azure
Foundry.

### Simple Coding Agent

See [examples/simple-coding-agent](examples/simple-coding-agent/README.md) for a tiny interactive coding-agent example with one `bash` tool.

### Complete

```rust,no_run
use ai::{Context, Message, Result, complete_simple, content_text, providers::openai};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder()
        .message(Message::user_text("Write a haiku about Rust."))
        .build();

    let message = complete_simple(model, context, None).await?;
    println!("{}", content_text(&message.content));
    Ok(())
}
```

### Streaming

```rust,no_run
use futures::StreamExt;

use ai::{AssistantMessageEvent, Context, Message, Result, providers::openai, stream_simple};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder()
        .message(Message::user_text("Write a haiku about Rust."))
        .build();

    let mut events = stream_simple(model, context, None)?;
    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::TextDelta { delta, .. } = event {
            print!("{delta}");
        }
    }

    Ok(())
}
```

### Embeddings

Use `embed` for one string and `embed_many` for multiple strings. Embeddings
are an ai.rs extra (not in Pi).

```rust,no_run
use ai::{Result, embed, embed_many, providers::openai};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai
        .embedding_model("text-embedding-3-small")
        .build_embedding()?;

    let one = embed(model.clone(), "hello", None).await?;
    let batch = embed_many(model, ["first", "second"], None).await?;

    println!("single: {:?}, batch: {}", one.embedding, batch.embeddings.len());
    Ok(())
}
```

### Provider Handles

```rust,no_run
use ai::{Result, providers::{anthropic, github_copilot, openai}};

fn main() -> Result<()> {
    // OpenAI Responses (OPENAI_API_KEY).
    let _openai_responses_from_env = openai::from_env()?;
    let _openai_responses_with_key = openai::builder()
        .api_key(Some("sk-..."))
        .responses()
        .build()?;

    // OpenAI Chat Completions, and OpenAI-compatible servers such as Ollama.
    let _openai_chat_with_key = openai::builder()
        .api_key(Some("sk-..."))
        .chat_completions()
        .build()?;
    let _ollama_chat = openai::builder()
        .provider_id("ollama")
        .base_url("http://localhost:11434/v1")
        .chat_completions()
        .build()?;

    // Anthropic (ANTHROPIC_API_KEY).
    let _anthropic_from_env = anthropic::from_env()?;
    let _anthropic_with_key = anthropic::builder().api_key("sk-ant-...").build()?;

    // GitHub Copilot (COPILOT_GITHUB_TOKEN, or `login_github_copilot`).
    let _copilot = github_copilot::from_env()?;
    Ok(())
}
```

### Image Generation

```rust,no_run
use ai::{ImagesContext, Result, generate_images, providers::openai};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.image_model("gpt-image-2").build_image()?;
    let context = ImagesContext::builder()
        .text("Generate a small watercolor robot reading a book.")
        .build();

    let images = generate_images(model, context, None).await?;
    println!("{} images", images.output.len());
    Ok(())
}
```

For llama.cpp, MLX, Ollama, or another OpenAI-compatible image endpoint, use
the OpenAI provider with the server's base URL:

```rust,no_run
use ai::{ImagesContext, Result, generate_images, providers::openai};

#[tokio::main]
async fn main() -> Result<()> {
    let ollama = openai::builder()
        .provider_id("ollama")
        .base_url("http://localhost:11434/v1")
        .build()?;
    let model = ollama.image_model("x/z-image-turbo").build_image()?;
    let context = ImagesContext::builder().text("Generate a robot.").build();

    let images = generate_images(model, context, None).await?;
    println!("{:?}", images.stop_reason);
    Ok(())
}
```

OpenRouter image models are available through `providers::openrouter`
(`openrouter.model("google/gemini-3-pro-image").build_image()?`).

### Agent

Use `Agent` when you want conversation state, awaited event subscribers,
abort, and steering/follow-up queues. The system prompt and tools live in
the transcript as system messages.

```rust,no_run
use ai::{
    Agent, AgentEvent, AgentOptions, AssistantMessageEvent, providers::anthropic,
    stream_simple_fn,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let anthropic = anthropic::from_env()?;
    let model = anthropic.model("claude-sonnet-4-5").build()?;
    let agent = Agent::new(
        AgentOptions::builder(model)
            .system_prompt("You are a concise coding assistant.")
            .stream_fn(stream_simple_fn())
            .build(),
    );

    let subscription = agent.subscribe(|event, signal| async move {
        if signal.is_cancelled() {
            return Ok(());
        }

        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = event
        {
            print!("{delta}");
        }

        Ok(())
    });

    agent
        .prompt_text("Explain ownership in one paragraph.", Vec::new())
        .await?;

    subscription.unsubscribe();
    Ok(())
}
```

Keep the subscription handle alive while the listener remains registered.
Dropping the handle also unsubscribes.

### Low-Level Agent Loop

```rust,no_run
use futures::StreamExt;

use ai::{
    AgentContext, AgentEvent, AgentLoopConfig, AssistantMessageEvent, Message, SystemMessage,
    agent_loop, providers::anthropic, stream_simple_fn,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let anthropic = anthropic::from_env()?;
    let model = anthropic.model("claude-sonnet-4-5").build()?;
    let context = AgentContext::builder()
        .message(SystemMessage {
            content: "You are a concise coding assistant.".into(),
            ..Default::default()
        })
        .build();

    let mut events = agent_loop(
        vec![Message::user_text("Explain ownership in one paragraph.")],
        context,
        AgentLoopConfig::new(model),
        None,
        Some(stream_simple_fn()),
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

    let _new_messages = events.result().await?;
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
