# API And Streaming Plan

Goal: keep pi's `model, context, options` concept while making the Rust API
native. Provider handles create executable models; provider HTTP streaming
should use Rust `Stream`s instead of background `tokio::spawn + mpsc`.

## API Shape

```rust
use ai::{complete_simple, providers::openai, Context, Message, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::from_env()?;
    let model = openai.model("gpt-5.5").build()?;
    let context = Context::builder()
        .system_prompt("You are concise.")
        .message(Message::user_text("What is Rust ownership?"))
        .build();
    let message = complete_simple(model, context, None).await?;
    println!("{}", message.text());
    Ok(())
}
```

Provider builders stay explicit:
`openai::builder().api_key(...).base_url(...).chat_completions().build()?`.
OpenAI supports `.responses()` and `.chat_completions()`. `from_env()` defaults
to Responses. Tools support struct literals and `Tool::builder(...)`.

## Stream Direction

Do not preserve `stream.result().await`. Main already returns a normal fallible
boxed stream, and that is the better Rust shape.

```rust
pub type AssistantEventStream =
    futures::stream::BoxStream<'static, Result<AssistantMessageEvent>>;

pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantEventStream>;
```

Use `complete_simple(model, context, options).await` for the final message.
Streaming callers consume `Result<AssistantMessageEvent>` items directly.

Error timing: setup failures before a stream exists return `Err(Error)` from
`stream_simple`; runtime failures after a stream exists yield `Err(Error)` from
the stream only for transport/decoder failures that cannot be represented as a
provider message. Pi parity requires provider-declared failures and cancellation
to end with `AssistantMessageEvent::Error` carrying the final `AssistantMessage`
with `stopReason` `error` or `aborted`. `complete_simple` drains until `Done`,
`Error`, or a transport `Err`.

## SSE

Do not copy main's line parser. Reuse `crates/ai/src/utils/sse.rs`.
`reqwest` gives `Response::bytes_stream()` but does not parse SSE.

Provider streams should be poll-driven: send the request inside
`async_stream::try_stream!`, pass the response to `sse::events(...)`, yield
`AssistantMessageEvent`s, and finish with `Done` or `Error`.

Dropping the stream should drop the response body and stop work.

## SSE CRLF Framing Safety

Audit the SSE parser for CRLF framing confusion, not HTTP header injection.
The current parser splits on either `\r` or `\n`; if `\r\n` is split across
network chunks, the next `\n` can look like an empty line and flush an event
too early.

Audit and test:

- Split `\r\n` across chunks must behave the same as unsplit `\r\n`.
- Multi-line `data:` frames must not flush until the real blank line.
- Raw CR/LF inside the SSE byte stream is framing, not JSON content.
- Escaped JSON strings like `"a\\r\\nb"` must survive unchanged.
- `event:` and `data:` injection attempts via raw CR/LF should become separate
  SSE fields, never be folded into one JSON payload.
- Add bounded behavior for very long unterminated lines or events.

## Migration Order

1. Replace `AssistantMessageEventStream` with `AssistantEventStream`.
2. Convert `complete_simple` to drain events and return the final assistant
   message from either `Done` or terminal provider `Error` events.
3. Convert OpenAI Chat Completions to `try_stream!` + `utils::sse::events`.
4. Convert OpenAI Responses and Anthropic the same way.
5. Keep `tokio::spawn` for real concurrency: agent event streams, OAuth callback
   server, mock servers, and explicit concurrency tests.
6. Update `crates/ai/README.md` examples after internals settle.
