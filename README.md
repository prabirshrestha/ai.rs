# ai

# Examples

## Ollama via OpenAI Client

```rust
use ai::chat_completions::{ChatCompletion, ChatCompletionRequestBuilder, Messages};

#[tokio::main]
async fn main() -> ai::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let openai =
        ai::clients::openai::Client::from_url("ollama_api_key", "http://localhost:11434/v1")?;

    let request = &ChatCompletionRequestBuilder::default()
        .model("llama3.2".to_string())
        .messages(vec![
            Message::system("Your are a helpful assistant."),
            Message::user("Tell me a joke".to_string()),
        ])
        .build()?;

    let response = openai.complete(&request).await?;

    dbg!(&response);

    Ok(())
}
```
