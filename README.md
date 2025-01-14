# ai

# Using the library

```
cargo add ai
```

# Example

## Chat Completion API (OpenAI)

```rust
use ai::chat_completions::{ChatCompletion, ChatCompletionRequestBuilder, Messages};

#[tokio::main]
async fn main() -> ai::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let openai =
        ai::clients::openai::Client::new("open_api_key")?;

    let request = &ChatCompletionRequestBuilder::default()
        .model("gpt-4o-mini".to_string())
        .messages(vec![
            Message::system("Your are a helpful assistant."),
            Message::user("Tell me a joke"),
        ])
        .build()?;

    let response = openai.complete(&request).await?;
    println!("{}", &response.choices[0].message.content);

    dbg!(&response);

    Ok(())
}
```

## Clients

### OpenAI

```sh
cargo add ai --features=openai_client
```

```rust
let openai = ai::clients::openai::Client::new("open_api_key")?;
let openai = ai::clients::openai::Client::from_url("open_api_key", "http://api.openai.com/v1")?;
let openai = ai::clients::openai::Client::from_env()?;
```

### Ollama

```sh
cargo add ai --features=ollama_client
```

```rust
let ollama = ai::clients::ollama::Client::new()?;
let ollama = ai::clients::ollama::Client::from_url("http://localhost:11434")?;
```
