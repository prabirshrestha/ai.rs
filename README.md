# ai

AI library for Rust primarily targeting OpenAI and Ollama APIs with more to come. This is work in progress.

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
    let openai =
        ai::clients::openai::Client::new("open_api_key")?;

    let request = &ChatCompletionRequestBuilder::default()
        .model("gpt-4o-mini".to_string())
        .messages(vec![
            Message::system("Your are a helpful assistant."),
            Message::user("Tell me a joke"),
        ])
        .build()?;

    let response = openai.chat_completions(&request).await?;
    println!("{}", &response.choices[0].message.content);

    dbg!(&response);

    Ok(())
}
```

## Dynamic Clients based on the runtime

Use `<T: Client + ?Sized>` to support both dynamic or static dispatch.

```rust
async fn summarize<T: Client + ?Sized>(client: &T, text: &str) -> ai::Result<String> {
    let request = &ChatCompletionRequestBuilder::default()
        .model("llama3.2".into())
        .messages(vec![
            Message::system("Your are a helpful assistant."),
            Message::user(format!("Summarize the following text: {}", text)),
        ])
        .build()?;

    let response = client.chat_completions(&request).await?;

    Ok(response.choices[0].message.content.to_owned())
}

#[tokio::main]
async fn main() -> ai::Result<()> {
    let client: Box<dyn Client> = if let Ok(openai_api_key) = std::env::var("OPENAI_API_KEY") {
        let openai = ai::clients::openai::Client::new(&openai_api_key)?;
        Box::new(openai)
    } else {
        let ollama = ai::clients::ollama::Client::new()?;
        Box::new(ollama)
    };

    let summary = summarize(&*client, "Sky is blue because it is blue.").await?;
    println!("{}", &summary);

    Ok(())
}
```

For `struct` use `Box<dyn Client>` to support dynamic dispatch.

```rust
struct Summarizer {
    client: Box<dyn Client>,
}

impl Summarizer {
    pub fn new(client: Box<dyn Client>) -> Self {
        Self { client }
    }

    pub async fn summarize(&self, text: &str) -> ai::Result<String> {
        let request = &ChatCompletionRequestBuilder::default()
            .model("llama3.2".into())
            .messages(vec![
                Message::system("Your are a helpful assistant."),
                Message::user("What is the capital of France? Return in JSON."),
            ])
            .build()?;

        let response = self.client.chat_completions(request).await?;

        Ok(response.choices[0].message.content.to_owned())
    }
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

#### Gemini API via OpenAI

```rust
let gemini = ai::clients::openai::ClientBuilder::default()
    .http_client(
        reqwest::Client::builder()
            .http1_title_case_headers()
            .build()?,
    )
    .api_key("gemini_api_key".into())
    .base_url("https://generativelanguage.googleapis.com/v1beta/openai".into())
    .build()?;
```

### Ollama

Suggest using openai client instead of ollama for maximum compatibility.

```sh
cargo add ai --features=ollama_client
```

```rust
let ollama = ai::clients::ollama::Client::new()?;
let ollama = ai::clients::ollama::Client::from_url("http://localhost:11434")?;
```

# Cargo Feature

| Feature               | Description                                   |
|-----------------------|-----------------------------------------------|
| `openai_client`       | Enable OpenAI client                          |
| `ollama_client`       | Enable Ollama client                          |
| `native_tls`          | Enable native TLS for reqwest http client     |
| `rustls_tls`          | Enable rustls TLS for reqwest http client     |

# LICENSE

MIT
