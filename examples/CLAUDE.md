# CLAUDE.md - AI Crate Usage Guide

This file provides guidance for using the `ai` crate in your Rust projects. The examples in this directory demonstrate various usage patterns and capabilities.

## Quick Start

Add the `ai` crate to your `Cargo.toml`:

```toml
[dependencies]
ai = { path = "path/to/ai/crate" }
# or from crates.io (when published)
# ai = "0.2.14"
tokio = { version = "1.43.0", features = ["full"] }
futures = "0.3.31"  # Only needed for streaming
tokio-util = "0.7.13"  # Only needed for cancellation
```

## Basic Usage

### Chat Completions

```rust
use ai::{
    chat_completions::{ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Create client (supports OpenAI, Azure OpenAI, Ollama)
    let client = ai::clients::openai::Client::from_env()?;
    
    let request = ChatCompletionRequestBuilder::default()
        .model("gpt-4")
        .messages(vec![
            ChatCompletionMessage::System("You are a helpful assistant".into()),
            ChatCompletionMessage::User("Tell me a joke.".into()),
        ])
        .build()?;

    let response = client.chat_completions(&request).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    
    Ok(())
}
```

### Streaming Chat Completions

```rust
use futures::StreamExt;

let mut stream = client.stream_chat_completions(&request).await?;
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    if !chunk.choices.is_empty() {
        if let Some(content) = &chunk.choices[0].delta.content {
            print!("{}", content);
        }
    }
}
```

### Streaming with Cancellation

```rust
use tokio_util::sync::CancellationToken;

let cancellation_token = CancellationToken::new();
let mut stream = client.stream_chat_completions_with_cancellation_token(&request, cancellation_token.clone()).await?;

// Cancel from another task
tokio::spawn(async move {
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    cancellation_token.cancel();
});

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    // Process chunk...
}
```

## Embeddings

```rust
use ai::embeddings::{EmbeddingsRequestBuilder, Embeddings};

let client = ai::clients::openai::Client::from_env()?;

let request = EmbeddingsRequestBuilder::default()
    .model("text-embedding-3-small")
    .input(vec!["Hello world".to_string()])
    .build()?;

// Get standard float embeddings
let response = client.create_embeddings(&request).await?;
println!("Embedding dimensions: {}", response.data[0].embedding.len());

// Get base64 encoded embeddings
let base64_response = client.create_base64_embeddings(&request).await?;
println!("Base64 embedding: {}", base64_response.data[0].embedding);
```

## Client Configuration

### OpenAI Client
```rust
// From environment variable OPENAI_API_KEY
let client = ai::clients::openai::Client::from_env()?;

// With explicit API key
let client = ai::clients::openai::Client::new("your-api-key")?;

// Custom base URL (for Ollama, etc.)
let client = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;
```

### Azure OpenAI Client
```rust
// From environment variables
let client = ai::clients::azure_openai::Client::from_env()?;

// With builder pattern
let client = ai::clients::azure_openai::ClientBuilder::default()
    .auth(ai::clients::azure_openai::Auth::BearerToken("token".into()))
    .api_version("2024-02-15-preview")
    .base_url("https://resourcename.openai.azure.com")
    .build()?;

// Pass deployment_id as model in ChatCompletionRequest
```

### Ollama Client

Recommend using OpenAI client instead of the dedicated Ollama client for maximum compatibility:

```rust
// Using OpenAI client (recommended)
let client = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;

// Dedicated Ollama client (requires ollama_client feature, not recommended)
let client = ai::clients::ollama::Client::new()?;
let client = ai::clients::ollama::Client::from_url("http://localhost:11434")?;
```

### Gemini API via OpenAI
```rust
let client = ai::clients::openai::ClientBuilder::default()
    .http_client(
        reqwest::Client::builder()
            .http1_title_case_headers()
            .build()?,
    )
    .api_key("gemini_api_key".into())
    .base_url("https://generativelanguage.googleapis.com/v1beta/openai".into())
    .build()?;
```

### Dynamic Client Selection
```rust
use ai::clients::Client;

let client: Box<dyn Client> = match provider {
    "openai" => Box::new(ai::clients::openai::Client::from_env()?),
    "azure" => Box::new(ai::clients::azure_openai::Client::from_env()?),
    _ => return Err("Unknown provider".into()),
};
```

## Graph API

The graph module provides a workflow execution framework for complex AI workflows, similar to LangGraph:

```rust
use ai::graph::{Graph, START, END};
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct State {
    message: String,
    count: i32,
}

let graph = Graph::new()
    .add_node("process", |mut state: State| async move {
        state.message = format!("Processed: {}", state.message);
        Ok(state)
    })
    .add_node("validate", |state: State| async move {
        println!("Validating: {}", state.message);
        Ok(state)
    })
    .add_edge(START, "process")
    .add_conditional_edges(
        "process",
        |state: State| async move {
            if state.count > 5 {
                "end".to_string()
            } else {
                "validate".to_string()
            }
        },
        {
            let mut mapping = HashMap::new();
            mapping.insert("validate", "validate");
            mapping.insert("end", END);
            mapping
        },
    )
    .add_edge("validate", END);

let compiled_graph = graph.compile()?;
let result = compiled_graph.execute(initial_state).await?;
```

### Graph Visualization
```rust
// Generate Mermaid diagram
println!("{}", compiled_graph.draw_mermaid());
```

## Features

Enable/disable clients using Cargo features:

```toml
[dependencies]
ai = { 
    path = "path/to/ai/crate",
    default-features = false,
    features = ["openai_client", "rustls_tls"]
}
```

Available features:
- `openai_client` (default) - OpenAI API support
- `azure_openai_client` (default) - Azure OpenAI support
- `ollama_client` - Ollama support
- `rustls_tls` (default) - Rustls TLS backend
- `native_tls` - Native TLS backend

## Error Handling

```rust
use ai::{Error, Result};

match client.chat_completions(&request).await {
    Ok(response) => println!("Success: {}", response.choices[0].message.content.as_ref().unwrap()),
    Err(Error::HttpError(status)) => eprintln!("HTTP error: {}", status),
    Err(Error::JsonError(e)) => eprintln!("JSON error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Message Types

The crate uses an enum-based message system:

```rust
use ai::chat_completions::ChatCompletionMessage;

let messages = vec![
    ChatCompletionMessage::System("You are helpful".into()),
    ChatCompletionMessage::User("Hello".into()),
    ChatCompletionMessage::Assistant("Hi there!".into()),
];

// Convenience conversion from tuples (not recommended - use explicit types)
let message: ChatCompletionMessage = ("user", "Hello").into();
```

## Tool Calling

```rust
use ai::chat_completions::{
    ChatCompletionTool, ChatCompletionToolFunctionDefinitionBuilder,
    ChatCompletionMessage, FinishReason
};
use serde_json::json;

// Define tools
let tools = vec![
    ChatCompletionTool::Function(
        ChatCompletionToolFunctionDefinitionBuilder::default()
            .name("search_web")
            .description("Search the web for information")
            .parameters(json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }))
            .build()?
    )
];

let request = ChatCompletionRequestBuilder::default()
    .model("gpt-4")
    .messages(vec![
        ChatCompletionMessage::User("Search for information about Rust programming".into())
    ])
    .tools(tools)
    .build()?;

let response = client.chat_completions(&request).await?;

// Check if the model wants to call a function
if let Some(choice) = response.choices.first() {
    if choice.finish_reason == Some(FinishReason::ToolCalls) {
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                match tool_call {
                    ai::chat_completions::ChatCompletionMessageToolCall::Function { function } => {
                        match function.name.as_str() {
                            "search_web" => {
                                // Parse function arguments
                                let args: serde_json::Value = serde_json::from_str(&function.arguments)?;
                                let query = args["query"].as_str().unwrap_or("");
                                
                                // Execute your function logic here
                                let search_result = format!("Search results for: {}", query);
                                println!("Function called: {} with query: {}", function.name, query);
                                
                                // Add function result back to conversation
                                // (You would typically continue the conversation with the result)
                            }
                            _ => println!("Unknown function: {}", function.name),
                        }
                    }
                }
            }
        }
    }
}
```

## Environment Variables

Set these environment variables for authentication:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```