use std::io::Write;
use tokio::time::{sleep, Duration};
use tokio_util::sync::CancellationToken;

use ai::chat_completions::{
    ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder,
    ChatCompletionRequestStreamOptionsBuilder,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai::Result<()> {
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;

    let cancel_token = CancellationToken::new();

    let request = ChatCompletionRequestBuilder::default()
        .model("gemma3")
        .messages(vec![ChatCompletionMessage::User(
            "Write a paragraph about LLM.".into(),
        )])
        .cancellation_token(cancel_token.clone())
        .stream_options(
            ChatCompletionRequestStreamOptionsBuilder::default()
                .include_usage(true)
                .build()?,
        )
        .build()?;

    let mut stream = openai.stream_chat_completions(&request).await?;

    tokio::spawn({
        let cancel_token = cancel_token.clone();
        async move {
            sleep(Duration::from_millis(200)).await;
            cancel_token.cancel();
            println!("\n\nCancelled after 200ms timeout!");
        }
    });

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if !chunk.choices.is_empty() {
            if let Some(content) = &chunk.choices[0].delta.content {
                print!("{}", content);
                std::io::stdout().flush()?;
            }
        }

        if chunk.usage.is_some() {
            println!("\n\nUsage: {:#?}", chunk.usage.unwrap());
        }
    }

    Ok(())
}
