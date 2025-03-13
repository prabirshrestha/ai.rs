use std::io::Write;

use ai::chat_completions::{
    ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder,
    ChatCompletionRequestStreamOptionsBuilder,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai::Result<()> {
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;

    let request = ChatCompletionRequestBuilder::default()
        .model("gemma3")
        .messages(vec![ChatCompletionMessage::User(
            "Write a paragraph about LLM.".into(),
        )])
        .stream_options(
            ChatCompletionRequestStreamOptionsBuilder::default()
                .include_usage(true)
                .build()?,
        )
        .build()?;

    let mut stream = openai.stream_chat_completions(&request).await?;

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
