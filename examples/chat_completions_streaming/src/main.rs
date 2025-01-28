use std::io::Write;

use ai::chat_completions::{ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder};
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai::Result<()> {
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;

    let request = ChatCompletionRequestBuilder::default()
        .model("llama3.2")
        .messages(vec![ChatCompletionMessage::User(
            "Write a paragraph about LLM.".into(),
        )])
        .build()?;

    let mut stream = openai.stream_chat_completions(&request).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(content) = &chunk.choices[0].delta.content {
            print!("{}", content);
            std::io::stdout().flush()?;
        }
    }

    Ok(())
}
