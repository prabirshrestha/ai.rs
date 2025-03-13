use ai::chat_completions::{ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> ai::Result<()> {
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;
    let mut conversation_history = vec![ChatCompletionMessage::System(
        "You are a helpful assistant. Be consise and clear in your responses.".into(),
    )];

    println!("Chat initialized. Type 'exit' to quit.");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;

        let user_input = user_input.trim();
        if user_input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        conversation_history.push(ChatCompletionMessage::User(user_input.into()));

        let request = ChatCompletionRequestBuilder::default()
            .model("gemma3")
            .messages(conversation_history.clone())
            .build()?;

        print!("\nAssistant: ");
        io::stdout().flush()?;

        let mut response_content = String::new();
        let mut stream = openai.stream_chat_completions(&request).await?;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if let Some(content) = &chunk.choices[0].delta.content {
                print!("{}", content);
                io::stdout().flush()?;
                response_content.push_str(content);
            }
        }
        println!("\n");

        conversation_history.push(ChatCompletionMessage::Assistant(response_content.into()));

        // Keep conversation history within reasonable limits
        if conversation_history.len() > 20 {
            conversation_history.drain(1..3); // Remove oldest Q&A pair, keep system prompt
        }
    }

    Ok(())
}
