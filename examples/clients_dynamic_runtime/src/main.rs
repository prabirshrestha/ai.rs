use ai::{
    chat_completions::{ChatCompletionMessage, ChatCompletionRequestBuilder},
    clients::Client,
    Result,
};

async fn summarize<T: Client + ?Sized>(client: &T, model: &str, text: &str) -> Result<String> {
    // Use `<T: Client + ?Sized>` to support both dynamic or static dispatch.

    let request = ChatCompletionRequestBuilder::default()
        .model(model)
        .messages(vec![
            ChatCompletionMessage::System(
                "You are a helpful assistant that helps in summarize content provided by the user. If asked for anything else reply with I cannot do that."
                    .into(),
            ),
            ChatCompletionMessage::User(format!("Summarize the following text: {text}").into()),
        ])
        .build()?;
    let response = client.chat_completions(&request).await?;

    Ok(response.choices[0]
        .message
        .content
        .to_owned()
        .unwrap_or_default())
}

#[tokio::main]
async fn main() -> Result<()> {
    let (client, model): (Box<dyn Client>, String) =
        if let Ok(open_api_key) = std::env::var("OPEN_API_KEY") {
            let openai = ai::clients::openai::Client::new(&open_api_key)?;
            (Box::new(openai), "gpt-4o-mini".to_string())
        } else {
            let ollama = ai::clients::ollama::Client::new()?;
            (Box::new(ollama), "llama3.2".to_string())
        };

    let summary = summarize(&*client, &model, "The sky is blue because it is blue.").await?;

    println!("Summary: {summary}");

    Ok(())
}
