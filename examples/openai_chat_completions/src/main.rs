use ai::{
    chat_completions::{ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;
    // let openai = ai::clients::openai::Client::from_env()?;
    // let openai = ai::clients::openai::Client::new("api_key")?;

    let request = ChatCompletionRequestBuilder::default()
        .model("gemma3")
        .messages(vec![
            ChatCompletionMessage::System("You are a helpful assistant".into()),
            ChatCompletionMessage::User("Tell me a joke.".into()),
        ])
        .build()?;

    let response = openai.chat_completions(&request).await?;

    println!("{}", &response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
