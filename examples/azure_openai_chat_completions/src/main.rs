use ai::{
    chat_completions::{ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Select either BearerToken or ApiKey for authentication.
    // To get bearer token, you can use the following command:
    // az account get-access-token --resource https://cognitiveservices.azure.com
    let azure_openai = ai::clients::azure_openai::ClientBuilder::default()
        .auth(ai::clients::azure_openai::Auth::BearerToken("token".into()))
        // .auth(ai::clients::azure_openai::Auth::ApiKey(
        //     std::env::var(ai::clients::azure_openai::AZURE_OPENAI_API_KEY_ENV_VAR)
        //         .map_err(|e| Error::EnvVarError(ai::clients::azure_openai::AZURE_OPENAI_API_KEY_ENV_VAR.to_string(), e))?
        //         .into(),
        // ))
        .api_version("2024-02-15-preview")
        .base_url("https://resourcename.openai.azure.com")
        .build()?;

    let request = ChatCompletionRequestBuilder::default()
        .model("gpt-4o-mini") // This is the deployment_id in Azure OpenAI
        .messages(vec![
            ChatCompletionMessage::System("You are a helpful assistant".into()),
            ChatCompletionMessage::User("Tell me a joke.".into()),
        ])
        .build()?;

    let response = azure_openai.chat_completions(&request).await?;

    println!("{}", &response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
