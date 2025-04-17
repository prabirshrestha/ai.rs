use ai::{
    embeddings::{Embeddings, EmbeddingsRequestBuilder},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // You can use any of these client initialization methods
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;
    // let openai = ai::clients::openai::Client::from_env()?;
    // let openai = ai::clients::openai::Client::new("api_key")?;

    let request = EmbeddingsRequestBuilder::default()
        .model("nomic-embed-text")
        // .model("text-embedding-3-small")
        .input(vec!["Hello, world!".to_string()])
        .build()
        .map_err(|e| ai::Error::UnknownError(format!("Failed to build request: {}", e)))?;

    // Get standard float embeddings
    let response = openai.create_embeddings(&request).await?;

    println!("Model: {}", response.model);
    println!("Embedding dimensions: {}", response.data[0].embedding.len());
    println!("First few values: {:?}", &response.data[0].embedding[..5]);
    println!("Tokens used: {}", response.usage.total_tokens);

    // Get base64 encoded embeddings
    // let base64_response = openai.create_base64_embeddings(&request).await?;
    // println!("\nBase64 embedding: {}", base64_response.data[0].embedding);

    Ok(())
}
