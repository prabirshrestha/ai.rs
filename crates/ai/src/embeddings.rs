use crate::types::{Embedding, EmbeddingBatch, EmbeddingOptions, Model};
use crate::{Error, Result};

pub async fn embed(
    model: Model,
    input: impl Into<String>,
    options: Option<EmbeddingOptions>,
) -> Result<Embedding> {
    let batch = embed_many(model, [input.into()], options).await?;
    let embedding = batch.embeddings.into_iter().next().ok_or_else(|| {
        Error::InvalidProviderResponse("embedding response contained no data".to_string())
    })?;
    Ok(Embedding {
        embedding,
        model: batch.model,
        usage: batch.usage,
    })
}

pub async fn embed_many<I, S>(
    model: Model,
    inputs: I,
    options: Option<EmbeddingOptions>,
) -> Result<EmbeddingBatch>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let inputs = inputs.into_iter().map(Into::into).collect::<Vec<_>>();
    if inputs.is_empty() {
        return Err(Error::Validation(
            "embedding input must contain at least one string".to_string(),
        ));
    }
    let api = model
        .embedding_api()
        .ok_or_else(|| Error::unsupported_capability(model.provider.clone(), "embedding models"))?;
    api.embed_many(model, inputs, options.unwrap_or_default())
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_many_rejects_empty_input_before_provider_dispatch() {
        let error = embed_many(Model::default(), Vec::<String>::new(), None)
            .await
            .expect_err("empty input should fail");

        assert!(matches!(error, Error::Validation(_)));
    }
}
