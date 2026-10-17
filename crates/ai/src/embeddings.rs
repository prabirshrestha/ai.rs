use crate::types::{
    Embedding, EmbeddingBatch, EmbeddingInput, EmbeddingOptions, Model, ModelInput,
};
use crate::{Error, Result};

/// Embeds one input. See [`embed_many`].
///
/// # Errors
///
/// As [`embed_many`].
pub async fn embed(
    model: Model,
    input: impl Into<EmbeddingInput>,
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

/// Embeds several inputs in one request; vectors come back in input order.
///
/// # Errors
///
/// [`Error::Validation`] before any request when there is no input, an
/// input is empty or has a MIME type that does not match its kind, or the
/// model does not declare image or audio input in `Model.input`. Otherwise
/// the provider's error.
pub async fn embed_many<I, S>(
    model: Model,
    inputs: I,
    options: Option<EmbeddingOptions>,
) -> Result<EmbeddingBatch>
where
    I: IntoIterator<Item = S>,
    S: Into<EmbeddingInput>,
{
    let inputs = inputs.into_iter().map(Into::into).collect::<Vec<_>>();
    if inputs.is_empty() {
        return Err(Error::Validation(
            "embedding input must contain at least one item".to_string(),
        ));
    }
    for input in &inputs {
        input.validate()?;
        let modality = input.modality();
        if modality != ModelInput::Text && !model.input.contains(&modality) {
            return Err(Error::Validation(format!(
                "model {} does not take {} input; declare it with ModelBuilder::input",
                model.id,
                match modality {
                    ModelInput::Image => "image",
                    _ => "audio",
                }
            )));
        }
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

    #[tokio::test]
    async fn embed_many_rejects_media_the_model_does_not_declare() {
        let model = crate::providers::openai::builder()
            .api_key(Some("test-key"))
            .base_url("http://127.0.0.1:9/v1")
            .build()
            .expect("provider")
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .expect("model");

        let error = embed_many(model, [EmbeddingInput::image(vec![1u8], "image/png")], None)
            .await
            .expect_err("text-only model");

        assert!(matches!(error, Error::Validation(message) if message.contains("image")));
    }

    #[tokio::test]
    async fn embed_many_rejects_empty_and_mismatched_inputs() {
        for input in [
            EmbeddingInput::text(""),
            EmbeddingInput::image(Vec::<u8>::new(), "image/png"),
            EmbeddingInput::image(vec![1u8], "application/octet-stream"),
            EmbeddingInput::audio(vec![1u8], "image/png"),
        ] {
            let error = embed_many(Model::default(), [input.clone()], None)
                .await
                .expect_err("invalid input");
            assert!(matches!(error, Error::Validation(_)), "{input:?}");
        }
    }

    #[test]
    fn inputs_convert_from_strings_and_chat_images() {
        assert_eq!(EmbeddingInput::from("a"), EmbeddingInput::text("a"));
        assert_eq!(
            EmbeddingInput::from(String::from("a")),
            EmbeddingInput::text("a")
        );
        assert_eq!(
            EmbeddingInput::from(&String::from("a")),
            EmbeddingInput::text("a")
        );
        assert_eq!(
            EmbeddingInput::from(std::borrow::Cow::Borrowed("a")),
            EmbeddingInput::text("a")
        );
        let image = crate::ImageContent {
            data: "/wD+".to_string(),
            mime_type: "image/png".to_string(),
        };
        assert_eq!(
            EmbeddingInput::try_from(image).expect("base64"),
            EmbeddingInput::image(vec![0xffu8, 0x00, 0xfe], "image/png")
        );
        let bad = crate::ImageContent {
            data: "not base64!".to_string(),
            mime_type: "image/png".to_string(),
        };
        assert!(EmbeddingInput::try_from(bad).is_err());
    }

    #[test]
    fn debug_shows_byte_count_not_bytes() {
        let debug = format!("{:?}", EmbeddingInput::image(vec![7u8; 3], "image/png"));
        assert_eq!(debug, "Image { mime_type: \"image/png\", bytes: 3 }");
    }
}
