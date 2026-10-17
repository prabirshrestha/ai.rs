//! Port of `utils/model-operations.ts`. The embedding helpers are ai.rs
//! extras, shaped like the image ones.

use indexmap::IndexMap;

use crate::types::{
    AnyModel, AssistantImages, ClassifierModel, ClassifierResult, ClassifierStopReason,
    EmbeddingModel, EmbeddingsResult, EmbeddingsStopReason, ImageModel, ImagesStopReason, Model,
    ModelType,
};
use crate::utils::models_error::{ModelsError, ModelsErrorCode};

/// The type of a model. Models without `type` are chat models.
pub fn get_model_type(model: &AnyModel) -> ModelType {
    match model {
        AnyModel::Chat(model) => model.model_type.unwrap_or(ModelType::Chat),
        AnyModel::Image(_) => ModelType::Image,
        AnyModel::Classifier(_) => ModelType::Classifier,
        AnyModel::Embedding(_) => ModelType::Embedding,
    }
}

/// Runtime-checked model type test, including legacy chat models without `type`.
pub fn is_model_type(model: &AnyModel, model_type: ModelType) -> bool {
    get_model_type(model) == model_type
}

pub fn assert_chat_model(model: &AnyModel) -> Result<&Model, ModelsError> {
    match model {
        AnyModel::Chat(chat) if is_model_type(model, ModelType::Chat) => Ok(chat),
        _ => Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!(
                "Model {}/{} is not a chat model",
                model.provider(),
                model.id()
            ),
        )),
    }
}

pub fn assert_image_model(model: &AnyModel) -> Result<&ImageModel, ModelsError> {
    match model {
        AnyModel::Image(image) => Ok(image),
        _ => Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!(
                "Model {}/{} is not an image model",
                model.provider(),
                model.id()
            ),
        )),
    }
}

/// `imageErrorResult()`: an error (or aborted) `AssistantImages` for `model`.
pub fn image_error_result(
    model: &ImageModel,
    error: impl std::fmt::Display,
    aborted: bool,
) -> AssistantImages {
    AssistantImages {
        stop_reason: if aborted {
            ImagesStopReason::Aborted
        } else {
            ImagesStopReason::Error
        },
        error_message: Some(error.to_string()),
        ..AssistantImages::empty_for(model)
    }
}

pub fn assert_classifier_model(model: &AnyModel) -> Result<&ClassifierModel, ModelsError> {
    match model {
        AnyModel::Classifier(classifier) => Ok(classifier),
        _ => Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!(
                "Model {}/{} is not a classifier model",
                model.provider(),
                model.id()
            ),
        )),
    }
}

/// `classifierErrorResult()`: an error (or aborted) `ClassifierResult` for `model`.
pub fn classifier_error_result(
    model: &ClassifierModel,
    error: impl std::fmt::Display,
    aborted: bool,
) -> ClassifierResult {
    ClassifierResult {
        answers: IndexMap::new(),
        stop_reason: if aborted {
            ClassifierStopReason::Aborted
        } else {
            ClassifierStopReason::Error
        },
        error_message: Some(error.to_string()),
        ..ClassifierResult::empty_for(model)
    }
}

pub fn assert_embedding_model(model: &AnyModel) -> Result<&EmbeddingModel, ModelsError> {
    match model {
        AnyModel::Embedding(embedding) => Ok(embedding),
        _ => Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!(
                "Model {}/{} is not an embedding model",
                model.provider(),
                model.id()
            ),
        )),
    }
}

/// An error (or aborted) [`EmbeddingsResult`] for `model`, like
/// [`image_error_result`].
pub fn embeddings_error_result(
    model: &EmbeddingModel,
    error: impl std::fmt::Display,
    aborted: bool,
) -> EmbeddingsResult {
    EmbeddingsResult {
        stop_reason: if aborted {
            EmbeddingsStopReason::Aborted
        } else {
            EmbeddingsStopReason::Error
        },
        error_message: Some(error.to_string()),
        ..EmbeddingsResult::empty_for(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrows_legacy_chat_models_and_image_models() {
        let chat = AnyModel::Chat(Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            ..Default::default()
        });
        assert!(is_model_type(&chat, ModelType::Chat));
        assert!(assert_chat_model(&chat).is_ok());
        assert_eq!(
            assert_image_model(&chat).unwrap_err().to_string(),
            "Model p/m is not an image model"
        );
        let image = AnyModel::Image(ImageModel::default());
        assert_eq!(get_model_type(&image), ModelType::Image);
        assert!(assert_chat_model(&image).is_err());
        let embedding = AnyModel::Embedding(EmbeddingModel::default());
        assert_eq!(get_model_type(&embedding), ModelType::Embedding);
        assert!(assert_embedding_model(&embedding).is_ok());
        assert!(assert_embedding_model(&image).is_err());
        assert!(assert_image_model(&embedding).is_err());
        let classifier = AnyModel::Classifier(ClassifierModel {
            id: "c".to_string(),
            provider: "p".to_string(),
            ..Default::default()
        });
        assert_eq!(get_model_type(&classifier), ModelType::Classifier);
        assert!(assert_classifier_model(&classifier).is_ok());
        assert_eq!(
            assert_classifier_model(&chat).unwrap_err().to_string(),
            "Model p/m is not a classifier model"
        );
    }
}
