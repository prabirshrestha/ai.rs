use std::sync::Arc;

use reqwest::header::{HeaderName, HeaderValue};

use crate::event_stream::AssistantEventStream;
use crate::types::{
    Context, Model, ModelCompat, ModelCost, ModelInput, SimpleStreamOptions, StreamOptions,
};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub language_models: bool,
    pub image_models: bool,
}

pub trait Provider: dyn_clone::DynClone + Send + Sync + 'static {
    fn id(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities;

    fn model(&self, id: &str) -> ModelBuilder {
        ModelBuilder::unsupported(self.id(), id)
    }
}

dyn_clone::clone_trait_object!(Provider);

pub trait LanguageModelApi: dyn_clone::DynClone + Send + Sync + 'static {
    fn id(&self) -> &str;

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantEventStream>;

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantEventStream>;
}

dyn_clone::clone_trait_object!(LanguageModelApi);

#[derive(Clone)]
pub struct ModelBuilder {
    model: Model,
}

impl ModelBuilder {
    pub fn unsupported(provider_id: &str, id: &str) -> Self {
        Self {
            model: Model {
                id: id.to_string(),
                provider: provider_id.to_string(),
                ..Model::default()
            },
        }
    }

    pub fn new(provider_id: &str, id: &str, api: Arc<dyn LanguageModelApi>) -> Self {
        Self {
            model: Model {
                id: id.to_string(),
                name: id.to_string(),
                api: api.id().to_string(),
                provider: provider_id.to_string(),
                language_api: Some(api),
                ..Model::default()
            },
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.model.name = name.into();
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.model.base_url = base_url.into();
        self
    }

    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.model.reasoning = reasoning;
        self
    }

    pub fn input(mut self, input: impl Into<Vec<ModelInput>>) -> Self {
        self.model.input = input.into();
        self
    }

    pub fn cost(mut self, cost: ModelCost) -> Self {
        self.model.cost = cost;
        self
    }

    pub fn context_window(mut self, context_window: u32) -> Self {
        self.model.context_window = context_window;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.model.max_tokens = max_tokens;
        self
    }

    pub fn compat(mut self, compat: ModelCompat) -> Self {
        self.model.compat = compat;
        self
    }

    pub fn headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.model.headers.extend(headers);
        self
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let value = value.into();
        let _parsed_name = name
            .parse::<HeaderName>()
            .map_err(|error| crate::Error::Provider(format!("invalid header name: {error}")))?;
        let _parsed_value = HeaderValue::from_str(&value)
            .map_err(|error| crate::Error::InvalidHeaderValue(name.clone(), error))?;
        self.model.headers.insert(name, value);
        Ok(self)
    }

    pub fn build(self) -> Result<Model> {
        if self.model.language_api.is_none() {
            return Err(Error::unsupported_capability(
                self.model.provider,
                "language models",
            ));
        }
        Ok(self.model)
    }
}

#[cfg(test)]
mod tests {
    use crate::providers::openai;

    use super::*;

    #[test]
    fn dyn_provider_can_build_and_clone_language_models() {
        let provider: Box<dyn Provider> = Box::new(
            openai::builder()
                .api_key("test-key")
                .chat_completions()
                .build()
                .expect("provider"),
        );
        let cloned = dyn_clone::clone_box(&*provider);

        let model = cloned.model("gpt-5.5").build().expect("model");

        assert_eq!(model.id(), "gpt-5.5");
        assert_eq!(model.provider_id(), "openai");
        assert_eq!(model.api_id(), "openai-completions");
    }
}
