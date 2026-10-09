//! Rust addition: the `provider.model("id").build()` builder kept from the
//! pre-1.0 API. Provider handles seed it with the catalog entry (or a
//! default shape for ids missing from the catalog); requests for the built
//! model go through the handle's `Models` collection. [`ImageModelBuilder`]
//! does the same for image models (`provider.model("id").build_image()`).

use indexmap::IndexMap;
use reqwest::header::{HeaderName, HeaderValue};

use crate::types::{
    ImageModel, Model, ModelCompat, ModelCost, ModelInput, ModelOutput, ThinkingLevelMap,
};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct ModelBuilder {
    model: Model,
}

impl ModelBuilder {
    pub fn new(model: Model) -> Self {
        Self { model }
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

    pub fn thinking_level_map(mut self, thinking_level_map: ThinkingLevelMap) -> Self {
        self.model.thinking_level_map = Some(thinking_level_map);
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
        self.model.compat = Some(compat);
        self
    }

    pub fn headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.model
            .headers
            .get_or_insert_with(IndexMap::new)
            .extend(headers);
        self
    }

    /// Add one static header, validating its name and value.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let value = value.into();
        validate_header(&name, &value)?;
        self.model
            .headers
            .get_or_insert_with(IndexMap::new)
            .insert(name, value);
        Ok(self)
    }

    pub fn build(self) -> Result<Model> {
        Ok(self.model)
    }
}

fn validate_header(name: &str, value: &str) -> Result<()> {
    name.parse::<HeaderName>()
        .map_err(|error| Error::message(format!("invalid header name: {error}")))?;
    HeaderValue::from_str(value)
        .map_err(|error| Error::InvalidHeaderValue(name.to_string(), error))?;
    Ok(())
}

/// Builder for image models from provider handles (pre-1.0
/// `provider.model("id").build_image()`).
#[derive(Debug, Clone)]
pub struct ImageModelBuilder {
    model: ImageModel,
}

impl ImageModelBuilder {
    pub fn new(model: ImageModel) -> Self {
        Self { model }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.model.name = name.into();
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.model.base_url = base_url.into();
        self
    }

    pub fn input(mut self, input: impl Into<Vec<ModelInput>>) -> Self {
        self.model.input = input.into();
        self
    }

    /// Output modalities. Image models always produce images; add
    /// `ModelOutput::Text` when the model can also return text.
    pub fn output(mut self, output: impl Into<Vec<ModelOutput>>) -> Self {
        self.model.output = output.into();
        self
    }

    pub fn cost(mut self, cost: ModelCost) -> Self {
        self.model.cost = cost;
        self
    }

    pub fn headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.model
            .headers
            .get_or_insert_with(IndexMap::new)
            .extend(headers);
        self
    }

    /// Add one static header, validating its name and value.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let value = value.into();
        validate_header(&name, &value)?;
        self.model
            .headers
            .get_or_insert_with(IndexMap::new)
            .insert(name, value);
        Ok(self)
    }

    pub fn build_image(self) -> Result<ImageModel> {
        if !self.model.output.contains(&ModelOutput::Image) {
            return Err(Error::message(format!(
                "Image model {}/{} must output images",
                self.model.provider, self.model.id
            )));
        }
        Ok(self.model)
    }

    /// Alias of [`ImageModelBuilder::build_image`].
    pub fn build(self) -> Result<ImageModel> {
        self.build_image()
    }
}
