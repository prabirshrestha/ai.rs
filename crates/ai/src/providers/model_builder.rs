//! Rust addition: the `provider.model("id").build()` builder kept from the
//! pre-1.0 API. Provider handles seed it with the catalog entry (or a
//! default shape for ids missing from the catalog) and bind the result to the
//! handle's `Models` collection.

use indexmap::IndexMap;
use reqwest::header::{HeaderName, HeaderValue};

use crate::types::{Model, ModelCompat, ModelCost, ModelInput, ThinkingLevelMap};
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
        name.parse::<HeaderName>()
            .map_err(|error| Error::message(format!("invalid header name: {error}")))?;
        HeaderValue::from_str(&value)
            .map_err(|error| Error::InvalidHeaderValue(name.clone(), error))?;
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
