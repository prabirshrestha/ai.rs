//! Mirrors of `@ai-sdk/provider-utils` `Schema` / `FlexibleSchema` and `Tool`.
//!
//! [rust] A `FlexibleSchema` (Zod, Standard Schema or `jsonSchema()`) becomes
//! [`Schema`]: the JSON Schema plus an optional validator, which is what the
//! AI SDK reduces every flexible schema to (`asSchema`).

use std::fmt;
use std::sync::Arc;

use crate::harness::errors::BoxError;
use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::messages::ProviderOptions;

/// `ValidationResult<T>` with the value as JSON.
pub type ValidationResult = Result<JsonValue, BoxError>;

/// Validator attached to a [`Schema`].
pub type SchemaValidator = Arc<dyn Fn(&JsonValue) -> ValidationResult + Send + Sync>;

/// `Schema` (`jsonSchema(...)` / `asSchema(...)`).
#[derive(Clone)]
pub struct Schema {
    /// `jsonSchema: JSONSchema7`.
    pub json_schema: JsonValue,
    /// `validate?`.
    pub validate: Option<SchemaValidator>,
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Schema")
            .field("json_schema", &self.json_schema)
            .field("validate", &self.validate.is_some())
            .finish()
    }
}

/// `jsonSchema(schema, { validate })`.
pub fn json_schema(schema: JsonValue) -> Schema {
    Schema {
        json_schema: schema,
        validate: None,
    }
}

impl Schema {
    /// Attach a validator.
    pub fn with_validate(
        mut self,
        validate: impl Fn(&JsonValue) -> ValidationResult + Send + Sync + 'static,
    ) -> Self {
        self.validate = Some(Arc::new(validate));
        self
    }

    /// `safeValidateTypes` without a validator passes the value through.
    pub fn validate_value(&self, value: &JsonValue) -> ValidationResult {
        match &self.validate {
            Some(v) => v(value),
            None => Ok(value.clone()),
        }
    }
}

/// `Tool['type']` and the fields that depend on it.
#[derive(Debug, Clone, Default)]
pub enum ToolType {
    /// `type?: undefined | 'function'`.
    #[default]
    Function,
    /// `type: 'dynamic'`.
    Dynamic,
    /// `type: 'provider'`.
    Provider {
        id: String,
        args: JsonObject,
        is_provider_executed: bool,
        supports_deferred_results: Option<bool>,
    },
}

/// The AI SDK `Tool`.
///
/// Phase 1 carries the declarative fields. The function-valued fields
/// (`execute`, `needsApproval`, `onInput*`, `toModelOutput`, a function
/// `description`) are added with the agent in phase 3.
#[derive(Debug, Clone)]
pub struct Tool {
    pub r#type: ToolType,
    pub description: Option<String>,
    pub title: Option<String>,
    pub defer_loading: Option<bool>,
    pub provider_options: Option<ProviderOptions>,
    pub metadata: Option<JsonObject>,
    pub input_schema: Schema,
    pub context_schema: Option<Schema>,
    pub output_schema: Option<Schema>,
    pub strict: Option<bool>,
    pub input_examples: Option<Vec<JsonValue>>,
}

impl Tool {
    /// `tool({ inputSchema })`.
    pub fn new(input_schema: Schema) -> Self {
        Self {
            r#type: ToolType::Function,
            description: None,
            title: None,
            defer_loading: None,
            provider_options: None,
            metadata: None,
            input_schema,
            context_schema: None,
            output_schema: None,
            strict: None,
            input_examples: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_output_schema(mut self, schema: Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }
}
