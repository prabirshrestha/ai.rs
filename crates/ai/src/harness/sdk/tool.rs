//! Mirrors of `@ai-sdk/provider-utils` `Schema` / `FlexibleSchema` and `Tool`.
//!
//! [rust] A `FlexibleSchema` (Zod, Standard Schema or `jsonSchema()`) becomes
//! [`Schema`]: the JSON Schema plus an optional validator, which is what the
//! AI SDK reduces every flexible schema to (`asSchema`).

use std::fmt;
use std::sync::Arc;

use futures::StreamExt as _;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sandbox::SandboxSession;
use crate::harness::sdk::messages::{ModelMessage, ProviderOptions, ToolResultOutput};

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

/// `ToolExecutionOptions<CONTEXT>`.
#[derive(Clone)]
pub struct ToolExecutionOptions {
    pub tool_call_id: String,
    pub messages: Vec<ModelMessage>,
    pub abort_signal: Option<CancellationToken>,
    /// `context: CONTEXT` (the validated tool context).
    pub context: JsonValue,
    pub experimental_sandbox: Option<Arc<dyn SandboxSession>>,
}

/// What `execute` returns: `PromiseLike<OUTPUT> | AsyncIterable<OUTPUT>`.
pub enum ToolExecuteOutput {
    /// A single output.
    Single(BoxFuture<'static, HarnessResult<JsonValue>>),
    /// Preliminary outputs; the last one is final.
    Stream(BoxStream<'static, HarnessResult<JsonValue>>),
}

/// `ToolExecuteFunction`.
#[derive(Clone)]
pub struct ToolExecuteFn(
    pub Arc<dyn Fn(JsonValue, ToolExecutionOptions) -> ToolExecuteOutput + Send + Sync>,
);

impl fmt::Debug for ToolExecuteFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ToolExecuteFn")
    }
}

/// `toModelOutput` options.
#[derive(Debug, Clone)]
pub struct ToModelOutputOptions {
    pub tool_call_id: String,
    pub input: JsonValue,
    pub output: JsonValue,
}

/// `toModelOutput`.
#[derive(Clone)]
pub struct ToModelOutputFn(
    pub  Arc<
        dyn Fn(ToModelOutputOptions) -> BoxFuture<'static, HarnessResult<ToolResultOutput>>
            + Send
            + Sync,
    >,
);

impl fmt::Debug for ToModelOutputFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ToModelOutputFn")
    }
}

/// `ToolSet`: tools keyed by name, in declaration order.
pub type ToolSet = indexmap::IndexMap<String, Tool>;

/// The AI SDK `Tool`.
///
/// [rust] The harness agent never calls `needsApproval` or the `onInput*`
/// hooks (tool approval is the agent's `toolApproval` setting), so those
/// function fields are not mirrored.
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
    /// `execute?`.
    pub execute: Option<ToolExecuteFn>,
    /// `toModelOutput?`.
    pub to_model_output: Option<ToModelOutputFn>,
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
            execute: None,
            to_model_output: None,
        }
    }

    /// Set `execute` to an async function.
    pub fn with_execute<F, Fut>(mut self, execute: F) -> Self
    where
        F: Fn(JsonValue, ToolExecutionOptions) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = HarnessResult<JsonValue>> + Send + 'static,
    {
        self.execute = Some(ToolExecuteFn(Arc::new(move |input, options| {
            ToolExecuteOutput::Single(Box::pin(execute(input, options)))
        })));
        self
    }

    /// Set `execute` to an async generator (preliminary outputs).
    pub fn with_streaming_execute<F>(mut self, execute: F) -> Self
    where
        F: Fn(JsonValue, ToolExecutionOptions) -> BoxStream<'static, HarnessResult<JsonValue>>
            + Send
            + Sync
            + 'static,
    {
        self.execute = Some(ToolExecuteFn(Arc::new(move |input, options| {
            ToolExecuteOutput::Stream(execute(input, options))
        })));
        self
    }

    /// Set `toModelOutput`.
    pub fn with_to_model_output<F, Fut>(mut self, to_model_output: F) -> Self
    where
        F: Fn(ToModelOutputOptions) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = HarnessResult<ToolResultOutput>> + Send + 'static,
    {
        self.to_model_output = Some(ToModelOutputFn(Arc::new(move |options| {
            Box::pin(to_model_output(options))
        })));
        self
    }

    pub fn with_context_schema(mut self, schema: Schema) -> Self {
        self.context_schema = Some(schema);
        self
    }

    pub fn with_type(mut self, r#type: ToolType) -> Self {
        self.r#type = r#type;
        self
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

/// `isExecutableTool(tool)`.
pub fn is_executable_tool(tool: Option<&Tool>) -> bool {
    tool.is_some_and(|t| t.execute.is_some())
}

/// One item of `executeTool`.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecuteToolPart {
    Preliminary(JsonValue),
    Final(JsonValue),
}

/// `executeTool({ tool, input, options })`.
///
/// Panics if the tool has no `execute` (TS requires an `ExecutableTool`).
pub fn execute_tool(
    tool: &Tool,
    input: JsonValue,
    options: ToolExecutionOptions,
) -> BoxStream<'static, HarnessResult<ExecuteToolPart>> {
    let execute = tool
        .execute
        .clone()
        .expect("executeTool requires an executable tool");
    match (execute.0)(input, options) {
        ToolExecuteOutput::Single(fut) => {
            futures::stream::once(async move { fut.await.map(ExecuteToolPart::Final) }).boxed()
        }
        ToolExecuteOutput::Stream(stream) => {
            let stream = async_stream::stream! {
                let mut stream = stream;
                let mut last = JsonValue::Null;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(output) => {
                            last = output.clone();
                            yield Ok(ExecuteToolPart::Preliminary(output));
                        }
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    }
                }
                yield Ok(ExecuteToolPart::Final(last));
            };
            stream.boxed()
        }
    }
}

/// `errorMode` of `createToolModelOutput`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolModelOutputErrorMode {
    None,
    Text,
    Json,
}

/// `createToolModelOutput({ toolCallId, input, output, tool, errorMode })`.
pub async fn create_tool_model_output(
    tool_call_id: &str,
    input: &JsonValue,
    output: &JsonValue,
    tool: Option<&Tool>,
    error_mode: ToolModelOutputErrorMode,
) -> HarnessResult<ToolResultOutput> {
    match error_mode {
        ToolModelOutputErrorMode::Text => {
            return Ok(ToolResultOutput::ErrorText {
                value: crate::harness::sdk::error_value::ErrorValue::Value(output.clone())
                    .get_error_message(),
                provider_options: None,
            });
        }
        ToolModelOutputErrorMode::Json => {
            return Ok(ToolResultOutput::ErrorJson {
                value: output.clone(),
                provider_options: None,
            });
        }
        ToolModelOutputErrorMode::None => {}
    }
    if let Some(to_model_output) = tool.and_then(|t| t.to_model_output.as_ref()) {
        return (to_model_output.0)(ToModelOutputOptions {
            tool_call_id: tool_call_id.to_string(),
            input: input.clone(),
            output: output.clone(),
        })
        .await;
    }
    Ok(match output {
        JsonValue::String(s) => ToolResultOutput::Text {
            value: s.clone(),
            provider_options: None,
        },
        other => ToolResultOutput::Json {
            value: other.clone(),
            provider_options: None,
        },
    })
}

/// `createToolModelOutput({ output: error, errorMode })` for a tool error
/// (`errorMode` `'text'` or `'json'`).
pub fn create_tool_error_model_output(
    error: &crate::harness::sdk::error_value::ErrorValue,
    error_mode: ToolModelOutputErrorMode,
) -> ToolResultOutput {
    match error_mode {
        ToolModelOutputErrorMode::Json => ToolResultOutput::ErrorJson {
            value: error.to_js_json(),
            provider_options: None,
        },
        _ => ToolResultOutput::ErrorText {
            value: error.get_error_message(),
            provider_options: None,
        },
    }
}
