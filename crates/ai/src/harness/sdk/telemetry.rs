//! Port of the `ai` telemetry layer the harness agent uses: the
//! generate-text lifecycle events, `Telemetry` integrations,
//! `TelemetryOptions`, `registerTelemetry`, `mergeCallbacks`,
//! `createTelemetryDispatcher` and `createRestrictedTelemetryDispatcher`.
//!
//! [rust] Integrations receive events as JSON objects
//! ([`TelemetryEvent`]), which is what TS's loosely typed
//! `InferTelemetryEvent<EVENT>` amounts to and lets `filterTelemetryEvent`
//! be ported field for field. Lifecycle callbacks receive typed events.
//! `redactTelemetryCredentials` and the diagnostics tracing channels have no
//! Rust counterpart and are not ported.

use std::sync::{Arc, OnceLock};

use futures::future::BoxFuture;
use parking_lot::RwLock;
use serde::Serialize;
use serde_json::json;

use crate::harness::errors::HarnessResult;
use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::content::{
    ContentPart, Context, FinishReason, ProviderMetadata, StepResult, ToolOutput, ToolsContext,
    TypedToolCall,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::messages::ModelMessage;
use crate::harness::sdk::output::SharedOutput;
use crate::harness::sdk::tool::{ToolSet, ToolType};
use crate::harness::sdk::usage::LanguageModelUsage;

/// `VERSION` of the `ai` package this port mirrors.
pub const AI_VERSION: &str = "7.0.177";

/// `InferTelemetryEvent<EVENT>` as JSON.
pub type TelemetryEvent = JsonObject;

/// `Callback<EVENT>`.
pub type Callback<E> = Arc<dyn Fn(E) -> BoxFuture<'static, ()> + Send + Sync>;

/// Build a [`Callback`] from an async closure.
pub fn callback<E, F, Fut>(f: F) -> Callback<E>
where
    F: Fn(E) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    Arc::new(move |e| Box::pin(f(e)))
}

/// `mergeCallbacks(...callbacks)`: runs every callback concurrently and
/// waits for all of them.
pub fn merge_callbacks<E: Clone + Send + 'static>(
    callbacks: Vec<Option<Callback<E>>>,
) -> Callback<E> {
    let callbacks: Vec<Callback<E>> = callbacks.into_iter().flatten().collect();
    Arc::new(move |event: E| {
        let futures: Vec<_> = callbacks.iter().map(|c| c(event.clone())).collect();
        Box::pin(async move {
            futures::future::join_all(futures).await;
        })
    })
}

/// `notify(event, ...callbacks)` (turn telemetry).
pub async fn notify<E: Clone>(event: E, callbacks: &[Option<&Callback<E>>]) {
    let futures: Vec<_> = callbacks
        .iter()
        .flatten()
        .map(|c| c(event.clone()))
        .collect();
    futures::future::join_all(futures).await;
}

fn tools_to_json(tools: &ToolSet) -> JsonValue {
    let mut map = JsonObject::new();
    for (name, tool) in tools {
        let mut t = JsonObject::new();
        match &tool.r#type {
            ToolType::Function => {}
            ToolType::Dynamic => {
                t.insert("type".into(), "dynamic".into());
            }
            ToolType::Provider { id, args, .. } => {
                t.insert("type".into(), "provider".into());
                t.insert("id".into(), id.clone().into());
                t.insert("args".into(), JsonValue::Object(args.clone()));
            }
        }
        if let Some(d) = &tool.description {
            t.insert("description".into(), d.clone().into());
        }
        if let Some(d) = &tool.title {
            t.insert("title".into(), d.clone().into());
        }
        t.insert("inputSchema".into(), tool.input_schema.json_schema.clone());
        map.insert(name.clone(), JsonValue::Object(t));
    }
    JsonValue::Object(map)
}

fn serialize_tools<S: serde::Serializer>(tools: &ToolSet, s: S) -> Result<S::Ok, S::Error> {
    tools_to_json(tools).serialize(s)
}

fn serialize_output<S: serde::Serializer>(
    output: &Option<SharedOutput>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match output {
        Some(o) => json!({ "name": o.name() }).serialize(s),
        None => s.serialize_none(),
    }
}

/// `GenerateTextStartEvent` (the fields the harness sets).
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextStartEvent {
    pub call_id: String,
    pub operation_id: String,
    pub provider: String,
    pub model_id: String,
    #[serde(serialize_with = "serialize_tools")]
    pub tools: ToolSet,
    pub active_tools: Vec<String>,
    pub tool_order: Vec<String>,
    pub max_retries: u32,
    #[serde(
        serialize_with = "serialize_output",
        skip_serializing_if = "Option::is_none"
    )]
    pub output: Option<SharedOutput>,
    pub tools_context: ToolsContext,
    pub runtime_context: Context,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub messages: Vec<ModelMessage>,
}

/// `GenerateTextStepStartEvent`.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextStepStartEvent {
    pub call_id: String,
    pub provider: String,
    pub model_id: String,
    pub step_number: u32,
    #[serde(serialize_with = "serialize_tools")]
    pub tools: ToolSet,
    pub active_tools: Vec<String>,
    pub tool_order: Vec<String>,
    #[serde(serialize_with = "serialize_steps")]
    pub steps: Vec<StepResult>,
    #[serde(
        serialize_with = "serialize_output",
        skip_serializing_if = "Option::is_none"
    )]
    pub output: Option<SharedOutput>,
    pub runtime_context: Context,
    pub tools_context: ToolsContext,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub messages: Vec<ModelMessage>,
}

/// `LanguageModelCallStartEvent`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelCallStartEvent {
    pub call_id: String,
    pub provider: String,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub messages: Vec<ModelMessage>,
    pub tools: Option<Vec<JsonObject>>,
}

/// `LanguageModelCallEndEvent['performance']`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelCallPerformance {
    pub response_time_ms: f64,
    pub effective_output_tokens_per_second: f64,
    pub output_tokens_per_second: Option<f64>,
    pub input_tokens_per_second: Option<f64>,
    pub effective_total_tokens_per_second: f64,
    pub time_to_first_output_ms: Option<f64>,
}

/// `LanguageModelCallEndEvent`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelCallEndEvent {
    pub call_id: String,
    pub provider: String,
    pub model_id: String,
    pub finish_reason: FinishReason,
    pub usage: LanguageModelUsage,
    pub content: Vec<ContentPart>,
    pub response_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    pub performance: LanguageModelCallPerformance,
}

/// `ToolExecutionStartEvent`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionStartEvent {
    pub call_id: String,
    pub messages: Vec<ModelMessage>,
    pub tool_call: TypedToolCall,
    pub tool_context: Option<JsonValue>,
}

/// `ToolExecutionEndEvent`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionEndEvent {
    pub call_id: String,
    pub messages: Vec<ModelMessage>,
    pub tool_call: TypedToolCall,
    pub tool_context: Option<JsonValue>,
    pub tool_output: ToolOutput,
    pub tool_execution_ms: f64,
}

/// `GenerateTextEndEvent`.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextEndEvent {
    pub call_id: String,
    pub step_number: u32,
    pub model: crate::harness::sdk::content::StepModel,
    pub tools_context: ToolsContext,
    pub runtime_context: Context,
    pub content: Vec<ContentPart>,
    pub text: String,
    #[serde(serialize_with = "serialize_reasoning")]
    pub reasoning: Vec<crate::harness::sdk::messages::ReasoningPart>,
    pub reasoning_text: Option<String>,
    pub files: Vec<JsonValue>,
    pub sources: Vec<JsonValue>,
    #[serde(serialize_with = "serialize_tool_calls")]
    pub tool_calls: Vec<TypedToolCall>,
    #[serde(serialize_with = "serialize_tool_calls")]
    pub static_tool_calls: Vec<TypedToolCall>,
    #[serde(serialize_with = "serialize_tool_calls")]
    pub dynamic_tool_calls: Vec<TypedToolCall>,
    #[serde(serialize_with = "serialize_tool_results")]
    pub tool_results: Vec<crate::harness::sdk::content::TypedToolResult>,
    #[serde(serialize_with = "serialize_tool_results")]
    pub static_tool_results: Vec<crate::harness::sdk::content::TypedToolResult>,
    #[serde(serialize_with = "serialize_tool_results")]
    pub dynamic_tool_results: Vec<crate::harness::sdk::content::TypedToolResult>,
    pub finish_reason: FinishReason,
    pub raw_finish_reason: Option<String>,
    pub usage: LanguageModelUsage,
    pub total_usage: LanguageModelUsage,
    pub warnings: Vec<crate::harness::sdk::content::CallWarning>,
    pub request: crate::harness::sdk::content::LanguageModelRequestMetadata,
    pub response: crate::harness::sdk::content::LanguageModelResponseMetadata,
    pub provider_metadata: Option<ProviderMetadata>,
    pub response_messages: Vec<ModelMessage>,
    #[serde(serialize_with = "serialize_steps")]
    pub steps: Vec<StepResult>,
    #[serde(serialize_with = "serialize_step")]
    pub final_step: StepResult,
}

/// `GenerateTextAbortEvent`.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextAbortEvent {
    pub call_id: String,
    #[serde(serialize_with = "serialize_steps")]
    pub steps: Vec<StepResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<ErrorValue>,
}

/// The `onError` event.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextErrorEvent {
    pub call_id: String,
    pub error: ErrorValue,
}

/// A step result as telemetry sees it: its fields plus the getters.
pub fn step_to_json(step: &StepResult) -> JsonValue {
    let mut value = serde_json::to_value(step).unwrap_or(JsonValue::Null);
    if let JsonValue::Object(map) = &mut value {
        let put = |map: &mut JsonObject, k: &str, v: JsonValue| {
            map.insert(k.into(), v);
        };
        put(map, "text", step.text().into());
        put(map, "reasoning", tagged(&step.reasoning(), "reasoning"));
        if let Some(r) = step.reasoning_text() {
            put(map, "reasoningText", r.into());
        }
        put(map, "files", json!([]));
        put(map, "sources", json!([]));
        put(map, "toolCalls", tagged(&step.tool_calls(), "tool-call"));
        put(
            map,
            "staticToolCalls",
            tagged(&step.static_tool_calls(), "tool-call"),
        );
        put(
            map,
            "dynamicToolCalls",
            tagged(&step.dynamic_tool_calls(), "tool-call"),
        );
        put(
            map,
            "toolResults",
            tagged(&step.tool_results(), "tool-result"),
        );
        put(
            map,
            "staticToolResults",
            tagged(&step.static_tool_results(), "tool-result"),
        );
        put(
            map,
            "dynamicToolResults",
            tagged(&step.dynamic_tool_results(), "tool-result"),
        );
    }
    value
}

/// Serialize `items` as an array of objects tagged with `type`, as the AI
/// SDK's content parts are (`{ type: 'tool-call', ... }`).
fn tagged<T: Serialize>(items: &[T], ty: &str) -> JsonValue {
    JsonValue::Array(
        items
            .iter()
            .map(|item| {
                let mut value = serde_json::to_value(item).unwrap_or(JsonValue::Null);
                if let JsonValue::Object(map) = &mut value {
                    map.insert("type".into(), ty.into());
                }
                value
            })
            .collect(),
    )
}

fn serialize_reasoning<S: serde::Serializer>(
    items: &[crate::harness::sdk::messages::ReasoningPart],
    s: S,
) -> Result<S::Ok, S::Error> {
    tagged(items, "reasoning").serialize(s)
}

fn serialize_tool_calls<S: serde::Serializer>(
    items: &[TypedToolCall],
    s: S,
) -> Result<S::Ok, S::Error> {
    tagged(items, "tool-call").serialize(s)
}

fn serialize_tool_results<S: serde::Serializer>(
    items: &[crate::harness::sdk::content::TypedToolResult],
    s: S,
) -> Result<S::Ok, S::Error> {
    tagged(items, "tool-result").serialize(s)
}

fn serialize_step<S: serde::Serializer>(step: &StepResult, s: S) -> Result<S::Ok, S::Error> {
    step_to_json(step).serialize(s)
}

fn serialize_steps<S: serde::Serializer>(steps: &[StepResult], s: S) -> Result<S::Ok, S::Error> {
    JsonValue::Array(steps.iter().map(step_to_json).collect()).serialize(s)
}

/// An `executeTool` thunk.
pub type ExecuteThunk = Box<dyn FnOnce() -> BoxFuture<'static, HarnessResult<JsonValue>> + Send>;

fn noop() -> BoxFuture<'static, ()> {
    Box::pin(async {})
}

/// `Telemetry`: a telemetry integration.
///
/// Every callback is optional (a no-op by default) and receives the event
/// as JSON.
pub trait Telemetry: Send + Sync {
    fn on_start(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_step_start(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_language_model_call_start(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_language_model_call_end(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_tool_execution_start(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_tool_execution_end(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_step_end(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    /// Deprecated alias of `onStepEnd`; dispatched alongside it.
    fn on_step_finish(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_step_error(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_end(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_abort(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    fn on_error(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        noop()
    }
    /// `executeTool`: wrap a tool execution. The default runs it.
    fn execute_tool(
        &self,
        _event: &TelemetryEvent,
        execute: ExecuteThunk,
    ) -> BoxFuture<'static, HarnessResult<JsonValue>> {
        execute()
    }
    /// Whether `executeTool` is implemented.
    fn supports_execute_tool(&self) -> bool {
        false
    }
    /// The `HarnessDiagnosticConsumer` side of a reporter
    /// (`ingestDiagnostic`).
    fn as_diagnostic_consumer(
        &self,
    ) -> Option<&dyn crate::harness::agent::observability::HarnessDiagnosticConsumer> {
        None
    }
}

/// A shared integration.
pub type SharedTelemetry = Arc<dyn Telemetry>;

/// `IncludedContext`: which context keys telemetry may see.
pub type IncludedContext = indexmap::IndexMap<String, bool>;
/// `IncludedToolsContext`: per tool [`IncludedContext`].
pub type IncludedToolsContext = indexmap::IndexMap<String, Option<IncludedContext>>;

/// `TelemetryOptions`.
#[derive(Clone, Default)]
pub struct TelemetryOptions {
    pub is_enabled: Option<bool>,
    pub record_inputs: Option<bool>,
    pub record_outputs: Option<bool>,
    pub function_id: Option<String>,
    pub include_runtime_context: Option<IncludedContext>,
    pub include_tools_context: Option<IncludedToolsContext>,
    /// `integrations?: Arrayable<Telemetry>`.
    pub integrations: Option<Vec<SharedTelemetry>>,
}

impl std::fmt::Debug for TelemetryOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryOptions")
            .field("is_enabled", &self.is_enabled)
            .field("record_inputs", &self.record_inputs)
            .field("record_outputs", &self.record_outputs)
            .field("function_id", &self.function_id)
            .field("integrations", &self.integrations.as_ref().map(|i| i.len()))
            .finish()
    }
}

fn global_integrations() -> &'static RwLock<Vec<SharedTelemetry>> {
    static GLOBAL: OnceLock<RwLock<Vec<SharedTelemetry>>> = OnceLock::new();
    GLOBAL.get_or_init(|| RwLock::new(Vec::new()))
}

/// `registerTelemetry(...integrations)`.
pub fn register_telemetry(integrations: impl IntoIterator<Item = SharedTelemetry>) {
    global_integrations().write().extend(integrations);
}

/// `getGlobalTelemetryIntegrations()`.
pub fn get_global_telemetry_integrations() -> Vec<SharedTelemetry> {
    global_integrations().read().clone()
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn clear_global_telemetry_integrations() {
    global_integrations().write().clear();
}

/// `filterIncludedContext({ context, includeContext })`.
pub fn filter_included_context(
    context: Option<&JsonValue>,
    include: Option<&IncludedContext>,
) -> JsonObject {
    match context {
        Some(JsonValue::Object(map)) => map
            .iter()
            .filter(|(k, _)| include.and_then(|i| i.get(*k)).copied() == Some(true))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        _ => JsonObject::new(),
    }
}

/// `filterToolContext({ toolName, toolContext, includeToolsContext })`.
pub fn filter_tool_context(
    tool_name: &str,
    tool_context: Option<&JsonValue>,
    include: Option<&IncludedToolsContext>,
) -> JsonObject {
    let tool_include = include
        .and_then(|i| i.get(tool_name))
        .and_then(|i| i.as_ref());
    filter_included_context(tool_context, tool_include)
}

/// `filterToolsContext({ toolsContext, includeToolsContext })`.
pub fn filter_tools_context(
    tools_context: &JsonObject,
    include: Option<&IncludedToolsContext>,
) -> JsonObject {
    let Some(include) = include else {
        return JsonObject::new();
    };
    tools_context
        .iter()
        .map(|(name, ctx)| {
            (
                name.clone(),
                JsonValue::Object(filter_tool_context(name, Some(ctx), Some(include))),
            )
        })
        .collect()
}

fn operation_failed() -> JsonValue {
    json!({ "name": "Error", "message": "AI SDK operation failed" })
}

/// `filterTelemetryEvent(event, { recordInputs, recordOutputs })` on JSON.
pub fn filter_telemetry_event(
    event: &JsonValue,
    record_inputs: Option<bool>,
    record_outputs: Option<bool>,
) -> JsonValue {
    let hide_inputs = record_inputs == Some(false);
    let hide_outputs = record_outputs == Some(false);
    let JsonValue::Object(original) = event else {
        return event.clone();
    };
    if !hide_inputs && !hide_outputs {
        return event.clone();
    }
    let mut filtered = original.clone();
    let replace = |filtered: &mut JsonObject, fields: Vec<(&str, Option<JsonValue>)>| {
        for (key, value) in fields {
            if original.contains_key(key) {
                match value {
                    Some(v) => {
                        filtered.insert(key.into(), v);
                    }
                    None => {
                        filtered.remove(key);
                    }
                }
            }
        }
    };
    let is_speech =
        filtered.get("operationId").and_then(|v| v.as_str()) == Some("ai.generateSpeech");
    if hide_inputs {
        let output_is_object = matches!(
            filtered.get("output"),
            Some(JsonValue::Object(_) | JsonValue::Array(_) | JsonValue::Null)
        );
        let state = match filtered.get("state") {
            Some(JsonValue::Array(_)) => json!([]),
            Some(JsonValue::String(_)) => json!(""),
            _ => json!({}),
        };
        let mut fields: Vec<(&str, Option<JsonValue>)> = vec![
            ("instructions", None),
            ("system", None),
            ("prompt", None),
            ("messages", Some(json!([]))),
            ("promptMessages", Some(json!([]))),
            ("experimental_steeringMessages", Some(json!([]))),
            ("tools", None),
            ("stepTools", None),
            ("schema", None),
            ("schemaName", None),
            ("schemaDescription", None),
            ("providerOptions", Some(json!({}))),
            ("value", Some(json!(""))),
            ("values", Some(json!([]))),
            ("query", Some(json!(""))),
            ("documents", Some(json!([]))),
            ("state", Some(state)),
            ("questions", Some(json!({}))),
        ];
        if output_is_object {
            fields.push(("output", None));
        }
        if is_speech {
            fields.push(("text", Some(json!(""))));
        }
        replace(&mut filtered, fields);
        for key in ["toolCall", "toolOutput"] {
            if let Some(JsonValue::Object(t)) = filtered.get_mut(key) {
                t.remove("input");
            }
        }
        if let Some(JsonValue::Array(ranking)) = filtered.get_mut("ranking") {
            for entry in ranking.iter_mut() {
                if let JsonValue::Object(e) = entry
                    && let Some(doc) = e.get("document")
                {
                    let empty = if doc.is_string() {
                        json!("")
                    } else {
                        json!({})
                    };
                    e.insert("document".into(), empty);
                }
            }
        }
    }
    if hide_outputs {
        let reasoning = if matches!(filtered.get("reasoning"), Some(JsonValue::Array(_))) {
            Some(json!([]))
        } else {
            None
        };
        let mut fields: Vec<(&str, Option<JsonValue>)> = vec![
            ("content", Some(json!([]))),
            ("reasoning", reasoning),
            ("reasoningText", None),
            ("files", Some(json!([]))),
            ("sources", Some(json!([]))),
            ("toolCalls", Some(json!([]))),
            ("staticToolCalls", Some(json!([]))),
            ("dynamicToolCalls", Some(json!([]))),
            ("toolResults", Some(json!([]))),
            ("staticToolResults", Some(json!([]))),
            ("dynamicToolResults", Some(json!([]))),
            ("responseMessages", Some(json!([]))),
            ("object", None),
            ("objectText", Some(json!(""))),
            ("embedding", Some(json!([]))),
            ("embeddings", Some(json!([]))),
            ("ranking", Some(json!([]))),
            ("answers", Some(json!({}))),
            ("segments", Some(json!([]))),
            ("words", Some(json!([]))),
        ];
        if !is_speech {
            fields.push(("text", Some(json!(""))));
        }
        replace(&mut filtered, fields);
        if let Some(JsonValue::Object(t)) = filtered.get_mut("toolOutput") {
            t.remove("output");
            t.remove("error");
        }
    }
    filtered.remove("providerMetadata");
    if let Some(error) = filtered.get("error")
        && !error.is_null()
    {
        filtered.insert("error".into(), operation_failed());
    }
    for key in ["toolCall", "toolOutput"] {
        if let Some(JsonValue::Object(t)) = filtered.get_mut(key) {
            t.remove("providerMetadata");
            if let Some(e) = t.get("error")
                && !e.is_null()
            {
                t.insert("error".into(), operation_failed());
            }
        }
    }
    for key in ["request", "response"] {
        let hide = if key == "request" {
            hide_inputs
        } else {
            hide_outputs
        };
        let filter = |m: &mut JsonValue| {
            if let JsonValue::Object(meta) = m {
                meta.remove("body");
                if hide {
                    if meta.contains_key("messages") {
                        meta.insert("messages".into(), json!([]));
                    }
                    meta.remove("instructions");
                }
            }
        };
        match filtered.get_mut(key) {
            Some(JsonValue::Array(items)) => items.iter_mut().for_each(filter),
            Some(m) => filter(m),
            None => {}
        }
    }
    if let Some(JsonValue::Array(steps)) = filtered.get_mut("steps") {
        for step in steps.iter_mut() {
            *step = filter_telemetry_event(step, record_inputs, record_outputs);
        }
    }
    if original.contains_key("finalStep")
        && let Some(step) = filtered.get("finalStep").cloned()
    {
        filtered.insert(
            "finalStep".into(),
            filter_telemetry_event(&step, record_inputs, record_outputs),
        );
    }
    JsonValue::Object(filtered)
}

/// Which integration callback to dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryCallbackKey {
    OnStart,
    OnStepStart,
    OnLanguageModelCallStart,
    OnLanguageModelCallEnd,
    OnToolExecutionStart,
    OnToolExecutionEnd,
    OnStepEnd,
    OnStepFinish,
    OnStepError,
    OnEnd,
    OnAbort,
    OnError,
}

fn dispatch(
    t: &dyn Telemetry,
    key: TelemetryCallbackKey,
    e: &TelemetryEvent,
) -> BoxFuture<'static, ()> {
    use TelemetryCallbackKey::*;
    match key {
        OnStart => t.on_start(e),
        OnStepStart => t.on_step_start(e),
        OnLanguageModelCallStart => t.on_language_model_call_start(e),
        OnLanguageModelCallEnd => t.on_language_model_call_end(e),
        OnToolExecutionStart => t.on_tool_execution_start(e),
        OnToolExecutionEnd => t.on_tool_execution_end(e),
        OnStepEnd => t.on_step_end(e),
        OnStepFinish => t.on_step_finish(e),
        OnStepError => t.on_step_error(e),
        OnEnd => t.on_end(e),
        OnAbort => t.on_abort(e),
        OnError => t.on_error(e),
    }
}

/// `createTelemetryDispatcher({ telemetry })`.
#[derive(Clone, Default)]
pub struct TelemetryDispatcher {
    integrations: Vec<SharedTelemetry>,
    record_inputs: Option<bool>,
    record_outputs: Option<bool>,
    function_id: Option<String>,
}

impl TelemetryDispatcher {
    /// `createTelemetryDispatcher({ telemetry })`.
    pub fn new(telemetry: Option<&TelemetryOptions>) -> Self {
        if telemetry.and_then(|t| t.is_enabled) == Some(false) {
            return Self::default();
        }
        let integrations = match telemetry.and_then(|t| t.integrations.clone()) {
            Some(local) => local,
            None => get_global_telemetry_integrations(),
        };
        Self {
            integrations,
            record_inputs: telemetry.and_then(|t| t.record_inputs),
            record_outputs: telemetry.and_then(|t| t.record_outputs),
            function_id: telemetry.and_then(|t| t.function_id.clone()),
        }
    }

    /// The integrations events go to.
    pub fn integrations(&self) -> &[SharedTelemetry] {
        &self.integrations
    }

    /// `augmentEvent(event, telemetryMetadata)`.
    pub fn augment(&self, event: JsonValue) -> TelemetryEvent {
        let mut value = event;
        if let JsonValue::Object(map) = &mut value {
            if let Some(v) = self.record_inputs {
                map.insert("recordInputs".into(), v.into());
            }
            if let Some(v) = self.record_outputs {
                map.insert("recordOutputs".into(), v.into());
            }
            if let Some(v) = &self.function_id {
                map.insert("functionId".into(), v.clone().into());
            }
            map.insert("aiVersion".into(), AI_VERSION.into());
        }
        match filter_telemetry_event(&value, self.record_inputs, self.record_outputs) {
            JsonValue::Object(map) => map,
            _ => JsonObject::new(),
        }
    }

    /// Dispatch an event to every integration's `key` callback.
    pub async fn emit(&self, key: TelemetryCallbackKey, event: JsonValue) {
        if self.integrations.is_empty() {
            return;
        }
        let event = self.augment(event);
        let futures: Vec<_> = self
            .integrations
            .iter()
            .map(|t| dispatch(&**t, key, &event))
            .collect();
        futures::future::join_all(futures).await;
    }

    /// `onStepEnd`: `mergeCallbacks(onStepEnd, onStepFinish)`.
    pub async fn emit_step_end(&self, event: JsonValue) {
        if self.integrations.is_empty() {
            return;
        }
        let event = self.augment(event);
        let futures: Vec<_> = self
            .integrations
            .iter()
            .flat_map(|t| [t.on_step_end(&event), t.on_step_finish(&event)])
            .collect();
        futures::future::join_all(futures).await;
    }

    /// Whether any integration wraps tool execution.
    pub fn has_execute_tool(&self) -> bool {
        self.integrations.iter().any(|t| t.supports_execute_tool())
    }

    /// `executeTool({ callId, toolCallId, execute })`: the first integration
    /// is the innermost wrapper.
    pub async fn execute_tool(
        &self,
        call_id: &str,
        tool_call_id: &str,
        execute: ExecuteThunk,
    ) -> HarnessResult<JsonValue> {
        let event = self.augment(json!({ "callId": call_id, "toolCallId": tool_call_id }));
        let mut wrapped = execute;
        for integration in self
            .integrations
            .iter()
            .filter(|t| t.supports_execute_tool())
        {
            let inner = wrapped;
            let integration = integration.clone();
            let event = event.clone();
            wrapped = Box::new(move || integration.execute_tool(&event, inner));
        }
        wrapped().await
    }
}

fn restrict_step(
    step: &StepResult,
    include_runtime: Option<&IncludedContext>,
    include_tools: Option<&IncludedToolsContext>,
) -> StepResult {
    let mut step = step.clone();
    step.runtime_context = filter_included_context(
        Some(&JsonValue::Object(step.runtime_context.clone())),
        include_runtime,
    );
    step.tools_context = filter_tools_context(&step.tools_context, include_tools);
    step
}

/// `createRestrictedTelemetryDispatcher({ telemetry, includeRuntimeContext,
/// includeToolsContext })`: filters runtime and tool context before
/// events reach integrations.
#[derive(Clone, Default)]
pub struct RestrictedTelemetryDispatcher {
    inner: TelemetryDispatcher,
    include_runtime_context: Option<IncludedContext>,
    include_tools_context: Option<IncludedToolsContext>,
}

impl RestrictedTelemetryDispatcher {
    pub fn new(telemetry: Option<&TelemetryOptions>) -> Self {
        Self {
            inner: TelemetryDispatcher::new(telemetry),
            include_runtime_context: telemetry.and_then(|t| t.include_runtime_context.clone()),
            include_tools_context: telemetry.and_then(|t| t.include_tools_context.clone()),
        }
    }

    fn restrict(&self, step: &StepResult) -> StepResult {
        restrict_step(
            step,
            self.include_runtime_context.as_ref(),
            self.include_tools_context.as_ref(),
        )
    }

    fn runtime(&self, context: &Context) -> JsonObject {
        filter_included_context(
            Some(&JsonValue::Object(context.clone())),
            self.include_runtime_context.as_ref(),
        )
    }

    fn tools(&self, context: &ToolsContext) -> JsonObject {
        filter_tools_context(context, self.include_tools_context.as_ref())
    }

    fn to_json<T: Serialize>(value: &T) -> JsonValue {
        serde_json::to_value(value).unwrap_or(JsonValue::Null)
    }

    pub async fn on_start(&self, event: &GenerateTextStartEvent) {
        let mut e = event.clone();
        e.runtime_context = self.runtime(&e.runtime_context);
        e.tools_context = self.tools(&e.tools_context);
        self.inner
            .emit(TelemetryCallbackKey::OnStart, Self::to_json(&e))
            .await;
    }

    pub async fn on_step_start(&self, event: &GenerateTextStepStartEvent) {
        let mut e = event.clone();
        e.runtime_context = self.runtime(&e.runtime_context);
        e.tools_context = self.tools(&e.tools_context);
        e.steps = e.steps.iter().map(|s| self.restrict(s)).collect();
        self.inner
            .emit(TelemetryCallbackKey::OnStepStart, Self::to_json(&e))
            .await;
    }

    pub async fn on_language_model_call_start(&self, event: &LanguageModelCallStartEvent) {
        self.inner
            .emit(
                TelemetryCallbackKey::OnLanguageModelCallStart,
                Self::to_json(event),
            )
            .await;
    }

    pub async fn on_language_model_call_end(&self, event: &LanguageModelCallEndEvent) {
        self.inner
            .emit(
                TelemetryCallbackKey::OnLanguageModelCallEnd,
                Self::to_json(event),
            )
            .await;
    }

    pub async fn on_tool_execution_start(&self, event: &ToolExecutionStartEvent) {
        let mut json = Self::to_json(event);
        json["toolContext"] = JsonValue::Object(filter_tool_context(
            &event.tool_call.tool_name,
            event.tool_context.as_ref(),
            self.include_tools_context.as_ref(),
        ));
        self.inner
            .emit(TelemetryCallbackKey::OnToolExecutionStart, json)
            .await;
    }

    pub async fn on_tool_execution_end(&self, event: &ToolExecutionEndEvent) {
        let mut json = Self::to_json(event);
        json["toolContext"] = JsonValue::Object(filter_tool_context(
            &event.tool_call.tool_name,
            event.tool_context.as_ref(),
            self.include_tools_context.as_ref(),
        ));
        self.inner
            .emit(TelemetryCallbackKey::OnToolExecutionEnd, json)
            .await;
    }

    pub async fn on_step_end(&self, step: &StepResult) {
        self.inner
            .emit_step_end(step_to_json(&self.restrict(step)))
            .await;
    }

    pub async fn on_end(&self, event: &GenerateTextEndEvent) {
        let mut e = event.clone();
        e.runtime_context = self.runtime(&e.runtime_context);
        e.tools_context = self.tools(&e.tools_context);
        e.steps = e.steps.iter().map(|s| self.restrict(s)).collect();
        if let Some(last) = e.steps.last() {
            e.final_step = last.clone();
        }
        self.inner
            .emit(TelemetryCallbackKey::OnEnd, Self::to_json(&e))
            .await;
    }

    pub async fn on_abort(&self, event: &GenerateTextAbortEvent) {
        let mut e = event.clone();
        e.steps = e.steps.iter().map(|s| self.restrict(s)).collect();
        self.inner
            .emit(TelemetryCallbackKey::OnAbort, Self::to_json(&e))
            .await;
    }

    pub async fn on_error(&self, event: &GenerateTextErrorEvent) {
        self.inner
            .emit(TelemetryCallbackKey::OnError, Self::to_json(event))
            .await;
    }

    pub fn has_execute_tool(&self) -> bool {
        self.inner.has_execute_tool()
    }

    pub async fn execute_tool(
        &self,
        call_id: &str,
        tool_call_id: &str,
        execute: ExecuteThunk,
    ) -> HarnessResult<JsonValue> {
        self.inner
            .execute_tool(call_id, tool_call_id, execute)
            .await
    }

    /// The integrations events go to.
    pub fn integrations(&self) -> &[SharedTelemetry] {
        self.inner.integrations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_hides_inputs_and_outputs() {
        let event = json!({
            "callId": "c",
            "instructions": "secret",
            "messages": [{"role": "user", "content": "hi"}],
            "content": [{"type": "text", "text": "x"}],
            "text": "x",
            "toolCall": {"toolName": "t", "input": {"a": 1}, "providerMetadata": {"p": {}}},
            "toolOutput": {"type": "tool-result", "input": {}, "output": 1},
            "providerMetadata": {"p": {}},
            "response": {"id": "r", "messages": [1], "body": "b"},
            "steps": [{"content": [1], "text": "y"}]
        });
        let filtered = filter_telemetry_event(&event, Some(false), Some(false));
        assert_eq!(
            filtered,
            json!({
                "callId": "c",
                "messages": [],
                "content": [],
                "text": "",
                "toolCall": {"toolName": "t"},
                "toolOutput": {"type": "tool-result"},
                "response": {"id": "r", "messages": []},
                "steps": [{"content": [], "text": ""}]
            })
        );
        assert_eq!(filter_telemetry_event(&event, None, None), event);
    }

    #[test]
    fn filter_masks_errors_when_filtering() {
        let event = json!({"callId": "c", "error": {"name": "Error", "message": "secret"}});
        assert_eq!(
            filter_telemetry_event(&event, Some(false), None),
            json!({"callId": "c", "error": {"name": "Error", "message": "AI SDK operation failed"}})
        );
    }

    #[test]
    fn filters_included_context() {
        let mut include = IncludedContext::new();
        include.insert("a".into(), true);
        include.insert("b".into(), false);
        assert_eq!(
            JsonValue::Object(filter_included_context(
                Some(&json!({"a": 1, "b": 2, "c": 3})),
                Some(&include)
            )),
            json!({"a": 1})
        );
        assert!(filter_included_context(None, Some(&include)).is_empty());
        let ctx = json!({"t": {"a": 1}}).as_object().cloned().unwrap();
        assert!(filter_tools_context(&ctx, None).is_empty());
        let mut tools = IncludedToolsContext::new();
        tools.insert("t".into(), Some(include));
        assert_eq!(
            JsonValue::Object(filter_tools_context(&ctx, Some(&tools))),
            json!({"t": {"a": 1}})
        );
    }

    #[tokio::test]
    async fn merge_callbacks_runs_every_callback() {
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let a = {
            let seen = seen.clone();
            callback(move |e: u32| {
                let seen = seen.clone();
                async move { seen.lock().push(("a", e)) }
            })
        };
        let b = {
            let seen = seen.clone();
            callback(move |e: u32| {
                let seen = seen.clone();
                async move { seen.lock().push(("b", e)) }
            })
        };
        merge_callbacks(vec![Some(a), None, Some(b)])(7).await;
        assert_eq!(*seen.lock(), vec![("a", 7), ("b", 7)]);
    }

    struct Recorder(Arc<parking_lot::Mutex<Vec<TelemetryEvent>>>);
    impl Telemetry for Recorder {
        fn on_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.0.lock().push(e.clone());
            Box::pin(async {})
        }
    }

    #[tokio::test]
    async fn dispatcher_augments_events_and_respects_is_enabled() {
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let options = TelemetryOptions {
            function_id: Some("f".into()),
            record_inputs: Some(true),
            integrations: Some(vec![Arc::new(Recorder(seen.clone()))]),
            ..Default::default()
        };
        TelemetryDispatcher::new(Some(&options))
            .emit(TelemetryCallbackKey::OnStart, json!({"callId": "c"}))
            .await;
        assert_eq!(
            JsonValue::Object(seen.lock()[0].clone()),
            json!({"callId": "c", "recordInputs": true, "functionId": "f", "aiVersion": AI_VERSION})
        );
        let disabled = TelemetryOptions {
            is_enabled: Some(false),
            ..options
        };
        TelemetryDispatcher::new(Some(&disabled))
            .emit(TelemetryCallbackKey::OnStart, json!({"callId": "c"}))
            .await;
        assert_eq!(seen.lock().len(), 1);
    }
}
