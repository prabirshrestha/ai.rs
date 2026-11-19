//! Port of `agent/internal/turn-telemetry.ts`.

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::harness::errors::HarnessResult;
use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::content::{
    ContentPart, Context, FinishReason, ProviderMetadata, StepResult, ToolOutput, ToolsContext,
    TypedToolCall,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::messages::ModelMessage;
use crate::harness::sdk::output::SharedOutput;
use crate::harness::sdk::telemetry::*;
use crate::harness::sdk::tool::ToolSet;
use crate::harness::sdk::usage::LanguageModelUsage;
use crate::harness::spec::tools::HarnessV1ToolSpec;

/// `HarnessAgentLifecycleCallbacks`.
#[derive(Clone, Default)]
pub struct HarnessAgentLifecycleCallbacks {
    pub on_start: Option<Callback<GenerateTextStartEvent>>,
    pub on_step_start: Option<Callback<GenerateTextStepStartEvent>>,
    pub on_language_model_call_start: Option<Callback<LanguageModelCallStartEvent>>,
    pub on_language_model_call_end: Option<Callback<LanguageModelCallEndEvent>>,
    pub on_tool_execution_start: Option<Callback<ToolExecutionStartEvent>>,
    pub on_tool_execution_end: Option<Callback<ToolExecutionEndEvent>>,
    pub on_step_end: Option<Callback<StepResult>>,
    pub on_end: Option<Callback<GenerateTextEndEvent>>,
}

/// `createTurnLifecycle` options.
#[derive(Clone)]
pub struct TurnLifecycleOptions {
    pub call_id: String,
    pub telemetry: Option<TelemetryOptions>,
    pub callbacks: HarnessAgentLifecycleCallbacks,
    pub harness_id: String,
    pub model_id: Option<String>,
    pub instructions: Option<String>,
    pub tools: ToolSet,
    pub tools_context: ToolsContext,
    pub active_tool_names: Vec<String>,
    pub tool_specs: Vec<HarnessV1ToolSpec>,
    pub messages: Vec<ModelMessage>,
    pub runtime_context: Context,
    pub output: Option<SharedOutput>,
}

#[derive(Default)]
struct State {
    model_id: String,
    started: bool,
    step_open: bool,
    step_number: u32,
    ended: bool,
    model_call_started_at: Option<Instant>,
    completed_steps: Vec<StepResult>,
}

/// `TurnLifecycle`: drives the user's lifecycle callbacks and the telemetry
/// integrations from a harness turn, with `streamText`'s event shapes.
pub struct TurnLifecycle {
    options: TurnLifecycleOptions,
    telemetry: RestrictedTelemetryDispatcher,
    provider: String,
    language_model_tools: Option<Vec<JsonObject>>,
    state: Mutex<State>,
}

/// `createTurnLifecycle(options)`.
pub fn create_turn_lifecycle(options: TurnLifecycleOptions) -> TurnLifecycle {
    let telemetry = RestrictedTelemetryDispatcher::new(options.telemetry.as_ref());
    let provider = format!("harness:{}", options.harness_id);
    let language_model_tools = if options.tool_specs.is_empty() {
        None
    } else {
        Some(
            options
                .tool_specs
                .iter()
                .map(|tool| {
                    let mut t = JsonObject::new();
                    t.insert("type".into(), "function".into());
                    t.insert("name".into(), tool.name.clone().into());
                    if let Some(d) = &tool.description {
                        t.insert("description".into(), d.clone().into());
                    }
                    if let Some(s) = &tool.input_schema {
                        t.insert("inputSchema".into(), s.clone());
                    }
                    t
                })
                .collect(),
        )
    };
    let state = State {
        model_id: options.model_id.clone().unwrap_or_default(),
        ..Default::default()
    };
    TurnLifecycle {
        options,
        telemetry,
        provider,
        language_model_tools,
        state: Mutex::new(state),
    }
}

/// `languageModelCallEnd` input.
#[derive(Debug, Clone)]
pub struct LanguageModelCallEndInput {
    pub finish_reason: FinishReason,
    pub usage: LanguageModelUsage,
    pub content: Vec<ContentPart>,
    pub provider_metadata: Option<ProviderMetadata>,
}

/// `notify(event, callback)` for one optional callback.
async fn call<E>(callback: Option<&Callback<E>>, event: E) {
    if let Some(callback) = callback {
        callback(event).await;
    }
}

impl TurnLifecycle {
    /// `start(modelId?)`.
    pub async fn start(&self, override_model_id: Option<String>) {
        let model_id = {
            let mut s = self.state.lock();
            if s.started {
                return;
            }
            if let Some(m) = override_model_id {
                s.model_id = m;
            }
            s.started = true;
            s.model_id.clone()
        };
        let o = &self.options;
        let event = GenerateTextStartEvent {
            call_id: o.call_id.clone(),
            operation_id: "ai.harness".into(),
            provider: self.provider.clone(),
            model_id,
            tools: o.tools.clone(),
            active_tools: o.active_tool_names.clone(),
            tool_order: vec![],
            max_retries: 0,
            output: o.output.clone(),
            tools_context: o.tools_context.clone(),
            runtime_context: o.runtime_context.clone(),
            instructions: o.instructions.clone(),
            messages: o.messages.clone(),
        };
        futures::join!(
            call(o.callbacks.on_start.as_ref(), event.clone()),
            self.telemetry.on_start(&event)
        );
    }

    fn is_started(&self) -> bool {
        self.state.lock().started
    }

    /// `ensureStepOpen()`.
    pub async fn ensure_step_open(&self) {
        if !self.is_started() {
            self.start(None).await;
        }
        let (model_id, step_number, steps) = {
            let mut s = self.state.lock();
            if s.step_open || s.ended {
                return;
            }
            s.step_open = true;
            (s.model_id.clone(), s.step_number, s.completed_steps.clone())
        };
        let o = &self.options;
        let step_start = GenerateTextStepStartEvent {
            call_id: o.call_id.clone(),
            provider: self.provider.clone(),
            model_id: model_id.clone(),
            step_number,
            tools: o.tools.clone(),
            active_tools: o.active_tool_names.clone(),
            tool_order: vec![],
            steps,
            output: o.output.clone(),
            runtime_context: o.runtime_context.clone(),
            tools_context: o.tools_context.clone(),
            instructions: o.instructions.clone(),
            messages: o.messages.clone(),
        };
        futures::join!(
            call(o.callbacks.on_step_start.as_ref(), step_start.clone()),
            self.telemetry.on_step_start(&step_start)
        );

        let model_start = LanguageModelCallStartEvent {
            call_id: o.call_id.clone(),
            provider: self.provider.clone(),
            model_id,
            instructions: o.instructions.clone(),
            messages: o.messages.clone(),
            tools: self.language_model_tools.clone(),
        };
        self.state.lock().model_call_started_at = Some(Instant::now());
        futures::join!(
            call(
                o.callbacks.on_language_model_call_start.as_ref(),
                model_start.clone()
            ),
            self.telemetry.on_language_model_call_start(&model_start)
        );
    }

    /// `languageModelCallEnd(input)`.
    pub async fn language_model_call_end(&self, input: LanguageModelCallEndInput) {
        self.ensure_step_open().await;
        let (model_id, step_number, started_at) = {
            let s = self.state.lock();
            (s.model_id.clone(), s.step_number, s.model_call_started_at)
        };
        let o = &self.options;
        let event = LanguageModelCallEndEvent {
            call_id: o.call_id.clone(),
            provider: self.provider.clone(),
            model_id,
            finish_reason: input.finish_reason,
            usage: input.usage,
            content: input.content,
            response_id: format!("{}-{step_number}", o.call_id),
            provider_metadata: input.provider_metadata,
            performance: LanguageModelCallPerformance {
                response_time_ms: started_at
                    .map(|t| t.elapsed().as_millis() as f64)
                    .unwrap_or(0.0),
                ..Default::default()
            },
        };
        futures::join!(
            call(
                o.callbacks.on_language_model_call_end.as_ref(),
                event.clone()
            ),
            self.telemetry.on_language_model_call_end(&event)
        );
    }

    /// `toolExecutionStart({ toolCall })`.
    pub async fn tool_execution_start(&self, tool_call: TypedToolCall) {
        let o = &self.options;
        let event = ToolExecutionStartEvent {
            call_id: o.call_id.clone(),
            messages: o.messages.clone(),
            tool_call,
            tool_context: None,
        };
        futures::join!(
            call(o.callbacks.on_tool_execution_start.as_ref(), event.clone()),
            self.telemetry.on_tool_execution_start(&event)
        );
    }

    /// `toolExecutionEnd({ toolCall, toolOutput, toolExecutionMs })`.
    pub async fn tool_execution_end(
        &self,
        tool_call: TypedToolCall,
        tool_output: ToolOutput,
        tool_execution_ms: f64,
    ) {
        let o = &self.options;
        let event = ToolExecutionEndEvent {
            call_id: o.call_id.clone(),
            messages: o.messages.clone(),
            tool_call,
            tool_context: None,
            tool_output,
            tool_execution_ms,
        };
        futures::join!(
            call(o.callbacks.on_tool_execution_end.as_ref(), event.clone()),
            self.telemetry.on_tool_execution_end(&event)
        );
    }

    /// `stepEnd(step)`.
    pub async fn step_end(&self, step: StepResult) {
        {
            let mut s = self.state.lock();
            if !s.step_open {
                return;
            }
            s.completed_steps.push(step.clone());
        }
        futures::join!(
            call(self.options.callbacks.on_step_end.as_ref(), step.clone()),
            self.telemetry.on_step_end(&step)
        );
        let mut s = self.state.lock();
        s.step_open = false;
        s.step_number += 1;
    }

    /// `end({ steps, usage })`.
    pub async fn end(&self, steps: &[StepResult], usage: LanguageModelUsage) {
        if !self.is_started() {
            self.start(None).await;
        }
        {
            let mut s = self.state.lock();
            if s.ended || steps.is_empty() {
                return;
            }
            s.ended = true;
        }
        let final_step = steps.last().expect("non-empty").clone();
        let o = &self.options;
        let event = GenerateTextEndEvent {
            call_id: o.call_id.clone(),
            step_number: final_step.step_number,
            model: final_step.model.clone(),
            tools_context: final_step.tools_context.clone(),
            runtime_context: final_step.runtime_context.clone(),
            content: steps.iter().flat_map(|s| s.content.clone()).collect(),
            text: final_step.text(),
            reasoning: final_step.reasoning(),
            reasoning_text: final_step.reasoning_text(),
            files: vec![],
            sources: vec![],
            tool_calls: steps.iter().flat_map(|s| s.tool_calls()).collect(),
            static_tool_calls: steps.iter().flat_map(|s| s.static_tool_calls()).collect(),
            dynamic_tool_calls: steps.iter().flat_map(|s| s.dynamic_tool_calls()).collect(),
            tool_results: steps.iter().flat_map(|s| s.tool_results()).collect(),
            static_tool_results: steps.iter().flat_map(|s| s.static_tool_results()).collect(),
            dynamic_tool_results: steps
                .iter()
                .flat_map(|s| s.dynamic_tool_results())
                .collect(),
            finish_reason: final_step.finish_reason,
            raw_finish_reason: final_step.raw_finish_reason.clone(),
            usage: usage.clone(),
            total_usage: usage,
            warnings: steps
                .iter()
                .flat_map(|s| s.warnings.clone().unwrap_or_default())
                .collect(),
            request: final_step.request.clone(),
            response: final_step.response.clone(),
            provider_metadata: final_step.provider_metadata.clone(),
            response_messages: steps
                .iter()
                .flat_map(|s| s.response.messages.clone())
                .collect(),
            steps: steps.to_vec(),
            final_step,
        };
        futures::join!(
            call(o.callbacks.on_end.as_ref(), event.clone()),
            self.telemetry.on_end(&event)
        );
    }

    /// `executeTool({ toolCallId, execute })`: runs `execute` inside the
    /// integrations' `executeTool` wrappers, when there are any.
    ///
    /// [rust] Integrations wrap a JSON thunk; the typed result travels
    /// beside it, and the wrappers see `null`.
    pub async fn execute_tool<T: Send + 'static>(
        &self,
        tool_call_id: &str,
        execute: BoxFuture<'static, HarnessResult<T>>,
    ) -> HarnessResult<T> {
        if !self.telemetry.has_execute_tool() {
            return execute.await;
        }
        let slot: Arc<Mutex<Option<T>>> = Arc::default();
        let out = slot.clone();
        self.telemetry
            .execute_tool(
                &self.options.call_id,
                tool_call_id,
                Box::new(move || {
                    Box::pin(async move {
                        let value = execute.await?;
                        *out.lock() = Some(value);
                        Ok(JsonValue::Null)
                    })
                }),
            )
            .await?;
        let value = slot.lock().take();
        value.ok_or_else(|| "executeTool integration did not run the tool".into())
    }

    /// `abort(reason?)`.
    pub async fn abort(&self, reason: Option<ErrorValue>) {
        if self.state.lock().ended {
            return;
        }
        if !self.is_started() {
            self.start(None).await;
        }
        let steps = {
            let mut s = self.state.lock();
            s.ended = true;
            s.completed_steps.clone()
        };
        self.telemetry
            .on_abort(&GenerateTextAbortEvent {
                call_id: self.options.call_id.clone(),
                steps,
                reason,
            })
            .await;
    }

    /// `error(error)`.
    pub async fn error(&self, error: ErrorValue) {
        if self.state.lock().ended {
            return;
        }
        if !self.is_started() {
            self.start(None).await;
        }
        self.state.lock().ended = true;
        self.telemetry
            .on_error(&GenerateTextErrorEvent {
                call_id: self.options.call_id.clone(),
                error,
            })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sdk::content::{LanguageModelResponseMetadata, StepModel};
    use crate::harness::sdk::messages::UserModelMessage;
    use crate::harness::sdk::usage::create_null_language_model_usage;
    use serde_json::json;
    use std::time::UNIX_EPOCH;

    fn create_step(step_number: u32) -> StepResult {
        StepResult {
            call_id: "call-1".into(),
            step_number,
            model: StepModel {
                provider: "harness:mock".into(),
                model_id: "mock-model".into(),
            },
            tools_context: Default::default(),
            runtime_context: Default::default(),
            content: vec![],
            finish_reason: FinishReason::Stop,
            raw_finish_reason: Some("stop".into()),
            usage: create_null_language_model_usage(),
            performance: Default::default(),
            warnings: None,
            request: Default::default(),
            response: LanguageModelResponseMetadata {
                id: format!("response-{step_number}"),
                timestamp: UNIX_EPOCH,
                model_id: "mock-model".into(),
                headers: None,
                messages: vec![],
                body: None,
            },
            provider_metadata: None,
        }
    }

    #[derive(Default)]
    struct Recorder {
        events: Mutex<Vec<(&'static str, TelemetryEvent)>>,
    }

    impl Recorder {
        fn push(&self, name: &'static str, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.events.lock().push((name, e.clone()));
            Box::pin(async {})
        }

        fn of(&self, name: &str) -> Vec<TelemetryEvent> {
            self.events
                .lock()
                .iter()
                .filter(|(n, _)| *n == name)
                .map(|(_, e)| e.clone())
                .collect()
        }
    }

    impl Telemetry for Recorder {
        fn on_step_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.push("onStepStart", e)
        }
        fn on_step_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.push("onStepEnd", e)
        }
        fn on_abort(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.push("onAbort", e)
        }
        fn on_error(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.push("onError", e)
        }
        fn on_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
            self.push("onEnd", e)
        }
    }

    fn lifecycle(recorder: Arc<Recorder>) -> TurnLifecycle {
        create_turn_lifecycle(TurnLifecycleOptions {
            call_id: "call-1".into(),
            telemetry: Some(TelemetryOptions {
                integrations: Some(vec![recorder]),
                ..Default::default()
            }),
            callbacks: Default::default(),
            harness_id: "mock".into(),
            model_id: Some("mock-model".into()),
            instructions: None,
            tools: Default::default(),
            tools_context: Default::default(),
            active_tool_names: vec![],
            tool_specs: vec![],
            messages: vec![ModelMessage::User(UserModelMessage {
                content: crate::harness::sdk::messages::UserContent::Text("go".into()),
                provider_options: None,
            })],
            runtime_context: Default::default(),
            output: None,
        })
    }

    #[tokio::test]
    async fn includes_the_current_step_number_on_on_step_end_events() {
        let recorder = Arc::new(Recorder::default());
        let lifecycle = lifecycle(recorder.clone());
        for step_number in [0, 1] {
            lifecycle.ensure_step_open().await;
            lifecycle.step_end(create_step(step_number)).await;
        }
        let numbers = |name| -> Vec<JsonValue> {
            recorder
                .of(name)
                .iter()
                .map(|e| e["stepNumber"].clone())
                .collect()
        };
        assert_eq!(numbers("onStepStart"), vec![json!(0), json!(1)]);
        assert_eq!(numbers("onStepEnd"), vec![json!(0), json!(1)]);
    }

    #[tokio::test]
    async fn includes_the_call_id_when_dispatching_errors() {
        let recorder = Arc::new(Recorder::default());
        let lifecycle = lifecycle(recorder.clone());
        lifecycle.error(ErrorValue::message("bridge exited")).await;
        let errors = recorder.of("onError");
        assert_eq!(errors.len(), 1);
        let e = &errors[0];
        assert!(e["aiVersion"].is_string());
        assert_eq!(e["callId"], json!("call-1"));
        assert_eq!(e["error"]["message"], json!("bridge exited"));
        assert_eq!(e.len(), 3);
    }

    #[tokio::test]
    async fn dispatches_aborts_and_settles_the_lifecycle() {
        let recorder = Arc::new(Recorder::default());
        let lifecycle = lifecycle(recorder.clone());
        let step = create_step(0);
        lifecycle.ensure_step_open().await;
        lifecycle.step_end(step.clone()).await;
        lifecycle.abort(Some(ErrorValue::message("stopped"))).await;
        lifecycle.error(ErrorValue::message("late error")).await;
        lifecycle
            .end(&[step], create_null_language_model_usage())
            .await;

        let aborts = recorder.of("onAbort");
        assert_eq!(aborts.len(), 1);
        let a = &aborts[0];
        assert!(a["aiVersion"].is_string());
        assert_eq!(a["callId"], json!("call-1"));
        assert_eq!(a["steps"].as_array().unwrap().len(), 1);
        assert_eq!(a["steps"][0]["callId"], json!("call-1"));
        assert_eq!(a["steps"][0]["stepNumber"], json!(0));
        assert_eq!(a["reason"]["message"], json!("stopped"));
        assert!(recorder.of("onError").is_empty());
        assert!(recorder.of("onEnd").is_empty());
    }
}
