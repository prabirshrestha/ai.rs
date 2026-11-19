//! `harness-agent.test.ts`: runtime context, lifecycle callbacks, the agent
//! surface, models and headers.

use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::sdk::content::StepResult;
use crate::harness::sdk::stop_condition::is_step_count;
use crate::harness::sdk::telemetry::{Telemetry, TelemetryEvent, TelemetryOptions, callback};

fn context(value: JsonValue) -> serde_json::Map<String, JsonValue> {
    obj(value)
}

/// A telemetry integration whose `onEnd` records the event's
/// `runtimeContext`.
struct RuntimeContextRecorder(Arc<Mutex<Vec<JsonValue>>>);

impl Telemetry for RuntimeContextRecorder {
    fn on_end(&self, event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.lock().push(
            event
                .get("runtimeContext")
                .cloned()
                .unwrap_or(JsonValue::Null),
        );
        Box::pin(async {})
    }
}

fn prepare_runtime_context(
    f: impl Fn(&mut HarnessAgentPrepareCallOptions) + Send + Sync + 'static,
) -> PrepareCallFn {
    Arc::new(move |mut call| {
        f(&mut call);
        Box::pin(async move { Ok(call) })
    })
}

#[tokio::test]
async fn uses_prepared_runtime_context_for_each_prompt_turn_and_filters_telemetry() {
    let configured_context = context(json!({ "requestId": "default", "secret": "configured" }));
    type Contexts = Arc<Mutex<Vec<Option<serde_json::Map<String, JsonValue>>>>>;
    let prepared_inputs: Contexts = Arc::default();
    let lifecycle_contexts: Arc<Mutex<Vec<serde_json::Map<String, JsonValue>>>> = Arc::default();
    let telemetry_contexts: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let mock = MockHarnessOptions::script(|| text_turn("text-1", "ok")).build();
    let mut s = settings(&mock);
    s.runtime_context = Some(configured_context.clone());
    s.call_options_schema = Some(z_object(&[("requestId", "string")]));
    let inputs = prepared_inputs.clone();
    s.prepare_call = Some(prepare_runtime_context(move |call| {
        inputs.lock().push(call.runtime_context.clone());
        let request_id = call.options.as_ref().unwrap()["requestId"].clone();
        call.runtime_context = Some(context(
            json!({ "requestId": request_id, "secret": "private" }),
        ));
    }));
    let push = |log: &Arc<Mutex<Vec<serde_json::Map<String, JsonValue>>>>| log.clone();
    let log = push(&lifecycle_contexts);
    s.on_start = Some(callback(
        move |e: crate::harness::sdk::telemetry::GenerateTextStartEvent| {
            log.lock().push(e.runtime_context);
            async {}
        },
    ));
    let log = push(&lifecycle_contexts);
    s.on_step_start = Some(callback(
        move |e: crate::harness::sdk::telemetry::GenerateTextStepStartEvent| {
            log.lock().push(e.runtime_context);
            async {}
        },
    ));
    let log = push(&lifecycle_contexts);
    s.on_step_end = Some(callback(move |e: StepResult| {
        log.lock().push(e.runtime_context);
        async {}
    }));
    let log = push(&lifecycle_contexts);
    s.on_end = Some(callback(
        move |e: crate::harness::sdk::telemetry::GenerateTextEndEvent| {
            log.lock().push(e.runtime_context);
            async {}
        },
    ));
    s.telemetry = Some(TelemetryOptions {
        include_runtime_context: Some(IndexMap::from([("requestId".to_string(), true)])),
        integrations: Some(vec![Arc::new(RuntimeContextRecorder(
            telemetry_contexts.clone(),
        ))]),
        ..Default::default()
    });
    let agent = agent(s);
    let session = new_session(&agent).await;

    let mut first = call(&session, "first");
    first.options = Some(json!({ "requestId": "req-1" }));
    let generated = agent.generate(first).await.unwrap();
    let mut second = call(&session, "second");
    second.options = Some(json!({ "requestId": "req-2" }));
    let streamed = agent.stream(second).await.unwrap();
    streamed.consume_stream().await;
    let streamed_final = streamed.final_step().await.unwrap();

    assert_eq!(
        *prepared_inputs.lock(),
        vec![Some(configured_context.clone()), Some(configured_context)]
    );
    assert_eq!(
        generated.final_step().runtime_context,
        context(json!({ "requestId": "req-1", "secret": "private" }))
    );
    assert_eq!(
        streamed_final.runtime_context,
        context(json!({ "requestId": "req-2", "secret": "private" }))
    );
    let mut expected = vec![generated.final_step().runtime_context.clone(); 4];
    expected.extend(vec![streamed_final.runtime_context.clone(); 4]);
    assert_eq!(*lifecycle_contexts.lock(), expected);
    assert_eq!(
        *telemetry_contexts.lock(),
        vec![
            json!({ "requestId": "req-1" }),
            json!({ "requestId": "req-2" })
        ]
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn defaults_to_empty_runtime_context_when_prepare_call_clears_it() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let mut s = settings(&mock);
    s.runtime_context = Some(context(json!({ "requestId": "configured" })));
    s.prepare_call = Some(prepare_runtime_context(|call| call.runtime_context = None));
    let agent = agent(s);
    let session = new_session(&agent).await;
    let result = agent.generate(call(&session, "go")).await.unwrap();

    assert!(result.final_step().runtime_context.is_empty());
    session.destroy().await.unwrap();
}

fn continuing_mock() -> Mock {
    MockHarnessOptions::script(one_step_then_more)
        .with_continue(|| text_turn("text-1", "done"))
        .build()
}

/// An agent that prepares `{ requestId: 'prepared' }` and stops after one
/// step, with a count of its `prepareCall` calls.
fn prepared_context_agent(mock: &Mock) -> (HarnessAgent, Arc<Mutex<usize>>) {
    let prepare_calls: Arc<Mutex<usize>> = Arc::default();
    let calls = prepare_calls.clone();
    let mut s = settings(mock);
    s.runtime_context = Some(context(json!({ "requestId": "default" })));
    s.prepare_call = Some(prepare_runtime_context(move |call| {
        *calls.lock() += 1;
        call.runtime_context = Some(context(json!({ "requestId": "prepared" })));
    }));
    s.stop_when = Some(vec![is_step_count(1)]);
    (agent(s), prepare_calls)
}

#[tokio::test]
async fn preserves_prepared_runtime_context_during_an_in_memory_continuation() {
    let prepared_context = context(json!({ "requestId": "prepared" }));
    let mock = continuing_mock();
    let (agent, prepare_calls) = prepared_context_agent(&mock);
    let session = new_session(&agent).await;

    let first = agent.generate(call(&session, "go")).await.unwrap();
    assert!(session.has_unfinished_turn());
    assert_eq!(first.final_step().runtime_context, prepared_context);

    let continued = agent
        .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();
    assert_eq!(continued.final_step().runtime_context, prepared_context);
    assert_eq!(*prepare_calls.lock(), 1);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn uses_rebound_context_or_constructor_fallback_after_recreating_a_suspended_turn_without_serializing_runtime_context()
 {
    for (rebind, expected_request_id) in [(true, "prepared"), (false, "default")] {
        let prepared_context = context(json!({ "requestId": "prepared" }));
        let mock = continuing_mock();
        let (agent, prepare_calls) = prepared_context_agent(&mock);
        let sandbox = make_sandbox_session();
        let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();

        let first = agent.generate(call(&session, "go")).await.unwrap();
        assert_eq!(first.final_step().runtime_context, prepared_context);
        let session_id = session.session_id().to_string();
        let continue_from = session.suspend_turn().await.unwrap();
        let state = to_json(&continue_from);
        assert!(state.get("runtimeContext").is_none());
        assert!(
            state
                .get("turnSettings")
                .is_none_or(|settings| settings.get("runtimeContext").is_none())
        );

        let mut options = with_sandbox(&sandbox);
        options.session_id = Some(session_id);
        options.continue_from = Some(continue_from.clone());
        if rebind {
            options.runtime_context = Some(prepared_context.clone());
        }
        let session = agent.create_session(options).await.unwrap();
        let continued = agent
            .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
            .await
            .unwrap();
        assert_eq!(
            continued.final_step().runtime_context,
            context(json!({ "requestId": expected_request_id }))
        );
        assert_eq!(*prepare_calls.lock(), 1);
        session.destroy().await.unwrap();
    }
}

#[tokio::test]
async fn forwards_configured_runtime_context_through_every_public_turn_entry_point() {
    let runtime_context = context(json!({ "conversationId": "conversation-1" }));
    let lifecycle_contexts: Arc<Mutex<Vec<serde_json::Map<String, JsonValue>>>> = Arc::default();
    let telemetry_contexts: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let completed_turn = || {
        let mut values = vec![json!({ "type": "stream-start", "modelId": "mock-model" })];
        values.extend([
            json!({ "type": "text-start", "id": "text-1" }),
            json!({ "type": "text-delta", "id": "text-1", "delta": "completed" }),
            json!({ "type": "text-end", "id": "text-1" }),
            json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "stop" }, "usage": zero_usage() }),
            json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "stop" }, "totalUsage": zero_usage() }),
        ]);
        parts(values)
    };
    let mock = MockHarnessOptions::script(completed_turn)
        .with_continue(completed_turn)
        .build();
    let mut s = settings(&mock);
    s.runtime_context = Some(runtime_context.clone());
    s.telemetry = Some(TelemetryOptions {
        include_runtime_context: Some(IndexMap::from([("conversationId".to_string(), true)])),
        integrations: Some(vec![Arc::new(RuntimeContextRecorder(
            telemetry_contexts.clone(),
        ))]),
        ..Default::default()
    });
    let log = lifecycle_contexts.clone();
    s.on_end = Some(callback(
        move |e: crate::harness::sdk::telemetry::GenerateTextEndEvent| {
            log.lock().push(e.runtime_context);
            async {}
        },
    ));
    let agent = agent(s);

    let generate_session = new_session(&agent).await;
    let generated = agent
        .generate(call(&generate_session, "generate"))
        .await
        .unwrap();
    assert_eq!(generated.final_step().runtime_context, runtime_context);
    generate_session.destroy().await.unwrap();

    let stream_session = new_session(&agent).await;
    let streamed = agent.stream(call(&stream_session, "stream")).await.unwrap();
    streamed.consume_stream().await;
    assert_eq!(
        streamed.final_step().await.unwrap().runtime_context,
        runtime_context
    );
    stream_session.destroy().await.unwrap();

    let continue_options = || HarnessAgentCreateSessionOptions {
        continue_from: Some(mock_continue_state()),
        ..with_sandbox(&make_sandbox_session())
    };
    let continue_generate_session = agent.create_session(continue_options()).await.unwrap();
    let continued_generation = agent
        .continue_generate(HarnessAgentContinueParameters::new(
            continue_generate_session.clone(),
        ))
        .await
        .unwrap();
    assert_eq!(
        continued_generation.final_step().runtime_context,
        runtime_context
    );
    continue_generate_session.destroy().await.unwrap();

    let continue_stream_session = agent.create_session(continue_options()).await.unwrap();
    let continued_stream = agent
        .continue_stream(HarnessAgentContinueParameters::new(
            continue_stream_session.clone(),
        ))
        .await
        .unwrap();
    continued_stream.consume_stream().await;
    assert_eq!(
        continued_stream.final_step().await.unwrap().runtime_context,
        runtime_context
    );
    continue_stream_session.destroy().await.unwrap();

    assert_eq!(*lifecycle_contexts.lock(), vec![runtime_context.clone(); 4]);
    assert_eq!(
        *telemetry_contexts.lock(),
        vec![JsonValue::Object(runtime_context); 4]
    );
}

#[tokio::test]
async fn runs_lifecycle_callbacks_in_order_and_merges_settings_before_call_callbacks() {
    let builtin_tools = tools(vec![("bash", tool(&[("command", "string")]))]);
    let mut options = MockHarnessOptions::script(|| {
        let mut values = vec![
            json!({ "type": "stream-start", "modelId": "resolved-model" }),
            json!({ "type": "tool-call", "toolCallId": "call-1", "toolName": "bash", "input": json!({ "command": "pwd" }).to_string(), "providerExecuted": true }),
            json!({ "type": "tool-result", "toolCallId": "call-1", "toolName": "bash", "result": { "output": "/work" } }),
        ];
        values.extend(stop_finish());
        parts(values)
    });
    options.builtin_tools = builtin_tools;
    let mock = options.build();
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let call_ids: Arc<Mutex<Vec<String>>> = Arc::default();
    let step_from_callback: Arc<Mutex<Option<StepResult>>> = Arc::default();
    let final_step_from_callback: Arc<Mutex<Option<StepResult>>> = Arc::default();

    macro_rules! record {
        ($name:expr, $ty:ty) => {{
            let (events, call_ids) = (events.clone(), call_ids.clone());
            Some(callback(move |e: $ty| {
                events.lock().push($name.to_string());
                call_ids.lock().push(e.call_id.clone());
                async {}
            }))
        }};
    }
    use crate::harness::sdk::telemetry::{
        GenerateTextEndEvent, GenerateTextStartEvent, GenerateTextStepStartEvent,
        LanguageModelCallEndEvent, LanguageModelCallStartEvent, ToolExecutionEndEvent,
        ToolExecutionStartEvent,
    };

    let mut s = settings(&mock);
    s.model = Some("requested-model".into());
    s.on_start = record!("settings:start", GenerateTextStartEvent);
    s.on_step_start = record!("settings:step-start", GenerateTextStepStartEvent);
    let (ev, ids) = (events.clone(), call_ids.clone());
    s.on_language_model_call_start = Some(callback(move |e: LanguageModelCallStartEvent| {
        ev.lock().push(format!("model-start:{}", e.model_id));
        ids.lock().push(e.call_id);
        async {}
    }));
    let (ev, ids) = (events.clone(), call_ids.clone());
    s.on_language_model_call_end = Some(callback(move |e: LanguageModelCallEndEvent| {
        let last = e
            .content
            .last()
            .map(|c| to_json(c)["type"].as_str().unwrap().to_string());
        ev.lock()
            .push(format!("model-end:{}", last.unwrap_or_default()));
        ids.lock().push(e.call_id);
        async {}
    }));
    s.on_tool_execution_start = record!("settings:tool-start", ToolExecutionStartEvent);
    let (ev, ids) = (events.clone(), call_ids.clone());
    s.on_tool_execution_end = Some(callback(move |e: ToolExecutionEndEvent| {
        ev.lock().push(format!(
            "settings:tool-end:{}",
            to_json(&e.tool_output)["type"].as_str().unwrap()
        ));
        ids.lock().push(e.call_id);
        async {}
    }));
    let (ev, ids, step) = (events.clone(), call_ids.clone(), step_from_callback.clone());
    s.on_step_end = Some(callback(move |e: StepResult| {
        ev.lock().push("settings:step-end".into());
        ids.lock().push(e.call_id.clone());
        *step.lock() = Some(e);
        async {}
    }));
    let (ev, ids, final_step) = (
        events.clone(),
        call_ids.clone(),
        final_step_from_callback.clone(),
    );
    s.on_end = Some(callback(move |e: GenerateTextEndEvent| {
        ev.lock().push("settings:end".into());
        ids.lock().push(e.call_id.clone());
        *final_step.lock() = Some(e.final_step);
        async {}
    }));
    let agent = agent(s);
    let session = new_session(&agent).await;

    let mut params = call(&session, "run pwd");
    params.on_start = record!("call:start", GenerateTextStartEvent);
    params.on_step_start = record!("call:step-start", GenerateTextStepStartEvent);
    params.on_tool_execution_start = record!("call:tool-start", ToolExecutionStartEvent);
    params.on_tool_execution_end = record!("call:tool-end", ToolExecutionEndEvent);
    params.on_step_end = record!("call:step-end", StepResult);
    params.on_end = record!("call:end", GenerateTextEndEvent);
    let result = agent.generate(params).await.unwrap();

    assert_eq!(
        *events.lock(),
        vec![
            "settings:start",
            "call:start",
            "settings:step-start",
            "call:step-start",
            "model-start:resolved-model",
            "settings:tool-start",
            "call:tool-start",
            "settings:tool-end:tool-result",
            "call:tool-end",
            "model-end:tool-result",
            "settings:step-end",
            "call:step-end",
            "settings:end",
            "call:end",
        ]
    );
    let unique: std::collections::HashSet<String> = call_ids.lock().iter().cloned().collect();
    assert_eq!(unique.len(), 1);
    assert_eq!(Some(&result.steps[0]), step_from_callback.lock().as_ref());
    assert_eq!(
        Some(result.final_step()),
        final_step_from_callback.lock().as_ref()
    );
    assert_eq!(
        to_json(&result.final_step().model),
        json!({ "provider": "harness:mock", "modelId": "resolved-model" })
    );
    session.destroy().await.unwrap();
}

// [rust] 'ignores lifecycle callback failures' is not ported: a Rust
// lifecycle `Callback` returns `()`, so it cannot fail.

#[test]
fn exposes_the_ai_sdk_agent_contract_surface() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let mut s = settings(&mock);
    s.id = Some("a1".into());
    let agent = agent(s);
    assert_eq!(agent.version, "agent-v1");
    assert_eq!(agent.id.as_deref(), Some("a1"));
    assert_eq!(agent.harness_id(), "mock");
    assert!(agent.tools.is_empty());
}

#[test]
fn rejects_a_caller_defined_question_tool_when_the_harness_owns_that_name() {
    let mut options = MockHarnessOptions::script(Vec::new);
    options.builtin_tools = tools(vec![("askUserQuestions", tool(&[("questions", "array")]))]);
    let mock = options.build();
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("askUserQuestions", tool(&[]))]));

    assert_eq!(
        err_message(HarnessAgent::new(s)),
        "HarnessAgent tool name 'askUserQuestions' is reserved for harness question requests."
    );
}

#[test]
fn allows_that_caller_defined_name_when_the_harness_has_no_question_tool() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("askUserQuestions", tool(&[]))]));

    assert!(HarnessAgent::new(s).is_ok());
}

#[tokio::test]
async fn passes_the_configured_model_to_each_turn() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let mut s = settings(&mock);
    s.model = Some("harness-specific-model".into());
    let agent = agent(s);

    let session = new_session(&agent).await;
    agent.generate(call(&session, "Hello")).await.unwrap();

    // [rust] `HarnessV1StartOptions` has no `model` field, so `doStart`
    // cannot receive one.
    assert_eq!(
        mock.record.prompt_turns.lock()[0].settings.model.as_deref(),
        Some("harness-specific-model")
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn passes_an_undefined_model_to_a_turn_when_no_model_is_configured() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let agent = agent(settings(&mock));

    let session = new_session(&agent).await;
    agent.generate(call(&session, "Hello")).await.unwrap();

    assert_eq!(mock.record.prompt_turns.lock()[0].settings.model, None);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn normalizes_and_snapshots_headers_before_passing_them_to_do_start() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let mut s = settings(&mock);
    // [rust] The settings own their headers, so the caller cannot mutate
    // them after construction.
    s.headers = Some(IndexMap::from([
        ("X-Tenant".to_string(), Some("acme".to_string())),
        ("X-Optional".to_string(), None),
    ]));
    let agent = agent(s);

    let session = new_session(&agent).await;

    assert_eq!(
        mock.start_call(0).headers,
        Some(IndexMap::from([(
            "x-tenant".to_string(),
            "acme".to_string()
        )]))
    );
    session.destroy().await.unwrap();
}

#[test]
fn rejects_the_managed_header() {
    for header in [
        "authorization",
        "Authorization",
        "x-api-key",
        "X-API-Key",
        "user-agent",
        "User-Agent",
        "x-client-app",
        "X-Client-App",
    ] {
        let mock = MockHarnessOptions::script(finish_events).build();
        let mut s = settings(&mock);
        s.headers = Some(IndexMap::from([(
            header.to_string(),
            Some("caller-value".to_string()),
        )]));

        assert_eq!(
            err_message(HarnessAgent::new(s)),
            format!(
                "HarnessAgent: `headers` must not include the managed header `{}`.",
                header.to_lowercase()
            )
        );
    }
}

#[test]
fn rejects_a_managed_header_with_an_undefined_value() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let mut s = settings(&mock);
    s.headers = Some(IndexMap::from([("Authorization".to_string(), None)]));

    assert_eq!(
        err_message(HarnessAgent::new(s)),
        "HarnessAgent: `headers` must not include the managed header `authorization`."
    );
}

#[tokio::test]
async fn passes_stable_headers_when_resuming_a_session() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let sandbox = make_sandbox_session();
    let mut s = settings(&mock);
    s.headers = Some(IndexMap::from([(
        "x-tenant".to_string(),
        Some("acme".to_string()),
    )]));
    let agent = agent(s);
    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("session-1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let resume_from = session.stop().await.unwrap();

    let resumed_session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("session-1".into()),
            resume_from: Some(resume_from.clone()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();

    let second = mock.start_call(1);
    assert_eq!(
        second.headers,
        Some(IndexMap::from([(
            "x-tenant".to_string(),
            "acme".to_string()
        )]))
    );
    assert_eq!(to_json(&second.resume_from), to_json(&Some(resume_from)));
    resumed_session.destroy().await.unwrap();
}
