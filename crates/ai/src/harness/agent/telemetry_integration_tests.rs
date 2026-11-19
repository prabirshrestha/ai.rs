//! Port of `agent/telemetry-integration.test.ts`.
//!
//! [rust] The OpenTelemetry span tests ("filters runtime context from
//! exported spans", "exports normalized OpenTelemetry spans from
//! HarnessAgent turns", "omits turn input attributes when recordInputs is
//! false") are not ported: they exercise `@ai-sdk/otel`, a separate package
//! with no ai.rs counterpart.

use super::*;
use crate::harness::sdk::telemetry::{
    SharedTelemetry, Telemetry, TelemetryEvent, TelemetryOptions,
    clear_global_telemetry_integrations, register_telemetry,
};

fn usage() -> JsonValue {
    json!({
        "inputTokens": { "total": 5, "noCache": 5 },
        "outputTokens": { "total": 2, "text": 2 },
    })
}

fn stop() -> JsonValue {
    json!({ "unified": "stop", "raw": "stop" })
}

fn finish_step(reason: JsonValue) -> JsonValue {
    json!({ "type": "finish-step", "finishReason": reason, "usage": usage() })
}

fn finish() -> JsonValue {
    json!({ "type": "finish", "finishReason": stop(), "totalUsage": usage() })
}

fn text(id: &str, delta: &str) -> Vec<JsonValue> {
    vec![
        json!({ "type": "text-start", "id": id }),
        json!({ "type": "text-delta", "id": id, "delta": delta }),
        json!({ "type": "text-end", "id": id }),
    ]
}

fn completed_turn_script(model_id: &str) -> Vec<JsonValue> {
    let mut script = vec![json!({ "type": "stream-start", "modelId": model_id })];
    script.extend(text("t1", "done"));
    script.push(finish_step(stop()));
    script.push(finish());
    script
}

/// `scriptedHarness(script)`.
fn scripted_harness(script: Vec<JsonValue>) -> Mock {
    let script = parts(script);
    MockHarnessOptions::script(move || script.clone()).build()
}

/// `recordingIntegration()`.
#[derive(Default)]
struct Recorder {
    /// `(method, callId)`.
    calls: Mutex<Vec<(String, JsonValue)>>,
    events: Mutex<IndexMap<String, Vec<JsonValue>>>,
}

impl Recorder {
    fn rec(&self, method: &str, event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let event = JsonValue::Object(event.clone());
        self.calls.lock().push((
            method.to_string(),
            event.get("callId").cloned().unwrap_or(JsonValue::Null),
        ));
        self.events
            .lock()
            .entry(method.to_string())
            .or_default()
            .push(event);
        Box::pin(async {})
    }
    fn methods(&self) -> Vec<String> {
        self.calls.lock().iter().map(|(m, _)| m.clone()).collect()
    }
    fn last(&self, method: &str) -> JsonValue {
        self.events.lock()[method].last().unwrap().clone()
    }
    fn list(&self, method: &str) -> Vec<JsonValue> {
        self.events.lock()[method].clone()
    }
}

struct Recording(Arc<Recorder>);

impl Telemetry for Recording {
    fn on_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onStart", e)
    }
    fn on_step_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onStepStart", e)
    }
    fn on_language_model_call_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onLanguageModelCallStart", e)
    }
    fn on_language_model_call_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onLanguageModelCallEnd", e)
    }
    fn on_tool_execution_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onToolExecutionStart", e)
    }
    fn on_tool_execution_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onToolExecutionEnd", e)
    }
    fn on_step_finish(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onStepFinish", e)
    }
    fn on_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.0.rec("onEnd", e)
    }
}

fn recording_integration() -> (SharedTelemetry, Arc<Recorder>) {
    let recorder = Arc::new(Recorder::default());
    (Arc::new(Recording(recorder.clone())), recorder)
}

fn with_integration(integration: SharedTelemetry) -> Option<TelemetryOptions> {
    Some(TelemetryOptions {
        integrations: Some(vec![integration]),
        ..Default::default()
    })
}

/// Drop `null`s recursively: TS `toEqual` treats `undefined` keys as absent.
fn defined(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(o) => JsonValue::Object(
            o.iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), defined(v)))
                .collect(),
        ),
        JsonValue::Array(a) => JsonValue::Array(a.iter().map(defined).collect()),
        other => other.clone(),
    }
}

const LIFECYCLE: [&str; 6] = [
    "onStart",
    "onStepStart",
    "onLanguageModelCallStart",
    "onLanguageModelCallEnd",
    "onStepFinish",
    "onEnd",
];

#[tokio::test]
async fn delivers_global_telemetry_without_per_call_settings() {
    let (integration, recorder) = recording_integration();
    register_telemetry([integration]);
    let mock = scripted_harness(completed_turn_script("global-model"));
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.generate(call(&session, "go")).await;
    clear_global_telemetry_integrations();
    session.destroy().await.unwrap();

    assert_eq!(result.unwrap().text(), "done");
    // [rust] The global registry is process-wide and tests run in
    // parallel, so keep only this turn's events (by its callId).
    let call_id = recorder
        .list("onStart")
        .into_iter()
        .find(|e| e["modelId"] == "global-model")
        .map(|e| e["callId"].clone())
        .expect("onStart for this turn");
    let methods: Vec<String> = recorder
        .calls
        .lock()
        .iter()
        .filter(|(_, id)| *id == call_id)
        .map(|(m, _)| m.clone())
        .collect();
    assert_eq!(methods, LIFECYCLE);
}

#[tokio::test]
async fn filters_runtime_context_for_direct_integrations() {
    for (name, include, expected) in [
        ("when inclusion is omitted", None, json!({})),
        (
            "when the property is disabled",
            Some(IndexMap::from([("conversationId".to_string(), false)])),
            json!({}),
        ),
        (
            "when the property is enabled",
            Some(IndexMap::from([("conversationId".to_string(), true)])),
            json!({ "conversationId": "conversation-1" }),
        ),
    ] {
        let mock = scripted_harness(completed_turn_script("context-model"));
        let (integration, recorder) = recording_integration();
        let mut s = settings(&mock);
        s.runtime_context = Some(obj(json!({ "conversationId": "conversation-1" })));
        s.telemetry = Some(TelemetryOptions {
            integrations: Some(vec![integration]),
            include_runtime_context: include,
            ..Default::default()
        });
        let agent = agent(s);
        let session = new_session(&agent).await;
        agent.generate(call(&session, "go")).await.unwrap();
        session.destroy().await.unwrap();

        let end = recorder.last("onEnd");
        let mut contexts = vec![
            recorder.last("onStart")["runtimeContext"].clone(),
            recorder.last("onStepStart")["runtimeContext"].clone(),
            recorder.last("onStepFinish")["runtimeContext"].clone(),
            end["runtimeContext"].clone(),
            end["finalStep"]["runtimeContext"].clone(),
        ];
        contexts.extend(
            end["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|step| step["runtimeContext"].clone()),
        );
        assert_eq!(contexts, vec![expected; 6], "{name}");
    }
}

#[tokio::test]
async fn drives_the_telemetry_lifecycle_from_the_turn_stream() {
    let mut script = vec![json!({ "type": "stream-start" })];
    script.extend(text("t1", "hi"));
    script.extend([
        json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "bash", "input": "{\"command\":\"ls\"}", "providerExecuted": true }),
        json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "bash", "result": { "output": "ok" } }),
        json!({
            "type": "finish-step",
            "finishReason": stop(),
            "usage": usage(),
            "harnessMetadata": { "gateway": { "generationId": "generation-id" } },
        }),
        finish(),
    ]);
    let mock = scripted_harness(script);
    let (integration, recorder) = recording_integration();
    let mut s = settings(&mock);
    s.telemetry = with_integration(integration);
    let agent = agent(s);
    let session = new_session(&agent).await;
    assert!(recorder.calls.lock().is_empty());
    agent.generate(call(&session, "go")).await.unwrap();
    session.destroy().await.unwrap();

    assert_eq!(
        recorder.methods(),
        [
            "onStart",
            "onStepStart",
            "onLanguageModelCallStart",
            "onToolExecutionStart",
            "onToolExecutionEnd",
            "onLanguageModelCallEnd",
            "onStepFinish",
            "onEnd",
        ]
    );

    // The model's output content reaches the inference-call end (gen_ai
    // output messages): the streamed text and the tool-call, captured
    // non-lossily.
    let lm_end = recorder.last("onLanguageModelCallEnd");
    let content = lm_end["content"].as_array().unwrap();
    assert_eq!(content.len(), 3, "{content:?}");
    assert_eq!(content[0], json!({ "type": "text", "text": "hi" }));
    assert!(matches(
        &content[1],
        &json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "bash", "input": { "command": "ls" } })
    ));
    assert!(matches(
        &content[2],
        &json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "bash", "output": { "output": "ok" } })
    ));
    let performance = &lm_end["performance"];
    assert!(performance["responseTimeMs"].is_number(), "{performance}");
    assert!(performance["timeToFirstOutputMs"].is_null());
    assert!(performance["timeBetweenOutputChunksMs"].is_null());
    assert_eq!(lm_end["finishReason"], "stop");
    assert_eq!(
        lm_end["providerMetadata"],
        json!({ "gateway": { "generationId": "generation-id" } })
    );
    assert_eq!(
        defined(&lm_end["usage"]),
        json!({
            "inputTokens": 5,
            "inputTokenDetails": { "noCacheTokens": 5 },
            "outputTokens": 2,
            "outputTokenDetails": { "textTokens": 2 },
            "totalTokens": 7,
        })
    );

    let end = recorder.last("onEnd");
    assert_eq!(end["text"], "hi");
    assert_eq!(end["finalStep"]["reasoning"], json!([]));
    assert_eq!(
        end["finalStep"]["providerMetadata"],
        json!({ "gateway": { "generationId": "generation-id" } })
    );
    let tool_calls = end["toolCalls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert!(matches(
        &tool_calls[0],
        &json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "bash", "input": { "command": "ls" } })
    ));
    assert_eq!(end["files"], json!([]));
    assert_eq!(end["finishReason"], "stop");
    assert_eq!(defined(&end["usage"]), defined(&lm_end["usage"]));
    // The input prompt is on the operation start (gen_ai input messages).
    assert_eq!(
        recorder.last("onStart")["messages"],
        json!([{ "role": "user", "content": "go" }])
    );
    // Every lifecycle event shares one correlating callId.
    let call_ids: std::collections::HashSet<String> = recorder
        .calls
        .lock()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    assert_eq!(call_ids.len(), 1);
    let call_id = &recorder.calls.lock()[0].1;
    assert!(call_id.as_str().is_some_and(|id| !id.is_empty()));
}

#[tokio::test]
async fn reports_fresh_turn_settings_without_exposing_skills() {
    let mut script = vec![json!({ "type": "stream-start" })];
    script.extend(text("t1", "done"));
    script.push(finish_step(stop()));
    script.push(finish());
    let mock = scripted_harness(script);
    let echo = tool(&[("value", "string")]).with_description("Echo a value.");
    let reverse = tool(&[("value", "string")]).with_description("Reverse a value.");
    let (integration, recorder) = recording_integration();
    let mut s = settings(&mock);
    s.telemetry = with_integration(integration);
    s.call_options_schema = Some(z_object(&[("tenant", "enum:alpha|beta")]));
    s.prepare_call = Some(Arc::new(move |mut call: HarnessAgentPrepareCallOptions| {
        let tenant = call.options.as_ref().unwrap()["tenant"]
            .as_str()
            .unwrap()
            .to_string();
        call.model = Some(format!("model-{tenant}"));
        call.skills = Some(vec![HarnessV1Skill {
            name: format!("{tenant}-skill"),
            description: format!("{tenant} skill"),
            content: format!("{tenant} skill instructions"),
            files: None,
        }]);
        call.instructions = Some(format!("{tenant} instructions").into());
        call.tools = Some(if tenant == "alpha" {
            tools(vec![("echo", echo.clone())])
        } else {
            tools(vec![("reverse", reverse.clone())])
        });
        Box::pin(async move { Ok(call) })
    }));
    let agent = agent(s);
    let session = new_session(&agent).await;
    for (prompt, tenant) in [("first", "alpha"), ("second", "beta")] {
        let mut params = call(&session, prompt);
        params.options = Some(json!({ "tenant": tenant }));
        agent.generate(params).await.unwrap();
    }
    session.destroy().await.unwrap();

    let summary = |events: Vec<JsonValue>| -> Vec<JsonValue> {
        events
            .iter()
            .map(|e| {
                json!({
                    "modelId": e["modelId"],
                    "instructions": e["instructions"],
                    "toolNames": e["tools"].as_object().map(|t| t.keys().cloned().collect::<Vec<_>>()),
                    "activeTools": e["activeTools"],
                })
            })
            .collect()
    };
    let starts = summary(recorder.list("onStart"));
    assert_eq!(
        starts,
        vec![
            json!({ "activeTools": ["echo"], "instructions": "alpha instructions", "modelId": "model-alpha", "toolNames": ["echo"] }),
            json!({ "activeTools": ["reverse"], "instructions": "beta instructions", "modelId": "model-beta", "toolNames": ["reverse"] }),
        ]
    );
    assert_eq!(summary(recorder.list("onStepStart")), starts);

    let model_call_starts: Vec<JsonValue> = recorder
        .list("onLanguageModelCallStart")
        .iter()
        .map(|e| json!({ "modelId": e["modelId"], "instructions": e["instructions"], "tools": e["tools"] }))
        .collect();
    // [rust] zod's JSON Schema adds `"$schema": draft-07`; the `z_object`
    // stand-in emits the schema without it.
    let definition = |name: &str, description: &str| {
        json!({
            "description": description,
            "inputSchema": {
                "additionalProperties": false,
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
                "type": "object",
            },
            "name": name,
            "type": "function",
        })
    };
    assert_eq!(
        model_call_starts,
        vec![
            json!({ "instructions": "alpha instructions", "modelId": "model-alpha", "tools": [definition("echo", "Echo a value.")] }),
            json!({ "instructions": "beta instructions", "modelId": "model-beta", "tools": [definition("reverse", "Reverse a value.")] }),
        ]
    );
    assert!(
        recorder
            .events
            .lock()
            .values()
            .flatten()
            .all(|event| event.get("skills").is_none())
    );
}

#[tokio::test]
async fn uses_final_step_text_and_reasoning_on_the_telemetry_end_event() {
    let reasoning = |id: &str, delta: &str| {
        vec![
            json!({ "type": "reasoning-start", "id": id }),
            json!({ "type": "reasoning-delta", "id": id, "delta": delta }),
            json!({ "type": "reasoning-end", "id": id }),
        ]
    };
    let mut script = vec![json!({ "type": "stream-start" })];
    script.extend(text("t1", "draft"));
    script.extend(reasoning("r1", "first thought"));
    script.push(finish_step(
        json!({ "unified": "tool-calls", "raw": "tool_calls" }),
    ));
    script.extend(text("t2", "final"));
    script.extend(reasoning("r2", "final thought"));
    script.push(finish_step(stop()));
    script.push(finish());
    let mock = scripted_harness(script);
    let (integration, recorder) = recording_integration();
    let mut s = settings(&mock);
    s.telemetry = with_integration(integration);
    let agent = agent(s);
    let session = new_session(&agent).await;
    agent.generate(call(&session, "go")).await.unwrap();
    session.destroy().await.unwrap();

    let end = recorder.last("onEnd");
    assert_eq!(end["text"], "final");
    assert_eq!(
        end["finalStep"]["reasoning"],
        json!([{ "type": "reasoning", "text": "final thought" }])
    );
}

#[tokio::test]
async fn fires_no_telemetry_when_settings_telemetry_is_unset() {
    let mock = scripted_harness(vec![json!({ "type": "stream-start" }), finish()]);
    // Integration created, but the agent has no telemetry settings → opt-out.
    let (_integration, recorder) = recording_integration();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    agent.generate(call(&session, "go")).await.unwrap();
    session.destroy().await.unwrap();
    assert!(recorder.calls.lock().is_empty());
}
