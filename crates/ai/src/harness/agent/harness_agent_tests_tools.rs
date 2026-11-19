//! `harness-agent.test.ts`: host tools, tool context, and tool filtering.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::agent::internal::run_prompt::tests::tool_outputs::{
    Calls, recording_tool, submitted_json,
};
use crate::harness::sdk::parse_tool_call::NoSuchToolError;
use crate::harness::sdk::telemetry::{ToolExecutionEndEvent, callback};
use crate::harness::sdk::tool::Tool;
use crate::harness::spec::tools::HarnessV1BuiltinToolFiltering;

fn tool_call(id: &str, name: &str, input: JsonValue) -> JsonValue {
    json!({ "type": "tool-call", "toolCallId": id, "toolName": name, "input": input.to_string() })
}

fn with_finish(values: Vec<JsonValue>) -> Vec<HarnessV1StreamPart> {
    let mut script = parts(values);
    script.extend(finish_events());
    script
}

fn submissions(mock: &Mock) -> Vec<JsonValue> {
    mock.tool_results().iter().map(submitted_json).collect()
}

#[tokio::test]
async fn host_side_tools_are_executed_and_the_result_is_submitted_back() {
    let mock = MockHarnessOptions::script(|| {
        let mut values = vec![tool_call("c1", "echo", json!({ "value": "ping" }))];
        values.extend([
            json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": zero_usage() }),
            json!({ "type": "finish", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "totalUsage": zero_usage() }),
        ]);
        parts(values)
    })
    .build();

    let echo = tool(&[("value", "string")])
        .with_description("Echo a string")
        .with_execute(|input, _| async move { Ok(json!({ "echoed": input["value"] })) });

    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("echo", echo)]));
    let agent = agent(s);
    let session = new_session(&agent).await;
    let result = agent.generate(call(&session, "go")).await.unwrap();

    assert_eq!(
        submissions(&mock),
        vec![json!({ "toolCallId": "c1", "output": { "echoed": "ping" } })]
    );
    let tool_calls = result.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(to_json(&tool_calls[0])["toolName"], "echo");

    session.destroy().await.unwrap();
}

/// `lookupAccount`: a tool with a `{ userId }` context schema that returns
/// its context's `userId`, recording each call's input and context.
fn lookup_account() -> (Tool, Calls, Calls) {
    let (inputs, contexts): (Calls, Calls) = (Arc::default(), Arc::default());
    let (i, c) = (inputs.clone(), contexts.clone());
    let tool = Tool::new(z_object(&[]))
        .with_context_schema(z_object(&[("userId", "string")]))
        .with_execute(move |input, options| {
            i.lock().push(input);
            c.lock().push(options.context.clone());
            async move { Ok(json!({ "userId": options.context["userId"] })) }
        });
    (tool, inputs, contexts)
}

fn user_context(user_id: JsonValue) -> serde_json::Map<String, JsonValue> {
    obj(json!({ "lookupAccount": { "userId": user_id } }))
}

/// Settings whose `prepareCall` binds `lookupAccount`'s context to the
/// call's `userId` option.
fn context_agent(mock: &Mock, lookup: Tool) -> HarnessAgent {
    let mut s = settings(mock);
    s.tools = Some(tools(vec![("lookupAccount", lookup)]));
    s.tools_context = Some(user_context(json!("initial-user")));
    s.call_options_schema = Some(z_object(&[("userId", "string")]));
    s.prepare_call = Some(Arc::new(|mut call: HarnessAgentPrepareCallOptions| {
        let user_id = call.options.as_ref().unwrap()["userId"].clone();
        call.tools_context = user_context(user_id);
        Box::pin(async move { Ok(call) })
    }));
    agent(s)
}

#[tokio::test]
async fn passes_prepare_call_tool_context_to_host_tools_and_step_results() {
    let mock = MockHarnessOptions::script(|| {
        with_finish(vec![tool_call("c1", "lookupAccount", json!({}))])
    })
    .build();
    let (lookup, inputs, contexts) = lookup_account();
    let agent = context_agent(&mock, lookup);
    let session = new_session(&agent).await;

    let mut params = call(&session, "go");
    params.options = Some(json!({ "userId": "user-123" }));
    let result = agent.generate(params).await.unwrap();

    assert_eq!(*inputs.lock(), vec![json!({})]);
    assert_eq!(*contexts.lock(), vec![json!({ "userId": "user-123" })]);
    assert_eq!(
        submissions(&mock),
        vec![json!({ "toolCallId": "c1", "output": { "userId": "user-123" } })]
    );
    assert_eq!(
        result.steps[0].tools_context,
        user_context(json!("user-123"))
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn rebinds_prepare_call_tool_context_after_recreating_a_suspended_session() {
    let initial_prompt_done = Gate::default();
    let mut options = MockHarnessOptions::script(Vec::new)
        .with_continue(|| with_finish(vec![tool_call("c1", "lookupAccount", json!({}))]));
    let gate = initial_prompt_done.clone();
    options.prompt_done = Some(Arc::new(move |_| {
        let gate = gate.clone();
        Box::pin(async move {
            gate.wait().await;
            Ok(())
        })
    }));
    let gate = initial_prompt_done.clone();
    options.on_suspend_turn = Some(Arc::new(move || {
        gate.open();
        Box::pin(async { Ok(()) })
    }));
    let mock = options.build();
    let (lookup, inputs, contexts) = lookup_account();
    let agent = context_agent(&mock, lookup);
    let sandbox = make_sandbox_session();
    let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();
    let mut params = call(&session, "go");
    params.options = Some(json!({ "userId": "user-123" }));
    let first = agent.stream(params).await.unwrap();
    let first_consumption = tokio::spawn(async move { first.consume_stream().await });
    let session_id = session.session_id().to_string();
    let continue_from = session.suspend_turn().await.unwrap();
    first_consumption.await.unwrap();

    // Context remains host-only instead of being serialized with turn state.
    assert!(
        to_json(&continue_from.turn_settings)
            .get("toolsContext")
            .is_none()
    );

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some(session_id),
            continue_from: Some(continue_from),
            tools_context: Some(user_context(json!("user-123"))),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    agent
        .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();

    assert_eq!(*inputs.lock(), vec![json!({})]);
    assert_eq!(*contexts.lock(), vec![json!({ "userId": "user-123" })]);
    assert_eq!(
        submissions(&mock),
        vec![json!({ "toolCallId": "c1", "output": { "userId": "user-123" } })]
    );

    session.destroy().await.unwrap();
}

fn guarded_lookup() -> (Tool, Calls) {
    let (tool, calls) = recording_tool(z_object(&[]), |_| Ok(json!({ "ok": true })));
    (
        tool.with_context_schema(z_object(&[("userId", "string")])),
        calls,
    )
}

#[tokio::test]
async fn rejects_missing_required_host_tool_context_before_execution() {
    let mock = MockHarnessOptions::script(|| {
        with_finish(vec![tool_call("c1", "lookupAccount", json!({}))])
    })
    .build();
    let (lookup, calls) = guarded_lookup();
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("lookupAccount", lookup)]));
    s.tools_context = Some(Default::default());
    let agent = agent(s);
    let session = new_session(&agent).await;

    agent.generate(call(&session, "go")).await.unwrap();

    assert!(calls.lock().is_empty());
    assert_eq!(
        submissions(&mock),
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "Tool context validation failed." }, "isError": true })
        ]
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn validates_host_tool_context_without_disclosing_it_to_the_model() {
    let mock = MockHarnessOptions::script(|| {
        with_finish(vec![tool_call("c1", "lookupAccount", json!({}))])
    })
    .build();
    let (lookup, calls) = guarded_lookup();
    let host_validation_error: Arc<Mutex<Option<String>>> = Arc::default();
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("lookupAccount", lookup)]));
    s.tools_context = Some(obj(
        json!({ "lookupAccount": { "userId": 123, "apiKey": "host-secret" } }),
    ));
    let error = host_validation_error.clone();
    s.on_tool_execution_end = Some(callback(move |event: ToolExecutionEndEvent| {
        let output = to_json(&event.tool_output);
        if output["type"] == "tool-error" {
            *error.lock() = Some(output["error"].to_string());
        }
        async {}
    }));
    let agent = agent(s);
    let session = new_session(&agent).await;

    agent.generate(call(&session, "go")).await.unwrap();

    assert!(calls.lock().is_empty());
    let submitted = submissions(&mock);
    assert_eq!(
        submitted,
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "Tool context validation failed." }, "isError": true })
        ]
    );
    assert!(!json!(submitted).to_string().contains("host-secret"));
    let host_error = host_validation_error.lock().clone().expect("a tool error");
    assert!(host_error.contains("host-secret"), "{host_error}");

    session.destroy().await.unwrap();
}

#[test]
fn rejects_active_tools_and_inactive_tools_together_at_runtime() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let mut s = settings(&mock);
    s.active_tools = Some(Vec::new());
    s.inactive_tools = Some(Vec::new());

    let message = err_message(HarnessAgent::new(s));
    assert!(
        message.contains("either `activeTools` or `inactiveTools`"),
        "{message}"
    );
}

#[test]
fn rejects_unknown_active_tool_names() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let mut s = settings(&mock);
    s.active_tools = Some(vec!["missing".into()]);

    let Err(error) = HarnessAgent::new(s) else {
        panic!("expected an error");
    };
    assert!(error.downcast_ref::<NoSuchToolError>().is_some(), "{error}");
}

fn echo_and_hidden() -> ToolSet {
    tools(vec![
        (
            "echo",
            tool(&[("value", "string")])
                .with_execute(|input, _| async move { Ok(json!({ "echoed": input["value"] })) }),
        ),
        (
            "hidden",
            tool(&[("value", "string")])
                .with_execute(|input, _| async move { Ok(json!({ "hidden": input["value"] })) }),
        ),
    ])
}

fn hidden_call_mock() -> Mock {
    MockHarnessOptions::script(|| {
        with_finish(vec![tool_call("c1", "hidden", json!({ "value": "ping" }))])
    })
    .build()
}

fn received_tool_names(mock: &Mock) -> Vec<String> {
    mock.record.prompt_turns.lock()[0]
        .settings
        .tools
        .iter()
        .map(|spec| spec.name.clone())
        .collect()
}

#[tokio::test]
async fn active_tools_filters_custom_tool_specs_and_blocks_inactive_custom_execution() {
    let mock = hidden_call_mock();
    let mut s = settings(&mock);
    s.tools = Some(echo_and_hidden());
    s.active_tools = Some(vec!["echo".into()]);
    let agent = agent(s);
    let session = new_session(&agent).await;

    agent.generate(call(&session, "go")).await.unwrap();

    assert_eq!(received_tool_names(&mock), vec!["echo"]);
    let denied = json!({
        "type": "execution-denied",
        "reason": "Tool 'hidden' is inactive due to the HarnessAgent tool filtering policy.",
    });
    assert_eq!(
        submissions(&mock),
        vec![json!({
            "toolCallId": "c1",
            "output": denied,
            "toolResult": { "type": "tool-result", "toolCallId": "c1", "toolName": "hidden", "output": denied },
        })]
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn inactive_tools_filters_custom_tool_specs_and_blocks_inactive_custom_execution() {
    let mock = hidden_call_mock();
    let mut s = settings(&mock);
    s.tools = Some(echo_and_hidden());
    s.inactive_tools = Some(vec!["hidden".into()]);
    let agent = agent(s);
    let session = new_session(&agent).await;

    agent.generate(call(&session, "go")).await.unwrap();

    assert_eq!(received_tool_names(&mock), vec!["echo"]);
    assert_eq!(mock.tool_results()[0].output["type"], "execution-denied");
    session.destroy().await.unwrap();
}

fn bash_builtins() -> ToolSet {
    tools(vec![("bash", tool(&[("command", "string")]))])
}

#[test]
fn rejects_builtin_filtering_when_the_harness_cannot_enforce_it() {
    let mut options = MockHarnessOptions::script(Vec::new);
    options.builtin_tools = bash_builtins();
    let mock = options.build();
    let mut s = settings(&mock);
    s.active_tools = Some(Vec::new());

    let Err(error) = HarnessAgent::new(s) else {
        panic!("expected an error");
    };
    assert!(matches!(
        error.downcast_ref::<HarnessError>(),
        Some(HarnessError::CapabilityUnsupported { .. })
    ));
}

#[tokio::test]
async fn passes_builtin_filtering_policy_to_approval_capable_harnesses() {
    let mut options = MockHarnessOptions::script(finish_events);
    options.builtin_tools = bash_builtins();
    options.supports_builtin_tool_approvals = Some(true);
    let mock = options.build();
    let mut s = settings(&mock);
    s.active_tools = Some(Vec::new());
    let agent = agent(s);
    let session = new_session(&agent).await;

    assert_eq!(
        mock.start_call(0).builtin_tool_filtering,
        Some(HarnessV1BuiltinToolFiltering::Allow {
            tool_names: Vec::new()
        })
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn auto_denies_inactive_builtin_approvals_without_approval_stream_parts() {
    let mut options = MockHarnessOptions::script(|| {
        with_finish(vec![
            json!({ "type": "tool-call", "toolCallId": "b1", "toolName": "bash", "input": json!({ "command": "pwd" }).to_string(), "providerExecuted": true }),
            json!({ "type": "tool-approval-request", "approvalId": "b1", "toolCallId": "b1" }),
        ])
    });
    options.builtin_tools = bash_builtins();
    options.supports_builtin_tool_approvals = Some(true);
    let mock = options.build();
    let mut s = settings(&mock);
    s.inactive_tools = Some(vec!["bash".into()]);
    let agent = agent(s);
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "go")).await.unwrap();
    let part_types: Vec<String> = futures::StreamExt::collect::<Vec<_>>(futures::StreamExt::map(
        result.full_stream(),
        |part| part.to_json()["type"].as_str().unwrap().to_string(),
    ))
    .await;

    assert!(!part_types.contains(&"tool-approval-request".to_string()));
    assert!(!part_types.contains(&"tool-approval-response".to_string()));
    assert_eq!(
        mock.tool_approvals(),
        vec![HarnessV1SubmitToolApproval {
            approval_id: "b1".into(),
            approved: false,
            reason: Some(
                "Tool 'bash' is inactive due to the HarnessAgent tool filtering policy.".into()
            ),
        }]
    );
    session.destroy().await.unwrap();
}
