//! `harness-agent.test.ts`: retained host results across approval pauses,
//! prompt normalization, and the session lifecycle methods.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use serde_json::json;

use super::*;
use crate::harness::agent::internal::run_prompt::tests::tool_outputs::{
    approval_config, recording_tool, submitted_json,
};
use crate::harness::sdk::messages::ModelMessage;
use crate::harness::spec::tools::HarnessV1PermissionMode;

fn submissions(mock: &Mock) -> Vec<JsonValue> {
    mock.tool_results().iter().map(submitted_json).collect()
}

fn host_input_step() -> JsonValue {
    json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "host-input" }, "usage": zero_usage() })
}

/// `[...responseMessages, { role: 'tool', content: [approval response] }]`.
fn with_approval(mut messages: Vec<ModelMessage>, approval_id: &str) -> Vec<ModelMessage> {
    messages.push(
        serde_json::from_value(json!({
            "role": "tool",
            "content": [{ "type": "tool-approval-response", "approvalId": approval_id, "approved": true }],
        }))
        .unwrap(),
    );
    messages
}

fn approval(approval_id: &str) -> HarnessV1SubmitToolApproval {
    HarnessV1SubmitToolApproval {
        approval_id: approval_id.into(),
        approved: true,
        reason: None,
    }
}

#[tokio::test]
async fn retains_an_unechoed_host_result_across_repeated_approval_pauses_and_session_resume() {
    let tickets = json!([{ "id": "AI-23503" }]);
    let gates = PromptGates::default();
    let continuation_count = Arc::new(AtomicUsize::new(0));
    let host_call = json!({ "type": "tool-call", "toolCallId": "host-1", "toolName": "listProjectTickets", "input": "{}", "providerExecuted": false });
    let script_host_call = host_call.clone();
    let continue_host_call = host_call.clone();
    let continue_tickets = tickets.clone();
    let mut options = MockHarnessOptions::script(move || {
        parts(vec![
            json!({ "type": "stream-start", "modelId": "claude-code-sequence" }),
            json!({ "type": "tool-call", "toolCallId": "bash-1", "toolName": "bash", "nativeName": "Bash", "input": "{\"command\":\"pwd\"}", "providerExecuted": true }),
            script_host_call.clone(),
            json!({ "type": "tool-approval-request", "approvalId": "bash-1", "toolCallId": "bash-1" }),
            host_input_step(),
        ])
    })
    .with_continue(move || {
        if continuation_count.fetch_add(1, Ordering::SeqCst) == 0 {
            return parts(vec![
                continue_host_call.clone(),
                json!({ "type": "tool-call", "toolCallId": "bash-2", "toolName": "bash", "nativeName": "Bash", "input": "{\"command\":\"git status\"}", "providerExecuted": true }),
                json!({ "type": "tool-approval-request", "approvalId": "bash-2", "toolCallId": "bash-2" }),
                host_input_step(),
            ]);
        }
        let host_result = json!({ "type": "tool-result", "toolCallId": "host-1", "toolName": "listProjectTickets", "result": continue_tickets });
        let mut script = parts(vec![
            continue_host_call.clone(),
            host_result.clone(),
            host_result,
            continue_host_call.clone(),
        ]);
        script.extend(finish_events());
        script
    });
    options.builtin_tools = tools(vec![("bash", tool(&[("command", "string")]))]);
    options.supports_builtin_tool_approvals = Some(true);
    options.prompt_done = Some(gates.prompt_done());
    let mock = options.build();
    let result_tickets = tickets.clone();
    let (list_tickets, executions) =
        recording_tool(z_object(&[]), move |_| Ok(result_tickets.clone()));
    let mut s = settings(&mock);
    s.permission_mode = Some(HarnessV1PermissionMode::AllowReads);
    s.tools = Some(tools(vec![("listProjectTickets", list_tickets)]));
    s.tool_approval = Some(approval_config(json!({ "listProjectTickets": "approved" })));
    let agent = agent(s);
    let sandbox = make_sandbox_session();
    let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();

    let first = agent
        .stream(call(
            &session,
            "Run Bash and list the project tickets in the same response.",
        ))
        .await
        .unwrap();
    first.consume_stream().await;

    assert!(session.has_unfinished_turn());
    assert_eq!(executions.lock().len(), 1);
    let host_submission = json!({ "toolCallId": "host-1", "output": tickets });
    assert_eq!(submissions(&mock), vec![host_submission.clone()]);
    let first_content = to_json(&first.steps().await.unwrap()[0].content);
    assert!(
        matches(
            &first_content,
            &json!([
                { "type": "tool-call", "toolCallId": "bash-1" },
                { "type": "tool-call", "toolCallId": "host-1" },
                { "type": "tool-approval-request", "approvalId": "bash-1" },
                { "type": "tool-result", "toolCallId": "host-1", "output": tickets },
            ])
        ),
        "{first_content}"
    );

    let continued = agent
        .stream(
            HarnessAgentCallParameters::new(session.clone()).with_messages(with_approval(
                first.response_messages().await.unwrap(),
                "bash-1",
            )),
        )
        .await
        .unwrap();
    continued.consume_stream().await;

    assert_eq!(mock.tool_approvals(), vec![approval("bash-1")]);
    assert_eq!(executions.lock().len(), 1);
    assert!(session.has_unfinished_turn());
    assert_eq!(
        submissions(&mock),
        vec![host_submission.clone(), host_submission.clone()]
    );
    assert!(
        continued
            .steps()
            .await
            .unwrap()
            .iter()
            .all(|step| step.tool_results().is_empty())
    );

    let resume_from = session.detach().await.unwrap();
    assert_eq!(
        to_json(
            &resume_from
                .continue_from
                .as_ref()
                .unwrap()
                .pending_tool_results
        ),
        json!([{
            "toolCallId": "host-1",
            "toolName": "listProjectTickets",
            "input": "{}",
            "completedResult": { "output": tickets },
        }])
    );
    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            resume_from: Some(resume_from),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let completed = agent
        .stream(
            HarnessAgentCallParameters::new(session.clone()).with_messages(with_approval(
                continued.response_messages().await.unwrap(),
                "bash-2",
            )),
        )
        .await
        .unwrap();
    let completed_parts: Vec<JsonValue> =
        completed.full_stream().map(|p| p.to_json()).collect().await;

    assert_eq!(executions.lock().len(), 1);
    let submitted = submissions(&mock);
    assert_eq!(submitted.len(), 3);
    assert_eq!(submitted[2], submitted[0]);
    assert_eq!(
        mock.tool_approvals(),
        vec![approval("bash-1"), approval("bash-2")]
    );
    assert!(!session.has_unfinished_turn());
    assert!(
        completed_parts
            .iter()
            .all(|part| part.get("toolCallId").is_none_or(|id| id != "host-1"))
    );
    assert_eq!(
        to_json(&completed.finish_reason().await.unwrap()),
        json!("tool-calls")
    );

    gates.finish_all();
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn retains_a_host_approval_result_across_a_subsequent_approval_pause_and_session_resume() {
    let continuation_count = Arc::new(AtomicUsize::new(0));
    let mock = MockHarnessOptions::script(|| {
        parts(vec![
            json!({ "type": "tool-call", "toolCallId": "host-1", "toolName": "first_write", "input": json!({ "value": "first" }).to_string() }),
            host_input_step(),
        ])
    })
    .with_continue(move || {
        if continuation_count.fetch_add(1, Ordering::SeqCst) == 0 {
            return parts(vec![
                json!({ "type": "tool-call", "toolCallId": "host-2", "toolName": "second_write", "input": json!({ "value": "second" }).to_string() }),
                host_input_step(),
            ]);
        }
        finish_events()
    })
    .build();
    let echo_value = |input: &JsonValue| Ok(json!({ "value": input["value"] }));
    let (first_write, first_executions) =
        recording_tool(z_object(&[("value", "string")]), echo_value);
    let (second_write, second_executions) =
        recording_tool(z_object(&[("value", "string")]), echo_value);
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![
        ("first_write", first_write),
        ("second_write", second_write),
    ]));
    s.tool_approval = Some(approval_config(
        json!({ "first_write": "user-approval", "second_write": "user-approval" }),
    ));
    let agent = agent(s);
    let sandbox = make_sandbox_session();
    let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();

    let approval_ids = |result: &crate::harness::agent::internal::harness_stream_text_result::HarnessStreamTextResult| {
        result.full_stream().filter_map(|part| async move {
            let part = part.to_json();
            (part["type"] == "tool-approval-request").then(|| part["approvalId"].as_str().unwrap().to_string())
        })
    };

    let first = agent
        .stream(call(&session, "Write both values."))
        .await
        .unwrap();
    let first_approval_id = approval_ids(&first).collect::<Vec<_>>().await.pop();
    let first_approval_id = first_approval_id.expect("a first approval request");

    let continued = agent
        .stream(
            HarnessAgentCallParameters::new(session.clone()).with_messages(with_approval(
                first.response_messages().await.unwrap(),
                &first_approval_id,
            )),
        )
        .await
        .unwrap();
    let second_approval_id = approval_ids(&continued).collect::<Vec<_>>().await.pop();
    let second_approval_id = second_approval_id.expect("a second approval request");
    assert_eq!(first_executions.lock().len(), 1);
    assert!(second_executions.lock().is_empty());
    let submitted = submissions(&mock);
    assert_eq!(submitted.len(), 1);
    assert!(matches(
        &submitted[0],
        &json!({ "toolCallId": "host-1", "output": { "value": "first" } })
    ));

    let resume_from = session.detach().await.unwrap();
    let nested = to_json(resume_from.continue_from.as_ref().unwrap());
    let approvals = nested["pendingToolApprovals"].as_array().unwrap();
    assert_eq!(approvals.len(), 1);
    assert!(matches(
        &approvals[0],
        &json!({ "toolCallId": "host-2", "toolName": "second_write", "kind": "custom" })
    ));
    let results = nested["pendingToolResults"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["toolCallId"], "host-1");
    assert_eq!(results[0]["toolName"], "first_write");
    assert_eq!(results[0]["input"], json!({ "value": "first" }).to_string());
    assert_eq!(
        results[0]["completedResult"]["output"],
        json!({ "value": "first" })
    );

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            resume_from: Some(resume_from),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let completed = agent
        .stream(
            HarnessAgentCallParameters::new(session.clone()).with_messages(with_approval(
                continued.response_messages().await.unwrap(),
                &second_approval_id,
            )),
        )
        .await
        .unwrap();
    completed.consume_stream().await;

    assert_eq!(first_executions.lock().len(), 1);
    assert_eq!(second_executions.lock().len(), 1);
    let submitted = submissions(&mock);
    assert_eq!(
        submitted
            .iter()
            .filter(|r| r["toolCallId"] == "host-1")
            .count(),
        2
    );
    let host_2: Vec<&JsonValue> = submitted
        .iter()
        .filter(|r| r["toolCallId"] == "host-2")
        .collect();
    assert_eq!(host_2.len(), 1);
    assert_eq!(host_2[0]["output"], json!({ "value": "second" }));
    assert!(!session.has_unfinished_turn());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn session_detach_after_a_tool_approval_pause_returns_resume_state_with_nested_continuation_state()
 {
    let mock = MockHarnessOptions::script(|| {
        parts(vec![json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": json!({ "city": "SF" }).to_string() })])
    })
    .build();
    let weather = tool(&[("city", "string")])
        .with_description("Get weather")
        .with_execute(|input, _| async move { Ok(json!({ "city": input["city"] })) });
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("weather", weather)]));
    s.tool_approval = Some(approval_config(json!({ "weather": "user-approval" })));
    let agent = agent(s);
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "go")).await.unwrap();

    let types: Vec<String> = result
        .full_stream()
        .map(|part| part.to_json()["type"].as_str().unwrap().to_string())
        .collect()
        .await;
    let state = to_json(&session.detach().await.unwrap());

    assert!(types.contains(&"tool-approval-request".to_string()));
    assert!(matches(
        &state,
        &json!({
            "type": "resume-session",
            "harnessId": "mock",
            "specificationVersion": "harness-v1",
            "data": {},
        })
    ));
    let continue_from = &state["continueFrom"];
    assert!(matches(
        continue_from,
        &json!({ "type": "continue-turn", "harnessId": "mock", "specificationVersion": "harness-v1", "data": {} })
    ));
    let approvals = continue_from["pendingToolApprovals"].as_array().unwrap();
    assert_eq!(approvals.len(), 1);
    assert!(approvals[0]["approvalId"].is_string());
    let mut approval = approvals[0].clone();
    approval.as_object_mut().unwrap().remove("approvalId");
    assert_eq!(
        approval,
        json!({
            "toolCallId": "c1",
            "toolName": "weather",
            "input": json!({ "city": "SF" }).to_string(),
            "kind": "custom",
            "providerExecuted": false,
        })
    );
    assert_eq!(Mock::count(&mock.record.detach_calls), 0);
    assert_eq!(Mock::count(&mock.record.suspend_calls), 1);
}

fn done_turn(text: &str) -> Vec<HarnessV1StreamPart> {
    let finish_reason = json!({ "unified": "stop" });
    parts(vec![
        json!({ "type": "text-start", "id": "t1" }),
        json!({ "type": "text-delta", "id": "t1", "delta": text }),
        json!({ "type": "text-end", "id": "t1" }),
        json!({ "type": "finish-step", "finishReason": finish_reason, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": finish_reason, "totalUsage": zero_usage() }),
    ])
}

#[tokio::test]
async fn session_detach_from_ui_stream_on_finish_uses_between_turn_resume_state_after_normal_completion()
 {
    use crate::harness::sdk::ui_message_stream::{
        UIMessageStreamOnEndEvent, UIMessageStreamOptions,
    };

    let mock = MockHarnessOptions::script(|| done_turn("done")).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "go")).await.unwrap();
    let detach_state: Arc<Mutex<Option<HarnessV1ResumeSessionState>>> = Arc::default();

    let (state, detaching) = (detach_state.clone(), session.clone());
    let ui_stream = result.to_ui_message_stream(UIMessageStreamOptions {
        on_finish: Some(Arc::new(move |_: UIMessageStreamOnEndEvent| {
            let (state, session) = (state.clone(), detaching.clone());
            Box::pin(async move {
                *state.lock() = Some(session.detach().await.unwrap());
            })
        })),
        ..Default::default()
    });
    let chunks: Vec<_> = ui_stream.collect().await;
    assert!(chunks.iter().all(Result::is_ok));

    assert_eq!(
        to_json(&*detach_state.lock()),
        json!({
            "type": "resume-session",
            "harnessId": "mock",
            "specificationVersion": "harness-v1",
            "data": {},
        })
    );
    assert_eq!(Mock::count(&mock.record.detach_calls), 1);
    assert_eq!(Mock::count(&mock.record.suspend_calls), 0);
}

#[tokio::test]
async fn to_ui_message_stream_emits_a_start_chunk_carrying_the_persistence_mode_message_id() {
    use crate::harness::sdk::ui_message_stream::{UIMessage, UIMessageStreamOptions};

    let mock = MockHarnessOptions::script(|| done_turn("pong")).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "ping")).await.unwrap();

    // Persistence mode injects the server-generated message id into the
    // stream's `start` chunk; it never synthesizes the chunk itself.
    let original: UIMessage = serde_json::from_value(
        json!({ "id": "user-1", "role": "user", "parts": [{ "type": "text", "text": "ping" }] }),
    )
    .unwrap();
    let chunks: Vec<JsonValue> = result
        .to_ui_message_stream(UIMessageStreamOptions {
            original_messages: Some(vec![original]),
            generate_message_id: Some(Arc::new(|| "msg-server-id".to_string())),
            ..Default::default()
        })
        .map(|chunk| chunk.unwrap().to_json())
        .collect()
        .await;

    assert_eq!(
        chunks[0],
        json!({ "type": "start", "messageId": "msg-server-id" })
    );
    session.destroy().await.unwrap();
}

fn finish_only() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "finish-step", "finishReason": { "unified": "stop" }, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "stop" }, "totalUsage": zero_usage() }),
    ])
}

#[tokio::test]
async fn a_single_session_can_drive_multiple_generate_turns() {
    let mock = MockHarnessOptions::script(|| {
        let mut script = parts(vec![
            json!({ "type": "text-delta", "id": "t", "delta": "ok" }),
        ]);
        script.extend(finish_only());
        script
    })
    .build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    agent.generate(call(&session, "one")).await.unwrap();
    agent.generate(call(&session, "two")).await.unwrap();

    assert_eq!(mock.prompts().len(), 2);
    assert_eq!(Mock::count(&mock.record.destroy_calls), 0);

    session.destroy().await.unwrap();
    assert_eq!(Mock::count(&mock.record.destroy_calls), 1);
}

#[tokio::test]
async fn session_destroy_is_idempotent_and_rejects_further_turns() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    session.destroy().await.unwrap();
    session.destroy().await.unwrap();
    assert_eq!(Mock::count(&mock.record.destroy_calls), 1);

    let message = err_message(agent.generate(call(&session, "after destroy")).await);
    assert!(message.contains("has ended"), "{message}");
}

fn messages(value: JsonValue) -> Vec<ModelMessage> {
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn normalizes_prompt_input_string_passes_through_message_array_is_reduced_to_the_last_user_message()
 {
    let mock = MockHarnessOptions::script(finish_only).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    agent
        .generate(call(&session, "plain string"))
        .await
        .unwrap();
    agent
        .generate(
            HarnessAgentCallParameters::new(session.clone()).with_messages(messages(json!([
                { "role": "system", "content": "be terse" },
                { "role": "user", "content": "older user turn — dropped" },
                { "role": "assistant", "content": "older assistant turn — dropped" },
                { "role": "user", "content": "latest user turn" },
            ]))),
        )
        .await
        .unwrap();
    agent
        .generate(
            HarnessAgentCallParameters::new(session.clone()).with_prompt(messages(json!([
                { "role": "user", "content": "discarded" },
                { "role": "assistant", "content": "discarded too" },
                { "role": "user", "content": [{ "type": "text", "text": "final turn" }] },
            ]))),
        )
        .await
        .unwrap();

    let prompts: Vec<JsonValue> = mock
        .prompts()
        .iter()
        .map(|prompt| match prompt {
            HarnessV1Prompt::Text(text) => json!(text),
            HarnessV1Prompt::Message(message) => {
                let mut value = to_json(message);
                value["role"] = json!("user");
                value
            }
        })
        .collect();
    assert_eq!(
        prompts,
        vec![
            json!("plain string"),
            json!({ "role": "user", "content": "latest user turn" }),
            json!({ "role": "user", "content": [{ "type": "text", "text": "final turn" }] }),
        ]
    );

    let message = err_message(
        agent
            .generate(
                HarnessAgentCallParameters::new(session.clone()).with_messages(messages(json!([
                    { "role": "system", "content": "no user message here" },
                    { "role": "assistant", "content": "nothing for the harness to run" },
                ]))),
            )
            .await,
    );
    assert!(
        message.contains("at least one `role: \"user\"` entry"),
        "{message}"
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn session_detach_parks_without_stopping_or_destroying_the_sandbox() {
    let lifecycle = make_lifecycle_session(LifecycleOptions::default()).await;

    assert_eq!(
        lifecycle.session.detach().await.unwrap(),
        lifecycle.resume_state
    );

    assert_eq!(lifecycle.detaches(), 1);
    assert_eq!(lifecycle.stops(), 0);
    assert_eq!(lifecycle.destroys(), 0);
    assert_eq!(lifecycle.sandbox.stops(), 0);
    assert_eq!(lifecycle.sandbox.destroys(), 0);
}

#[tokio::test]
async fn session_detach_keeps_the_local_handle_active_when_detaching_fails() {
    let lifecycle = make_lifecycle_session(LifecycleOptions {
        detach_error: Some("could not persist resume state".into()),
        ..Default::default()
    })
    .await;

    assert_eq!(
        err_message(lifecycle.session.detach().await),
        "could not persist resume state"
    );
    assert_eq!(
        to_json(&lifecycle.session.stop().await.unwrap())["type"],
        "resume-session"
    );
    assert_eq!(lifecycle.detaches(), 1);
    assert_eq!(lifecycle.stops(), 1);
}

#[tokio::test]
async fn session_stop_saves_state_and_stops_the_sandbox() {
    let lifecycle = make_lifecycle_session(LifecycleOptions::default()).await;

    assert_eq!(
        lifecycle.session.stop().await.unwrap(),
        lifecycle.resume_state
    );

    assert_eq!(lifecycle.detaches(), 0);
    assert_eq!(lifecycle.stops(), 1);
    assert_eq!(lifecycle.destroys(), 0);
    assert_eq!(lifecycle.sandbox.stops(), 1);
    assert_eq!(lifecycle.sandbox.destroys(), 0);
}

#[tokio::test]
async fn session_stop_does_not_stop_a_caller_owned_sandbox() {
    let lifecycle = make_lifecycle_session(LifecycleOptions {
        owns_sandbox_lifecycle: Some(false),
        ..Default::default()
    })
    .await;

    lifecycle.session.stop().await.unwrap();

    assert_eq!(lifecycle.stops(), 1);
    assert_eq!(lifecycle.sandbox.stops(), 0);
    assert_eq!(lifecycle.sandbox.destroys(), 0);
}

#[tokio::test]
async fn session_stop_wraps_unfinished_turns_as_nested_continuation_state_and_stops_the_sandbox() {
    let lifecycle = make_lifecycle_session(LifecycleOptions {
        turn_state: Some(HarnessAgentTurnState::Suspended),
        ..Default::default()
    })
    .await;

    let mut expected = mock_resume_state();
    expected.data = lifecycle.continue_state.data.clone();
    expected.continue_from = Some(lifecycle.continue_state.clone());
    assert_eq!(lifecycle.session.stop().await.unwrap(), expected);

    assert_eq!(lifecycle.stops(), 0);
    assert_eq!(lifecycle.destroys(), 0);
    assert_eq!(lifecycle.sandbox.stops(), 1);
    assert_eq!(lifecycle.sandbox.destroys(), 0);
}

#[tokio::test]
async fn session_suspend_turn_returns_raw_continuation_state_and_detaches_the_local_handle() {
    let lifecycle = make_lifecycle_session(LifecycleOptions {
        turn_state: Some(HarnessAgentTurnState::Running),
        ..Default::default()
    })
    .await;

    assert_eq!(
        lifecycle.session.suspend_turn().await.unwrap(),
        lifecycle.continue_state
    );

    assert_eq!(lifecycle.detaches(), 0);
    assert_eq!(lifecycle.stops(), 0);
    assert_eq!(lifecycle.sandbox.stops(), 0);
    let message = err_message(lifecycle.session.suspend_turn().await);
    assert!(message.contains("not active"), "{message}");
}

#[tokio::test]
async fn session_has_unfinished_turn_reflects_every_turn_lifecycle_state() {
    for (turn_state, expected) in [
        (HarnessAgentTurnState::Idle, false),
        (HarnessAgentTurnState::Running, true),
        (HarnessAgentTurnState::AwaitingApproval, true),
        (HarnessAgentTurnState::AwaitingToolResult, true),
        (HarnessAgentTurnState::Suspended, true),
    ] {
        let lifecycle = make_lifecycle_session(LifecycleOptions {
            turn_state: Some(turn_state),
            ..Default::default()
        })
        .await;
        assert_eq!(lifecycle.session.has_unfinished_turn(), expected);
        lifecycle.session.destroy().await.unwrap();
    }
}

#[tokio::test]
async fn session_destroy_destroys_the_sandbox_without_saving_state() {
    let lifecycle = make_lifecycle_session(LifecycleOptions::default()).await;

    lifecycle.session.destroy().await.unwrap();

    assert_eq!(lifecycle.detaches(), 0);
    assert_eq!(lifecycle.stops(), 0);
    assert_eq!(lifecycle.destroys(), 1);
    assert_eq!(lifecycle.sandbox.stops(), 0);
    assert_eq!(lifecycle.sandbox.destroys(), 1);
}

#[tokio::test]
async fn session_compact_forwards_to_the_harness_session_do_compact_then_throws_once_ended() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    session.compact(None).await.unwrap();
    session
        .compact(Some("keep the error trace".into()))
        .await
        .unwrap();
    assert_eq!(
        *mock.record.compact_calls.lock(),
        vec![None, Some("keep the error trace".to_string())]
    );

    session.destroy().await.unwrap();
    let message = err_message(session.compact(None).await);
    assert!(message.to_lowercase().contains("ended"), "{message}");
}

fn bridge_schema() -> crate::harness::sdk::tool::Schema {
    z_object(&[("bridge", "object?")])
}

fn state_with(data: JsonValue) -> HarnessV1ResumeSessionState {
    let mut state = mock_resume_state();
    state.data = data;
    state
}

#[tokio::test]
async fn session_detach_returns_validated_coords_surfaces_resume_status_and_ends_the_local_handle()
{
    let bridge = json!({ "bridge": { "port": 5001, "token": "t", "lastSeenEventId": 3 } });
    let mut options = MockHarnessOptions::script(|| parts(vec![json!({ "type": "finish" })]));
    options.is_resume = true;
    options.lifecycle_state_schema = Some(bridge_schema());
    options.detach_state = Some(state_with(bridge.clone()));
    let mut suspend_state = mock_continue_state();
    suspend_state.data = bridge.clone();
    options.suspend_state = Some(suspend_state);
    let mock = options.build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    assert!(session.is_resume());

    let handle = session.detach().await.unwrap();
    assert_eq!(handle, state_with(bridge));
    assert_eq!(Mock::count(&mock.record.stop_calls), 0);
    assert_eq!(Mock::count(&mock.record.destroy_calls), 0);

    let message = err_message(agent.generate(call(&session, "after detach")).await);
    assert!(message.contains("has ended"), "{message}");
}

#[tokio::test]
async fn session_stop_returns_validated_state_and_ends_the_local_handle() {
    let saved = json!({ "value": "saved" });
    let mut options = MockHarnessOptions::script(Vec::new);
    options.lifecycle_state_schema = Some(z_object(&[("value", "string")]));
    options.detach_state = Some(state_with(saved.clone()));
    options.stop_state = Some(state_with(saved.clone()));
    let mut suspend_state = mock_continue_state();
    suspend_state.data = saved.clone();
    options.suspend_state = Some(suspend_state);
    let mock = options.build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    assert_eq!(session.stop().await.unwrap(), state_with(saved));
    assert_eq!(Mock::count(&mock.record.stop_calls), 1);
    assert_eq!(Mock::count(&mock.record.destroy_calls), 0);
    let message = err_message(agent.generate(call(&session, "after stop")).await);
    assert!(message.contains("has ended"), "{message}");
}
