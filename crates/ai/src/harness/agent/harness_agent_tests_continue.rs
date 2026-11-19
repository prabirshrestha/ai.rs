//! `harness-agent.test.ts`: continuing an unfinished turn.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use serde_json::json;

use super::*;
use crate::harness::agent::internal::run_prompt::tests::tool_outputs::{
    custom_approval, submitted_json,
};
use crate::harness::sdk::messages::{ModelMessage, SystemModelMessage, ToolResultPart};
use crate::harness::spec::tools::HarnessV1Skill;

fn undefined_usage_finish() -> Vec<JsonValue> {
    vec![
        json!({ "type": "finish-step", "finishReason": { "unified": "stop" }, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "stop" }, "totalUsage": zero_usage() }),
    ]
}

fn continue_from_options() -> HarnessAgentCreateSessionOptions {
    HarnessAgentCreateSessionOptions {
        continue_from: Some(mock_continue_state()),
        ..with_sandbox(&make_sandbox_session())
    }
}

#[tokio::test]
async fn continue_stream_continues_an_in_flight_turn_and_streams_translated_parts() {
    let mock = MockHarnessOptions::script(Vec::new)
        .with_continue(|| {
            let mut values = vec![
                json!({ "type": "stream-start" }),
                json!({ "type": "text-delta", "id": "t1", "delta": "Still running" }),
            ];
            values.extend(undefined_usage_finish());
            parts(values)
        })
        .build();

    let mut s = settings(&mock);
    s.instructions = Some(HarnessAgentInstructions::System(SystemModelMessage {
        content: "Be concise.".into(),
        provider_options: Some(serde_json::from_value(json!({ "test": { "cache": true } })).unwrap()),
    }));
    let agent = agent(s);
    let session = agent.create_session(continue_from_options()).await.unwrap();
    let result = agent
        .continue_stream(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();

    let types: Vec<String> = result
        .full_stream()
        .map(|part| part.to_json()["type"].as_str().unwrap().to_string())
        .collect()
        .await;

    for ty in ["text-delta", "finish-step", "finish"] {
        assert!(types.contains(&ty.to_string()), "{types:?}");
    }
    assert_eq!(result.text().await.unwrap(), "Still running");
    assert!(mock.prompts().is_empty());
    assert_eq!(mock.continue_count(), 1);
    assert_eq!(
        mock.record.continue_turns.lock()[0].settings.instructions.as_deref(),
        Some("Be concise.")
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn continue_generate_continues_an_in_flight_turn_and_returns_generated_text() {
    let usage = json!({ "inputTokens": { "total": 3, "noCache": 3 }, "outputTokens": { "total": 1, "text": 1 } });
    let mock = MockHarnessOptions::script(Vec::new)
        .with_continue(move || {
            parts(vec![
                json!({ "type": "stream-start" }),
                json!({ "type": "text-start", "id": "t1" }),
                json!({ "type": "text-delta", "id": "t1", "delta": "Completed" }),
                json!({ "type": "text-end", "id": "t1" }),
                json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": usage }),
                json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": usage }),
            ])
        })
        .build();

    let agent = agent(settings(&mock));
    let session = agent.create_session(continue_from_options()).await.unwrap();
    let result = agent
        .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();

    assert_eq!(result.text(), "Completed");
    assert_eq!(to_json(&result.finish_reason()), json!("stop"));
    assert_eq!(result.raw_finish_reason().as_deref(), Some("end_turn"));
    assert_eq!(result.usage.input_tokens, Some(3));
    assert_eq!(result.usage.output_tokens, Some(1));
    assert!(mock.prompts().is_empty());
    assert_eq!(mock.continue_count(), 1);

    session.destroy().await.unwrap();
}

fn weather_tool() -> crate::harness::sdk::tool::Tool {
    tool(&[("city", "string")]).with_description("Get weather")
}

fn lima_call() -> JsonValue {
    json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": json!({ "city": "Lima" }).to_string() })
}

fn lima_result_part() -> ToolResultPart {
    serde_json::from_value(json!({
        "toolCallId": "c1",
        "toolName": "weather",
        "output": { "type": "json", "value": { "city": "Lima", "celsius": 19 } },
    }))
    .unwrap()
}

fn lima_submission() -> JsonValue {
    json!({
        "toolCallId": "c1",
        "output": { "city": "Lima", "celsius": 19 },
        "toolResult": {
            "type": "tool-result",
            "toolCallId": "c1",
            "toolName": "weather",
            "output": { "type": "json", "value": { "city": "Lima", "celsius": 19 } },
        },
    })
}

#[tokio::test]
async fn serializes_and_resumes_a_client_side_tool_result_pause() {
    let mock = MockHarnessOptions::script(|| parts(vec![lima_call()]))
        .with_continue(|| {
            parts(vec![
                lima_call(),
                json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "weather", "result": { "city": "Lima", "celsius": 19 } }),
                json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": zero_usage() }),
                json!({ "type": "text-delta", "id": "t1", "delta": "It is 19°C in Lima." }),
                json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": zero_usage() }),
                json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": zero_usage() }),
            ])
        })
        .build();
    let prepare_call_count = Arc::new(AtomicUsize::new(0));
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("weather", weather_tool())]));
    s.call_options_schema = Some(z_object(&[("city", "string")]));
    let count = prepare_call_count.clone();
    s.prepare_call = Some(Arc::new(move |mut call: HarnessAgentPrepareCallOptions| {
        count.fetch_add(1, Ordering::SeqCst);
        let city = call.options.as_ref().unwrap()["city"].as_str().unwrap().to_string();
        call.skills = Some(vec![HarnessV1Skill {
            name: format!("{city}-weather"),
            description: "Weather guidance.".into(),
            content: "Use the weather tool.".into(),
            files: None,
        }]);
        call.instructions = Some(HarnessAgentInstructions::Text(format!("Be concise about {city}.")));
        call.tools = Some(tools(vec![("weather", weather_tool())]));
        Box::pin(async move { Ok(call) })
    }));
    let agent = agent(s);
    let sandbox = make_sandbox_session();
    let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();

    let mut params = call(&session, "Check Lima weather");
    params.options = Some(json!({ "city": "lima" }));
    let first = agent.stream(params).await.unwrap();
    first.consume_stream().await;

    assert!(session.has_unfinished_turn());
    assert_eq!(first.steps().await.unwrap().len(), 1);
    assert!(mock.tool_results().is_empty());

    let session_id = session.session_id().to_string();
    let continue_from = session.suspend_turn().await.unwrap();
    assert!(session.has_unfinished_turn());
    assert_eq!(
        to_json(&continue_from.pending_tool_results),
        json!([{ "toolCallId": "c1", "toolName": "weather", "input": json!({ "city": "Lima" }).to_string() }])
    );
    assert!(matches(
        &to_json(&continue_from.turn_settings),
        &json!({
            "skills": [{ "name": "lima-weather", "description": "Weather guidance.", "content": "Use the weather tool." }],
            "instructions": "Be concise about lima.",
            "tools": [{ "name": "weather", "description": "Get weather", "inputSchema": to_json(&weather_tool().input_schema.json_schema) }],
        })
    ));

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some(session_id),
            continue_from: Some(continue_from),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    assert!(session.has_unfinished_turn());
    let mut params = HarnessAgentContinueParameters::new(session.clone());
    params.tool_result_continuations = vec![lima_result_part()];
    let continued = agent.continue_stream(params).await.unwrap();
    let continued_part_types: Vec<String> = continued
        .full_stream()
        .map(|part| part.to_json()["type"].as_str().unwrap().to_string())
        .collect()
        .await;

    assert_eq!(
        mock.tool_results().iter().map(submitted_json).collect::<Vec<_>>(),
        vec![lima_submission()]
    );
    assert_eq!(
        continued_part_types,
        vec!["start", "start-step", "text-delta", "finish-step", "finish"]
    );
    assert_eq!(
        continued
            .steps()
            .await
            .unwrap()
            .iter()
            .map(|step| step.text())
            .collect::<Vec<_>>(),
        vec!["It is 19°C in Lima."]
    );
    assert!(!session.has_unfinished_turn());
    assert_eq!(prepare_call_count.load(Ordering::SeqCst), 1);
    let continued_settings = &mock.record.continue_turns.lock()[0].settings;
    assert_eq!(continued_settings.skills[0].name, "lima-weather");
    assert_eq!(continued_settings.instructions.as_deref(), Some("Be concise about lima."));
    assert_eq!(continued_settings.tools[0].name, "weather");

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn collects_a_client_side_tool_result_from_messages() {
    let mock = MockHarnessOptions::script(|| parts(vec![lima_call()]))
        .with_continue(finish_events)
        .build();
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("weather", weather_tool())]));
    let agent = agent(s);
    let session = new_session(&agent).await;

    let first = agent.stream(call(&session, "Check Lima weather")).await.unwrap();
    first.consume_stream().await;
    let messages: Vec<ModelMessage> = serde_json::from_value(json!([{
        "role": "tool",
        "content": [{
            "type": "tool-result",
            "toolCallId": "c1",
            "toolName": "weather",
            "output": { "type": "json", "value": { "city": "Lima", "celsius": 19 } },
        }],
    }]))
    .unwrap();
    let continued = agent
        .stream(HarnessAgentCallParameters::new(session.clone()).with_messages(messages))
        .await
        .unwrap();
    continued.consume_stream().await;

    assert_eq!(
        mock.tool_results().iter().map(submitted_json).collect::<Vec<_>>(),
        vec![lima_submission()]
    );
    assert!(!session.has_unfinished_turn());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn continues_a_restored_parallel_approval_batch_with_mixed_ui_decisions() {
    let executions = Arc::new(AtomicUsize::new(0));
    let mock = MockHarnessOptions::script(Vec::new)
        .with_continue(finish_events)
        .build();
    let count = executions.clone();
    let weather = tool(&[("city", "string")]).with_execute(move |input, _| {
        count.fetch_add(1, Ordering::SeqCst);
        async move { Ok(json!({ "city": input["city"] })) }
    });
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("weather", weather)]));
    let agent = agent(s);
    let mut continue_from = mock_continue_state();
    continue_from.pending_tool_approvals = Some(
        ["Paris", "Tokyo"]
            .iter()
            .enumerate()
            .map(|(index, city)| {
                custom_approval(
                    &format!("approval-{index}"),
                    &format!("call-{index}"),
                    "weather",
                    &json!({ "city": city }).to_string(),
                )
            })
            .collect(),
    );
    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            continue_from: Some(continue_from),
            ..with_sandbox(&make_sandbox_session())
        })
        .await
        .unwrap();

    // [rust] `convertToModelMessages` comes with the UI message stream port;
    // these are the model messages it produces for the two
    // `approval-responded` tool parts.
    let messages: Vec<ModelMessage> = serde_json::from_value(json!([
        { "role": "assistant", "content": [
            { "type": "tool-call", "toolCallId": "call-0", "toolName": "weather", "input": { "city": "Paris" } },
            { "type": "tool-approval-request", "approvalId": "approval-0", "toolCallId": "call-0" },
            { "type": "tool-call", "toolCallId": "call-1", "toolName": "weather", "input": { "city": "Tokyo" } },
            { "type": "tool-approval-request", "approvalId": "approval-1", "toolCallId": "call-1" },
        ]},
        { "role": "tool", "content": [
            { "type": "tool-approval-response", "approvalId": "approval-0", "approved": true },
            { "type": "tool-approval-response", "approvalId": "approval-1", "approved": false, "reason": "User denied the lookup." },
            { "type": "tool-result", "toolCallId": "call-1", "toolName": "weather",
              "output": { "type": "execution-denied", "reason": "User denied the lookup." } },
        ]},
    ]))
    .unwrap();
    let result = agent
        .stream(HarnessAgentCallParameters::new(session.clone()).with_messages(messages))
        .await
        .unwrap();
    let decisions: Vec<bool> = result
        .full_stream()
        .filter_map(|part| async move {
            let part = part.to_json();
            (part["type"] == "tool-approval-response").then(|| part["approved"].as_bool().unwrap())
        })
        .collect()
        .await;

    assert_eq!(decisions, vec![true, false]);
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(
        mock.tool_results().iter().map(submitted_json).collect::<Vec<_>>(),
        vec![
            json!({ "output": { "city": "Paris" }, "toolCallId": "call-0" }),
            json!({
                "output": { "reason": "User denied the lookup.", "type": "execution-denied" },
                "toolCallId": "call-1",
                "toolResult": {
                    "output": { "reason": "User denied the lookup.", "type": "execution-denied" },
                    "toolCallId": "call-1",
                    "toolName": "weather",
                    "type": "tool-result",
                },
            }),
        ]
    );
    assert!(!session.has_unfinished_turn());
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn continue_stream_rejects_when_there_is_no_unfinished_turn() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let message = err_message(
        agent
            .continue_stream(HarnessAgentContinueParameters::new(session.clone()))
            .await,
    );
    assert!(message.contains("no unfinished turn to continue"), "{message}");

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn resume_from_continue_from_resumes_a_session_that_must_continue_before_accepting_a_new_prompt() {
    let mock = MockHarnessOptions::script(|| {
        let mut values = vec![json!({ "type": "text-delta", "id": "t2", "delta": "after" })];
        values.extend(undefined_usage_finish());
        parts(values)
    })
    .with_continue(|| {
        let mut values = vec![json!({ "type": "text-delta", "id": "t1", "delta": "continued" })];
        values.extend(undefined_usage_finish());
        parts(values)
    })
    .build();

    let agent = agent(settings(&mock));
    let mut resume_from = mock_resume_state();
    resume_from.continue_from = Some(mock_continue_state());
    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            resume_from: Some(resume_from),
            ..with_sandbox(&make_sandbox_session())
        })
        .await
        .unwrap();

    let message = err_message(agent.generate(call(&session, "new prompt too early")).await);
    assert!(message.contains("must be continued"), "{message}");

    let continued = agent
        .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();
    assert_eq!(continued.text(), "continued");
    let after = agent.generate(call(&session, "new prompt now")).await.unwrap();
    assert_eq!(after.text(), "after");

    assert_eq!(mock.continue_count(), 1);
    let prompts = mock.prompts();
    assert_eq!(prompts.len(), 1);
    assert!(matches!(&prompts[0], HarnessV1Prompt::Text(t) if t == "new prompt now"));

    session.destroy().await.unwrap();
}
