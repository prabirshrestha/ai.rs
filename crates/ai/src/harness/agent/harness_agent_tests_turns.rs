//! `harness-agent.test.ts`: turn preparation, steering, `stopWhen`, and
//! generated results.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::agent::internal::run_prompt::tests::tool_outputs::approval_config;
use crate::harness::agent::internal::run_prompt::tests::{FakeSession, SubmitFn};
use crate::harness::sdk::messages::SystemModelMessage;
use crate::harness::sdk::stop_condition::{StopCondition, is_step_count, stop_condition};
use crate::harness::sdk::tool::Tool;
use crate::harness::spec::tools::HarnessV1Skill;

fn prompt_text(prompt: &HarnessV1Prompt) -> String {
    match prompt {
        HarnessV1Prompt::Text(text) => text.clone(),
        other => panic!("expected a text prompt, got {other:?}"),
    }
}

#[tokio::test]
async fn prepares_model_skills_instructions_tools_and_the_prompt_for_each_fresh_turn() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let echo = tool(&[("value", "string")])
        .with_description("Echo a value.")
        .with_execute(|input, _| async move { Ok(input["value"].clone()) });
    let prepare_calls: Arc<Mutex<Vec<Option<JsonValue>>>> = Arc::default();
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("echo", echo.clone())]));
    s.call_options_schema = Some(z_object(&[("tenant", "string")]));
    let calls = prepare_calls.clone();
    s.prepare_call = Some(Arc::new(move |mut call: HarnessAgentPrepareCallOptions| {
        calls.lock().push(call.options.clone());
        let tenant = call.options.as_ref().unwrap()["tenant"]
            .as_str()
            .unwrap()
            .to_string();
        let prompt = match &call.prompt {
            Some(HarnessAgentCallPrompt::Text(text)) => text.clone(),
            _ => unreachable!(),
        };
        call.prompt = Some(format!("{prompt} for {tenant}").into());
        call.model = Some(format!("model-{tenant}"));
        call.skills = Some(vec![HarnessV1Skill {
            name: tenant.clone(),
            description: format!("{tenant} skill"),
            content: format!("{tenant} instructions"),
            files: None,
        }]);
        call.instructions = Some(HarnessAgentInstructions::System(SystemModelMessage {
            content: format!("Serve {tenant}"),
            provider_options: Some(
                serde_json::from_value(json!({ "test": { "cache": true } })).unwrap(),
            ),
        }));
        call.tools = (tenant == "alpha").then(|| tools(vec![("echo", echo.clone())]));
        Box::pin(async move { Ok(call) })
    }));
    let agent = agent(s);
    let session = new_session(&agent).await;

    for tenant in ["alpha", "beta"] {
        let mut params = call(&session, "Hello");
        params.options = Some(json!({ "tenant": tenant }));
        agent.generate(params).await.unwrap();
    }

    // [rust] `HarnessV1StartOptions` has no model, skills, instructions, or
    // tools, so `doStart` cannot receive them.
    let prompt_turns = mock.record.prompt_turns.lock().clone();
    assert_eq!(prompt_turns.len(), 2);
    for (turn, tenant, tools) in [
        (
            &prompt_turns[0],
            "alpha",
            json!([{ "name": "echo", "description": "Echo a value." }]),
        ),
        (&prompt_turns[1], "beta", json!([])),
    ] {
        assert_eq!(prompt_text(&turn.prompt), format!("Hello for {tenant}"));
        assert_eq!(turn.settings.model, Some(format!("model-{tenant}")));
        assert_eq!(turn.settings.instructions, Some(format!("Serve {tenant}")));
        assert!(matches(
            &to_json(&turn.settings.skills),
            &json!([{ "name": tenant }])
        ));
        assert!(matches(&to_json(&turn.settings.tools), &tools));
    }
    assert_eq!(prepare_calls.lock().len(), 2);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn validates_custom_call_options_before_preparing_a_turn() {
    let mock = MockHarnessOptions::script(finish_events).build();
    let mut s = settings(&mock);
    s.call_options_schema = Some(z_object(&[("tenant", "string")]));
    s.prepare_call = Some(Arc::new(|mut call: HarnessAgentPrepareCallOptions| {
        let tenant = call.options.as_ref().unwrap()["tenant"].to_string();
        call.prompt = Some(format!("Hello for {tenant}").into());
        Box::pin(async move { Ok(call) })
    }));
    let agent = agent(s);
    let session = new_session(&agent).await;

    let mut params = call(&session, "Hello");
    params.options = Some(json!({ "tenant": 123 }));
    assert!(agent.generate(params).await.is_err());
    session.destroy().await.unwrap();
}

fn steering_mock(gates: &PromptGates, supports_steering: bool) -> Mock {
    let mut options = MockHarnessOptions::script(Vec::new);
    options.supports_steering = supports_steering;
    options.prompt_done = Some(gates.prompt_done());
    options.build()
}

#[tokio::test]
async fn experimental_steer_submits_a_message_to_the_running_turn() {
    let gates = PromptGates::default();
    let mock = steering_mock(&gates, true);
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "Start.")).await.unwrap();

    agent
        .experimental_steer(&session, "Change course.")
        .await
        .unwrap();

    assert_eq!(mock.user_messages(), vec!["Change course."]);
    gates.finish_all();
    result.consume_stream().await;
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn experimental_steer_turn_exposes_the_session_level_steering_api() {
    let gates = PromptGates::default();
    let mock = steering_mock(&gates, true);
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "Start.")).await.unwrap();

    session
        .experimental_steer_turn("Change course.".into())
        .await
        .unwrap();

    assert_eq!(mock.user_messages(), vec!["Change course."]);
    gates.finish_all();
    result.consume_stream().await;
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn experimental_steer_reports_an_unsupported_harness_capability() {
    let gates = PromptGates::default();
    let mock = steering_mock(&gates, false);
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "Start.")).await.unwrap();

    let error = agent
        .experimental_steer(&session, "Change course.")
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<HarnessError>(),
        Some(HarnessError::CapabilityUnsupported { .. })
    ));

    gates.finish_all();
    result.consume_stream().await;
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn experimental_steer_rejects_when_the_session_has_no_running_turn() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let message = err_message(agent.experimental_steer(&session, "Change course.").await);
    assert!(
        message.contains("has no running turn to steer"),
        "{message}"
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn experimental_steer_rejects_while_the_turn_awaits_tool_approval() {
    let mut options = MockHarnessOptions::script(|| {
        parts(vec![json!({
            "type": "tool-call", "toolCallId": "call-1", "toolName": "weather",
            "input": json!({ "city": "Paris" }).to_string(),
        })])
    });
    options.supports_steering = true;
    let mock = options.build();
    let weather = tool(&[("city", "string")])
        .with_execute(|input, _| async move { Ok(json!({ "city": input["city"] })) });
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("weather", weather)]));
    s.tool_approval = Some(approval_config(json!({ "weather": "user-approval" })));
    let agent = agent(s);
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "Start.")).await.unwrap();
    result.consume_stream().await;

    let message = err_message(agent.experimental_steer(&session, "Change course.").await);
    assert!(
        message.contains("has no running turn to steer"),
        "{message}"
    );
    assert!(mock.user_messages().is_empty());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn experimental_steer_rejects_after_the_active_turn_is_suspended() {
    let gates = PromptGates::default();
    let mut options = MockHarnessOptions::script(Vec::new);
    options.supports_steering = true;
    options.prompt_done = Some(gates.prompt_done());
    let on_suspend = gates.clone();
    options.on_suspend_turn = Some(Arc::new(move || {
        on_suspend.finish_all();
        Box::pin(async { Ok(()) })
    }));
    let mock = options.build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "Start.")).await.unwrap();

    session.suspend_turn().await.unwrap();
    let message = err_message(agent.experimental_steer(&session, "Change course.").await);
    assert!(
        message.contains("has ended and cannot be reused"),
        "{message}"
    );
    assert!(mock.user_messages().is_empty());

    gates.finish_all();
    result.consume_stream().await;
}

#[tokio::test]
async fn experimental_steer_targets_the_current_turn_when_a_session_is_reused() {
    let gates = PromptGates::default();
    let mock = steering_mock(&gates, true);
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let first = agent.stream(call(&session, "First.")).await.unwrap();
    agent
        .experimental_steer(&session, "Steer first.")
        .await
        .unwrap();
    gates.finish_next();
    first.consume_stream().await;

    let second = agent.stream(call(&session, "Second.")).await.unwrap();
    agent
        .experimental_steer(&session, "Steer second.")
        .await
        .unwrap();
    gates.finish_next();
    second.consume_stream().await;

    assert_eq!(mock.user_messages(), vec!["Steer first.", "Steer second."]);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_limit_steps_when_stop_when_is_omitted() {
    let mock = MockHarnessOptions::script(|| {
        let mut script = vec![finish_events().remove(0); 21];
        script.push(finish_events().remove(1));
        script
    })
    .build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let result = agent.generate(call(&session, "keep going")).await.unwrap();

    assert_eq!(result.steps.len(), 21);
    assert!(!session.has_unfinished_turn());
    assert_eq!(Mock::count(&mock.record.suspend_calls), 0);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn generate_stops_after_a_configured_step_and_reuses_its_captured_suspension_state() {
    let predicate_step_counts: Arc<Mutex<Vec<usize>>> = Arc::default();
    let mock = MockHarnessOptions::script(|| {
        vec![
            finish_events().remove(0),
            finish_events().remove(0),
            finish_events().remove(1),
        ]
    })
    .build();
    let counts = predicate_step_counts.clone();
    let mut s = settings(&mock);
    let async_condition: StopCondition = Arc::new(|steps| {
        let done = steps.len() == 1;
        Box::pin(async move { done })
    });
    s.stop_when = Some(vec![
        stop_condition(move |steps| {
            counts.lock().push(steps.len());
            false
        }),
        async_condition,
    ]);
    let agent = agent(s);
    let session = new_session(&agent).await;

    let result = agent.generate(call(&session, "one step")).await.unwrap();

    assert_eq!(result.steps.len(), 1);
    assert_eq!(*predicate_step_counts.lock(), vec![1]);
    assert!(session.has_unfinished_turn());
    assert_eq!(Mock::count(&mock.record.suspend_calls), 1);

    assert_eq!(
        to_json(&session.suspend_turn().await.unwrap()),
        json!({
            "type": "continue-turn",
            "harnessId": "mock",
            "specificationVersion": "harness-v1",
            "data": {},
            "turnSettings": { "skills": [], "tools": [] },
        })
    );
    assert_eq!(Mock::count(&mock.record.suspend_calls), 1);
}

#[tokio::test]
async fn stream_and_continued_turns_apply_stop_when_independently() {
    let continuation_count = Arc::new(Mutex::new(0));
    let count = continuation_count.clone();
    let mock = MockHarnessOptions::script(one_step_then_more)
        .with_continue(move || {
            let mut count = count.lock();
            *count += 1;
            if *count <= 2 {
                one_step_then_more()
            } else {
                vec![finish_events().remove(1)]
            }
        })
        .build();
    let mut s = settings(&mock);
    s.stop_when = Some(vec![is_step_count(1)]);
    let agent = agent(s);

    let sandbox = make_sandbox_session();
    let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();
    let first = agent
        .stream(call(&session, "one step at a time"))
        .await
        .unwrap();
    first.consume_stream().await;
    assert_eq!(first.steps().await.unwrap().len(), 1);
    let mut continue_from = session.suspend_turn().await.unwrap();

    let recreate = |session: &HarnessAgentSession, continue_from: HarnessV1ContinueTurnState| {
        HarnessAgentCreateSessionOptions {
            session_id: Some(session.session_id().to_string()),
            continue_from: Some(continue_from),
            ..with_sandbox(&sandbox)
        }
    };
    let session = agent
        .create_session(recreate(&session, continue_from))
        .await
        .unwrap();
    let second = agent
        .continue_stream(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();
    second.consume_stream().await;
    assert_eq!(second.steps().await.unwrap().len(), 1);
    continue_from = session.suspend_turn().await.unwrap();

    let session = agent
        .create_session(recreate(&session, continue_from))
        .await
        .unwrap();
    let third = agent
        .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();
    assert_eq!(third.steps.len(), 1);
    continue_from = session.suspend_turn().await.unwrap();

    let session = agent
        .create_session(recreate(&session, continue_from))
        .await
        .unwrap();
    let terminal = agent
        .continue_generate(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();

    assert_eq!(terminal.steps.len(), 0);
    assert!(!session.has_unfinished_turn());
    assert_eq!(Mock::count(&mock.record.suspend_calls), 3);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn stop_when_lets_a_terminal_text_only_step_finish_the_turn_naturally() {
    let mock = MockHarnessOptions::script(|| {
        let mut values = vec![json!({ "type": "text-delta", "id": "t1", "delta": "done" })];
        values.extend(stop_finish());
        parts(values)
    })
    .build();
    let mut s = settings(&mock);
    s.stop_when = Some(vec![is_step_count(1)]);
    let agent = agent(s);
    let session = new_session(&agent).await;

    let result = agent.generate(call(&session, "finish")).await.unwrap();

    assert_eq!(result.text(), "done");
    assert_eq!(result.steps.len(), 1);
    assert_eq!(to_json(&result.finish_reason()), json!("stop"));
    assert!(!session.has_unfinished_turn());
    assert_eq!(Mock::count(&mock.record.suspend_calls), 0);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn generate_returns_text_and_steps_for_a_simple_text_only_turn() {
    let usage = json!({
        "inputTokens": { "total": 5, "noCache": 5 },
        "outputTokens": { "total": 2, "text": 2 },
    });
    let mock = MockHarnessOptions::script(move || {
        parts(vec![
            json!({ "type": "stream-start" }),
            json!({ "type": "text-start", "id": "t1" }),
            json!({ "type": "text-delta", "id": "t1", "delta": "Hello, " }),
            json!({ "type": "text-delta", "id": "t1", "delta": "world." }),
            json!({ "type": "text-end", "id": "t1" }),
            json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": usage }),
            json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": usage }),
        ])
    })
    .build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.generate(call(&session, "hi")).await.unwrap();

    assert_eq!(result.text(), "Hello, world.");
    assert_eq!(to_json(&result.finish_reason()), json!("stop"));
    assert_eq!(result.raw_finish_reason().as_deref(), Some("end_turn"));
    assert_eq!(result.usage.input_tokens, Some(5));
    assert_eq!(result.usage.output_tokens, Some(2));
    assert_eq!(result.usage.total_tokens, Some(7));
    assert_eq!(result.steps.len(), 1);
    assert_eq!(result.final_step().text(), "Hello, world.");
    assert!(result.tool_calls().is_empty());
    assert!(result.tool_results().is_empty());
    assert_eq!(result.response_messages.len(), 1);
    assert_eq!(to_json(&result.response_messages[0])["role"], "assistant");

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn preserves_interleaved_text_and_reasoning_blocks_by_stream_id() {
    let metadata = |block: &str| json!({ "mock": { "block": block } });
    let mock = MockHarnessOptions::script(move || {
        let mut values = vec![
            json!({ "type": "text-start", "id": "text-a", "harnessMetadata": metadata("text-a") }),
            json!({ "type": "text-delta", "id": "text-a", "delta": "A1" }),
            json!({ "type": "text-start", "id": "text-b", "harnessMetadata": metadata("text-b") }),
            json!({ "type": "text-delta", "id": "text-b", "delta": "B1" }),
            json!({ "type": "text-delta", "id": "text-a", "delta": "A2" }),
            json!({ "type": "text-end", "id": "text-a" }),
            json!({ "type": "text-end", "id": "text-b" }),
            json!({ "type": "reasoning-start", "id": "reasoning-a", "harnessMetadata": metadata("reasoning-a") }),
            json!({ "type": "reasoning-delta", "id": "reasoning-a", "delta": "R1" }),
            json!({ "type": "reasoning-start", "id": "reasoning-b", "harnessMetadata": metadata("reasoning-b") }),
            json!({ "type": "reasoning-delta", "id": "reasoning-b", "delta": "S1" }),
            json!({ "type": "reasoning-delta", "id": "reasoning-a", "delta": "R2" }),
            json!({ "type": "reasoning-end", "id": "reasoning-a" }),
            json!({ "type": "reasoning-end", "id": "reasoning-b" }),
        ];
        values.extend(stop_finish());
        parts(values)
    })
    .build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let result = agent.stream(call(&session, "hi")).await.unwrap();
    result.consume_stream().await;

    assert_eq!(
        to_json(&result.final_step().await.unwrap().content),
        json!([
            { "type": "text", "text": "A1A2", "providerMetadata": metadata("text-a") },
            { "type": "text", "text": "B1", "providerMetadata": metadata("text-b") },
            { "type": "reasoning", "text": "R1R2", "providerMetadata": metadata("reasoning-a") },
            { "type": "reasoning", "text": "S1", "providerMetadata": metadata("reasoning-b") },
        ])
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn preserves_response_message_order_across_tool_and_text_steps() {
    let (done, resolve_done) = deferred_done();
    let mut session = FakeSession::new(parts(vec![json!({
        "type": "tool-call", "toolCallId": "lookup-call", "toolName": "lookup", "input": "{}",
    })]));
    session.done = Some(done);
    let on_submit: SubmitFn = Arc::new(move |input, emit| {
        let resolve_done = resolve_done.clone();
        Box::pin(async move {
            if !input.output.is_number() {
                return Err(plain_error("lookup must return a number"));
            }
            for event in parts(vec![
                json!({ "type": "tool-result", "toolCallId": input.tool_call_id, "toolName": "lookup", "result": input.output }),
                json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_calls" }, "usage": zero_usage() }),
                json!({ "type": "text-delta", "id": "answer", "delta": "The answer is 42." }),
                json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "stop" }, "usage": zero_usage() }),
                json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "stop" }, "totalUsage": zero_usage() }),
            ]) {
                emit(event);
            }
            resolve(&resolve_done);
            Ok(())
        })
    });
    session.on_submit = Some(on_submit);
    let harness = SessionHarness::new(session);
    let mut s = HarnessAgentSettings::new(harness);
    s.tools = Some(tools(vec![(
        "lookup",
        Tool::new(z_object(&[])).with_execute(|_, _| async { Ok(json!(42)) }),
    )]));
    let agent = agent(s);
    let session = new_session(&agent).await;

    let result = agent.generate(call(&session, "Look it up")).await.unwrap();

    assert_eq!(result.steps.len(), 2);
    assert_eq!(
        to_json(&result.response_messages),
        json!([
            {
                "role": "assistant",
                "content": [{ "type": "tool-call", "toolCallId": "lookup-call", "toolName": "lookup", "input": {} }],
            },
            {
                "role": "tool",
                "content": [{
                    "type": "tool-result", "toolCallId": "lookup-call", "toolName": "lookup",
                    "output": { "type": "json", "value": 42 },
                }],
            },
            {
                "role": "assistant",
                "content": [{ "type": "text", "text": "The answer is 42." }],
            },
        ])
    );

    session.destroy().await.unwrap();
}
