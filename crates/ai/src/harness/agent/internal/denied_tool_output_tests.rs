//! Port of `agent/internal/denied-tool-output.test.ts`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::future::FutureExt;
use serde_json::json;

use super::tests::tool_outputs::{approval_config, custom_approval, submitted_json};
use super::tests::{FakeSession, WORK_DIR, empty_usage, p, parts, sandbox, to_json};
use super::*;
use crate::harness::sdk::messages::ToolResultOutput;
use crate::harness::sdk::tool::Tool;

const REASON: &str = "weather disabled";
const FILTERED_REASON: &str =
    "Tool 'weather' is inactive due to the HarnessAgent tool filtering policy.";

fn finish_reason() -> JsonValue {
    json!({ "unified": "stop", "raw": "end_turn" })
}

fn finish_events() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "text-start", "id": "answer" }),
        json!({ "type": "text-delta", "id": "answer", "delta": "Weather is disabled." }),
        json!({ "type": "text-end", "id": "answer" }),
        json!({ "type": "finish-step", "finishReason": finish_reason(), "usage": empty_usage() }),
        json!({ "type": "finish", "finishReason": finish_reason(), "totalUsage": empty_usage() }),
    ])
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Source {
    Configured,
    Resume,
    Filtered,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Converter {
    Map,
    Validate,
    None,
}

/// `{ content: { text: string }[] }`.
fn is_weather_output(output: &JsonValue) -> bool {
    output["content"]
        .as_array()
        .is_some_and(|parts| parts.iter().all(|part| part["text"].is_string()))
}

fn weather_tool(converter: Converter) -> (Tool, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let executions = Arc::new(AtomicUsize::new(0));
    let conversions = Arc::new(AtomicUsize::new(0));
    let execute_count = executions.clone();
    let mut tool =
        Tool::new(super::tests::z_object(&[("city", "string")])).with_execute(move |_, _| {
            execute_count.fetch_add(1, Ordering::SeqCst);
            async { Ok(json!({ "content": [{ "text": "Sunny" }] })) }
        });
    if converter != Converter::None {
        let count = conversions.clone();
        tool = tool.with_to_model_output(move |options| {
            count.fetch_add(1, Ordering::SeqCst);
            let value = if converter == Converter::Validate {
                if is_weather_output(&options.output) {
                    "Valid weather".to_string()
                } else {
                    "Invalid weather result".to_string()
                }
            } else {
                options.output["content"]
                    .as_array()
                    .map(|parts| {
                        parts
                            .iter()
                            .map(|part| part["text"].as_str().unwrap_or_default())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default()
            };
            async move {
                Ok(ToolResultOutput::Text {
                    value,
                    provider_options: None,
                })
            }
        });
    }
    (tool, executions, conversions)
}

async fn run_case(source: Source, echo: bool, converter: Converter, configured: Option<&str>) {
    let label = format!("{source:?}, echo={echo}, converter={converter:?}, reason={configured:?}");
    let resume = source == Source::Resume;
    let reason = match source {
        Source::Filtered => Some(FILTERED_REASON.to_string()),
        _ => configured.map(str::to_string),
    };
    let mut denial = json!({ "type": "execution-denied" });
    if let Some(reason) = &reason {
        denial["reason"] = json!(reason);
    }

    let submissions: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>> = Arc::default();
    let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
    let done_tx = Arc::new(Mutex::new(Some(done_tx)));
    let script = if resume {
        Vec::new()
    } else {
        vec![p(
            json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": "{\"city\":\"SF\"}" }),
        )]
    };
    let log = submissions.clone();
    let mut session = FakeSession::new(script).with_submit(move |submission, emit| {
        if echo {
            emit(p(json!({
                "type": "tool-result",
                "toolCallId": submission.tool_call_id,
                "toolName": "weather",
                "result": submission.output,
            })));
        }
        log.lock().push(submission);
        if resume {
            emit(p(
                json!({ "type": "finish-step", "finishReason": finish_reason(), "usage": empty_usage() }),
            ));
        }
        for event in finish_events() {
            emit(event);
        }
        if let Some(tx) = done_tx.lock().take() {
            let _ = tx.send(());
        }
    });
    session.id = "denied-tool".into();
    session.done = Some(
        done_rx
            .map(|result| result.map_err(|e| e.to_string()))
            .boxed()
            .shared(),
    );

    let (weather, executions, conversions) = weather_tool(converter);
    let mut input = RunPromptInput::new(
        super::tests::fake_harness(),
        Arc::new(session),
        sandbox(),
        WORK_DIR,
    );
    input.mode = if resume {
        RunPromptMode::Continue
    } else {
        RunPromptMode::Prompt
    };
    input.prompt = Some("Get the weather".into());
    input.tools = super::tests::tools(vec![("weather", weather)]);
    if source == Source::Filtered {
        input.active_tools = Some(ToolSet::default());
    }
    input.tool_approval = Some(approval_config(if resume {
        json!({ "weather": "user-approval" })
    } else {
        let mut config = json!({ "type": "denied" });
        if let Some(reason) = configured {
            config["reason"] = json!(reason);
        }
        json!({ "weather": config })
    }));
    if resume {
        input.pending_tool_approvals = vec![custom_approval(
            "approval-1",
            "c1",
            "weather",
            "{\"city\":\"SF\"}",
        )];
        input.tool_approval_continuations = vec![ToolApprovalResponse {
            approval_id: "approval-1".into(),
            approved: false,
            reason: configured.map(str::to_string),
            provider_executed: None,
        }];
    }

    let output = run_prompt(input);
    // [rust] TS races the turn against a 1s timer and asserts it finished.
    let (stream_parts, messages) = tokio::time::timeout(Duration::from_secs(1), async {
        let stream_parts: Vec<JsonValue> = output
            .result
            .full_stream()
            .map(|part| part.to_json())
            .collect()
            .await;
        output.done.clone().await;
        let messages = to_json(&output.result.response_messages().await.unwrap());
        (stream_parts, messages)
    })
    .await
    .unwrap_or_else(|_| panic!("{label}: turn did not complete"));

    let submitted: Vec<JsonValue> = submissions.lock().iter().map(submitted_json).collect();
    assert_eq!(
        submitted,
        vec![json!({
            "toolCallId": "c1",
            "output": denial,
            "toolResult": {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "weather",
                "output": denial,
            },
        })],
        "{label}"
    );
    assert_eq!(executions.load(Ordering::SeqCst), 0, "{label}");
    assert!(
        super::tests::contains_matching(
            &stream_parts,
            json!({ "type": "text-delta", "text": "Weather is disabled." })
        ),
        "{label}"
    );
    assert!(
        stream_parts.iter().any(|p| p["type"] == "finish"),
        "{label}"
    );
    assert!(
        !stream_parts.iter().any(|p| p["type"] == "error"),
        "{label}"
    );
    if resume {
        let mut expected = json!({ "type": "tool-approval-response", "approved": false });
        if let Some(reason) = &reason {
            expected["reason"] = json!(reason);
        }
        assert!(
            super::tests::contains_matching(&stream_parts, expected),
            "{label}: {stream_parts:?}"
        );
    }
    let outputs: Vec<JsonValue> = messages
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .flat_map(|message| message["content"].as_array().cloned().unwrap_or_default())
        .filter(|part| part["type"] == "tool-result")
        .map(|part| part["output"].clone())
        .collect();
    let mut expected = Vec::new();
    if !resume {
        if source == Source::Configured {
            expected.push(denial.clone());
        }
        if echo {
            expected.push(denial.clone());
        }
    }
    assert_eq!(outputs, expected, "{label}");
    assert_eq!(conversions.load(Ordering::SeqCst), 0, "{label}");
}

#[tokio::test]
async fn run_prompt_denied_host_tool_output() {
    for source in [Source::Configured, Source::Resume, Source::Filtered] {
        for echo in [false, true] {
            for converter in [Converter::Map, Converter::Validate, Converter::None] {
                let reasons: &[Option<&str>] = if source == Source::Filtered {
                    &[Some(REASON)]
                } else {
                    &[Some(REASON), None]
                };
                for reason in reasons {
                    run_case(source, echo, converter, *reason).await;
                }
            }
        }
    }
}
