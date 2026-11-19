use std::sync::{Arc, Mutex as StdMutex};

use serde_json::json;

use super::*;
use crate::harness::adapters::pi::pi_test_support::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::session::{HarnessSandbox, HarnessTurnOptions};
use crate::harness::tools::{BuiltinToolFilter, PermissionMode};
use crate::providers::faux::{
    FauxAssistantMessageOptions, FauxProviderRegistration, faux_assistant_message, faux_tool_call,
    register_faux_provider,
};
use crate::{
    AgentEvent, AgentToolBuilder, AgentToolResult, AssistantContent, BeforeToolCallResult, Message,
    StopReason, ToolResultContent,
};

struct Fixture {
    _temp: TempDir,
    root: String,
    registration: FauxProviderRegistration,
    harness: NativeHarness,
}

fn fixture() -> Fixture {
    let temp = mkdtemp("native-harness-");
    let root = canonical(temp.path());
    let registration = register_faux_provider(None);
    let harness = NativeHarness::new(registration.get_model());
    Fixture {
        _temp: temp,
        root,
        registration,
        harness,
    }
}

impl Fixture {
    fn start_options(&self) -> HarnessStartOptions {
        HarnessStartOptions::new(
            "session-1",
            HarnessSandbox::Basic(Arc::new(LocalSandbox::new(self.root.clone()))),
            self.root.clone(),
        )
    }
}

fn tool_use(name: &str, args: serde_json::Value, id: &str) -> crate::AssistantMessage {
    faux_assistant_message(
        vec![faux_tool_call(name, args, Some(id.into()))],
        Some(FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..Default::default()
        }),
    )
}

fn tool_result_text(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some(
                result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ToolResultContent::Text(text) => Some(text.text.clone()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

fn last_text(messages: &[Message]) -> String {
    match messages.last() {
        Some(Message::Assistant(assistant)) => assistant
            .content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[tokio::test]
async fn runs_builtin_tools_in_the_sandbox_and_reports_harness_names() {
    let f = fixture();
    std::fs::write(format!("{}/a.txt", f.root), "x").unwrap();
    f.registration.set_responses([
        tool_use(
            "write",
            json!({ "file_path": "hello.txt", "content": "hi" }),
            "call-1",
        ),
        tool_use("find", json!({ "pattern": "*.txt" }), "call-2"),
        faux_assistant_message("done", None),
    ]);
    let session = f.harness.start(f.start_options()).await.unwrap();
    let turn = session
        .prompt(
            Message::user_text("write a file"),
            HarnessTurnOptions::default(),
        )
        .await
        .unwrap();
    let (events, messages) = turn.collect().await.unwrap();

    assert_eq!(
        std::fs::read_to_string(format!("{}/hello.txt", f.root)).unwrap(),
        "hi"
    );
    assert_eq!(
        tool_result_text(&messages),
        ["Wrote hello.txt", "a.txt\nhello.txt"]
    );
    assert_eq!(last_text(&messages), "done");

    let started: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            HarnessEvent::Agent(AgentEvent::ToolExecutionStart { tool_name, .. }) => {
                Some(tool_name.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(started, ["write", "glob"]);
    assert!(events.iter().any(|event| matches!(
        event,
        HarnessEvent::FileChange { kind: FileChangeKind::Create, path } if path == "hello.txt"
    )));
    f.registration.unregister();
}

#[tokio::test]
async fn resumes_from_the_saved_history() {
    let f = fixture();
    f.registration
        .set_responses([faux_assistant_message("first", None)]);
    let session = f.harness.start(f.start_options()).await.unwrap();
    assert!(!session.is_resume());
    let (_, first) = session
        .prompt(Message::user_text("one"), HarnessTurnOptions::default())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(first.len(), 2);
    let state = session.stop().await.unwrap();
    assert_eq!(state.harness_id, "ai");
    assert!(
        session
            .prompt(Message::user_text("again"), HarnessTurnOptions::default())
            .await
            .is_err()
    );

    let mut options = f.start_options();
    options.resume = Some(serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap());
    let resumed = f.harness.start(options).await.unwrap();
    assert!(resumed.is_resume());
    assert_eq!(resumed.messages().await.unwrap().len(), 2);
    f.registration
        .set_responses([faux_assistant_message("second", None)]);
    let (_, second) = resumed
        .prompt(Message::user_text("two"), HarnessTurnOptions::default())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(last_text(&second), "second");
    assert_eq!(resumed.messages().await.unwrap().len(), 4);
    f.registration.unregister();
}

#[tokio::test]
async fn asks_before_bash_when_edits_are_allowed() {
    let f = fixture();
    f.registration.set_responses([
        tool_use(
            "write",
            json!({ "file_path": "w.txt", "content": "ok" }),
            "call-1",
        ),
        tool_use("bash", json!({ "command": "touch ran.txt" }), "call-2"),
        faux_assistant_message("done", None),
    ]);
    let mut start = f.start_options();
    start.permission_mode = PermissionMode::AllowEdits;
    let session = f.harness.start(start).await.unwrap();
    let asked = Arc::new(StdMutex::new(Vec::new()));
    let mut options = HarnessTurnOptions::default();
    {
        let asked = asked.clone();
        options.before_tool_call = Some(Arc::new(move |context, _| {
            asked.lock().unwrap().push(context.tool_call.name.clone());
            Box::pin(async {
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: Some("denied".into()),
                }))
            })
        }));
    }
    let (_, messages) = session
        .prompt(Message::user_text("go"), options)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(*asked.lock().unwrap(), ["bash"]);
    assert!(std::path::Path::new(&format!("{}/w.txt", f.root)).exists());
    assert!(!std::path::Path::new(&format!("{}/ran.txt", f.root)).exists());
    assert_eq!(tool_result_text(&messages)[1], "denied");
    f.registration.unregister();
}

#[tokio::test]
async fn blocks_unapproved_builtins_without_a_handler() {
    let f = fixture();
    f.registration.set_responses([
        tool_use(
            "edit",
            json!({ "file_path": "x", "old_string": "a", "new_string": "b" }),
            "c",
        ),
        faux_assistant_message("done", None),
    ]);
    let mut start = f.start_options();
    start.permission_mode = PermissionMode::AllowReads;
    let session = f.harness.start(start).await.unwrap();
    let (_, messages) = session
        .prompt(Message::user_text("go"), HarnessTurnOptions::default())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        tool_result_text(&messages),
        ["Tool 'edit' requires approval, but no approval handler was provided."]
    );
    f.registration.unregister();
}

#[tokio::test]
async fn filters_builtins_and_runs_host_tools() {
    let f = fixture();
    let seen_tools = Arc::new(StdMutex::new(Vec::new()));
    let harness = {
        let seen_tools = seen_tools.clone();
        let registration_model = f.registration.get_model();
        NativeHarness::new(registration_model).with_stream_fn(Arc::new(
            move |model, context, options| {
                seen_tools.lock().unwrap().push(
                    context
                        .tools
                        .iter()
                        .map(|tool| tool.name.clone())
                        .collect::<Vec<_>>(),
                );
                Box::pin(async move { crate::stream_simple(model, context, Some(options)) })
            },
        ))
    };
    f.registration.set_responses([
        tool_use("weather", json!({ "city": "Paris" }), "call-1"),
        faux_assistant_message("sunny", None),
    ]);
    let mut start = f.start_options();
    start.builtin_tool_filter = Some(BuiltinToolFilter::Allow {
        tool_names: vec!["read".into(), "glob".into()],
    });
    let session = harness.start(start).await.unwrap();
    let weather = AgentToolBuilder::new("weather")
        .description("Weather for a city")
        .parameters(json!({ "type": "object", "properties": { "city": { "type": "string" } } }))
        .execute(|args| async move {
            Ok(AgentToolResult::text(format!(
                "sunny in {}",
                args["city"].as_str().unwrap()
            )))
        })
        .build()
        .unwrap();
    let mut options = HarnessTurnOptions::default();
    options.settings.tools.push(weather);
    let (_, messages) = session
        .prompt(Message::user_text("weather?"), options)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(tool_result_text(&messages), ["sunny in Paris"]);
    assert_eq!(seen_tools.lock().unwrap()[0], ["read", "find", "weather"]);
    f.registration.unregister();
}

#[tokio::test]
async fn frames_instructions_into_the_first_message_only() {
    let f = fixture();
    f.registration.set_responses([
        faux_assistant_message("a", None),
        faux_assistant_message("b", None),
    ]);
    let session = f.harness.start(f.start_options()).await.unwrap();
    for text in ["first", "second"] {
        let mut options = HarnessTurnOptions::default();
        options.settings.instructions = Some("Be brief.".into());
        session
            .prompt(Message::user_text(text), options)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    let users: Vec<String> = session
        .messages()
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            Message::User(user) => match user.content {
                crate::UserMessageContent::Text(text) => Some(text),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(users[0].contains("<session-instructions>\nThe block below"));
    assert!(users[0].contains("Be brief.") && users[0].contains("<user-message>\nfirst\n"));
    assert_eq!(users[1], "second");
    f.registration.unregister();
}

#[tokio::test]
async fn rejects_unknown_models_and_host_tool_name_conflicts() {
    let f = fixture();
    let session = f.harness.start(f.start_options()).await.unwrap();
    let mut options = HarnessTurnOptions::default();
    options.settings.model = Some("missing".into());
    let error = session
        .prompt(Message::user_text("hi"), options)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Unknown model 'missing'"));

    let bash = AgentToolBuilder::new("bash")
        .description("Run a command")
        .execute(|_| async { Ok(AgentToolResult::text("")) })
        .build()
        .unwrap();
    let mut options = HarnessTurnOptions::default();
    options.settings.tools.push(bash);
    let error = session
        .prompt(Message::user_text("hi"), options)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("conflicts with a built-in tool"));
    f.registration.unregister();
}
