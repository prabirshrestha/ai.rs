use std::sync::{Arc, Mutex as StdMutex};

use serde_json::json;

use super::*;
use crate::harness::adapters::pi::pi_session_log::memory_storage::MemoryPiSessionStorage;
use crate::harness::adapters::pi::pi_test_support::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::session::{HarnessSandbox, HarnessTurnOptions};
use crate::harness::tools::HarnessSkill;
use crate::providers::faux::{
    FauxAssistantMessageOptions, FauxProviderRegistration, faux_assistant_message, faux_tool_call,
    register_faux_provider,
};
use crate::{AgentEvent, StopReason, ToolResultContent};

struct Fixture {
    _temp: TempDir,
    root: String,
    home: String,
    registration: FauxProviderRegistration,
    contexts: Arc<StdMutex<Vec<crate::Context>>>,
}

fn fixture() -> Fixture {
    let temp = mkdtemp("pi-harness-");
    let base = canonical(temp.path());
    let root = format!("{base}/work");
    let home = format!("{base}/home");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    Fixture {
        _temp: temp,
        root,
        home,
        registration: register_faux_provider(None),
        contexts: Arc::default(),
    }
}

impl Fixture {
    fn settings(&self) -> PiHarnessSettings {
        let contexts = self.contexts.clone();
        PiHarnessSettings {
            models: vec![self.registration.get_model()],
            model: Some(self.registration.get_model().id),
            env: Some(IndexMap::new()),
            stream_fn: Some(Arc::new(move |model, context, options| {
                contexts.lock().unwrap().push(context.clone());
                Box::pin(async move { crate::stream_simple(model, context, Some(options)) })
            })),
            ..Default::default()
        }
    }

    fn start_options(&self) -> HarnessStartOptions {
        HarnessStartOptions::new(
            "pi-session",
            HarnessSandbox::Basic(Arc::new(
                LocalSandbox::new(self.root.clone()).with_home(self.home.clone()),
            )),
            self.root.clone(),
        )
    }

    fn system_prompt(&self, index: usize) -> String {
        self.contexts.lock().unwrap()[index]
            .system_prompt
            .clone()
            .unwrap_or_default()
    }
}

fn tool_use(name: &str, args: serde_json::Value) -> crate::AssistantMessage {
    faux_assistant_message(
        vec![faux_tool_call(name, args, Some("call-1".into()))],
        Some(FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..Default::default()
        }),
    )
}

async fn run(
    session: &dyn HarnessSession,
    text: &str,
    options: HarnessTurnOptions,
) -> Vec<HarnessEvent> {
    session
        .prompt(Message::user_text(text), options)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .0
}

#[tokio::test]
async fn runs_pi_tools_with_pi_system_prompt_and_instructions() {
    let f = fixture();
    f.registration.set_responses([
        tool_use("write", json!({ "file_path": "a.txt", "content": "hi" })),
        faux_assistant_message("done", None),
    ]);
    let session = PiHarness::new(f.settings())
        .start(f.start_options())
        .await
        .unwrap();
    let mut options = HarnessTurnOptions::default();
    options.settings.instructions = Some("Always answer in French.".into());
    let events = run(session.as_ref(), "write a.txt", options).await;

    assert_eq!(
        std::fs::read_to_string(format!("{}/a.txt", f.root)).unwrap(),
        "hi"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        HarnessEvent::FileChange { kind: FileChangeKind::Create, path } if path == "a.txt"
    )));
    let prompt = f.system_prompt(0);
    assert!(prompt.starts_with(super::super::pi_system_prompt::PI_PREAMBLE));
    assert!(prompt.contains("<addendum>\nAlways answer in French.\n</addendum>"));
    assert!(prompt.ends_with(&format!("<cwd>\n{}\n</cwd>", f.root)));
    let tools: Vec<_> = f.contexts.lock().unwrap()[0]
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    assert_eq!(
        tools,
        ["read", "write", "edit", "bash", "grep", "find", "ls"]
    );
    // The instructions are not framed into the message.
    match &session.messages().await.unwrap()[0] {
        Message::User(user) => {
            assert!(matches!(&user.content, UserMessageContent::Text(t) if t == "write a.txt"))
        }
        other => panic!("{other:?}"),
    }
    f.registration.unregister();
}

#[tokio::test]
async fn writes_skills_to_the_sandbox_lists_them_and_expands_skill_commands() {
    let f = fixture();
    f.registration
        .set_responses([faux_assistant_message("ok", None)]);
    let session = PiHarness::new(f.settings())
        .start(f.start_options())
        .await
        .unwrap();
    let mut options = HarnessTurnOptions::default();
    options.settings.skills = vec![HarnessSkill {
        name: "release-notes".into(),
        description: "Use when writing release notes.".into(),
        content: "# Release notes\n\nUse three sections.".into(),
        files: None,
    }];
    run(session.as_ref(), "/skill:release-notes Draft v2.", options).await;

    let skill_path = format!("{}/.agents/skills/release-notes/SKILL.md", f.home);
    assert!(
        std::fs::read_to_string(&skill_path)
            .unwrap()
            .contains("Use three sections.")
    );
    assert!(
        f.system_prompt(0)
            .contains(&format!("<location>{skill_path}</location>"))
    );
    match &session.messages().await.unwrap()[0] {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(text) => {
                assert!(text.starts_with(&format!(
                    "<skill name=\"release-notes\" location=\"{skill_path}\">"
                )));
                assert!(text.ends_with("</skill>\n\nDraft v2."));
            }
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
    f.registration.unregister();
}

#[tokio::test]
async fn resumes_from_entries_and_from_a_session_storage() {
    let f = fixture();
    // Without a storage, the state holds the entries.
    f.registration
        .set_responses([faux_assistant_message("one", None)]);
    let harness = PiHarness::new(f.settings());
    let session = harness.start(f.start_options()).await.unwrap();
    run(session.as_ref(), "first", HarnessTurnOptions::default()).await;
    let state = session.stop().await.unwrap();
    let entries = state.data["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0]["type"], "session");
    assert_eq!(entries[0]["id"], "pi-session");
    assert_eq!(entries[1]["type"], "message");

    let mut options = f.start_options();
    options.resume = Some(state);
    let resumed = harness.start(options).await.unwrap();
    assert!(resumed.is_resume());
    assert_eq!(resumed.messages().await.unwrap().len(), 2);

    // With a storage, the state holds the leaf.
    let storage = Arc::new(MemoryPiSessionStorage::default());
    let mut settings = f.settings();
    settings.session_storage = Some(storage.clone());
    let harness = PiHarness::new(settings);
    f.registration
        .set_responses([faux_assistant_message("two", None)]);
    let session = harness.start(f.start_options()).await.unwrap();
    run(session.as_ref(), "second", HarnessTurnOptions::default()).await;
    let state = session.stop().await.unwrap();
    assert!(state.data.get("entries").is_none());
    let leaf = state.data["leafId"].as_str().unwrap().to_string();
    assert_eq!(storage.entries.lock().unwrap().len(), 2);

    f.registration
        .set_responses([faux_assistant_message("three", None)]);
    let mut options = f.start_options();
    options.resume = Some(state);
    let resumed = harness.start(options).await.unwrap();
    assert_eq!(resumed.messages().await.unwrap().len(), 2);
    run(resumed.as_ref(), "third", HarnessTurnOptions::default()).await;
    let state = resumed.detach().await.unwrap();
    assert_ne!(state.data["leafId"].as_str().unwrap(), leaf);
    assert_eq!(storage.entries.lock().unwrap().len(), 4);

    // A storage-backed state cannot load without the storage.
    let mut options = f.start_options();
    options.resume = Some(state);
    let error = PiHarness::new(f.settings())
        .start(options)
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("has no sessionStorage"));
    f.registration.unregister();
}

#[tokio::test]
async fn reports_find_as_glob_and_rejects_unknown_models() {
    let f = fixture();
    f.registration.set_responses([
        tool_use("find", json!({ "pattern": "*.md" })),
        faux_assistant_message("done", None),
    ]);
    let session = PiHarness::new(f.settings())
        .start(f.start_options())
        .await
        .unwrap();
    let events = run(session.as_ref(), "find", HarnessTurnOptions::default()).await;
    let ended: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            HarnessEvent::Agent(AgentEvent::ToolExecutionEnd {
                tool_name, result, ..
            }) => Some((tool_name.clone(), result.content.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(ended.len(), 1);
    assert_eq!(ended[0].0, "glob");
    assert!(matches!(&ended[0].1[0], ToolResultContent::Text(text) if text.text.is_empty()));

    let mut options = HarnessTurnOptions::default();
    options.settings.model = Some("nope".into());
    let error = session
        .prompt(Message::user_text("hi"), options)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Harness 'pi' has no model 'nope' in its catalog."
    );
    f.registration.unregister();
}
