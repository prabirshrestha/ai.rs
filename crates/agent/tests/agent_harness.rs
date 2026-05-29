use std::collections::HashMap;
use std::sync::Arc;

use agent::{
    AgentHarness, AgentHarnessAuth, AgentHarnessAuthFn, AgentHarnessErrorCode, AgentHarnessOptions,
    AgentHarnessResources, AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch,
    InMemorySessionRepo, MapPatch, NavigateTreeOptions, NodeExecutionEnv, SessionCreateOptions,
    SessionRepo, apply_stream_options_patch, create_failure_message, merge_headers,
    validate_unique_names,
};
use ai::{
    AssistantContent, CacheRetention, Message, Model, StopReason, TextContent, Transport,
    faux_assistant_message, register_faux_provider,
};
use serde_json::json;

fn auth_fn() -> AgentHarnessAuthFn {
    Arc::new(|_model| {
        Box::pin(async {
            Some(AgentHarnessAuth {
                api_key: "test-key".to_string(),
                headers: HashMap::new(),
            })
        })
    })
}

fn assistant_text(text: &str) -> Message {
    let mut message = faux_assistant_message(text, None);
    message.content = vec![AssistantContent::Text(TextContent {
        text: text.to_string(),
        text_signature: None,
    })];
    Message::Assistant(message)
}

#[test]
fn stream_option_patches_set_clear_merge_and_delete_values() {
    let base = AgentHarnessStreamOptions {
        transport: Some(Transport::Sse),
        timeout_ms: Some(100),
        max_retries: Some(2),
        max_retry_delay_ms: Some(50),
        headers: HashMap::from([
            ("keep".to_string(), "yes".to_string()),
            ("delete".to_string(), "old".to_string()),
        ]),
        metadata: Some(json!({
            "keep": true,
            "delete": true
        })),
        cache_retention: Some(CacheRetention::Short),
    };
    let patch = AgentHarnessStreamOptionsPatch {
        transport: Some(Some(Transport::Websocket)),
        timeout_ms: Some(None),
        headers: Some(MapPatch::Merge(HashMap::from([
            ("add".to_string(), Some("new".to_string())),
            ("delete".to_string(), None),
        ]))),
        metadata: Some(MapPatch::Merge(HashMap::from([
            ("add".to_string(), Some(json!(1))),
            ("delete".to_string(), None),
        ]))),
        ..Default::default()
    };

    let result = apply_stream_options_patch(&base, Some(&patch));

    assert_eq!(result.transport, Some(Transport::Websocket));
    assert_eq!(result.timeout_ms, None);
    assert_eq!(result.max_retries, Some(2));
    assert_eq!(result.headers.get("keep").map(String::as_str), Some("yes"));
    assert_eq!(result.headers.get("add").map(String::as_str), Some("new"));
    assert!(!result.headers.contains_key("delete"));
    assert_eq!(result.metadata, Some(json!({"keep": true, "add": 1})));

    let cleared = apply_stream_options_patch(
        &result,
        Some(&AgentHarnessStreamOptionsPatch {
            headers: Some(MapPatch::Clear),
            metadata: Some(MapPatch::Clear),
            cache_retention: Some(None),
            ..Default::default()
        }),
    );
    assert!(cleared.headers.is_empty());
    assert_eq!(cleared.metadata, None);
    assert_eq!(cleared.cache_retention, None);
}

#[test]
fn merges_headers_in_source_order() {
    let headers = merge_headers([
        Some(HashMap::from([("a".to_string(), "1".to_string())])),
        None,
        Some(HashMap::from([
            ("a".to_string(), "2".to_string()),
            ("b".to_string(), "3".to_string()),
        ])),
    ]);

    assert_eq!(headers.get("a").map(String::as_str), Some("2"));
    assert_eq!(headers.get("b").map(String::as_str), Some("3"));
}

#[test]
fn validates_duplicate_names() {
    let error =
        validate_unique_names(["read", "write", "read"], "Duplicate tool name(s)").unwrap_err();

    assert_eq!(error.code, AgentHarnessErrorCode::InvalidArgument);
    assert_eq!(error.message(), "Duplicate tool name(s): read");
}

#[test]
fn creates_failure_assistant_messages() {
    let model = Model {
        id: "gpt-test".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        ..Default::default()
    };
    let Message::Assistant(message) = create_failure_message(&model, "boom", true) else {
        panic!("expected assistant message");
    };

    assert_eq!(message.api, "openai-responses");
    assert_eq!(message.provider, "openai");
    assert_eq!(message.model, "gpt-test");
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("boom"));
    assert!(message.content.len() == 1);
}

#[tokio::test]
async fn harness_mutations_update_session_state() {
    let registration = register_faux_provider(None);
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    let mut options = AgentHarnessOptions::new(
        NodeExecutionEnv::new("."),
        session,
        registration.get_model(),
    );
    options.resources = AgentHarnessResources {
        skills: Vec::new(),
        prompt_templates: Vec::new(),
    };
    let mut harness = AgentHarness::new(options).unwrap();
    let mut next_model = registration.get_model();
    next_model.id = "next-model".to_string();

    harness.set_model(next_model.clone()).await.unwrap();
    harness
        .set_thinking_level(ai::ModelThinkingLevel::High)
        .await
        .unwrap();

    let context = harness.session().build_context().await.unwrap();
    assert_eq!(context.model.unwrap().model_id, "next-model");
    assert_eq!(context.thinking_level, "high");

    registration.unregister();
}

#[tokio::test]
async fn harness_compact_generates_and_persists_compaction() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("## Goal\nCompacted", None)]);
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    session
        .append_message(Message::user_text("old work"))
        .await
        .unwrap();
    let mut options = AgentHarnessOptions::new(
        NodeExecutionEnv::new("."),
        session,
        registration.get_model(),
    );
    options.get_api_key_and_headers = Some(auth_fn());
    let mut harness = AgentHarness::new(options).unwrap();

    let result = harness.compact(None).await.unwrap();

    assert!(result.summary.contains("## Goal\nCompacted"));
    let leaf = harness.session().get_leaf_id().await.unwrap().unwrap();
    let entry = harness.session().get_entry(&leaf).await.unwrap().unwrap();
    assert!(matches!(entry, agent::SessionTreeEntry::Compaction(_)));

    registration.unregister();
}

#[tokio::test]
async fn harness_navigate_tree_generates_branch_summary() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("## Goal\nBranch", None)]);
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    let _u1 = session
        .append_message(Message::user_text("first"))
        .await
        .unwrap();
    let a1 = session
        .append_message(assistant_text("answer"))
        .await
        .unwrap();
    let _u2 = session
        .append_message(Message::user_text("second"))
        .await
        .unwrap();
    let _a2 = session
        .append_message(assistant_text("other branch"))
        .await
        .unwrap();
    let mut options = AgentHarnessOptions::new(
        NodeExecutionEnv::new("."),
        session,
        registration.get_model(),
    );
    options.get_api_key_and_headers = Some(auth_fn());
    let mut harness = AgentHarness::new(options).unwrap();

    let result = harness
        .navigate_tree(
            &a1,
            NavigateTreeOptions {
                summarize: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let summary = result.summary_entry.unwrap();
    assert!(summary.summary.contains("## Goal\nBranch"));
    assert_eq!(
        harness.session().get_leaf_id().await.unwrap().as_deref(),
        Some(summary.id.as_str())
    );

    registration.unregister();
}

#[tokio::test]
async fn harness_navigate_tree_to_user_returns_editor_text() {
    let registration = register_faux_provider(None);
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    let _u1 = session
        .append_message(Message::user_text("first"))
        .await
        .unwrap();
    let a1 = session
        .append_message(assistant_text("answer"))
        .await
        .unwrap();
    let u2 = session
        .append_message(Message::user_text("second"))
        .await
        .unwrap();
    let _a2 = session
        .append_message(assistant_text("later"))
        .await
        .unwrap();
    let mut harness = AgentHarness::new(AgentHarnessOptions::new(
        NodeExecutionEnv::new("."),
        session,
        registration.get_model(),
    ))
    .unwrap();

    let result = harness
        .navigate_tree(&u2, NavigateTreeOptions::default())
        .await
        .unwrap();

    assert_eq!(result.editor_text.as_deref(), Some("second"));
    assert_eq!(
        harness.session().get_leaf_id().await.unwrap().as_deref(),
        Some(a1.as_str())
    );

    registration.unregister();
}
