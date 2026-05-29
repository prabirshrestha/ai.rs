use agent::{
    InMemorySessionRepo, InMemorySessionStorage, InMemorySessionStorageOptions, MoveToSummary,
    SessionCreateOptions, SessionErrorCode, SessionForkOptions, SessionForkPosition,
    SessionMetadata, SessionRepo, SessionStorage, SessionTreeEntry,
};
use ai::{
    AssistantContent, AssistantMessage, Message, StopReason, TextContent, Usage, UserContent,
    UserMessage, UserMessageContent,
};

fn assistant_text(provider: &str, model: &str, text: &str) -> Message {
    Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        api: "openai-responses".to_string(),
        provider: provider.to_string(),
        model: model.to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: ai::utils::time::now_millis(),
    })
}

fn message_text(message: &Message) -> Option<&str> {
    match message {
        Message::User(UserMessage {
            content: UserMessageContent::Text(text),
            ..
        }) => Some(text),
        Message::User(UserMessage {
            content: UserMessageContent::Parts(parts),
            ..
        }) => parts.iter().find_map(|part| match part {
            UserContent::Text(text) => Some(text.text.as_str()),
            UserContent::Image(_) => None,
        }),
        Message::Assistant(message) => message.content.iter().find_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }),
        Message::ToolResult(_) => None,
    }
}

#[tokio::test]
async fn session_context_tracks_state_and_messages() {
    let repo = InMemorySessionRepo::new();
    let session = repo
        .create(SessionCreateOptions {
            id: Some("session-1".to_string()),
        })
        .await
        .unwrap();

    session
        .append_model_change("openai", "gpt-5")
        .await
        .unwrap();
    session.append_thinking_level_change("high").await.unwrap();
    session
        .append_active_tools_change(vec!["shell".to_string(), "edit".to_string()])
        .await
        .unwrap();
    session
        .append_message(Message::user_text("hello"))
        .await
        .unwrap();
    session
        .append_message(assistant_text("anthropic", "claude-sonnet-4-5", "hi"))
        .await
        .unwrap();

    let context = session.build_context().await.unwrap();
    assert_eq!(context.thinking_level, "high");
    assert_eq!(
        context.active_tool_names.as_deref(),
        Some(["shell".to_string(), "edit".to_string()].as_slice())
    );
    assert_eq!(context.model.unwrap().provider, "anthropic");
    assert_eq!(
        context
            .messages
            .iter()
            .filter_map(message_text)
            .collect::<Vec<_>>(),
        ["hello", "hi"]
    );
}

#[tokio::test]
async fn labels_and_session_names_are_replayed_from_entries() {
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    let user_id = session
        .append_message(Message::user_text("important"))
        .await
        .unwrap();

    session
        .append_label(user_id.clone(), Some("  Keep this  ".to_string()))
        .await
        .unwrap();
    assert_eq!(
        session.get_label(&user_id).await.unwrap().as_deref(),
        Some("Keep this")
    );

    session.append_label(user_id.clone(), None).await.unwrap();
    assert_eq!(session.get_label(&user_id).await.unwrap(), None);

    session.append_session_name("  Research  ").await.unwrap();
    assert_eq!(
        session.get_session_name().await.unwrap().as_deref(),
        Some("Research")
    );
}

#[tokio::test]
async fn move_to_records_leaf_and_optional_branch_summary() {
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    let user_id = session
        .append_message(Message::user_text("root"))
        .await
        .unwrap();
    session
        .append_message(assistant_text("openai", "gpt-5", "discarded branch"))
        .await
        .unwrap();

    let summary_id = session
        .move_to(
            Some(user_id.clone()),
            Some(MoveToSummary {
                summary: "came back from discarded branch".to_string(),
                details: None,
                from_hook: Some(true),
            }),
        )
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        session.get_leaf_id().await.unwrap().as_deref(),
        Some(summary_id.as_str())
    );
    let branch = session.get_branch(None).await.unwrap();
    assert_eq!(
        branch.iter().map(SessionTreeEntry::id).collect::<Vec<_>>(),
        [user_id.as_str(), summary_id.as_str()]
    );

    let context = session.build_context().await.unwrap();
    let texts = context
        .messages
        .iter()
        .filter_map(message_text)
        .collect::<Vec<_>>();
    assert_eq!(texts.len(), 2);
    assert_eq!(texts[0], "root");
    assert!(texts[1].contains("came back from discarded branch"));
}

#[tokio::test]
async fn fork_uses_before_and_at_positions() {
    let repo = InMemorySessionRepo::new();
    let source = repo.create(SessionCreateOptions::default()).await.unwrap();
    let _user_1 = source
        .append_message(Message::user_text("u1"))
        .await
        .unwrap();
    source
        .append_message(assistant_text("openai", "gpt-5", "a1"))
        .await
        .unwrap();
    let user_2 = source
        .append_message(Message::user_text("u2"))
        .await
        .unwrap();
    let assistant_2 = source
        .append_message(assistant_text("openai", "gpt-5", "a2"))
        .await
        .unwrap();
    let metadata = source.get_metadata().await.unwrap();

    let before = repo
        .fork(
            metadata.clone(),
            SessionForkOptions {
                entry_id: Some(user_2.clone()),
                position: None,
                id: Some("before".to_string()),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        before
            .build_context()
            .await
            .unwrap()
            .messages
            .iter()
            .filter_map(message_text)
            .collect::<Vec<_>>(),
        ["u1", "a1"]
    );

    let at = repo
        .fork(
            metadata.clone(),
            SessionForkOptions {
                entry_id: Some(assistant_2.clone()),
                position: Some(SessionForkPosition::At),
                id: Some("at".to_string()),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        at.build_context()
            .await
            .unwrap()
            .messages
            .iter()
            .filter_map(message_text)
            .collect::<Vec<_>>(),
        ["u1", "a1", "u2", "a2"]
    );

    let err = match repo
        .fork(
            metadata,
            SessionForkOptions {
                entry_id: Some(assistant_2),
                position: None,
                id: None,
            },
        )
        .await
    {
        Ok(_) => panic!("fork before assistant should fail"),
        Err(err) => err,
    };
    assert_eq!(err.code, SessionErrorCode::InvalidForkTarget);
}

#[tokio::test]
async fn storage_reports_missing_leaf_and_invalid_parent() {
    let bad_leaf = SessionTreeEntry::Leaf(agent::LeafEntry {
        id: "leaf-entry".to_string(),
        parent_id: None,
        timestamp: "0".to_string(),
        target_id: Some("missing".to_string()),
    });
    let err = InMemorySessionStorage::with_options(InMemorySessionStorageOptions {
        entries: vec![bad_leaf],
        metadata: SessionMetadata {
            id: "bad".to_string(),
            created_at: "0".to_string(),
        },
    })
    .unwrap_err();
    assert_eq!(err.code, SessionErrorCode::InvalidSession);

    let storage = InMemorySessionStorage::new().unwrap();
    let missing = storage
        .get_path_to_root(Some("missing".to_string()))
        .await
        .unwrap_err();
    assert_eq!(missing.code, SessionErrorCode::NotFound);

    storage
        .append_entry(SessionTreeEntry::Message(agent::MessageEntry {
            id: "child".to_string(),
            parent_id: Some("missing-parent".to_string()),
            timestamp: "0".to_string(),
            message: Message::user_text("child"),
        }))
        .await
        .unwrap();
    let invalid = storage
        .get_path_to_root(Some("child".to_string()))
        .await
        .unwrap_err();
    assert_eq!(invalid.code, SessionErrorCode::InvalidSession);
}
