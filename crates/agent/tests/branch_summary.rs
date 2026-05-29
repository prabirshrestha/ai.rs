use agent::{
    BranchSummaryEntry, InMemorySessionRepo, SessionCreateOptions, SessionRepo, SessionTreeEntry,
    collect_entries_for_branch_summary, compute_file_lists, prepare_branch_entries,
};
use ai::{
    AssistantContent, AssistantMessage, Message, StopReason, TextContent, ToolCall,
    ToolResultContent, ToolResultMessage, Usage,
};
use serde_json::json;

fn assistant_text(text: &str) -> Message {
    Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "gpt-5".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 1,
    })
}

fn assistant_tool_call(name: &str, path: &str) -> Message {
    Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: format!("{name}-1"),
            name: name.to_string(),
            arguments: json!({ "path": path }),
            thought_signature: None,
        })],
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "gpt-5".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 1,
    })
}

#[tokio::test]
async fn collects_entries_from_old_leaf_to_common_ancestor() {
    let repo = InMemorySessionRepo::new();
    let session = repo.create(SessionCreateOptions::default()).await.unwrap();
    let _u1 = session
        .append_message(Message::user_text("u1"))
        .await
        .unwrap();
    let a1 = session.append_message(assistant_text("a1")).await.unwrap();
    let u2 = session
        .append_message(Message::user_text("u2"))
        .await
        .unwrap();
    let a2 = session.append_message(assistant_text("a2")).await.unwrap();

    let collected = collect_entries_for_branch_summary(&session, Some(&a2), &a1)
        .await
        .unwrap();
    assert_eq!(collected.common_ancestor_id.as_deref(), Some(a1.as_str()));
    assert_eq!(
        collected
            .entries
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        [u2.as_str(), a2.as_str()]
    );

    let empty = collect_entries_for_branch_summary(&session, None, &a1)
        .await
        .unwrap();
    assert!(empty.entries.is_empty());
    assert_eq!(empty.common_ancestor_id, None);
}

#[test]
fn prepares_branch_entries_and_file_operations() {
    let entries = vec![
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "u1".to_string(),
            parent_id: None,
            timestamp: "1".to_string(),
            message: Message::user_text("u1"),
        }),
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "tool".to_string(),
            parent_id: Some("u1".to_string()),
            timestamp: "1".to_string(),
            message: Message::ToolResult(ToolResultMessage {
                tool_call_id: "read-1".to_string(),
                tool_name: "read".to_string(),
                content: vec![ToolResultContent::text("ignored")],
                details: None,
                is_error: false,
                timestamp: 1,
            }),
        }),
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "a1".to_string(),
            parent_id: Some("tool".to_string()),
            timestamp: "1".to_string(),
            message: assistant_tool_call("read", "README.md"),
        }),
        SessionTreeEntry::BranchSummary(BranchSummaryEntry {
            id: "b1".to_string(),
            parent_id: Some("a1".to_string()),
            timestamp: "1".to_string(),
            from_id: "old".to_string(),
            summary: "branch".to_string(),
            details: Some(json!({
                "readFiles": ["src/lib.rs"],
                "modifiedFiles": ["src/main.rs"]
            })),
            from_hook: Some(false),
        }),
    ];

    let prepared = prepare_branch_entries(&entries, None);
    assert_eq!(prepared.messages.len(), 3);
    let (read_files, modified_files) = compute_file_lists(&prepared.file_ops);
    assert_eq!(read_files, ["README.md", "src/lib.rs"]);
    assert_eq!(modified_files, ["src/main.rs"]);
    assert!(prepared.total_tokens > 0);
}

#[test]
fn branch_preparation_respects_token_budget() {
    let entries = vec![
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "u1".to_string(),
            parent_id: None,
            timestamp: "1".to_string(),
            message: Message::user_text("old"),
        }),
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "u2".to_string(),
            parent_id: Some("u1".to_string()),
            timestamp: "1".to_string(),
            message: Message::user_text("x".repeat(100)),
        }),
    ];

    let prepared = prepare_branch_entries(&entries, Some(1));
    assert!(prepared.messages.is_empty());
    assert_eq!(prepared.total_tokens, 0);

    let summary_entries = vec![SessionTreeEntry::BranchSummary(BranchSummaryEntry {
        id: "b1".to_string(),
        parent_id: None,
        timestamp: "1".to_string(),
        from_id: "old".to_string(),
        summary: "x".repeat(100),
        details: None,
        from_hook: Some(false),
    })];
    let prepared = prepare_branch_entries(&summary_entries, Some(1));
    assert_eq!(prepared.messages.len(), 1);
}
