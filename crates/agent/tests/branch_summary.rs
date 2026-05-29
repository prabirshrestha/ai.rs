use std::sync::{Arc, Mutex};

use agent::{
    BranchSummaryEntry, BranchSummaryErrorCode, GenerateBranchSummaryOptions, InMemorySessionRepo,
    SUMMARIZATION_SYSTEM_PROMPT, SessionCreateOptions, SessionRepo, SessionTreeEntry,
    collect_entries_for_branch_summary, compute_file_lists, generate_branch_summary,
    prepare_branch_entries,
};
use ai::{
    AssistantContent, AssistantMessage, Context, FauxAssistantMessageOptions, FauxResponseStep,
    Message, StopReason, StreamOptions, TextContent, ToolCall, ToolResultContent,
    ToolResultMessage, Usage, UserContent, UserMessageContent, faux_assistant_message,
    register_faux_provider,
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

fn user_prompt_from_context(context: &Context) -> &str {
    let Message::User(message) = &context.messages[0] else {
        panic!("expected user summarization message");
    };
    let UserMessageContent::Parts(parts) = &message.content else {
        panic!("expected parts content");
    };
    let UserContent::Text(text) = &parts[0] else {
        panic!("expected text prompt");
    };
    &text.text
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

#[tokio::test]
async fn generates_branch_summary_with_prompt_options_and_file_details() {
    let registration = register_faux_provider(None);
    let captured: Arc<Mutex<Option<(Context, StreamOptions)>>> = Arc::new(Mutex::new(None));
    registration.set_responses([FauxResponseStep::factory({
        let captured = Arc::clone(&captured);
        move |context, options, _state, _model| {
            let captured = Arc::clone(&captured);
            async move {
                *captured.lock().unwrap() = Some((context, options));
                Ok(faux_assistant_message("## Goal\nSummarized branch", None))
            }
        }
    })]);

    let entries = vec![
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "u1".to_string(),
            parent_id: None,
            timestamp: "1".to_string(),
            message: Message::user_text("please inspect the repo"),
        }),
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "a1".to_string(),
            parent_id: Some("u1".to_string()),
            timestamp: "1".to_string(),
            message: assistant_tool_call("read", "README.md"),
        }),
        SessionTreeEntry::Message(agent::MessageEntry {
            id: "a2".to_string(),
            parent_id: Some("a1".to_string()),
            timestamp: "1".to_string(),
            message: assistant_tool_call("edit", "src/lib.rs"),
        }),
    ];
    let mut options = GenerateBranchSummaryOptions::new(registration.get_model(), "test-key");
    options
        .headers
        .insert("x-test".to_string(), "yes".to_string());
    options.custom_instructions = Some("focus on changed files".to_string());

    let result = generate_branch_summary(&entries, options).await.unwrap();

    assert!(
        result.summary.starts_with(
            "The user explored a different conversation branch before returning here."
        )
    );
    assert!(result.summary.contains("## Goal\nSummarized branch"));
    assert!(
        result
            .summary
            .contains("<read-files>\nREADME.md\n</read-files>")
    );
    assert!(
        result
            .summary
            .contains("<modified-files>\nsrc/lib.rs\n</modified-files>")
    );
    assert_eq!(result.read_files, vec!["README.md".to_string()]);
    assert_eq!(result.modified_files, vec!["src/lib.rs".to_string()]);

    let captured = captured.lock().unwrap().clone().unwrap();
    assert_eq!(
        captured.0.system_prompt.as_deref(),
        Some(SUMMARIZATION_SYSTEM_PROMPT)
    );
    assert_eq!(captured.1.max_tokens, Some(2048));
    assert_eq!(captured.1.api_key.as_deref(), Some("test-key"));
    assert_eq!(
        captured.1.headers.get("x-test").map(String::as_str),
        Some("yes")
    );
    let prompt = user_prompt_from_context(&captured.0);
    assert!(prompt.contains("<conversation>\n[User]: please inspect the repo"));
    assert!(prompt.contains("[Assistant tool calls]: read(path=\"README.md\")"));
    assert!(prompt.contains("Additional focus: focus on changed files"));

    registration.unregister();
}

#[tokio::test]
async fn generate_branch_summary_skips_provider_when_no_messages() {
    let registration = register_faux_provider(None);
    let result = generate_branch_summary(
        &[],
        GenerateBranchSummaryOptions::new(registration.get_model(), "test-key"),
    )
    .await
    .unwrap();

    assert_eq!(result.summary, "No content to summarize");
    assert!(result.read_files.is_empty());
    assert!(result.modified_files.is_empty());
    assert_eq!(registration.state.call_count(), 0);

    registration.unregister();
}

#[tokio::test]
async fn generate_branch_summary_maps_provider_errors() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message(
        "",
        Some(FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some("provider failed".to_string()),
            ..Default::default()
        }),
    )]);
    let entries = vec![SessionTreeEntry::Message(agent::MessageEntry {
        id: "u1".to_string(),
        parent_id: None,
        timestamp: "1".to_string(),
        message: Message::user_text("summarize this"),
    })];

    let error = generate_branch_summary(
        &entries,
        GenerateBranchSummaryOptions::new(registration.get_model(), "test-key"),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code, BranchSummaryErrorCode::SummarizationFailed);
    assert_eq!(error.message(), "Branch summary failed: provider failed");

    registration.unregister();
}
