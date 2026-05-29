use agent::{
    ActiveToolsChangeEntry, CompactionEntry, CompactionSettings, MessageEntry, SessionTreeEntry,
    calculate_context_tokens, estimate_context_tokens, estimate_tokens, find_cut_point,
    get_last_assistant_usage, prepare_compaction, should_compact,
};
use ai::{
    AssistantContent, AssistantMessage, Message, StopReason, TextContent, ToolCall, Usage,
    UsageCost,
};
use serde_json::json;

fn user_entry(id: &str, parent_id: Option<&str>, text: &str) -> SessionTreeEntry {
    SessionTreeEntry::Message(MessageEntry {
        id: id.to_string(),
        parent_id: parent_id.map(ToString::to_string),
        timestamp: "1".to_string(),
        message: Message::user_text(text),
    })
}

fn assistant_entry(
    id: &str,
    parent_id: Option<&str>,
    text: &str,
    usage: Option<Usage>,
) -> SessionTreeEntry {
    SessionTreeEntry::Message(MessageEntry {
        id: id.to_string(),
        parent_id: parent_id.map(ToString::to_string),
        timestamp: "1".to_string(),
        message: assistant_message(text, usage),
    })
}

fn assistant_message(text: &str, usage: Option<Usage>) -> Message {
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
        usage: usage.unwrap_or_default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 1,
    })
}

fn usage(total_tokens: u32) -> Usage {
    Usage {
        input: 10,
        output: 20,
        cache_read: 30,
        cache_write: 40,
        total_tokens,
        cost: UsageCost::default(),
    }
}

#[test]
fn estimates_context_tokens_from_usage_and_trailing_messages() {
    assert_eq!(calculate_context_tokens(&usage(0)), 100);
    assert_eq!(calculate_context_tokens(&usage(77)), 77);

    let messages = vec![
        Message::user_text("hello"),
        assistant_message("answer", Some(usage(50))),
        Message::user_text("trailing text"),
    ];
    let estimate = estimate_context_tokens(&messages);
    assert_eq!(estimate.usage_tokens, 50);
    assert_eq!(estimate.trailing_tokens, estimate_tokens(&messages[2]));
    assert_eq!(estimate.last_usage_index, Some(1));
    assert_eq!(estimate.tokens, 50 + estimate.trailing_tokens);
}

#[test]
fn uses_heuristic_tokens_without_assistant_usage() {
    let message = assistant_message("", None);
    let Message::Assistant(mut assistant) = message else {
        unreachable!();
    };
    assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "tool-1".to_string(),
        name: "read".to_string(),
        arguments: json!({"path": "src/lib.rs"}),
        thought_signature: None,
    })];
    let message = Message::Assistant(assistant);
    assert!(estimate_tokens(&message) > 0);

    let estimate = estimate_context_tokens(&[Message::user_text("abcd"), message]);
    assert_eq!(estimate.usage_tokens, 0);
    assert_eq!(estimate.last_usage_index, None);
    assert!(estimate.tokens >= 1);
}

#[test]
fn detects_compaction_thresholds_and_last_usage() {
    let settings = CompactionSettings {
        enabled: true,
        reserve_tokens: 100,
        keep_recent_tokens: 20,
    };
    assert!(should_compact(901, 1000, &settings));
    assert!(!should_compact(900, 1000, &settings));
    assert!(!should_compact(
        10_000,
        1000,
        &CompactionSettings {
            enabled: false,
            ..settings
        }
    ));

    let entries = vec![
        user_entry("u1", None, "hello"),
        assistant_entry("a1", Some("u1"), "answer", Some(usage(123))),
    ];
    assert_eq!(
        get_last_assistant_usage(&entries).unwrap().total_tokens,
        123
    );
}

#[test]
fn finds_split_turn_cut_points() {
    let entries = vec![
        user_entry("u1", None, "short"),
        assistant_entry("a1", Some("u1"), "short", None),
        user_entry("u2", Some("a1"), &"u".repeat(80)),
        assistant_entry("a2", Some("u2"), &"a".repeat(80), None),
    ];

    let cut = find_cut_point(&entries, 0, entries.len(), 10);
    assert_eq!(cut.first_kept_entry_index, 3);
    assert_eq!(cut.turn_start_index, Some(2));
    assert!(cut.is_split_turn);
}

#[test]
fn prepare_compaction_selects_history_and_turn_prefix() {
    let entries = vec![
        user_entry("u1", None, "short"),
        assistant_entry("a1", Some("u1"), "short", None),
        SessionTreeEntry::ActiveToolsChange(ActiveToolsChangeEntry {
            id: "tools".to_string(),
            parent_id: Some("a1".to_string()),
            timestamp: "1".to_string(),
            active_tool_names: vec!["read".to_string()],
        }),
        user_entry("u2", Some("tools"), &"u".repeat(80)),
        assistant_entry("a2", Some("u2"), &"a".repeat(80), None),
    ];
    let preparation = prepare_compaction(
        &entries,
        CompactionSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 10,
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(preparation.first_kept_entry_id, "a2");
    assert!(preparation.is_split_turn);
    assert_eq!(preparation.messages_to_summarize.len(), 2);
    assert_eq!(preparation.turn_prefix_messages.len(), 1);
    assert!(preparation.tokens_before > 0);
}

#[test]
fn prepare_compaction_carries_previous_summary_and_file_details() {
    let entries = vec![
        SessionTreeEntry::Compaction(CompactionEntry {
            id: "c1".to_string(),
            parent_id: None,
            timestamp: "1".to_string(),
            summary: "previous".to_string(),
            first_kept_entry_id: "u1".to_string(),
            tokens_before: 100,
            details: Some(json!({
                "readFiles": ["README.md"],
                "modifiedFiles": ["src/lib.rs"]
            })),
            from_hook: Some(false),
        }),
        user_entry("u1", Some("c1"), "hello"),
        assistant_entry("a1", Some("u1"), "world", None),
    ];

    let preparation = prepare_compaction(
        &entries,
        CompactionSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 1,
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(preparation.previous_summary.as_deref(), Some("previous"));
    assert!(preparation.file_ops.read.contains("README.md"));
    assert!(preparation.file_ops.edited.contains("src/lib.rs"));
}

#[test]
fn prepare_compaction_skips_empty_or_current_compaction_paths() {
    assert!(
        prepare_compaction(&[], agent::DEFAULT_COMPACTION_SETTINGS)
            .unwrap()
            .is_none()
    );
    assert!(
        prepare_compaction(
            &[SessionTreeEntry::Compaction(CompactionEntry {
                id: "c1".to_string(),
                parent_id: None,
                timestamp: "1".to_string(),
                summary: "summary".to_string(),
                first_kept_entry_id: "c1".to_string(),
                tokens_before: 1,
                details: None,
                from_hook: None,
            })],
            agent::DEFAULT_COMPACTION_SETTINGS,
        )
        .unwrap()
        .is_none()
    );
}
