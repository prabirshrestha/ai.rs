use agent::{
    BRANCH_SUMMARY_PREFIX, BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage,
    HarnessMessage, HarnessMessageContent, bash_execution_to_text, convert_harness_messages_to_llm,
    create_branch_summary_message, create_compaction_summary_message, create_custom_message,
};
use ai::{Message, UserContent, UserMessageContent};
use serde_json::json;

fn user_text(message: &Message) -> Option<&str> {
    let Message::User(user) = message else {
        return None;
    };
    match &user.content {
        UserMessageContent::Text(text) => Some(text.as_str()),
        UserMessageContent::Parts(parts) => parts.iter().find_map(|part| match part {
            UserContent::Text(text) => Some(text.text.as_str()),
            UserContent::Image(_) => None,
        }),
    }
}

#[test]
fn formats_bash_execution_messages_as_text() {
    let message = BashExecutionMessage {
        command: "cargo test".to_string(),
        output: "failed".to_string(),
        exit_code: Some(101),
        cancelled: false,
        truncated: true,
        full_output_path: Some("/tmp/full.log".to_string()),
        timestamp: 10,
        exclude_from_context: false,
    };

    assert_eq!(
        bash_execution_to_text(&message),
        "Ran `cargo test`\n```\nfailed\n```\n\nCommand exited with code 101\n\n[Output truncated. Full output: /tmp/full.log]"
    );

    let cancelled = BashExecutionMessage {
        output: String::new(),
        cancelled: true,
        exit_code: Some(1),
        truncated: false,
        full_output_path: None,
        ..message
    };
    assert_eq!(
        bash_execution_to_text(&cancelled),
        "Ran `cargo test`\n(no output)\n\n(command cancelled)"
    );
}

#[test]
fn creates_summary_and_custom_messages_from_timestamps() {
    let timestamp = "2026-05-29T12:34:56Z";
    let branch = create_branch_summary_message("summary", "entry-1", timestamp);
    let compaction = create_compaction_summary_message("compact", 42, timestamp);
    let custom = create_custom_message(
        "notice",
        HarnessMessageContent::Text("hello".to_string()),
        true,
        Some(json!({"x": 1})),
        timestamp,
    );

    assert_eq!(branch.from_id, "entry-1");
    assert_eq!(branch.timestamp, 1_780_058_096_000);
    assert_eq!(compaction.tokens_before, 42);
    assert_eq!(compaction.timestamp, branch.timestamp);
    assert_eq!(custom.timestamp, branch.timestamp);
}

#[test]
fn converts_harness_messages_to_llm_messages() {
    let messages = vec![
        HarnessMessage::BashExecution(BashExecutionMessage {
            command: "echo ok".to_string(),
            output: "ok".to_string(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1,
            exclude_from_context: false,
        }),
        HarnessMessage::BashExecution(BashExecutionMessage {
            command: "hidden".to_string(),
            output: "hidden".to_string(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 2,
            exclude_from_context: true,
        }),
        HarnessMessage::Custom(create_custom_message(
            "custom",
            HarnessMessageContent::Text("custom text".to_string()),
            true,
            None,
            "3",
        )),
        HarnessMessage::BranchSummary(BranchSummaryMessage {
            summary: "branch".to_string(),
            from_id: "x".to_string(),
            timestamp: 4,
        }),
        HarnessMessage::CompactionSummary(CompactionSummaryMessage {
            summary: "compact".to_string(),
            tokens_before: 100,
            timestamp: 5,
        }),
        HarnessMessage::Llm(Message::user_text("plain")),
    ];

    let llm_messages = convert_harness_messages_to_llm(&messages);
    let texts = llm_messages
        .iter()
        .filter_map(user_text)
        .collect::<Vec<_>>();
    assert_eq!(texts.len(), 5);
    assert!(texts[0].contains("Ran `echo ok`"));
    assert_eq!(texts[1], "custom text");
    assert!(texts[2].starts_with(BRANCH_SUMMARY_PREFIX));
    assert!(texts[3].contains("compact"));
    assert_eq!(texts[4], "plain");
}
