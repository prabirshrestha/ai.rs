use agent::{
    compute_file_lists, create_file_ops, extract_file_ops_from_message, format_file_operations,
    serialize_conversation,
};
use ai::{
    AssistantContent, AssistantMessage, Message, StopReason, TextContent, ThinkingContent,
    ToolCall, ToolResultContent, ToolResultMessage, Usage, UserContent,
};
use serde_json::json;

fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
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

fn tool_call(name: &str, args: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: format!("{name}-1"),
        name: name.to_string(),
        arguments: args,
        thought_signature: None,
    })
}

#[test]
fn extracts_and_formats_file_operations_from_assistant_tool_calls() {
    let message = assistant(vec![
        tool_call("read", json!({"path": "src/lib.rs"})),
        tool_call("write", json!({"path": "src/main.rs"})),
        tool_call("edit", json!({"path": "src/lib.rs"})),
        tool_call("unknown", json!({"path": "ignored"})),
        tool_call("read", json!({"path": 42})),
    ]);
    let mut ops = create_file_ops();
    extract_file_ops_from_message(&message, &mut ops);

    let (read_files, modified_files) = compute_file_lists(&ops);
    assert!(read_files.is_empty());
    assert_eq!(modified_files, ["src/lib.rs", "src/main.rs"]);
    assert_eq!(
        format_file_operations(&read_files, &modified_files),
        "\n\n<modified-files>\nsrc/lib.rs\nsrc/main.rs\n</modified-files>"
    );
}

#[test]
fn formats_read_only_and_modified_file_lists() {
    let mut ops = create_file_ops();
    ops.read.insert("README.md".to_string());
    ops.read.insert("src/lib.rs".to_string());
    ops.edited.insert("src/lib.rs".to_string());

    let (read_files, modified_files) = compute_file_lists(&ops);
    assert_eq!(read_files, ["README.md"]);
    assert_eq!(modified_files, ["src/lib.rs"]);
    assert_eq!(
        format_file_operations(&read_files, &modified_files),
        "\n\n<read-files>\nREADME.md\n</read-files>\n\n<modified-files>\nsrc/lib.rs\n</modified-files>"
    );
}

#[test]
fn serializes_conversation_for_summary_prompts() {
    let messages = vec![
        Message::User(ai::UserMessage {
            content: ai::UserMessageContent::Parts(vec![
                UserContent::text("hello"),
                UserContent::Image(ai::ImageContent {
                    data: "abc".to_string(),
                    mime_type: "image/png".to_string(),
                }),
                UserContent::text(" world"),
            ]),
            timestamp: 1,
        }),
        assistant(vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "reasoning".to_string(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantContent::Text(TextContent {
                text: "answer".to_string(),
                text_signature: None,
            }),
            tool_call("read", json!({"path": "src/lib.rs", "limit": 10})),
        ]),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "read-1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ToolResultContent::text("file contents")],
            details: None,
            is_error: false,
            timestamp: 2,
        }),
    ];

    let serialized = serialize_conversation(&messages);
    assert!(serialized.contains("[User]: hello world"));
    assert!(serialized.contains("[Assistant thinking]: reasoning"));
    assert!(serialized.contains("[Assistant]: answer"));
    assert!(
        serialized.contains("[Assistant tool calls]: read(limit=10, path=\"src/lib.rs\")")
            || serialized.contains("[Assistant tool calls]: read(path=\"src/lib.rs\", limit=10)")
    );
    assert!(serialized.contains("[Tool result]: file contents"));
}

#[test]
fn truncates_long_tool_results_in_serialized_conversation() {
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage {
        tool_call_id: "tool-1".to_string(),
        tool_name: "tool".to_string(),
        content: vec![ToolResultContent::text("x".repeat(2100))],
        details: None,
        is_error: false,
        timestamp: 1,
    })]);

    assert!(serialized.contains("[... 100 more characters truncated]"));
}
