use std::collections::HashMap;

use agent::{
    AgentHarnessErrorCode, AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch, MapPatch,
    apply_stream_options_patch, create_failure_message, merge_headers, validate_unique_names,
};
use ai::{CacheRetention, Message, Model, StopReason, Transport};
use serde_json::json;

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
