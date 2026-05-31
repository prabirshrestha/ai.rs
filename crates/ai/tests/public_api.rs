use ai::{
    AgentToolCall, AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay, AssistantContent,
    AssistantMessage, AssistantMessageEvent, CacheRetention, Context, OpenAICompletionsOptions,
    OpenAIResponsesAuthHeader, OpenAIResponsesOptions, SimpleStreamOptions, StopReason,
    TextContent, Usage, build_anthropic_payload, build_chat_completions_payload,
    build_copilot_dynamic_headers, build_responses_payload, clear_api_providers,
    convert_anthropic_messages, convert_openai_completions_messages,
    create_assistant_message_event_stream, get_api_provider, get_api_providers, get_oauth_api_key,
    get_oauth_provider, get_oauth_provider_info_list, get_oauth_providers,
    get_openai_completions_compat, get_openai_responses_compat, has_copilot_vision_input,
    infer_copilot_initiator, login_anthropic, register_builtin_api_providers,
    register_oauth_provider, repair_json, reset_api_providers, reset_oauth_providers,
    stream_anthropic, stream_openai_completions, stream_openai_responses, stream_simple_anthropic,
    stream_simple_openai_completions, stream_simple_openai_responses, unregister_oauth_provider,
    validate_tool_arguments, validate_tool_call,
};
use futures::StreamExt;
use std::collections::HashMap;

#[test]
fn focused_provider_symbols_are_exported_from_ai_crate() {
    let model = ai::Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
        ..Default::default()
    };
    let context = Context::default();

    let responses_options = OpenAIResponsesOptions::default();
    assert_eq!(
        OpenAIResponsesAuthHeader::default(),
        OpenAIResponsesAuthHeader::Bearer
    );
    let responses_model = ai::Model {
        api: "openai-responses".to_string(),
        ..model.clone()
    };
    let resolved_responses_compat = get_openai_responses_compat(&responses_model);
    let _responses_payload = build_responses_payload(
        &responses_model,
        &context,
        &responses_options,
        &resolved_responses_compat,
        CacheRetention::Short,
    );

    let chat_options = OpenAICompletionsOptions::default();
    let chat_compat = get_openai_completions_compat(&model);
    let _chat_messages = convert_openai_completions_messages(&model, &context, &chat_compat);
    let _chat_payload = build_chat_completions_payload(
        &model,
        &context,
        &chat_options,
        &chat_compat,
        CacheRetention::Short,
    );

    let anthropic_options = AnthropicOptions::default();
    assert!(anthropic_options.interleaved_thinking);
    let _effort = AnthropicEffort::High;
    let _display = AnthropicThinkingDisplay::Summarized;
    let anthropic_model = ai::Model {
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        ..model.clone()
    };
    let _anthropic_messages = convert_anthropic_messages(&[], &anthropic_model, false, None, false);
    let _anthropic_payload =
        build_anthropic_payload(&anthropic_model, &context, &anthropic_options, false, None);

    assert_eq!(infer_copilot_initiator(&[]), "user");
    assert!(!has_copilot_vision_input(&[]));
    assert_eq!(
        build_copilot_dynamic_headers(&[], false)
            .get("X-Initiator")
            .map(String::as_str),
        Some("user")
    );

    let _simple_responses: fn(
        ai::Model,
        Context,
        SimpleStreamOptions,
    ) -> ai::AssistantMessageEventStream = stream_simple_openai_responses;
    let _responses: fn(
        ai::Model,
        Context,
        OpenAIResponsesOptions,
    ) -> ai::AssistantMessageEventStream = stream_openai_responses;
    let _simple_chat: fn(
        ai::Model,
        Context,
        SimpleStreamOptions,
    ) -> ai::AssistantMessageEventStream = stream_simple_openai_completions;
    let _chat: fn(ai::Model, Context, OpenAICompletionsOptions) -> ai::AssistantMessageEventStream =
        stream_openai_completions;
    let _simple_anthropic: fn(
        ai::Model,
        Context,
        SimpleStreamOptions,
    ) -> ai::AssistantMessageEventStream = stream_simple_anthropic;
    let _anthropic: fn(ai::Model, Context, AnthropicOptions) -> ai::AssistantMessageEventStream =
        stream_anthropic;
}

#[tokio::test]
async fn assistant_event_stream_factory_is_exported_and_terminal() {
    let (mut sender, mut stream) = create_assistant_message_event_stream();
    let message = AssistantMessage {
        content: vec![AssistantContent::Text(TextContent {
            text: "done".to_string(),
            text_signature: None,
        })],
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o-mini".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 0,
    };

    sender.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: message.clone(),
    });
    sender.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Done { .. })
    ));
    assert!(stream.next().await.is_none());
    assert_eq!(stream.result().await.unwrap(), message);
    assert_eq!(stream.result().await.unwrap(), message);
}

#[test]
fn api_registry_and_builtin_provider_helpers_are_exported() {
    reset_api_providers();

    assert!(get_api_provider("openai-completions").is_some());
    assert!(get_api_provider("openai-responses").is_some());
    assert!(get_api_provider("anthropic-messages").is_some());
    assert!(get_api_provider("mistral-conversations").is_none());
    assert!(get_api_provider("azure-openai-responses").is_none());
    assert!(get_api_provider("openai-codex-responses").is_none());
    let api_order = get_api_providers()
        .into_iter()
        .map(|provider| provider.api)
        .collect::<Vec<_>>();
    assert_eq!(
        api_order,
        [
            "anthropic-messages",
            "openai-completions",
            "openai-responses"
        ]
    );

    clear_api_providers();
    assert!(get_api_provider("openai-completions").is_none());

    register_builtin_api_providers();
    assert!(get_api_provider("openai-responses").is_some());
    let provider_count = get_api_providers().len();
    register_builtin_api_providers();
    assert_eq!(get_api_providers().len(), provider_count);

    reset_api_providers();
}

#[test]
fn json_and_validation_helpers_are_exported() {
    let repaired = repair_json("{\"text\":\"hello\nworld\"}");
    assert_eq!(repaired, "{\"text\":\"hello\\nworld\"}");

    let parsed: serde_json::Value =
        ai::parse_json_with_repair("{\"text\":\"hello\nworld\"}").unwrap();
    assert_eq!(parsed["text"], "hello\nworld");
    assert_eq!(
        ai::parse_streaming_json(Some("{\"answer\": 42")),
        serde_json::json!({ "answer": 42 })
    );

    let _validate_tool_call: fn(&[ai::Tool], &ai::ToolCall) -> ai::Result<serde_json::Value> =
        validate_tool_call;
    let _validate_tool_arguments: fn(&ai::Tool, &ai::ToolCall) -> ai::Result<serde_json::Value> =
        validate_tool_arguments;
}

#[tokio::test]
async fn oauth_registry_helpers_are_exported() {
    reset_oauth_providers();

    let _auth_info = ai::OAuthAuthInfo {
        url: "https://example.com/auth".to_string(),
        instructions: Some("Open in browser".to_string()),
    };
    let _select_prompt = ai::OAuthSelectPrompt {
        message: "Choose provider".to_string(),
        options: vec![ai::OAuthSelectOption {
            id: "github-copilot".to_string(),
            label: "GitHub Copilot".to_string(),
        }],
    };

    let provider = get_oauth_provider("github-copilot").expect("copilot provider");
    assert_eq!(provider.id(), "github-copilot");
    assert_eq!(
        provider.get_api_key(&oauth_credentials("copilot-access")),
        "copilot-access"
    );
    assert!(
        get_oauth_provider_info_list()
            .iter()
            .any(|info| info.id == "github-copilot")
    );
    let oauth_ids = get_oauth_providers()
        .into_iter()
        .map(|provider| provider.id().to_string())
        .collect::<Vec<_>>();
    assert_eq!(oauth_ids, ["anthropic", "github-copilot"]);

    let mut credentials = HashMap::new();
    credentials.insert(
        "github-copilot".to_string(),
        oauth_credentials("copilot-access"),
    );
    let key = get_oauth_api_key("github-copilot", &credentials)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(key.api_key, "copilot-access");

    let _register: fn(ai::OAuthProvider) = register_oauth_provider;
    let _unregister: fn(&str) = unregister_oauth_provider;
    let _reset: fn() = reset_oauth_providers;
    let _login_anthropic = login_anthropic;

    reset_oauth_providers();
}

fn oauth_credentials(access: &str) -> ai::OAuthCredentials {
    ai::OAuthCredentials {
        refresh: "refresh".to_string(),
        access: access.to_string(),
        expires: u64::MAX,
        enterprise_url: None,
    }
}

#[test]
fn agent_type_aliases_are_exported_from_ai_crate() {
    let _tool_call: AgentToolCall = ai::ToolCall {
        id: "call-1".to_string(),
        name: "lookup".to_string(),
        arguments: serde_json::json!({ "query": "rust" }),
        thought_signature: None,
    };
    let level: ai::ThinkingLevel = ai::ThinkingLevel::Low;
    assert_eq!(
        ai::ModelThinkingLevel::from(level),
        ai::ModelThinkingLevel::Low
    );
}

#[test]
fn stream_options_include_websocket_connect_timeout() {
    let options = ai::StreamOptions {
        websocket_connect_timeout_ms: Some(5_000),
        ..Default::default()
    };

    assert_eq!(options.websocket_connect_timeout_ms, Some(5_000));
}
