use ai::{
    AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay, AssistantContent,
    AssistantMessage, AssistantMessageEvent, CacheRetention, Context, OpenAICompletionsOptions,
    OpenAIResponsesAuthHeader, OpenAIResponsesOptions, SimpleStreamOptions, StopReason,
    TextContent, Usage, build_anthropic_payload, build_chat_completions_payload,
    build_copilot_dynamic_headers, build_responses_payload, convert_anthropic_messages,
    convert_openai_completions_messages, create_assistant_message_event_stream,
    get_openai_completions_compat, get_openai_responses_compat, has_copilot_vision_input,
    infer_copilot_initiator, stream_anthropic, stream_openai_completions, stream_openai_responses,
    stream_simple_anthropic, stream_simple_openai_completions, stream_simple_openai_responses,
};
use futures::StreamExt;

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
}
