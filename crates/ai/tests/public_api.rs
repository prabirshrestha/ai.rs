use ai::{
    AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay, CacheRetention, Context,
    OpenAICompletionsOptions, OpenAIResponsesAuthHeader, OpenAIResponsesOptions,
    ResolvedOpenAIResponsesCompat, SimpleStreamOptions, build_anthropic_payload,
    build_chat_completions_payload, build_copilot_dynamic_headers, build_responses_payload,
    get_openai_completions_compat, has_copilot_vision_input, infer_copilot_initiator,
    stream_anthropic, stream_openai_completions, stream_openai_responses, stream_simple_anthropic,
    stream_simple_openai_completions, stream_simple_openai_responses,
};

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
    let _responses_payload = build_responses_payload(
        &ai::Model {
            api: "openai-responses".to_string(),
            ..model.clone()
        },
        &context,
        &responses_options,
        &ResolvedOpenAIResponsesCompat {
            send_session_id_header: true,
            supports_long_cache_retention: true,
        },
        CacheRetention::Short,
    );

    let chat_options = OpenAICompletionsOptions::default();
    let chat_compat = get_openai_completions_compat(&model);
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
    let _anthropic_payload = build_anthropic_payload(
        &ai::Model {
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            ..model.clone()
        },
        &context,
        &anthropic_options,
        false,
        None,
    );

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
