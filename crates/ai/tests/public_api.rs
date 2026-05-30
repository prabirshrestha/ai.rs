use ai::{
    AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay, Context, OpenAICompletionsOptions,
    OpenAIResponsesAuthHeader, OpenAIResponsesOptions, SimpleStreamOptions,
    build_copilot_dynamic_headers, has_copilot_vision_input, infer_copilot_initiator,
    stream_anthropic, stream_openai_completions, stream_openai_responses, stream_simple_anthropic,
    stream_simple_openai_completions, stream_simple_openai_responses,
};

#[test]
fn focused_provider_symbols_are_exported_from_ai_crate() {
    let _responses_options = OpenAIResponsesOptions::default();
    assert_eq!(
        OpenAIResponsesAuthHeader::default(),
        OpenAIResponsesAuthHeader::Bearer
    );

    let _chat_options = OpenAICompletionsOptions::default();

    let anthropic_options = AnthropicOptions::default();
    assert!(anthropic_options.interleaved_thinking);
    let _effort = AnthropicEffort::High;
    let _display = AnthropicThinkingDisplay::Summarized;

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
