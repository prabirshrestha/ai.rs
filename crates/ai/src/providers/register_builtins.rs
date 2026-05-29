use std::sync::OnceLock;

use crate::api_registry::{self, ApiProvider};
use crate::providers::{anthropic, azure_openai_responses, openai_completions, openai_responses};
use crate::types::{Context, Model, SimpleStreamOptions};
use crate::{AssistantMessageEventStream, Result};

pub fn ensure_builtins_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(register_builtins);
}

pub fn register_builtins() {
    api_registry::register_api_provider(
        ApiProvider {
            api: "openai-completions".to_string(),
            stream: api_registry::wrap_stream("openai-completions", |model, context, options| {
                Ok(openai_completions::stream_openai_completions(
                    model,
                    context,
                    openai_completions::OpenAICompletionsOptions {
                        base: options,
                        tool_choice: None,
                        reasoning_effort: None,
                    },
                ))
            }),
            stream_simple: api_registry::wrap_stream_simple(
                "openai-completions",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantMessageEventStream> {
                    Ok(openai_completions::stream_simple_openai_completions(
                        model, context, options,
                    ))
                },
            ),
        },
        Some("builtin".to_string()),
    );

    api_registry::register_api_provider(
        ApiProvider {
            api: "openai-responses".to_string(),
            stream: api_registry::wrap_stream("openai-responses", |model, context, options| {
                Ok(openai_responses::stream_openai_responses(
                    model,
                    context,
                    openai_responses::OpenAIResponsesOptions {
                        base: options,
                        reasoning_effort: None,
                        reasoning_summary: None,
                        service_tier: None,
                        ..Default::default()
                    },
                ))
            }),
            stream_simple: api_registry::wrap_stream_simple(
                "openai-responses",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantMessageEventStream> {
                    Ok(openai_responses::stream_simple_openai_responses(
                        model, context, options,
                    ))
                },
            ),
        },
        Some("builtin".to_string()),
    );

    api_registry::register_api_provider(
        ApiProvider {
            api: "azure-openai-responses".to_string(),
            stream: api_registry::wrap_stream(
                "azure-openai-responses",
                |model, context, options| {
                    Ok(azure_openai_responses::stream_azure_openai_responses(
                        model,
                        context,
                        azure_openai_responses::AzureOpenAIResponsesOptions {
                            base: options,
                            reasoning_effort: None,
                            reasoning_summary: None,
                            ..Default::default()
                        },
                    ))
                },
            ),
            stream_simple: api_registry::wrap_stream_simple(
                "azure-openai-responses",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantMessageEventStream> {
                    Ok(
                        azure_openai_responses::stream_simple_azure_openai_responses(
                            model, context, options,
                        ),
                    )
                },
            ),
        },
        Some("builtin".to_string()),
    );

    api_registry::register_api_provider(
        ApiProvider {
            api: "anthropic-messages".to_string(),
            stream: api_registry::wrap_stream("anthropic-messages", |model, context, options| {
                Ok(anthropic::stream_anthropic(
                    model,
                    context,
                    anthropic::AnthropicOptions {
                        base: options,
                        ..Default::default()
                    },
                ))
            }),
            stream_simple: api_registry::wrap_stream_simple(
                "anthropic-messages",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantMessageEventStream> {
                    Ok(anthropic::stream_simple_anthropic(model, context, options))
                },
            ),
        },
        Some("builtin".to_string()),
    );
}
