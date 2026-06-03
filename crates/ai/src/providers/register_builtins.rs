use std::sync::OnceLock;

use crate::api_registry::{self, ApiProvider};
use crate::providers::{anthropic, openai_completions, openai_responses};
use crate::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, StreamOptions};
use crate::{AssistantEventStream, Result};
use serde_json::Value;

pub use anthropic::{stream_anthropic, stream_simple_anthropic};
pub use openai_completions::{stream_openai_completions, stream_simple_openai_completions};
pub use openai_responses::{stream_openai_responses, stream_simple_openai_responses};

pub(crate) fn ensure_builtins_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(register_builtins);
}

pub fn register_builtin_api_providers() {
    register_builtins();
}

pub fn reset_api_providers() {
    api_registry::clear_api_providers();
    register_builtins();
}

fn register_builtins() {
    api_registry::register_api_provider(
        ApiProvider {
            api: "anthropic-messages".to_string(),
            stream: api_registry::wrap_stream("anthropic-messages", |model, context, options| {
                Ok(anthropic::stream_anthropic(
                    model,
                    context,
                    anthropic_options_from_stream_options(options),
                ))
            }),
            stream_simple: api_registry::wrap_stream_simple(
                "anthropic-messages",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantEventStream> {
                    Ok(anthropic::stream_simple_anthropic(model, context, options))
                },
            ),
        },
        Some("builtin".to_string()),
    );

    api_registry::register_api_provider(
        ApiProvider {
            api: "openai-completions".to_string(),
            stream: api_registry::wrap_stream("openai-completions", |model, context, options| {
                Ok(openai_completions::stream_openai_completions(
                    model,
                    context,
                    openai_completions_options_from_stream_options(options),
                ))
            }),
            stream_simple: api_registry::wrap_stream_simple(
                "openai-completions",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantEventStream> {
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
                    openai_responses_options_from_stream_options(options),
                ))
            }),
            stream_simple: api_registry::wrap_stream_simple(
                "openai-responses",
                |model: Model,
                 context: Context,
                 options: SimpleStreamOptions|
                 -> Result<AssistantEventStream> {
                    Ok(openai_responses::stream_simple_openai_responses(
                        model, context, options,
                    ))
                },
            ),
        },
        Some("builtin".to_string()),
    );
}

pub(crate) fn openai_completions_options_from_stream_options(
    options: StreamOptions,
) -> openai_completions::OpenAICompletionsOptions {
    let tool_choice = provider_option(&options, &["toolChoice"]).cloned();
    let reasoning_effort = openai_reasoning_effort(&options);
    openai_completions::OpenAICompletionsOptions {
        base: options,
        tool_choice,
        reasoning_effort,
    }
}

pub(crate) fn openai_responses_options_from_stream_options(
    options: StreamOptions,
) -> openai_responses::OpenAIResponsesOptions {
    let reasoning_effort = openai_reasoning_effort(&options);
    let reasoning_summary =
        provider_option(&options, &["reasoningSummary"]).and_then(reasoning_summary_option);
    let service_tier = provider_string(&options, &["serviceTier"]);
    openai_responses::OpenAIResponsesOptions {
        base: options,
        reasoning_effort,
        reasoning_summary,
        service_tier,
    }
}

pub(crate) fn anthropic_options_from_stream_options(
    options: StreamOptions,
) -> anthropic::AnthropicOptions {
    let thinking_enabled = provider_bool(&options, &["thinkingEnabled"]);
    let thinking_budget_tokens = provider_u32(&options, &["thinkingBudgetTokens"]);
    let effort = provider_anthropic_effort(&options, &["effort"]);
    let thinking_display = provider_anthropic_thinking_display(&options, &["thinkingDisplay"]);
    let interleaved_thinking = provider_bool(&options, &["interleavedThinking"]).unwrap_or(true);
    let tool_choice = provider_option(&options, &["toolChoice"]).cloned();
    anthropic::AnthropicOptions {
        base: options,
        thinking_enabled,
        thinking_budget_tokens,
        effort,
        thinking_display,
        interleaved_thinking,
        tool_choice,
    }
}

fn provider_option<'a>(options: &'a StreamOptions, names: &[&str]) -> Option<&'a Value> {
    names
        .iter()
        .find_map(|name| options.provider_options.get(*name))
}

fn provider_string(options: &StreamOptions, names: &[&str]) -> Option<String> {
    provider_option(options, names)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn provider_bool(options: &StreamOptions, names: &[&str]) -> Option<bool> {
    provider_option(options, names).and_then(Value::as_bool)
}

fn provider_u32(options: &StreamOptions, names: &[&str]) -> Option<u32> {
    provider_option(options, names)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn openai_reasoning_effort(options: &StreamOptions) -> Option<ModelThinkingLevel> {
    provider_string(options, &["reasoningEffort"])
        .and_then(|value| ModelThinkingLevel::parse(&value))
        .filter(|effort| *effort != ModelThinkingLevel::Off)
}

fn reasoning_summary_option(value: &Value) -> Option<Option<String>> {
    if value.is_null() {
        Some(None)
    } else {
        value.as_str().map(|value| Some(value.to_string()))
    }
}

fn provider_anthropic_effort(
    options: &StreamOptions,
    names: &[&str],
) -> Option<anthropic::AnthropicEffort> {
    match provider_string(options, names)?.as_str() {
        "low" => Some(anthropic::AnthropicEffort::Low),
        "medium" => Some(anthropic::AnthropicEffort::Medium),
        "high" => Some(anthropic::AnthropicEffort::High),
        "xhigh" => Some(anthropic::AnthropicEffort::Xhigh),
        "max" => Some(anthropic::AnthropicEffort::Max),
        _ => None,
    }
}

fn provider_anthropic_thinking_display(
    options: &StreamOptions,
    names: &[&str],
) -> Option<anthropic::AnthropicThinkingDisplay> {
    match provider_string(options, names)?.as_str() {
        "summarized" => Some(anthropic::AnthropicThinkingDisplay::Summarized),
        "omitted" => Some(anthropic::AnthropicThinkingDisplay::Omitted),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn generic_openai_completions_options_forward_provider_options() {
        let options = StreamOptions {
            provider_options: [
                ("toolChoice".to_string(), json!("required")),
                ("reasoningEffort".to_string(), json!("high")),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        let converted = openai_completions_options_from_stream_options(options);

        assert_eq!(converted.tool_choice, Some(json!("required")));
        assert_eq!(converted.reasoning_effort, Some(ModelThinkingLevel::High));
    }

    #[test]
    fn generic_provider_options_use_upstream_camel_case_names() {
        let options = StreamOptions {
            provider_options: [
                ("tool_choice".to_string(), json!("required")),
                ("reasoning_effort".to_string(), json!("high")),
                ("reasoning_summary".to_string(), json!("concise")),
                ("service_tier".to_string(), json!("flex")),
                ("thinking_enabled".to_string(), json!(true)),
                ("thinking_budget_tokens".to_string(), json!(4096)),
                ("thinking_display".to_string(), json!("omitted")),
                ("interleaved_thinking".to_string(), json!(false)),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        let completions = openai_completions_options_from_stream_options(options.clone());
        assert_eq!(completions.tool_choice, None);
        assert_eq!(completions.reasoning_effort, None);

        let responses = openai_responses_options_from_stream_options(options.clone());
        assert_eq!(responses.reasoning_effort, None);
        assert_eq!(responses.reasoning_summary, None);
        assert_eq!(responses.service_tier, None);

        let anthropic = anthropic_options_from_stream_options(options);
        assert_eq!(anthropic.thinking_enabled, None);
        assert_eq!(anthropic.thinking_budget_tokens, None);
        assert_eq!(anthropic.thinking_display, None);
        assert!(anthropic.interleaved_thinking);
        assert_eq!(anthropic.tool_choice, None);
    }

    #[test]
    fn generic_openai_options_do_not_forward_off_reasoning_effort() {
        let options = StreamOptions {
            provider_options: [("reasoningEffort".to_string(), json!("off"))]
                .into_iter()
                .collect(),
            ..Default::default()
        };

        let converted = openai_responses_options_from_stream_options(options);

        assert_eq!(converted.reasoning_effort, None);
    }

    #[test]
    fn generic_openai_responses_options_forward_provider_options() {
        let options = StreamOptions {
            provider_options: [
                ("reasoningSummary".to_string(), json!("concise")),
                ("serviceTier".to_string(), json!("flex")),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        let converted = openai_responses_options_from_stream_options(options);

        assert_eq!(
            converted.reasoning_summary,
            Some(Some("concise".to_string()))
        );
        assert_eq!(converted.service_tier.as_deref(), Some("flex"));
    }

    #[test]
    fn generic_anthropic_options_forward_provider_options() {
        let options = StreamOptions {
            provider_options: [
                ("thinkingEnabled".to_string(), json!(true)),
                ("thinkingBudgetTokens".to_string(), json!(4096)),
                ("effort".to_string(), json!("xhigh")),
                ("thinkingDisplay".to_string(), json!("omitted")),
                ("interleavedThinking".to_string(), json!(false)),
                (
                    "toolChoice".to_string(),
                    json!({"type": "tool", "name": "edit"}),
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        let converted = anthropic_options_from_stream_options(options);

        assert_eq!(converted.thinking_enabled, Some(true));
        assert_eq!(converted.thinking_budget_tokens, Some(4096));
        assert_eq!(converted.effort, Some(anthropic::AnthropicEffort::Xhigh));
        assert_eq!(
            converted.thinking_display,
            Some(anthropic::AnthropicThinkingDisplay::Omitted)
        );
        assert!(!converted.interleaved_thinking);
        assert_eq!(
            converted.tool_choice,
            Some(json!({"type": "tool", "name": "edit"}))
        );
    }
}
