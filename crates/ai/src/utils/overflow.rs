//! Port of `utils/overflow.ts`.

use std::sync::OnceLock;

use regex::{Regex, RegexBuilder};

use crate::types::{AssistantMessage, StopReason};

/// Regex patterns (case-insensitive) that detect context overflow errors
/// from different providers. See Pi's `OVERFLOW_PATTERNS` for the example
/// message behind each pattern.
const OVERFLOW_PATTERNS: &[&str] = &[
    r"prompt (?:is )?too long",   // Anthropic and z.ai token overflow
    r"prompt exceeds max length", // z.ai CN endpoint token overflow
    r"request_too_large",         // Anthropic request byte-size overflow (HTTP 413)
    r"input is too long for requested model", // Amazon Bedrock
    r"exceeds the context window", // OpenAI (Completions & Responses API)
    r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))", // OpenAI-compatible proxies (LiteLLM)
    r"input token count.*exceeds the maximum", // Google (Gemini)
    r"maximum prompt length is \d+",           // xAI (Grok)
    r"reduce the length of the messages",      // Groq
    r"maximum context length is \d+ tokens",   // OpenRouter (most backends)
    r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", // OpenRouter/Poolside
    r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", // Together AI
    r"exceeds the limit of \d+",           // GitHub Copilot
    r"exceeds the available context size", // llama.cpp server
    r"greater than the context length",    // LM Studio
    r"context window exceeds limit",       // MiniMax
    r"exceeded model token limit",         // Kimi For Coding
    r"too large for model with \d+ maximum context length", // Mistral
    r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", // DS4 server
    r"model_context_window_exceeded",      // z.ai non-standard finish_reason surfaced as error text
    r"prompt too long; exceeded (?:max )?context length", // Ollama explicit overflow error
    r"range of input length should be",    // DashScope / Qwen Token Plan
    r"context[_ ]length[_ ]exceeded",      // Generic fallback
    r"too many tokens",                    // Generic fallback
    r"token limit exceeded",               // Generic fallback
];

const CEREBRAS_BODYLESS_OVERFLOW_PATTERN: &str = r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)";

/// Patterns that indicate non-overflow errors (rate limiting, server errors).
const NON_OVERFLOW_PATTERNS: &[&str] = &[
    r"^(Throttling error|Service unavailable):", // AWS Bedrock non-overflow errors
    r"rate limit",                               // Generic rate limiting
    r"too many requests",                        // Generic HTTP 429 style
];

fn compile(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .expect("overflow pattern compiles")
}

fn overflow_regexes() -> &'static [Regex] {
    static REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();
    REGEXES.get_or_init(|| OVERFLOW_PATTERNS.iter().map(|p| compile(p)).collect())
}

fn non_overflow_regexes() -> &'static [Regex] {
    static REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();
    REGEXES.get_or_init(|| NON_OVERFLOW_PATTERNS.iter().map(|p| compile(p)).collect())
}

fn cerebras_bodyless_overflow_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| compile(CEREBRAS_BODYLESS_OVERFLOW_PATTERN))
}

/// Check if an assistant message represents a context overflow error.
///
/// Handles error-message overflow, silent overflow (usage above
/// `context_window`, z.ai style) and length-stop overflow (zero output with
/// a filled context, Xiaomi MiMo style).
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u32>) -> bool {
    // Case 1: Check error message patterns
    if message.stop_reason == StopReason::Error
        && let Some(error_message) = message
            .error_message
            .as_deref()
            .filter(|error| !error.is_empty())
    {
        let is_non_overflow = non_overflow_regexes()
            .iter()
            .any(|pattern| pattern.is_match(error_message));
        if !is_non_overflow {
            if overflow_regexes()
                .iter()
                .any(|pattern| pattern.is_match(error_message))
            {
                return true;
            }
            if message.provider == "cerebras"
                && cerebras_bodyless_overflow_regex().is_match(error_message)
            {
                return true;
            }
        }
    }

    let context_window = context_window.filter(|window| *window > 0);

    // Case 2: Silent overflow (z.ai style) - successful but usage exceeds context
    if let Some(context_window) = context_window
        && message.stop_reason == StopReason::Stop
    {
        let input_tokens = u64::from(message.usage.input) + u64::from(message.usage.cache_read);
        if input_tokens > u64::from(context_window) {
            return true;
        }
    }

    // Case 3: Length-stop overflow (Xiaomi MiMo style)
    if let Some(context_window) = context_window
        && message.stop_reason == StopReason::Length
        && message.usage.output == 0
    {
        let input_tokens = u64::from(message.usage.input) + u64::from(message.usage.cache_read);
        if input_tokens as f64 >= f64::from(context_window) * 0.99 {
            return true;
        }
    }

    false
}

/// Check whether a length stop ended below the caller or model's intended
/// output limit. `desired_max_output` must be the original limit before any
/// context-based clamping.
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u32) -> bool {
    message.stop_reason == StopReason::Length
        && desired_max_output > 0
        && message.usage.output < desired_max_output
}

/// Get the overflow patterns for testing purposes.
pub fn get_overflow_patterns() -> Vec<Regex> {
    overflow_regexes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Model, Usage};

    fn create_error_message(error_message: &str, provider: &str) -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(&Model {
            id: "qwen3.5:35b".to_string(),
            api: "openai-completions".to_string(),
            provider: provider.to_string(),
            ..Default::default()
        });
        message.stop_reason = StopReason::Error;
        message.error_message = Some(error_message.to_string());
        message
    }

    fn ollama(error_message: &str) -> AssistantMessage {
        create_error_message(error_message, "ollama")
    }

    #[test]
    fn detects_explicit_ollama_prompt_too_long_errors() {
        let message = ollama("400 `prompt too long; exceeded max context length by 100918 tokens`");
        assert!(is_context_overflow(&message, Some(32768)));
    }

    #[test]
    fn detects_zai_prompt_too_long_errors() {
        let message =
            create_error_message(r#"400 {"code":"1261","message":"Prompt too long"}"#, "zai");
        assert!(is_context_overflow(&message, Some(1048576)));
    }

    #[test]
    fn detects_zai_cn_endpoint_prompt_exceeds_max_length_errors() {
        let message = create_error_message(
            r#"400 {"code":"1261","message":"Prompt exceeds max length"}"#,
            "zai",
        );
        assert!(is_context_overflow(&message, Some(1048576)));
    }

    #[test]
    fn detects_together_ai_context_length_errors() {
        let message = ollama(
            "400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
        );
        assert!(is_context_overflow(&message, Some(262144)));
    }

    #[test]
    fn detects_litellm_wrapped_openai_maximum_context_length_errors() {
        let message = ollama(
            "Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: litellm.APIConnectionError: APIConnectionError: OpenAIException - Requested token count exceeds the model's maximum context length of 131072 tokens.",
        );
        assert!(is_context_overflow(&message, Some(131072)));
    }

    #[test]
    fn detects_openai_compatible_parenthesized_maximum_context_length_errors() {
        let message = ollama(
            "Error: 400 Input length (265330) exceeds model's maximum context length (262144).",
        );
        assert!(is_context_overflow(&message, Some(262144)));
    }

    #[test]
    fn detects_openrouter_poolside_maximum_allowed_input_length_errors() {
        let message = ollama(
            "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens.",
        );
        assert!(is_context_overflow(&message, Some(131072)));
    }

    #[test]
    fn detects_ds4_configured_context_size_errors() {
        let message = ollama(
            "400 Prompt has 256468 tokens, but the configured context size is 256000 tokens",
        );
        assert!(is_context_overflow(&message, Some(256000)));
        let comma_message = ollama(
            "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens",
        );
        assert!(is_context_overflow(&comma_message, Some(256000)));
    }

    #[test]
    fn does_not_treat_generic_non_overflow_ollama_errors_as_overflow() {
        let message = ollama("500 `model runner crashed unexpectedly`");
        assert!(!is_context_overflow(&message, Some(32768)));
    }

    #[test]
    fn only_treats_bodyless_400_and_413_errors_as_overflow_for_cerebras() {
        for error_message in ["400 status code (no body)", "413 status code (no body)"] {
            assert!(is_context_overflow(
                &create_error_message(error_message, "cerebras"),
                Some(131072)
            ));
            assert!(!is_context_overflow(
                &create_error_message(error_message, "opencode-go"),
                Some(1000000)
            ));
        }
    }

    #[test]
    fn does_not_treat_bedrock_throttling_too_many_tokens_as_overflow() {
        let message = ollama("Throttling error: Too many tokens, please wait before trying again.");
        assert!(!is_context_overflow(&message, Some(200000)));
    }

    #[test]
    fn does_not_treat_bedrock_service_unavailable_as_overflow() {
        let message = ollama("Service unavailable: The service is temporarily unavailable.");
        assert!(!is_context_overflow(&message, Some(200000)));
    }

    #[test]
    fn does_not_treat_generic_rate_limit_errors_as_overflow() {
        let message = ollama("Rate limit exceeded, please retry after 30 seconds.");
        assert!(!is_context_overflow(&message, Some(200000)));
    }

    #[test]
    fn does_not_treat_http_429_style_errors_as_overflow() {
        let message = ollama("Too many requests. Please slow down.");
        assert!(!is_context_overflow(&message, Some(200000)));
    }

    fn create_length_stop_message(
        input: u32,
        cache_read: u32,
        output: u32,
        cache_write: u32,
    ) -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(&Model {
            id: "test-model".to_string(),
            api: "openai-completions".to_string(),
            provider: "test-provider".to_string(),
            ..Default::default()
        });
        message.usage = Usage {
            input,
            output,
            cache_read,
            cache_write,
            total_tokens: input + cache_read + cache_write + output,
            ..Default::default()
        };
        message.stop_reason = StopReason::Length;
        message
    }

    #[test]
    fn detects_xiaomi_style_overflow() {
        let message = create_length_stop_message(58, 1048512, 0, 0);
        assert!(is_context_overflow(&message, Some(1048576)));
    }

    #[test]
    fn treats_a_length_stop_below_the_desired_output_limit_as_recoverable() {
        let message = create_length_stop_message(3, 253584, 16, 25554);
        assert!(is_recoverable_length(&message, 128000));
    }

    #[test]
    fn does_not_recover_a_length_stop_that_reached_the_desired_output_limit() {
        let message = create_length_stop_message(4062, 0, 1024, 0);
        assert!(!is_recoverable_length(&message, 1024));
    }

    #[test]
    fn treats_zero_output_length_stops_as_recoverable_without_context_metadata() {
        let message = create_length_stop_message(100, 0, 0, 0);
        assert!(is_recoverable_length(&message, 128000));
    }

    #[test]
    fn does_not_treat_normal_length_stops_with_output_as_context_overflow() {
        let message = create_length_stop_message(1000, 0, 4096, 0);
        assert!(!is_context_overflow(&message, Some(200000)));
    }

    #[test]
    fn does_not_treat_zero_output_length_stops_far_below_context_as_context_overflow() {
        let message = create_length_stop_message(100, 0, 0, 0);
        assert!(!is_context_overflow(&message, Some(200000)));
    }
}
