use std::sync::OnceLock;

use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

const OVERFLOW_PATTERNS: &[&str] = &[
    r"(?i)prompt is too long",
    r"(?i)request_too_large",
    r"(?i)input is too long for requested model",
    r"(?i)exceeds the context window",
    r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length of [\d,]+ tokens?",
    r"(?i)input token count.*exceeds the maximum",
    r"(?i)maximum prompt length is \d+",
    r"(?i)reduce the length of the messages",
    r"(?i)maximum context length is \d+ tokens",
    r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
    r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
    r"(?i)exceeds the limit of \d+",
    r"(?i)exceeds the available context size",
    r"(?i)greater than the context length",
    r"(?i)context window exceeds limit",
    r"(?i)exceeded model token limit",
    r"(?i)too large for model with \d+ maximum context length",
    r"(?i)model_context_window_exceeded",
    r"(?i)prompt too long; exceeded (?:max )?context length",
    r"(?i)context[_ ]length[_ ]exceeded",
    r"(?i)too many tokens",
    r"(?i)token limit exceeded",
    r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
];

const NON_OVERFLOW_PATTERNS: &[&str] = &[
    r"(?i)^(Throttling error|Service unavailable):",
    r"(?i)rate limit",
    r"(?i)too many requests",
];

pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u32>) -> bool {
    if message.stop_reason == StopReason::Error
        && let Some(error_message) = &message.error_message
    {
        let is_non_overflow = non_overflow_regexes()
            .iter()
            .any(|pattern| pattern.is_match(error_message));
        if !is_non_overflow
            && overflow_regexes()
                .iter()
                .any(|pattern| pattern.is_match(error_message))
        {
            return true;
        }
    }

    if let Some(context_window) = context_window {
        if message.stop_reason == StopReason::Stop {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if input_tokens > context_window {
                return true;
            }
        }

        if message.stop_reason == StopReason::Length && message.usage.output == 0 {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if (input_tokens as f64) >= (context_window as f64 * 0.99) {
                return true;
            }
        }
    }

    false
}

pub fn get_overflow_patterns() -> Vec<Regex> {
    overflow_regexes().to_vec()
}

fn overflow_regexes() -> &'static [Regex] {
    static REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();
    REGEXES.get_or_init(|| compile_patterns(OVERFLOW_PATTERNS))
}

fn non_overflow_regexes() -> &'static [Regex] {
    static REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();
    REGEXES.get_or_init(|| compile_patterns(NON_OVERFLOW_PATTERNS))
}

fn compile_patterns(patterns: &[&str]) -> Vec<Regex> {
    patterns
        .iter()
        .map(|pattern| Regex::new(pattern).expect("overflow regex should compile"))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::types::{Usage, UsageCost};

    use super::*;

    fn create_error_message(error_message: &str) -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: "openai-completions".to_string(),
            provider: "ollama".to_string(),
            model: "qwen3.5:35b".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            error_message: Some(error_message.to_string()),
            timestamp: crate::utils::time::now_millis(),
        }
    }

    fn create_length_stop_message(input: u32, cache_read: u32, output: u32) -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: "openai-completions".to_string(),
            provider: "xiaomi".to_string(),
            model: "mimo-v2.5-pro".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage {
                input,
                output,
                cache_read,
                cache_write: 0,
                total_tokens: input + cache_read + output,
                cost: UsageCost::default(),
            },
            stop_reason: StopReason::Length,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }

    #[test]
    fn detects_explicit_overflow_errors() {
        assert!(is_context_overflow(
            &create_error_message(
                "400 `prompt too long; exceeded max context length by 100918 tokens`"
            ),
            Some(32768)
        ));
        assert!(is_context_overflow(
            &create_error_message(
                "400 The input (516368 tokens) is longer than the model's context length (262144 tokens)."
            ),
            Some(262144)
        ));
        assert!(is_context_overflow(
            &create_error_message(
                "Requested token count exceeds the model's maximum context length of 131072 tokens."
            ),
            Some(131072)
        ));
        assert!(is_context_overflow(
            &create_error_message(
                "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens."
            ),
            Some(131072)
        ));
    }

    #[test]
    fn excludes_non_overflow_errors() {
        assert!(!is_context_overflow(
            &create_error_message("500 `model runner crashed unexpectedly`"),
            Some(32768)
        ));
        assert!(!is_context_overflow(
            &create_error_message(
                "Throttling error: Too many tokens, please wait before trying again."
            ),
            Some(200000)
        ));
        assert!(!is_context_overflow(
            &create_error_message("Service unavailable: The service is temporarily unavailable."),
            Some(200000)
        ));
        assert!(!is_context_overflow(
            &create_error_message("Rate limit exceeded, please retry after 30 seconds."),
            Some(200000)
        ));
        assert!(!is_context_overflow(
            &create_error_message("Too many requests. Please slow down."),
            Some(200000)
        ));
    }

    #[test]
    fn detects_silent_and_length_stop_overflow() {
        let mut silent = create_error_message("");
        silent.stop_reason = StopReason::Stop;
        silent.error_message = None;
        silent.usage.input = 200001;
        assert!(is_context_overflow(&silent, Some(200000)));

        assert!(is_context_overflow(
            &create_length_stop_message(58, 1048512, 0),
            Some(1048576)
        ));
        assert!(!is_context_overflow(
            &create_length_stop_message(1000, 0, 4096),
            Some(200000)
        ));
        assert!(!is_context_overflow(
            &create_length_stop_message(100, 0, 0),
            Some(200000)
        ));
    }
}
