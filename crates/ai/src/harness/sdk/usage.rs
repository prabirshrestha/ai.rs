//! Port of `ai` `types/usage.ts` (`LanguageModelUsage` and its helpers).

use serde::{Deserialize, Serialize};

use crate::harness::provider::{JsonObject, LanguageModelV4Usage};

/// `LanguageModelUsage['inputTokenDetails']`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputTokenDetails {
    pub no_cache_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}

/// `LanguageModelUsage['outputTokenDetails']`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputTokenDetails {
    pub text_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

/// `LanguageModelUsage`.
///
/// [rust] `undefined` counts serialize as `null`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelUsage {
    pub input_tokens: Option<u64>,
    pub input_token_details: InputTokenDetails,
    pub output_tokens: Option<u64>,
    pub output_token_details: OutputTokenDetails,
    pub total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<JsonObject>,
}

fn add_token_counts(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    if a.is_none() && b.is_none() {
        None
    } else {
        Some(a.unwrap_or(0) + b.unwrap_or(0))
    }
}

/// `asLanguageModelUsage(usage)`.
pub fn as_language_model_usage(usage: &LanguageModelV4Usage) -> LanguageModelUsage {
    LanguageModelUsage {
        input_tokens: usage.input_tokens.total,
        input_token_details: InputTokenDetails {
            no_cache_tokens: usage.input_tokens.no_cache,
            cache_read_tokens: usage.input_tokens.cache_read,
            cache_write_tokens: usage.input_tokens.cache_write,
        },
        output_tokens: usage.output_tokens.total,
        output_token_details: OutputTokenDetails {
            text_tokens: usage.output_tokens.text,
            reasoning_tokens: usage.output_tokens.reasoning,
        },
        total_tokens: add_token_counts(usage.input_tokens.total, usage.output_tokens.total),
        raw: usage.raw.clone(),
    }
}

/// `createNullLanguageModelUsage()`.
pub fn create_null_language_model_usage() -> LanguageModelUsage {
    LanguageModelUsage::default()
}

fn is_null_language_model_usage(usage: &LanguageModelUsage) -> bool {
    usage.input_tokens.is_none()
        && usage.input_token_details.no_cache_tokens.is_none()
        && usage.input_token_details.cache_read_tokens.is_none()
        && usage.input_token_details.cache_write_tokens.is_none()
        && usage.output_tokens.is_none()
        && usage.output_token_details.text_tokens.is_none()
        && usage.output_token_details.reasoning_tokens.is_none()
        && usage.total_tokens.is_none()
        && usage.raw.is_none()
}

/// `addLanguageModelUsage(usage1, usage2)`.
pub fn add_language_model_usage(
    usage1: &LanguageModelUsage,
    usage2: &LanguageModelUsage,
) -> LanguageModelUsage {
    let raw = if is_null_language_model_usage(usage1) {
        usage2.raw.clone()
    } else if is_null_language_model_usage(usage2) {
        usage1.raw.clone()
    } else {
        None
    };
    LanguageModelUsage {
        input_tokens: add_token_counts(usage1.input_tokens, usage2.input_tokens),
        input_token_details: InputTokenDetails {
            no_cache_tokens: add_token_counts(
                usage1.input_token_details.no_cache_tokens,
                usage2.input_token_details.no_cache_tokens,
            ),
            cache_read_tokens: add_token_counts(
                usage1.input_token_details.cache_read_tokens,
                usage2.input_token_details.cache_read_tokens,
            ),
            cache_write_tokens: add_token_counts(
                usage1.input_token_details.cache_write_tokens,
                usage2.input_token_details.cache_write_tokens,
            ),
        },
        output_tokens: add_token_counts(usage1.output_tokens, usage2.output_tokens),
        output_token_details: OutputTokenDetails {
            text_tokens: add_token_counts(
                usage1.output_token_details.text_tokens,
                usage2.output_token_details.text_tokens,
            ),
            reasoning_tokens: add_token_counts(
                usage1.output_token_details.reasoning_tokens,
                usage2.output_token_details.reasoning_tokens,
            ),
        },
        total_tokens: add_token_counts(usage1.total_tokens, usage2.total_tokens),
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::provider::{LanguageModelV4InputTokens, LanguageModelV4OutputTokens};

    fn v4(input: Option<u64>, output: Option<u64>) -> LanguageModelV4Usage {
        LanguageModelV4Usage {
            input_tokens: LanguageModelV4InputTokens {
                total: input,
                ..Default::default()
            },
            output_tokens: LanguageModelV4OutputTokens {
                total: output,
                ..Default::default()
            },
            raw: None,
        }
    }

    #[test]
    fn as_usage_sums_total_tokens() {
        let usage = as_language_model_usage(&v4(Some(3), Some(4)));
        assert_eq!(usage.total_tokens, Some(7));
        assert_eq!(as_language_model_usage(&v4(None, None)).total_tokens, None);
        assert_eq!(
            as_language_model_usage(&v4(Some(2), None)).total_tokens,
            Some(2)
        );
    }

    #[test]
    fn add_usage_keeps_undefined_when_both_undefined() {
        let a = as_language_model_usage(&v4(Some(1), None));
        let b = as_language_model_usage(&v4(Some(2), None));
        let sum = add_language_model_usage(&a, &b);
        assert_eq!(sum.input_tokens, Some(3));
        assert_eq!(sum.output_tokens, None);
        assert_eq!(sum.total_tokens, Some(3));
    }

    #[test]
    fn add_usage_keeps_raw_only_against_null_usage() {
        let mut raw = JsonObject::new();
        raw.insert("x".into(), 1.into());
        let mut a = as_language_model_usage(&v4(Some(1), None));
        a.raw = Some(raw.clone());
        let null = create_null_language_model_usage();
        assert_eq!(add_language_model_usage(&null, &a).raw, Some(raw.clone()));
        assert_eq!(add_language_model_usage(&a, &null).raw, Some(raw));
        assert_eq!(add_language_model_usage(&a, &a).raw, None);
    }
}
