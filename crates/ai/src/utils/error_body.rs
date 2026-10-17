//! Port of `utils/error-body.ts`: shared normalization for provider HTTP
//! error objects.
//!
//! Pi probes SDK-specific error fields. The Rust API modules raise
//! [`ProviderHttpError`] (status plus raw body) or other [`Error`]s, so the
//! probing reduces to reading those fields; [`ProviderErrorSource`] lets
//! callers describe other shapes (an SDK message plus a parsed JSON body).

use serde_json::Value;

use crate::Error;
use crate::utils::estimate::js_length;

pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedProviderError {
    /// HTTP status code, when one could be extracted.
    pub status: Option<u16>,
    /// Raw HTTP body reason, already trimmed and truncated to the cap.
    pub body: Option<String>,
    /// The error message, or the JSON of a non-`Error` thrown value.
    pub message: String,
    /// True when `message` already contains the body.
    pub message_carries_body: bool,
}

/// The body of a provider error: raw text or a parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderErrorBody {
    Text(String),
    Json(Value),
}

/// An error shape to normalize.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderErrorSource {
    /// An `Error` with a message and optional status/body fields.
    Error {
        message: String,
        status: Option<u16>,
        body: Option<ProviderErrorBody>,
    },
    /// A thrown non-`Error` value.
    Thrown(Value),
}

impl From<&Error> for ProviderErrorSource {
    fn from(error: &Error) -> Self {
        match error {
            Error::ProviderHttp(http) => Self::Error {
                message: http.message.clone(),
                status: http.status,
                body: http.body.clone().map(ProviderErrorBody::Text),
            },
            Error::Http(http) => Self::Error {
                message: error.to_string(),
                status: http.status().map(|status| status.as_u16()),
                body: None,
            },
            _ => Self::Error {
                message: error.to_string(),
                status: None,
                body: None,
            },
        }
    }
}

pub fn normalize_provider_error(error: impl Into<ProviderErrorSource>) -> NormalizedProviderError {
    match error.into() {
        ProviderErrorSource::Thrown(value) => NormalizedProviderError {
            status: None,
            body: None,
            message: safe_json_stringify(&value),
            message_carries_body: false,
        },
        ProviderErrorSource::Error {
            message,
            status,
            body,
        } => {
            let body = extract_body(body.as_ref());
            let message_carries_body = body.as_ref().is_none_or(|body| message.contains(body));
            NormalizedProviderError {
                status,
                body,
                message,
                message_carries_body,
            }
        }
    }
}

fn extract_body(body: Option<&ProviderErrorBody>) -> Option<String> {
    let body_text = match body? {
        ProviderErrorBody::Text(text) => text.clone(),
        ProviderErrorBody::Json(value) if is_plain_non_empty_object(value) => {
            safe_json_stringify(value)
        }
        ProviderErrorBody::Json(_) => return None,
    };
    let trimmed = body_text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS))
}

fn is_plain_non_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(|object| !object.is_empty())
}

/// Compose a display string from a normalized error:
/// `"<status>: <body>"` or `"<prefix> (<status>): <body>"`, or the message
/// when it already carries the body or no status/body was extracted.
pub fn format_provider_error(norm: &NormalizedProviderError, prefix: Option<&str>) -> String {
    match (norm.message_carries_body, norm.status, &norm.body) {
        (false, Some(status), Some(body)) => match prefix {
            Some(prefix) => format!("{prefix} ({status}): {body}"),
            None => format!("{status}: {body}"),
        },
        _ => match (prefix, norm.status) {
            (Some(prefix), Some(status)) => format!("{prefix} ({status}): {}", norm.message),
            _ => norm.message.clone(),
        },
    }
}

/// Truncate to `max_chars` JavaScript characters (UTF-16 code units).
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
    let length = js_length(text);
    if length <= max_chars {
        return text.to_string();
    }
    let mut units = 0;
    let mut end = 0;
    for (index, ch) in text.char_indices() {
        if units + ch.len_utf16() > max_chars {
            break;
        }
        units += ch.len_utf16();
        end = index + ch.len_utf8();
    }
    format!(
        "{}... [truncated {} chars]",
        &text[..end],
        length - max_chars
    )
}

pub fn safe_json_stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn sdk_error(message: &str, status: u16, body: ProviderErrorBody) -> ProviderErrorSource {
        ProviderErrorSource::Error {
            message: message.to_string(),
            status: Some(status),
            body: Some(body),
        }
    }

    #[test]
    fn extracts_status_and_body_from_a_text_body() {
        let norm = normalize_provider_error(sdk_error(
            "Mistral request failed",
            403,
            ProviderErrorBody::Text(r#"{"error":"blocked by gateway WAF"}"#.to_string()),
        ));
        assert_eq!(norm.status, Some(403));
        assert_eq!(
            norm.body.as_deref(),
            Some(r#"{"error":"blocked by gateway WAF"}"#)
        );
        assert!(!norm.message_carries_body);
    }

    #[test]
    fn reads_a_parsed_body_when_the_message_is_opaque() {
        let norm = normalize_provider_error(sdk_error(
            "403 status code (no body)",
            403,
            ProviderErrorBody::Json(json!({ "error": "blocked by gateway WAF" })),
        ));
        assert_eq!(
            norm.body.as_deref(),
            Some(r#"{"error":"blocked by gateway WAF"}"#)
        );
        assert!(!norm.message_carries_body);
    }

    #[test]
    fn preserves_the_message_when_it_already_carries_the_body() {
        let body = json!({ "error": { "code": 403, "message": "Permission denied" } }).to_string();
        let norm = normalize_provider_error(ProviderErrorSource::Error {
            message: body.clone(),
            status: Some(403),
            body: None,
        });
        assert!(norm.message_carries_body);
        assert_eq!(norm.message, body);
    }

    #[test]
    fn json_stringifies_a_non_error_thrown_value() {
        let norm =
            normalize_provider_error(ProviderErrorSource::Thrown(json!({ "reason": "boom" })));
        assert_eq!(norm.status, None);
        assert_eq!(norm.message, r#"{"reason":"boom"}"#);
        assert!(!norm.message_carries_body);
        assert_eq!(format_provider_error(&norm, None), r#"{"reason":"boom"}"#);
    }

    #[test]
    fn treats_an_empty_parsed_body_object_as_no_body() {
        let norm = normalize_provider_error(sdk_error(
            "403 status code (no body)",
            403,
            ProviderErrorBody::Json(json!({})),
        ));
        assert_eq!(norm.body, None);
        assert!(norm.message_carries_body);
    }

    #[test]
    fn truncates_the_body_at_the_cap() {
        let long_body = "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS + 50);
        let norm = normalize_provider_error(sdk_error(
            "failed",
            500,
            ProviderErrorBody::Text(long_body.clone()),
        ));
        let body = norm.body.unwrap();
        assert!(body.contains("... [truncated 50 chars]"));
        assert!(body.len() < long_body.len());
    }

    #[test]
    fn sets_message_carries_body_when_the_message_contains_the_body() {
        let norm = normalize_provider_error(sdk_error(
            "500: upstream exploded",
            500,
            ProviderErrorBody::Text("upstream exploded".to_string()),
        ));
        assert!(norm.message_carries_body);
    }

    #[test]
    fn formats_status_body_and_prefix() {
        let norm = normalize_provider_error(sdk_error(
            "403 status code (no body)",
            403,
            ProviderErrorBody::Json(json!({ "error": "blocked by gateway WAF" })),
        ));
        assert_eq!(
            format_provider_error(&norm, None),
            r#"403: {"error":"blocked by gateway WAF"}"#
        );
        assert_eq!(
            format_provider_error(&norm, Some("OpenAI API error")),
            r#"OpenAI API error (403): {"error":"blocked by gateway WAF"}"#
        );
        let body = json!({ "error": { "message": "Permission denied" } }).to_string();
        let carried = normalize_provider_error(ProviderErrorSource::Error {
            message: body.clone(),
            status: Some(403),
            body: None,
        });
        assert_eq!(
            format_provider_error(&carried, Some("OpenAI API error")),
            format!("OpenAI API error (403): {body}")
        );
    }

    #[test]
    fn normalizes_crate_provider_http_errors() {
        let error = Error::from(crate::utils::provider_retry::ProviderHttpError::new(
            400,
            Default::default(),
            Some("bad".to_string()),
        ));
        let norm = normalize_provider_error(&error);
        assert_eq!(norm.status, Some(400));
        assert_eq!(norm.message, "400 bad");
        assert!(norm.message_carries_body);
    }
}
