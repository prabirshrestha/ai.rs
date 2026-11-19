//! Parts every classifier API shares: one JSON POST with bearer auth, hooks
//! and retries, usage priced from the model, and error messages. Ports Pi's
//! `classifier-shared.ts`.

use serde_json::{Map, Value};

use crate::classifier::{ClassifierOptions, ClassifierResult};
use crate::models::calculate_cost;
use crate::providers::openai_embeddings::build_headers;
use crate::types::{Model, ProviderResponse, Usage};
use crate::utils::headers::headers_to_record;
use crate::utils::http::{request_timeout, send_request_with_retries_except};

/// Classifier requests retry twice unless the caller sets `max_retries`.
const DEFAULT_MAX_RETRIES: u32 = 2;

/// A classifier failure, before it becomes a result's `error_message`.
#[derive(Debug)]
pub(crate) enum ClassifierError {
    /// The service answered with an error status.
    Status { status: u16, body: String },
    /// Any other failure, with its whole message.
    Message(String),
}

pub(crate) type ClassifierOutcome<T> = std::result::Result<T, ClassifierError>;

impl ClassifierError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// The message Pi's `formatProviderError` gives: `{prefix} ({status}):
    /// {body}` for an HTTP failure, the bare message otherwise.
    pub(crate) fn format(&self, prefix: &str) -> String {
        match self {
            Self::Status { status, body } => format!("{prefix} ({status}): {body}"),
            Self::Message(message) => message.clone(),
        }
    }
}

impl From<crate::Error> for ClassifierError {
    fn from(error: crate::Error) -> Self {
        match error {
            crate::Error::ApiStatus { status, body } => Self::Status {
                status: status.as_u16(),
                body,
            },
            error => Self::Message(error.to_string()),
        }
    }
}

/// The credentials and client a classifier API was made with.
#[derive(Clone)]
pub(crate) struct ClassifierAuth {
    pub(crate) api_key: Option<String>,
    /// A custom base URL may serve classifiers with no key, as a local
    /// llama.cpp server does.
    pub(crate) allow_missing_api_key: bool,
    pub(crate) http_client: Option<reqwest::Client>,
}

impl ClassifierAuth {
    /// Fills the options' key and client from the API's own when unset.
    pub(crate) fn apply(&self, options: &mut ClassifierOptions) {
        if options
            .base
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
        {
            options.base.api_key = self.api_key.clone();
        }
        if options.base.http_client.is_none() {
            options.base.http_client = self.http_client.clone();
        }
    }
}

/// Posts one JSON classifier request and returns the parsed response body.
/// `no_retry_statuses` fail at once although they are normally retried.
pub(crate) async fn post_classifier_request(
    label: &str,
    url: &str,
    model: &Model,
    body: Value,
    auth: &ClassifierAuth,
    options: &ClassifierOptions,
    no_retry_statuses: &[u16],
) -> ClassifierOutcome<Value> {
    let api_key = options
        .base
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.trim().is_empty());
    if api_key.is_none() && !auth.allow_missing_api_key {
        return Err(ClassifierError::message(format!(
            "No API key for provider: {}",
            model.provider
        )));
    }
    let mut payload = body;
    if let Some(on_payload) = &options.base.on_payload
        && let Some(next) = on_payload(payload.clone(), model).await?
    {
        payload = next;
    }

    let mut base = options.base.clone();
    base.max_retries = Some(base.max_retries.unwrap_or(DEFAULT_MAX_RETRIES));
    let client = base.http_client.clone().unwrap_or_default();
    let headers = build_headers(api_key, &model.headers, &base.headers)?;
    let response = send_request_with_retries_except(&base, no_retry_statuses, || {
        client
            .post(url)
            .headers(headers.clone())
            .json(&payload)
            .timeout(request_timeout(base.timeout_ms))
    })
    .await
    .map_err(|error| timeout_error(error, base.timeout_ms))?;
    let status = response.status();
    let response_headers = response.headers().clone();
    let text = response
        .text()
        .await
        .map_err(|error| timeout_error(error.into(), base.timeout_ms))?;
    if !status.is_success() {
        return Err(ClassifierError::Status {
            status: status.as_u16(),
            body: text,
        });
    }
    let json = serde_json::from_str(&text).map_err(|error| {
        ClassifierError::message(format!("{label} returned invalid JSON: {error}"))
    })?;
    if let Some(on_response) = &options.base.on_response {
        on_response(
            ProviderResponse {
                status: status.as_u16(),
                headers: headers_to_record(&response_headers),
            },
            model,
        )
        .await?;
    }
    Ok(json)
}

fn timeout_error(error: crate::Error, timeout_ms: Option<u64>) -> ClassifierError {
    match &error {
        crate::Error::Http(http) if http.is_timeout() => ClassifierError::message(format!(
            "Request timed out after {}ms",
            request_timeout(timeout_ms).as_millis()
        )),
        _ => error.into(),
    }
}

/// Usage from a `{ input_tokens, output_tokens }` object, priced from the
/// model. A missing or malformed object leaves the result without usage.
pub(crate) fn parse_classifier_usage(value: Option<&Value>, model: &Model) -> Option<Usage> {
    let value = value?.as_object()?;
    if !value.contains_key("input_tokens") && !value.contains_key("output_tokens") {
        return None;
    }
    let input = token_count(value.get("input_tokens"));
    let output = token_count(value.get("output_tokens"));
    let mut usage = Usage {
        input,
        output,
        total_tokens: input.saturating_add(output),
        ..Usage::default()
    };
    calculate_cost(model, &mut usage);
    Some(usage)
}

fn token_count(value: Option<&Value>) -> u32 {
    value
        .and_then(Value::as_f64)
        .filter(|count| count.is_finite() && *count > 0.0)
        .map_or(0, |count| count.min(f64::from(u32::MAX)) as u32)
}

/// A finite number from a response field, or the error Pi gives.
pub(crate) fn required_number(
    label: &str,
    value: Option<&Value>,
    field: &str,
) -> ClassifierOutcome<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| ClassifierError::message(format!("{label} returned an invalid {field}")))
}

pub(crate) fn as_record(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

/// Turns the outcome of one classification into its result. Usage stays on
/// a failed result: a request with malformed or refused answers was still
/// billed.
pub(crate) fn finish(
    mut result: ClassifierResult,
    outcome: ClassifierOutcome<()>,
    label: &str,
    options: &ClassifierOptions,
    format: impl FnOnce(&ClassifierError, &str) -> String,
) -> ClassifierResult {
    if let Err(error) = outcome {
        result.answers.clear();
        result.stop_reason = if options.is_aborted() {
            crate::classifier::ClassifierStopReason::Aborted
        } else {
            crate::classifier::ClassifierStopReason::Error
        };
        result.error_message = Some(format(&error, label));
    }
    result
}
