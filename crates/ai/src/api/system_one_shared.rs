//! Port of `api/system-one-shared.ts`: TypeSafe System One classification
//! over a service-specific transport (TypeSafe, OpenRouter, Cloudflare
//! Workers AI).
//!
//! Divergences: `fetch` is a `reqwest` POST; token counts are whole numbers
//! (`Usage` holds `u32`s), so a fractional count is truncated.

use indexmap::IndexMap;
use reqwest::header::HeaderMap;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::models::calculate_cost_for;
use crate::types::{
    ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion,
    ClassifierResult, ClassifierStopReason, ProviderHeaders, ProviderResponse, Usage,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::headers::{
    apply_provider_headers, headers_to_record, provider_headers_to_record,
};
use crate::utils::http::http_client;
use crate::utils::provider_retry::{
    ProviderHttpError, ProviderRetryOptions, retry_provider_request,
};
use crate::{Error, Result};

/// TypeSafe System One request body without the transport-specific envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneWireRequest {
    pub state: Map<String, Value>,
    pub questions: Map<String, Value>,
}

impl SystemOneWireRequest {
    /// `{ state, questions }` as JSON object entries.
    pub fn into_entries(self) -> Map<String, Value> {
        let mut entries = Map::new();
        entries.insert("state".to_string(), Value::Object(self.state));
        entries.insert("questions".to_string(), Value::Object(self.questions));
        entries
    }
}

/// Differences between services that serve System One models.
pub struct SystemOneTransport {
    /// Classifier API implemented by this transport.
    pub api: &'static str,
    /// Service name used in error messages.
    pub label: &'static str,
    /// Absolute request URL.
    pub url: fn(&ClassifierModel) -> String,
    /// Wraps the System One request in the service's request envelope.
    pub payload: fn(&ClassifierModel, SystemOneWireRequest) -> Value,
    /// Extracts the System One output (`{ answers, usage }`) from the
    /// service's response envelope.
    pub output: fn(Value) -> Result<Map<String, Value>>,
}

/// `new URL(path, `${baseUrl without trailing slashes}/`)` for a relative
/// path.
pub(crate) fn join_url(base_url: &str, path: &str) -> String {
    format!("{}/{path}", base_url.trim_end_matches('/'))
}

fn http_error(label: &str, status: u16, headers: IndexMap<String, String>, body: String) -> Error {
    ProviderHttpError {
        status: Some(status),
        headers,
        body: Some(body),
        message: format!("{label} returned {status}"),
    }
    .into()
}

fn timeout_error(timeout_ms: u64) -> Error {
    ProviderHttpError {
        status: None,
        headers: IndexMap::new(),
        body: None,
        message: format!("Request timed out after {timeout_ms}ms"),
    }
    .into()
}

/// One POST of `payload` as JSON: the response status and headers and its
/// parsed JSON body. A non-2xx response is an HTTP error labelled `label`;
/// `timeout_ms` bounds the whole attempt, including reading the body.
pub(crate) async fn post_json(
    client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    payload: &Value,
    label: &str,
    timeout_ms: Option<u64>,
    signal: Option<&CancellationToken>,
) -> Result<(ProviderResponse, Value)> {
    let attempt = async {
        let response = client
            .post(url)
            .headers(headers.clone())
            .body(serde_json::to_string(payload)?)
            .send()
            .await?;
        let status = response.status().as_u16();
        let response_headers = headers_to_record(response.headers());
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(http_error(label, status, response_headers, body));
        }
        let body: Value = serde_json::from_str(&response.text().await?)?;
        Ok((
            ProviderResponse {
                status,
                headers: response_headers,
            },
            body,
        ))
    };
    let timed = async {
        match timeout_ms {
            Some(timeout_ms) => {
                tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), attempt)
                    .await
                    .map_err(|_| timeout_error(timeout_ms))?
            }
            None => attempt.await,
        }
    };
    match signal {
        Some(signal) => {
            if signal.is_cancelled() {
                return Err(Error::Aborted("Request was aborted".to_string()));
            }
            tokio::select! {
                _ = signal.cancelled() => Err(Error::Aborted("Request was aborted".to_string())),
                result = timed => result,
            }
        }
        None => timed.await,
    }
}

/// `providerHeadersToRecord(...)` as request headers.
pub(crate) fn request_header_map(sources: &[Option<&ProviderHeaders>]) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let merged: ProviderHeaders = provider_headers_to_record(sources)
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| (name, Some(value)))
        .collect();
    apply_provider_headers(&mut headers, &merged)?;
    Ok(headers)
}

fn required_number(label: &str, value: Option<&Value>, field: &str) -> Result<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| Error::message(format!("{label} returned an invalid {field}")))
}

fn probabilities(label: &str, value: Option<&Value>, id: &str) -> Result<IndexMap<String, f64>> {
    let Some(Value::Object(value)) = value else {
        return Err(Error::message(format!(
            "{label} returned invalid probabilities for {id}"
        )));
    };
    value
        .iter()
        .map(|(key, probability)| {
            Ok((
                key.clone(),
                required_number(
                    label,
                    Some(probability),
                    &format!("probability for {id}.{key}"),
                )?,
            ))
        })
        .collect()
}

fn parse_answers(
    label: &str,
    value: Option<&Value>,
    context: &ClassifierContext,
) -> Result<IndexMap<String, ClassifierAnswer>> {
    let Some(Value::Object(value)) = value else {
        return Err(Error::message(format!(
            "{label} returned an unexpected response"
        )));
    };
    let mut answers = IndexMap::new();
    for (id, question) in &context.questions {
        let Some(Value::Object(answer)) = value.get(id) else {
            return Err(Error::message(format!(
                "{label} did not return an answer for {id}"
            )));
        };
        let answer_type = answer.get("type").and_then(Value::as_str);
        let parsed = match question {
            ClassifierQuestion::Choice { .. } => {
                let (Some("choice"), Some(Value::String(choice))) =
                    (answer_type, answer.get("choice"))
                else {
                    return Err(Error::message(format!(
                        "{label} did not return a choice answer for {id}"
                    )));
                };
                ClassifierAnswer::Choice {
                    choice: choice.clone(),
                    probabilities: probabilities(label, answer.get("probabilities"), id)?,
                    confidence: required_number(
                        label,
                        answer.get("confidence"),
                        &format!("confidence for {id}"),
                    )?,
                }
            }
            ClassifierQuestion::Score { .. } => {
                if answer_type != Some("score") {
                    return Err(Error::message(format!(
                        "{label} did not return a score answer for {id}"
                    )));
                }
                ClassifierAnswer::Score {
                    score: required_number(label, answer.get("score"), &format!("score for {id}"))?,
                    confidence: required_number(
                        label,
                        answer.get("confidence"),
                        &format!("confidence for {id}"),
                    )?,
                }
            }
            ClassifierQuestion::Bool { .. } => {
                if answer_type != Some("noul") {
                    return Err(Error::message(format!(
                        "{label} did not return a bool answer for {id}"
                    )));
                }
                ClassifierAnswer::Bool {
                    probability: required_number(
                        label,
                        answer.get("noul"),
                        &format!("probability for {id}"),
                    )?,
                }
            }
        };
        answers.insert(id.clone(), parsed);
    }
    Ok(answers)
}

fn token_count(value: Option<&Value>) -> u32 {
    match value.and_then(Value::as_f64) {
        Some(count) if count.is_finite() && count > 0.0 => count as u32,
        _ => 0,
    }
}

/// Usage from System One's `{ input_tokens, output_tokens }`, priced from
/// the model catalog like chat usage. A missing or malformed usage object
/// leaves the result without usage instead of failing it.
fn parse_usage(value: Option<&Value>, model: &ClassifierModel) -> Option<Usage> {
    let Some(Value::Object(value)) = value else {
        return None;
    };
    if !value.contains_key("input_tokens") && !value.contains_key("output_tokens") {
        return None;
    }
    let input = token_count(value.get("input_tokens"));
    let output = token_count(value.get("output_tokens"));
    let mut usage = Usage {
        input,
        output,
        total_tokens: input + output,
        ..Default::default()
    };
    calculate_cost_for(&model.cost, &mut usage);
    Some(usage)
}

/// Maps public `bool` questions to TypeSafe's wire-level `noul` type.
fn wire_request(context: &ClassifierContext) -> SystemOneWireRequest {
    let questions = context
        .questions
        .iter()
        .map(|(id, question)| {
            let mut value = serde_json::to_value(question).unwrap_or(Value::Null);
            if matches!(question, ClassifierQuestion::Bool { .. }) {
                value["type"] = json!("noul");
            }
            (id.clone(), value)
        })
        .collect();
    SystemOneWireRequest {
        state: context.state.clone(),
        questions,
    }
}

fn request_headers(
    model: &ClassifierModel,
    api_key: &str,
    options_headers: Option<&ProviderHeaders>,
) -> Result<HeaderMap> {
    let mut base = ProviderHeaders::new();
    base.insert("authorization", Some(format!("Bearer {api_key}")));
    base.insert("content-type", Some("application/json".to_string()));
    let model_headers: Option<ProviderHeaders> = model.headers.clone().map(Into::into);
    request_header_map(&[Some(&base), model_headers.as_ref(), options_headers])
}

/// Runs one System One classification over the given transport.
pub async fn classify_system_one(
    transport: &SystemOneTransport,
    model: ClassifierModel,
    context: ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    let mut output = ClassifierResult::empty_for(&model);
    if let Err(error) = run(transport, &model, &context, &options, &mut output).await {
        output.stop_reason = if options
            .signal
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            ClassifierStopReason::Aborted
        } else {
            ClassifierStopReason::Error
        };
        output.error_message = Some(format_provider_error(
            &normalize_provider_error(&error),
            Some(&format!("{} error", transport.label)),
        ));
    }
    output
}

async fn run(
    transport: &SystemOneTransport,
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: &ClassifierOptions,
    output: &mut ClassifierResult,
) -> Result<()> {
    if model.api != transport.api {
        return Err(Error::message(format!(
            "Unsupported classifier API: {}",
            model.api
        )));
    }
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::message(format!(
            "No API key for provider: {}",
            model.provider
        )));
    };
    let mut payload = (transport.payload)(model, wire_request(context));
    if let Some(on_payload) = &options.on_payload
        && let Some(transformed) = on_payload(payload.clone(), model).await?
    {
        payload = transformed;
    }
    let client = http_client(options.http_client.as_ref());
    let url = (transport.url)(model);
    let headers = request_headers(model, api_key, options.headers.as_ref())?;
    let (response, body) = retry_provider_request(
        || {
            post_json(
                &client,
                &url,
                &headers,
                &payload,
                transport.label,
                options.timeout_ms,
                options.signal.as_ref(),
            )
        },
        &ProviderRetryOptions {
            max_retries: Some(options.max_retries.unwrap_or(2)),
            max_retry_delay_ms: options.max_retry_delay_ms,
            signal: options.signal.clone(),
        },
    )
    .await?;
    if let Some(on_response) = &options.on_response {
        on_response(response, model).await?;
    }
    let result = (transport.output)(body)?;
    // Set before parsing answers: a request with malformed answers was still billed.
    output.usage = parse_usage(result.get("usage"), model);
    output.answers = parse_answers(transport.label, result.get("answers"), context)?;
    Ok(())
}
