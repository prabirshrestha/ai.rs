//! Port of `api/typesafe-system-one.ts` (plus its `.lazy.ts`): TypeSafe's
//! native System One protocol. OpenRouter serves the same protocol, so both
//! providers use this API with different base URLs.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use super::system_one_shared::{
    SystemOneTransport, SystemOneWireRequest, classify_system_one, join_url,
};
use crate::types::{
    ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, KnownClassifierApi,
    ProviderClassifier,
};
use crate::{Error, Result};

fn output(body: Value) -> Result<Map<String, Value>> {
    match body {
        Value::Object(body) => Ok(body),
        _ => Err(Error::message(
            "System One API returned an unexpected response",
        )),
    }
}

fn payload(model: &ClassifierModel, request: SystemOneWireRequest) -> Value {
    let mut payload = Map::new();
    payload.insert("model".to_string(), Value::String(model.id.clone()));
    payload.extend(request.into_entries());
    Value::Object(payload)
}

const TRANSPORT: SystemOneTransport = SystemOneTransport {
    api: KnownClassifierApi::TypesafeSystemOne.as_str(),
    label: "System One API",
    url: |model| join_url(&model.base_url, "systemone"),
    payload,
    output,
};

/// TypeSafe System One classification with public `bool` values mapped to
/// wire-level `noul`.
pub async fn classify(
    model: ClassifierModel,
    context: ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    classify_system_one(&TRANSPORT, model, context, options).await
}

struct TypesafeSystemOneApi;

#[async_trait]
impl ProviderClassifier for TypesafeSystemOneApi {
    async fn classify(
        &self,
        model: ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> ClassifierResult {
        classify(model, context, options).await
    }
}

/// `typesafeSystemOneApi()`.
pub fn typesafe_system_one_api() -> Arc<dyn ProviderClassifier> {
    Arc::new(TypesafeSystemOneApi)
}

#[cfg(test)]
mod tests {
    //! Port of `test/typesafe-system-one.test.ts`. `fetch` mocks become a
    //! local mock server.

    use serde_json::json;

    use super::*;
    use crate::api::openai_client::test_support::{MockResponse, MockServer, serve_silent};
    use crate::types::{ClassifierAnswer, ClassifierStopReason, ProviderHeaders};

    fn model(base_url: &str) -> ClassifierModel {
        serde_json::from_value(json!({
            "type": "classifier",
            "id": "jev-latest",
            "name": "Jev",
            "api": "typesafe-system-one",
            "provider": "typesafe",
            "baseUrl": base_url,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 64000,
        }))
        .unwrap()
    }

    pub(crate) fn context() -> ClassifierContext {
        serde_json::from_value(json!({
            "state": { "text": "The deployment succeeded, thank you." },
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Classify the message",
                    "criteria": { "success": "Successful", "failure": "Failed" },
                },
                "satisfaction": {
                    "type": "score",
                    "instructions": "Score satisfaction",
                    "criteria": ["low", "neutral", "high"],
                },
                "approved": {
                    "type": "bool",
                    "instructions": "Does the user approve?",
                    "criteria": { "true": "Approval", "false": "No approval" },
                },
            },
        }))
        .unwrap()
    }

    pub(crate) fn wire_answers() -> Value {
        json!({
            "category": {
                "type": "choice",
                "choice": "success",
                "probabilities": { "success": 0.9, "failure": 0.1 },
                "confidence": 0.8,
            },
            "satisfaction": { "type": "score", "score": 2, "confidence": 0.7 },
            "approved": { "type": "noul", "noul": 0.95 },
        })
    }

    fn with_key() -> ClassifierOptions {
        ClassifierOptions {
            api_key: Some("secret".to_string()),
            ..Default::default()
        }
    }

    /// `https://api.typesafe.ai/v1/` with the mock server as host.
    fn base(server: &MockServer) -> String {
        format!("{}/", server.url)
    }

    #[tokio::test]
    async fn maps_public_bool_questions_and_answers_to_typesafe_noul_values() {
        let server = MockServer::start(vec![MockResponse::json(
            json!({ "answers": wire_answers() }),
        )])
        .await;
        let result = classify(
            model(&base(&server)),
            context(),
            ClassifierOptions {
                temperature: Some(1.5),
                ..with_key()
            },
        )
        .await;
        let priced_server = MockServer::start(vec![MockResponse::json(json!({
            "answers": wire_answers(),
            "usage": { "input_tokens": 308, "output_tokens": 23 },
        }))])
        .await;
        let mut priced_model = model(&base(&priced_server));
        priced_model.cost.input = 0.042;
        let priced_result = classify(priced_model, context(), with_key()).await;

        let requests = server.requests();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.path, "/v1/systemone");
        assert_eq!(request.body["model"], "jev-latest");
        assert_eq!(request.body["questions"]["category"]["type"], "choice");
        assert_eq!(request.body["questions"]["satisfaction"]["type"], "score");
        assert_eq!(request.body["questions"]["approved"]["type"], "noul");
        // System One has no temperature field; the option is ignored.
        assert!(request.body.get("temperature").is_none());
        assert_eq!(request.header("authorization"), Some("Bearer secret"));

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers["approved"],
            ClassifierAnswer::Bool { probability: 0.95 }
        );
        assert!(matches!(
            &result.answers["category"],
            ClassifierAnswer::Choice { choice, .. } if choice == "success"
        ));
        assert_eq!(
            result.answers["satisfaction"],
            ClassifierAnswer::Score {
                score: 2.0,
                confidence: 0.7
            }
        );
        assert!(result.usage.is_none());
        let usage = priced_result.usage.unwrap();
        assert_eq!(
            (usage.input, usage.output, usage.total_tokens),
            (308, 23, 331)
        );
        assert!((usage.cost.total - 0.000012936).abs() < 1e-12);
    }

    #[tokio::test]
    async fn posts_openrouter_system_one_requests_to_its_typesafe_compatible_endpoint() {
        // Response shape observed from the live OpenRouter endpoint.
        let server = MockServer::start(vec![MockResponse::json(json!({
            "id": "gen-dec-1",
            "provider": "TypeSafe",
            "answers": wire_answers(),
            "usage": { "input_tokens": 308, "output_tokens": 23, "cost": 0.000012936 },
        }))])
        .await;
        let mut open_router_model = model(&server.url);
        open_router_model.id = "typesafe/jev-1.13".to_string();
        open_router_model.provider = "openrouter".to_string();
        open_router_model.cost.input = 0.042;

        let result = classify(open_router_model, context(), with_key()).await;

        let request = server.last();
        assert_eq!(request.path, "/v1/systemone");
        assert_eq!(request.body["model"], "typesafe/jev-1.13");
        assert_eq!(request.body["state"], json!(context().state));
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers["approved"],
            ClassifierAnswer::Bool { probability: 0.95 }
        );
        // Priced from the catalog like chat usage; matches OpenRouter's reported cost.
        assert!((result.usage.unwrap().cost.total - 0.000012936).abs() < 1e-12);
    }

    #[tokio::test]
    async fn rejects_models_for_other_classifier_apis() {
        let server = MockServer::start(vec![MockResponse::json(
            json!({ "answers": wire_answers() }),
        )])
        .await;
        let mut other = model(&server.url);
        other.api = "cloudflare-workers-ai-system-one".to_string();
        let result = classify(other, context(), with_key()).await;

        assert!(server.requests().is_empty());
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert!(
            result
                .error_message
                .unwrap()
                .contains("Unsupported classifier API: cloudflare-workers-ai-system-one")
        );
    }

    #[tokio::test]
    async fn merges_headers_case_insensitively_and_supports_null_suppression() {
        let server = MockServer::start(vec![MockResponse::json(
            json!({ "answers": wire_answers() }),
        )])
        .await;
        let mut model_with_headers = model(&server.url);
        model_with_headers.headers = Some(
            [
                ("authorization".to_string(), "Bearer model".to_string()),
                ("X-Source".to_string(), "model".to_string()),
            ]
            .into_iter()
            .collect(),
        );

        let mut request_headers = ProviderHeaders::new();
        request_headers.insert("Authorization", Some("Bearer request".to_string()));
        request_headers.insert("x-source", Some("request".to_string()));
        classify(
            model_with_headers.clone(),
            context(),
            ClassifierOptions {
                headers: Some(request_headers),
                ..with_key()
            },
        )
        .await;
        let mut suppress = ProviderHeaders::new();
        suppress.insert("Authorization", None::<String>);
        classify(
            model_with_headers,
            context(),
            ClassifierOptions {
                headers: Some(suppress),
                ..with_key()
            },
        )
        .await;

        let requests = server.requests();
        assert_eq!(requests[0].header("authorization"), Some("Bearer request"));
        assert_eq!(requests[0].header("x-source"), Some("request"));
        assert_eq!(
            requests[0]
                .headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .count(),
            1
        );
        assert_eq!(requests[1].header("authorization"), None);
    }

    #[tokio::test]
    async fn preserves_prototype_sensitive_question_ids_in_answers() {
        let server = MockServer::start(vec![MockResponse::json(json!({
            "answers": { "__proto__": { "type": "noul", "noul": 0.75 } },
        }))])
        .await;
        let prototype_context: ClassifierContext = serde_json::from_value(json!({
            "state": {},
            "questions": {
                "__proto__": {
                    "type": "bool",
                    "instructions": "Is this true?",
                    "criteria": { "true": "Yes", "false": "No" },
                },
            },
        }))
        .unwrap();
        let result = classify(model(&server.url), prototype_context, with_key()).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(result.answers.keys().collect::<Vec<_>>(), ["__proto__"]);
        assert_eq!(
            result.answers["__proto__"],
            ClassifierAnswer::Bool { probability: 0.75 }
        );
        assert_eq!(
            serde_json::to_value(&result.answers).unwrap()["__proto__"],
            json!({ "type": "bool", "probability": 0.75 })
        );
    }

    #[tokio::test]
    async fn reports_request_timeouts_separately_from_caller_cancellation() {
        let url = serve_silent().await;
        let result = classify(
            model(&url),
            context(),
            ClassifierOptions {
                timeout_ms: Some(5),
                max_retries: Some(0),
                ..with_key()
            },
        )
        .await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Request timed out after 5ms")
        );
    }

    #[tokio::test]
    async fn creates_a_fresh_timeout_for_every_retry_attempt() {
        // Pi compares the per-attempt abort signals; here the second attempt
        // must succeed within its own timeout after the first one failed.
        let server = MockServer::start(vec![
            MockResponse::status(500, &[("retry-after-ms", "0")], "retry"),
            MockResponse::json(json!({ "answers": wire_answers() })),
        ])
        .await;
        let result = classify(
            model(&server.url),
            context(),
            ClassifierOptions {
                timeout_ms: Some(1000),
                max_retries: Some(1),
                ..with_key()
            },
        )
        .await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(server.requests().len(), 2);
    }

    #[tokio::test]
    async fn returns_malformed_responses_as_classifier_errors() {
        let server = MockServer::start(vec![MockResponse::json(json!({
            "answers": {},
            "usage": { "input_tokens": 10, "output_tokens": 2 },
        }))])
        .await;
        let result = classify(model(&server.url), context(), with_key()).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert!(result.answers.is_empty());
        assert!(
            result
                .error_message
                .unwrap()
                .contains("did not return an answer for category")
        );
        // The request was billed, so its usage is kept.
        let usage = result.usage.unwrap();
        assert_eq!((usage.input, usage.output), (10, 2));
    }

    #[tokio::test]
    async fn ignores_malformed_usage() {
        let server = MockServer::start(vec![
            MockResponse::json(json!({
                "answers": wire_answers(),
                "usage": { "input_tokens": "many", "output_tokens": 3 },
            })),
            MockResponse::json(json!({ "answers": wire_answers(), "usage": { "cost": 0.1 } })),
        ])
        .await;
        let result = classify(model(&server.url), context(), with_key()).await;
        let without_tokens = classify(model(&server.url), context(), with_key()).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        let usage = result.usage.unwrap();
        assert_eq!((usage.input, usage.output, usage.total_tokens), (0, 3, 3));
        assert_eq!(without_tokens.stop_reason, ClassifierStopReason::Stop);
        assert!(without_tokens.usage.is_none());
    }
}
