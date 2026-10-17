//! Port of `api/cloudflare-workers-ai-system-one.ts` (plus its `.lazy.ts`):
//! System One models on the Workers AI REST endpoint, `POST
//! /accounts/{account}/ai/run` with `{ model, input }`.
//!
//! The REST API wraps the model output in Cloudflare's API envelope.
//! Third-party models such as `typesafe/jev` add a run record:
//! `{ success, result: { state: "Completed", result: { answers, usage } } }`
//! (<https://developers.cloudflare.com/ai/models/typesafe/jev/>).
//! Cloudflare-hosted models such as `@cf/cloudflare/clef` return the output
//! directly: `{ success, result: { model, answers, usage } }`
//! (<https://developers.cloudflare.com/workers-ai/models/clef/>).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::system_one_shared::{
    SystemOneTransport, SystemOneWireRequest, classify_system_one, join_url,
};
use crate::types::{
    ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, KnownClassifierApi,
    ProviderClassifier,
};
use crate::{Error, Result};

const LABEL: &str = "Cloudflare Workers AI";

fn cloudflare_error_message(errors: Option<&Value>) -> String {
    if let Some(Value::Array(errors)) = errors {
        let messages: Vec<&str> = errors
            .iter()
            .filter_map(|error| error.get("message").and_then(Value::as_str))
            .collect();
        if !messages.is_empty() {
            return format!("{LABEL} error: {}", messages.join("; "));
        }
    }
    format!("{LABEL} request failed")
}

fn unexpected() -> Error {
    Error::message(format!("{LABEL} returned an unexpected response"))
}

fn output(body: Value) -> Result<Map<String, Value>> {
    let Value::Object(mut body) = body else {
        return Err(unexpected());
    };
    if body.get("success") == Some(&Value::Bool(false)) {
        return Err(Error::message(cloudflare_error_message(body.get("errors"))));
    }
    let Some(Value::Object(mut result)) = body.remove("result") else {
        return Err(unexpected());
    };
    if result.contains_key("answers") {
        return Ok(result);
    }
    if result.get("state").and_then(Value::as_str) != Some("Completed") {
        let state = match result.get("state") {
            Some(Value::String(state)) => state.clone(),
            None => "undefined".to_string(),
            Some(state) => state.to_string(),
        };
        return Err(Error::message(format!(
            "{LABEL} run did not complete (state: {state})"
        )));
    }
    match result.remove("result") {
        Some(Value::Object(result)) => Ok(result),
        _ => Err(unexpected()),
    }
}

fn payload(model: &ClassifierModel, request: SystemOneWireRequest) -> Value {
    json!({ "model": model.id, "input": Value::Object(request.into_entries()) })
}

const TRANSPORT: SystemOneTransport = SystemOneTransport {
    api: KnownClassifierApi::CloudflareWorkersAiSystemOne.as_str(),
    label: LABEL,
    url: |model| join_url(&model.base_url, "run"),
    payload,
    output,
};

/// Cloudflare Workers AI System One classification with public `bool`
/// values mapped to wire-level `noul`.
pub async fn classify(
    model: ClassifierModel,
    context: ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    classify_system_one(&TRANSPORT, model, context, options).await
}

struct CloudflareWorkersAiSystemOneApi;

#[async_trait]
impl ProviderClassifier for CloudflareWorkersAiSystemOneApi {
    async fn classify(
        &self,
        model: ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> ClassifierResult {
        classify(model, context, options).await
    }
}

/// `cloudflareWorkersAISystemOneApi()`.
pub fn cloudflare_workers_ai_system_one_api() -> Arc<dyn ProviderClassifier> {
    Arc::new(CloudflareWorkersAiSystemOneApi)
}

#[cfg(test)]
mod tests {
    //! Port of `test/cloudflare-workers-ai-system-one.test.ts`. The `fetch`
    //! mocks become a local mock server: requests go to a copy of the catalog
    //! model whose host is the mock server, keeping the
    //! `{CLOUDFLARE_ACCOUNT_ID}` placeholder so the provider still fills it.

    use serde_json::json;

    use super::*;
    use crate::api::cloudflare::CLOUDFLARE_WORKERS_AI_REST_BASE_URL;
    use crate::api::openai_client::test_support::{MockResponse, MockServer};
    use crate::models::{Models, create_models};
    use crate::providers::all::get_builtin_classifier_model;
    use crate::providers::cloudflare_workers_ai::cloudflare_workers_ai_provider;
    use crate::types::{ClassifierAnswer, ClassifierStopReason, ModelType, ProviderEnv};

    fn context() -> ClassifierContext {
        serde_json::from_value(json!({
            "state": { "message": "Help! My payouts have been failing for 3 days." },
            "questions": {
                "is_urgent": {
                    "type": "bool",
                    "instructions": "Does this convey urgency?",
                    "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" },
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": { "billing": "Payments", "technical": "Bugs" },
                },
            },
        }))
        .unwrap()
    }

    // Model output from https://developers.cloudflare.com/ai/models/typesafe/jev/
    fn jev_output() -> Value {
        json!({
            "model": "jev-1.13.0",
            "answers": {
                "is_urgent": { "type": "noul", "noul": 0.95 },
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.8,
                    "probabilities": { "billing": 0.87, "technical": 0.13 },
                },
            },
            "usage": { "input_tokens": 426, "output_tokens": 73 },
        })
    }

    // REST envelope observed from the live /ai/run endpoint.
    fn rest_response(state: &str, result: Value) -> Value {
        json!({
            "result": { "state": state, "result": result, "gatewayMetadata": { "keySource": "Unified" } },
            "success": true,
            "errors": [],
            "messages": [],
        })
    }

    // Cloudflare-hosted output observed from a live /ai/run call, question ids
    // renamed to match `context`. The envelope carries the output directly,
    // without a run record.
    fn clef_output() -> Value {
        json!({
            "model": "clef",
            "answers": {
                "is_urgent": { "type": "noul", "noul": 0.9912 },
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": { "billing": 0.1632, "technical": 0.8368 },
                    "confidence": 0.4538,
                },
            },
            "usage": { "input_tokens": 222, "output_tokens": 0 },
        })
    }

    fn setup(id: &str) -> (Models, ClassifierModel) {
        let models = create_models(Default::default());
        models.set_provider(cloudflare_workers_ai_provider());
        let model = models
            .get_model_of_type(ModelType::Classifier, "cloudflare-workers-ai", id)
            .and_then(|model| model.as_classifier().cloned())
            .unwrap_or_else(|| panic!("missing Cloudflare {id} model"));
        (models, model)
    }

    /// The catalog model with the mock server as its host.
    fn served(model: &ClassifierModel, server: &MockServer) -> ClassifierModel {
        let root = server.url.trim_end_matches("/v1");
        ClassifierModel {
            base_url: model.base_url.replace("https://api.cloudflare.com", root),
            ..model.clone()
        }
    }

    fn auth() -> ClassifierOptions {
        ClassifierOptions {
            api_key: Some("cf-key".to_string()),
            env: Some(ProviderEnv::from([(
                "CLOUDFLARE_ACCOUNT_ID".to_string(),
                "account-id".to_string(),
            )])),
            ..Default::default()
        }
    }

    #[test]
    fn exposes_jev_only_through_classifier_catalog_accessors() {
        let (models, jev) = setup("typesafe/jev");
        assert_eq!(
            Some(&jev),
            get_builtin_classifier_model("cloudflare-workers-ai", "typesafe/jev").as_ref()
        );
        assert_eq!(jev.api, "cloudflare-workers-ai-system-one");
        assert_eq!(jev.base_url, CLOUDFLARE_WORKERS_AI_REST_BASE_URL);
        assert!(
            models
                .get_model("cloudflare-workers-ai", "typesafe/jev")
                .is_none()
        );
    }

    #[tokio::test]
    async fn runs_jev_through_the_account_scoped_ai_run_endpoint() {
        let (models, jev) = setup("typesafe/jev");
        let server = MockServer::start(vec![MockResponse::json(rest_response(
            "Completed",
            jev_output(),
        ))])
        .await;

        let result = models
            .classify(&served(&jev, &server), &context(), auth())
            .await;

        let request = server.last();
        assert_eq!(request.path, "/client/v4/accounts/account-id/ai/run");
        assert_eq!(request.body["model"], "typesafe/jev");
        assert_eq!(request.body["input"]["state"], json!(context().state));
        assert_eq!(
            request.body["input"]["questions"]["is_urgent"]["type"],
            "noul"
        );
        assert_eq!(
            request.body["input"]["questions"]["department"]["type"],
            "choice"
        );
        assert_eq!(request.header("authorization"), Some("Bearer cf-key"));
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers["is_urgent"],
            ClassifierAnswer::Bool { probability: 0.95 }
        );
        assert!(matches!(
            &result.answers["department"],
            ClassifierAnswer::Choice { choice, confidence, .. }
                if choice == "billing" && *confidence == 0.8
        ));
        let usage = result.usage.unwrap();
        assert_eq!(
            (usage.input, usage.output, usage.total_tokens),
            (426, 73, 499)
        );
    }

    #[tokio::test]
    async fn runs_clef_models_through_ai_run_and_parses_their_direct_output() {
        for (id, input_price) in [
            ("@cf/cloudflare/clef", 0.24),
            ("@cf/cloudflare/clef-flash", 0.09),
        ] {
            let (models, clef) = setup(id);
            let server = MockServer::start(vec![MockResponse::json(json!({
                "result": clef_output(),
                "success": true,
                "errors": [],
                "messages": [],
            }))])
            .await;

            let result = models
                .classify(&served(&clef, &server), &context(), auth())
                .await;

            let request = server.last();
            assert_eq!(request.path, "/client/v4/accounts/account-id/ai/run");
            assert_eq!(request.body["model"], id);
            assert_eq!(request.body["input"]["state"], json!(context().state));
            assert_eq!(
                request.body["input"]["questions"]["is_urgent"]["type"],
                "noul"
            );
            assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
            assert_eq!(
                result.answers["is_urgent"],
                ClassifierAnswer::Bool {
                    probability: 0.9912
                }
            );
            assert!(matches!(
                &result.answers["department"],
                ClassifierAnswer::Choice { choice, confidence, .. }
                    if choice == "technical" && *confidence == 0.4538
            ));
            let usage = result.usage.unwrap();
            assert_eq!(
                (usage.input, usage.output, usage.total_tokens),
                (222, 0, 222)
            );
            assert!((usage.cost.input - 222.0 * input_price / 1_000_000.0).abs() < 1e-12);
        }
    }

    #[tokio::test]
    async fn reports_runs_that_did_not_complete() {
        let (models, jev) = setup("typesafe/jev");
        let server = MockServer::start(vec![MockResponse::json(rest_response(
            "Queued",
            Value::Null,
        ))])
        .await;
        let result = models
            .classify(&served(&jev, &server), &context(), auth())
            .await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert!(
            result
                .error_message
                .unwrap()
                .contains("run did not complete (state: Queued)")
        );
    }

    #[tokio::test]
    async fn reports_cloudflare_envelope_errors() {
        let (models, jev) = setup("typesafe/jev");
        let server = MockServer::start(vec![MockResponse::json(json!({
            "success": false,
            "errors": [{ "code": 5007, "message": "No such model" }],
            "result": null,
        }))])
        .await;
        let result = models
            .classify(&served(&jev, &server), &context(), auth())
            .await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert!(
            result
                .error_message
                .unwrap()
                .contains("Cloudflare Workers AI error: No such model")
        );
    }
}
