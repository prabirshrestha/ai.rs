//! Port of `fx-transport.test.ts`.
//!
//! [rust] Not ported: the replay cases (recorded responses, divergent
//! replays and new attempts), since the transport keeps no records.

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::harness::adapters::fx::libfx::fetch::{FxFetch, FxFetchError, FxHttpRequest, FxHttpResponse};
use crate::harness::session::ResponseFormat;

const FIXTURE: &str = include_str!("fixtures/gateway-0.sse.txt");
const OPENAI: &str = include_str!("fixtures/responses-openai.chunks.txt");
const XAI: &str = include_str!("fixtures/responses-xai.chunks.txt");
const GATEWAY_URL: &str = "https://ai-gateway.vercel.sh/v4/ai/language-model";

fn responses_fixtures(provider: FxProvider) -> Vec<String> {
    let lines = if provider == FxProvider::Codex { OPENAI } else { XAI };
    let mut responses = Vec::new();
    let mut body = String::new();
    for line in lines.trim().split('\n') {
        body.push_str(&format!("data: {line}\n\n"));
        if parse_fx_event(&format!("data: {line}")).and_then(|part| part["type"].as_str().map(str::to_string))
            == Some("response.completed".into())
        {
            responses.push(std::mem::take(&mut body));
        }
    }
    responses
}

type Log = Arc<Mutex<Vec<FxHttpRequest>>>;

/// A fetch that records requests and answers each with `respond`.
fn fetch_with(log: Log, respond: impl Fn(usize) -> Result<FxHttpResponse, FxFetchError> + Send + Sync + 'static) -> FxFetch {
    Arc::new(move |request, _| {
        let index = {
            let mut log = log.lock().unwrap();
            log.push(request);
            log.len() - 1
        };
        let response = respond(index);
        Box::pin(async move { response })
    })
}

fn transport(
    provider: FxProvider,
    base_url: Option<&str>,
    headers: &[(&str, &str)],
    response_format: Option<ResponseFormat>,
    fetch: FxFetch,
) -> (Arc<FxTransport>, Arc<Mutex<Vec<FxTransportSignal>>>) {
    let signals = Arc::new(Mutex::new(Vec::new()));
    let sink = signals.clone();
    let transport = FxTransport::new(FxTransportOptions {
        base_url: base_url.map(str::to_string),
        provider,
        headers: headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect::<IndexMap<_, _>>(),
        response_format,
        signals: Arc::new(move |signal| sink.lock().unwrap().push(signal)),
        fetch,
    });
    (transport, signals)
}

fn post(url: &str, body: &str, headers: &[(&str, &str)]) -> FxHttpRequest {
    FxHttpRequest {
        url: url.into(),
        method: "POST".into(),
        headers: headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect(),
        body: Some(Bytes::from(body.to_string())),
    }
}

fn gateway_request() -> FxHttpRequest {
    post(
        GATEWAY_URL,
        r#"{"prompt":[]}"#,
        &[("ai-language-model-id", "anthropic/claude-haiku-4-5")],
    )
}

fn tool_calls(signals: &[FxTransportSignal]) -> Vec<FxToolCall> {
    signals
        .iter()
        .filter_map(|signal| match signal {
            FxTransportSignal::ToolCalls { calls, .. } => Some(calls.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

#[tokio::test]
async fn extracts_complete_subscription_tool_batches_before_release() {
    for provider in [FxProvider::Codex, FxProvider::Grok] {
        let fixtures = responses_fixtures(provider);
        let served = fixtures.clone();
        let log = Log::default();
        let (transport, signals) = transport(
            provider,
            None,
            &[],
            None,
            fetch_with(log.clone(), move |index| Ok(FxHttpResponse::new(200, served[index].clone()))),
        );
        let fetch = transport.fetch();
        let endpoint = if provider == FxProvider::Codex {
            "https://chatgpt.com/backend-api/codex/responses"
        } else {
            "https://cli-chat-proxy.grok.com/v1/responses"
        };
        for (index, fixture) in fixtures.iter().enumerate() {
            let body = json!({ "model": "supplied-model", "input": [index] }).to_string();
            let response = fetch(post(endpoint, &body, &[]), CancellationToken::new()).await.unwrap();
            assert_eq!(&response.text().await.unwrap(), fixture);
        }
        let calls: Vec<(String, Value)> = tool_calls(&signals.lock().unwrap())
            .into_iter()
            .map(|call| {
                (
                    call.tool_name,
                    serde_json::from_str(call.input.as_str().unwrap()).unwrap(),
                )
            })
            .collect();
        if provider == FxProvider::Codex {
            assert_eq!(calls.len(), 3);
            assert_eq!(
                calls[0],
                ("calculator".into(), json!({ "a": 12, "b": 7, "op": "add" }))
            );
        } else {
            assert!(calls.is_empty());
        }
        let requests: Vec<Option<String>> = signals
            .lock()
            .unwrap()
            .iter()
            .filter_map(|signal| match signal {
                FxTransportSignal::Request { model_id, .. } => Some(model_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(requests.len(), fixtures.len());
        assert_eq!(requests[0].as_deref(), Some("supplied-model"));
        transport.stop().await;
        assert_eq!(log.lock().unwrap().len(), fixtures.len());
    }
}

#[tokio::test]
async fn reuses_the_request_index_after_a_rejected_subscription_response() {
    let fixture = responses_fixtures(FxProvider::Codex).remove(0);
    let (transport, signals) = transport(
        FxProvider::Codex,
        None,
        &[],
        None,
        fetch_with(Log::default(), move |index| {
            Ok(if index == 0 {
                FxHttpResponse::new(401, "rejected secret")
            } else {
                FxHttpResponse::new(200, fixture.clone())
            })
        }),
    );
    let fetch = transport.fetch();
    let endpoint = "https://chatgpt.com/backend-api/codex/responses";
    let rejected = fetch(post(endpoint, r#"{"model":"unchanged"}"#, &[]), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(rejected.status, 401);
    assert_eq!(rejected.text().await.unwrap(), "rejected secret");
    assert_eq!(transport.request_index(), None);
    fetch(post(endpoint, r#"{"model":"unchanged"}"#, &[]), CancellationToken::new())
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(transport.request_index(), Some(0));
    let signals = signals.lock().unwrap();
    assert_eq!(
        signals[1],
        FxTransportSignal::Rejected {
            index: 0,
            model_id: Some("unchanged".into()),
            status: 401,
        }
    );
    assert!(matches!(signals[2], FxTransportSignal::Request { index: 0, .. }));
    transport.stop().await;
}

#[tokio::test]
async fn keeps_tools_unpublished_when_a_response_is_truncated() {
    let codex = responses_fixtures(FxProvider::Codex).remove(0);
    let codex = codex[..codex.find(r#"data: {"type":"response.completed""#).unwrap()].to_string();
    let gateway = FIXTURE[..FIXTURE.find(r#"data: {"type":"finish""#).unwrap()].to_string();
    for (provider, url, body) in [
        (FxProvider::Codex, "https://chatgpt.com/backend-api/codex/responses", codex),
        (FxProvider::Gateway, GATEWAY_URL, gateway),
    ] {
        let (transport, signals) = transport(
            provider,
            Some("https://ai-gateway.vercel.sh"),
            &[],
            None,
            fetch_with(Log::default(), move |_| Ok(FxHttpResponse::new(200, body.clone()))),
        );
        let request = if provider == FxProvider::Gateway {
            gateway_request()
        } else {
            post(url, "{}", &[])
        };
        let error = transport.fetch()(request, CancellationToken::new())
            .await
            .unwrap()
            .text()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("before a terminal event"), "{error}");
        assert!(tool_calls(&signals.lock().unwrap()).is_empty());
        transport.stop().await;
    }
}

#[tokio::test]
async fn scopes_gateway_headers_away_from_subscription_requests() {
    let fixture = responses_fixtures(FxProvider::Grok).remove(0);
    let log = Log::default();
    let (transport, _) = transport(
        FxProvider::Grok,
        None,
        &[("authorization", "gateway-key"), ("x-client-app", "gateway")],
        Some(ResponseFormat::Json {
            schema: Some(json!({ "type": "object" })),
            name: None,
            description: None,
        }),
        fetch_with(log.clone(), move |_| Ok(FxHttpResponse::new(200, fixture.clone()))),
    );
    let fetch = transport.fetch();
    for url in [
        "https://auth.x.ai/oauth2/token",
        "https://api.x.ai/v1/models",
        "https://cli-chat-proxy.grok.com/v1/responses",
    ] {
        fetch(
            post(url, r#"{"model":"unchanged"}"#, &[("authorization", "subscription")]),
            CancellationToken::new(),
        )
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    }
    let seen: Vec<Value> = log
        .lock()
        .unwrap()
        .iter()
        .map(|request| {
            json!({
                "url": request.url,
                "headers": request.headers.iter().map(|(name, value)| (name.clone(), json!(value))).collect::<serde_json::Map<_, _>>(),
                "body": serde_json::from_str::<Value>(&request.body_text()).unwrap(),
            })
        })
        .collect();
    assert_eq!(
        seen,
        vec![
            json!({ "url": "https://auth.x.ai/oauth2/token", "headers": { "authorization": "subscription" }, "body": { "model": "unchanged" } }),
            json!({ "url": "https://api.x.ai/v1/models", "headers": { "authorization": "subscription" }, "body": { "model": "unchanged" } }),
            json!({
                "url": "https://cli-chat-proxy.grok.com/v1/responses",
                "headers": { "authorization": "subscription" },
                "body": {
                    "model": "unchanged",
                    "text": { "format": { "type": "json_schema", "name": "response", "schema": { "type": "object" }, "strict": true } },
                },
            }),
        ]
    );
    transport.stop().await;
}

#[tokio::test]
async fn publishes_the_tool_batch_before_releasing_tool_frames() {
    let (transport, signals) = transport(
        FxProvider::Gateway,
        Some("https://ai-gateway.vercel.sh"),
        &[],
        None,
        fetch_with(Log::default(), |_| Ok(FxHttpResponse::new(200, FIXTURE))),
    );
    let response = transport.fetch()(gateway_request(), CancellationToken::new()).await.unwrap();
    let mut body = response.body;
    let mut text = String::new();
    use futures::StreamExt as _;
    while let Some(chunk) = body.next().await {
        let chunk = String::from_utf8(chunk.unwrap().to_vec()).unwrap();
        if chunk.contains(r#""type":"tool-call""#) {
            // The batch is out before the core can see the call.
            let calls = tool_calls(&signals.lock().unwrap());
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].tool_call_id, "toolu_01KswTwrpijqi88ffraLQVsY");
            assert_eq!(calls[0].tool_name, "bash");
            assert_eq!(
                serde_json::from_str::<Value>(calls[0].input.as_str().unwrap()).unwrap(),
                json!({ "command": "printf LIBFX_FIXTURE" })
            );
        }
        text.push_str(&chunk);
    }
    assert_eq!(text.trim(), FIXTURE.trim());
    transport.stop().await;
}

#[tokio::test]
async fn keeps_request_indexes_stable_when_libfx_retries_a_failed_fetch() {
    let (transport, _) = transport(
        FxProvider::Gateway,
        Some("https://ai-gateway.vercel.sh"),
        &[],
        None,
        fetch_with(Log::default(), |index| {
            if index == 0 {
                Err(FxFetchError::Failed("fetch failed".into()))
            } else {
                Ok(FxHttpResponse::new(200, FIXTURE))
            }
        }),
    );
    let fetch = transport.fetch();
    let error = fetch(gateway_request(), CancellationToken::new()).await.unwrap_err();
    assert_eq!(error.to_string(), "fetch failed");
    fetch(gateway_request(), CancellationToken::new())
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(transport.request_index(), Some(0));
    transport.stop().await;
}

#[tokio::test]
async fn forwards_structured_output_and_attribution_to_the_configured_gateway() {
    let log = Log::default();
    let (transport, _) = transport(
        FxProvider::Gateway,
        Some("https://gateway.example"),
        &[("x-client-app", "ai-sdk-harness-fx/test")],
        Some(ResponseFormat::Json {
            schema: Some(json!({ "type": "object", "properties": { "answer": { "type": "string" } } })),
            name: None,
            description: None,
        }),
        fetch_with(log.clone(), |_| Ok(FxHttpResponse::new(200, FIXTURE))),
    );
    transport.fetch()(gateway_request(), CancellationToken::new())
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let log = log.lock().unwrap();
    assert_eq!(log[0].url, "https://gateway.example/v4/ai/language-model");
    assert_eq!(log[0].header("x-client-app"), Some("ai-sdk-harness-fx/test"));
    assert_eq!(
        serde_json::from_str::<Value>(&log[0].body_text()).unwrap(),
        json!({
            "prompt": [],
            "responseFormat": { "type": "json", "schema": { "type": "object", "properties": { "answer": { "type": "string" } } } },
        })
    );
    transport.stop().await;
}

#[test]
fn parses_sse_frames() {
    assert_eq!(parse_fx_event("event: x\r\ndata: {\"type\":\"finish\"}\r\n"), Some(json!({ "type": "finish" })));
    assert_eq!(parse_fx_event("data: [DONE]"), None);
}
