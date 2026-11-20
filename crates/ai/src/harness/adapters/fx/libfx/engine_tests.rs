//! The engine against fx's real WebAssembly core (`AI_FX_CORE_WASM`, or the
//! cached libfx download), with a fake network.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use serde_json::json;
use tokio::sync::mpsc;

use super::engine::*;
use super::fetch::{FxFetch, FxHttpRequest, FxHttpResponse};
use super::host;
use super::wasm::{FxCoreSource, load_core};

const MODELS: &str = include_str!("../fixtures/gateway-models.json");
const GATEWAY_0: &str = include_str!("../fixtures/gateway-0.sse.txt");
const GATEWAY_1: &str = include_str!("../fixtures/gateway-1.sse.txt");

/// fx's core, compiled once per process.
pub(crate) async fn core_module() -> Option<wasmtime::Module> {
    let bytes = match load_core(&FxCoreSource::Default).await {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("skipping: fx core unavailable: {error}");
            return None;
        }
    };
    let started = Instant::now();
    let module = host::compile(&bytes).unwrap();
    eprintln!("fx core ready in {:?}", started.elapsed());
    Some(module)
}

/// A network that answers fx's requests from fixtures.
pub(crate) fn fixture_fetch(
    responses: Vec<&'static str>,
    requests: Arc<StdMutex<Vec<FxHttpRequest>>>,
) -> FxFetch {
    let responses = Arc::new(StdMutex::new(responses.into_iter()));
    Arc::new(move |request: FxHttpRequest, _| {
        let responses = responses.clone();
        let requests = requests.clone();
        Box::pin(async move {
            if request.url == "https://x.ai/cli/stable" {
                return Ok(FxHttpResponse::new(200, "1.0.13\n"));
            }
            if request.method == "GET" {
                return Ok(FxHttpResponse::new(200, MODELS));
            }
            requests.lock().unwrap().push(request);
            let body = responses
                .lock()
                .unwrap()
                .next()
                .expect("an unexpected model request");
            Ok(FxHttpResponse::new(200, body).with_header("content-type", "text/event-stream"))
        })
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn runs_a_turn_with_a_host_tool_on_the_real_core() {
    let Some(module) = core_module().await else {
        return;
    };
    let requests = Arc::new(StdMutex::new(Vec::new()));
    let (events, mut receiver) = mpsc::unbounded_channel();
    let engine = WasmFxEngine::create(
        &module,
        FxEngineOptions {
            provider: FxEngineProvider::Gateway {
                api_key: "test-key".into(),
            },
            model: Some("anthropic/claude-haiku-4.5".into()),
            effort: None,
            fast: None,
            ultrafast: None,
            instructions: "Be brief.".into(),
            tools: vec![FxToolDescriptor {
                name: "bash".into(),
                description: "Run a command.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"],
                }),
            }],
            checkpoint: None,
            fetch: fixture_fetch(vec![GATEWAY_0, GATEWAY_1], requests.clone()),
            hooks: true,
            events,
        },
    )
    .await
    .unwrap();
    let first = engine.checkpoint().await.unwrap();
    assert!(!first.is_empty());
    let turn = engine
        .prompt(vec![FxPromptBlock::Text(
            "Run printf LIBFX_FIXTURE.".into(),
        )])
        .unwrap();
    let collector = tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Some(event) = receiver.recv().await {
            match event {
                FxEngineEvent::ToolCall {
                    name, input, reply, ..
                } => {
                    seen.push(format!("call {name} {input}"));
                    let _ = reply.send(FxToolResult::text("LIBFX_FIXTURE"));
                }
                FxEngineEvent::ModelEnd(end) => {
                    seen.push(format!("model.end {:?}", end.finish_reason))
                }
                FxEngineEvent::Turn(event) => seen.push(format!("{event:?}")),
            }
        }
        seen
    });
    let FxTurn { result, control } = turn;
    let result = tokio::time::timeout(std::time::Duration::from_secs(120), result)
        .await
        .expect("the turn ends")
        .unwrap();
    let checkpoint = engine.checkpoint().await.unwrap();
    assert!(!checkpoint.is_empty());
    engine.close().await;
    drop((engine, control));
    let seen = tokio::time::timeout(std::time::Duration::from_secs(5), collector)
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();
    eprintln!("{result:?}\n{seen:#?}");
    assert_eq!(result.stop_reason, "end_turn");
    assert!(seen.iter().any(|line| line.starts_with("call bash")));
    assert!(seen.iter().any(|line| line.contains("TextDelta")));
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].body_text().contains("LIBFX_FIXTURE"));
}
