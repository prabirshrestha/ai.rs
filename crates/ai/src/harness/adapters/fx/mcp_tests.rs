//! Port of `harness-fx/src/fx-mcp.test.ts`, plus the libfx adapter's naming
//! and result rules and a Streamable HTTP round trip.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

use super::*;
use crate::harness::adapters::fx::mcp_client::default_mcp_connector;

#[derive(Default)]
struct MockClient {
    instructions: Option<String>,
    catalog: Mutex<Vec<HarnessResult<Value>>>,
    list_calls: Mutex<Vec<Option<String>>>,
    calls: Mutex<Vec<(String, Value)>>,
    call_result: Mutex<Option<Value>>,
    closes: Mutex<usize>,
}

impl MockClient {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            instructions: Some("Server instructions".into()),
            ..Default::default()
        })
    }

    fn with_pages(pages: Vec<HarnessResult<Value>>) -> Arc<Self> {
        let client = Self::new();
        *client.catalog.lock().unwrap() = pages;
        client
    }

    fn closes(&self) -> usize {
        *self.closes.lock().unwrap()
    }
}

#[async_trait]
impl FxMcpClient for MockClient {
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    async fn list_tools(&self, cursor: Option<String>) -> HarnessResult<Value> {
        self.list_calls.lock().unwrap().push(cursor);
        let mut catalog = self.catalog.lock().unwrap();
        if catalog.is_empty() {
            Ok(json!({ "tools": [] }))
        } else {
            catalog.remove(0)
        }
    }

    async fn call_tool(&self, name: &str, arguments: Value, _signal: CancellationToken) -> HarnessResult<Value> {
        self.calls.lock().unwrap().push((name.into(), arguments));
        Ok(self.call_result.lock().unwrap().clone().unwrap_or(json!({ "content": [] })))
    }

    async fn close(&self) {
        *self.closes.lock().unwrap() += 1;
    }
}

type Connections = Arc<Mutex<Vec<FxMcpTransportConfig>>>;

/// A connector that hands out `clients` in order; `None` fails to connect.
fn connector(clients: Vec<Option<Arc<MockClient>>>) -> (FxMcpConnector, Connections) {
    let connections: Connections = Arc::default();
    let queue = Arc::new(Mutex::new(clients));
    let seen = connections.clone();
    let connector: FxMcpConnector = Arc::new(move |config, _signal| {
        seen.lock().unwrap().push(config);
        let next = queue.lock().unwrap().remove(0);
        Box::pin(async move {
            match next {
                Some(client) => Ok(client as Arc<dyn FxMcpClient>),
                None => Err(HarnessError::new("initialization failed").into()),
            }
        })
    });
    (connector, connections)
}

fn servers(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

#[tokio::test]
async fn configures_each_transport_type() {
    for (kind, expected) in [
        (None, FxMcpTransportKind::Http),
        (Some("http"), FxMcpTransportKind::Http),
        (Some("streamableHttp"), FxMcpTransportKind::Http),
        (Some("sse"), FxMcpTransportKind::Sse),
    ] {
        let client = MockClient::new();
        let (connector, connections) = connector(vec![Some(client.clone())]);
        let mut config = json!({
            "url": "https://mcp.context7.com/mcp",
            "headers": { "x-api-key": "test-key" },
            "disabled": false,
            "extraField": "accepted",
        });
        if let Some(kind) = kind {
            config["type"] = json!(kind);
        }
        let runtime = create_fx_mcp_runtime(Some(&servers(json!({ "context7": config }))), None, &connector)
            .await
            .unwrap();
        assert_eq!(
            connections.lock().unwrap().clone(),
            vec![FxMcpTransportConfig {
                kind: expected,
                url: "https://mcp.context7.com/mcp".into(),
                headers: Some(IndexMap::from([("x-api-key".to_string(), "test-key".to_string())])),
            }]
        );
        assert_eq!(runtime.instructions, "Server instructions");
        assert!(runtime.tools.is_empty());
        assert_eq!(client.list_calls.lock().unwrap().clone(), vec![None]);
        runtime.close().await;
        assert_eq!(client.closes(), 1);
    }
}

#[tokio::test]
async fn skips_disabled_entries_without_validating_other_fields() {
    let (connector, connections) = connector(vec![]);
    let runtime = create_fx_mcp_runtime(
        Some(&servers(json!({ "disabled": { "disabled": true, "headers": [], "type": "stdio" } }))),
        None,
        &connector,
    )
    .await
    .unwrap();
    assert!(connections.lock().unwrap().is_empty());
    assert!(runtime.tools.is_empty());
    runtime.close().await;
}

#[tokio::test]
async fn rejects_unsupported_server_configurations() {
    for config in [
        json!(null),
        json!("invalid"),
        json!([]),
        json!({}),
        json!({ "command": "server" }),
        json!({ "url": 123 }),
        json!({ "url": "https://mcp.context7.com/mcp", "type": "stdio" }),
        json!({ "url": "https://mcp.context7.com/mcp", "type": "websocket" }),
    ] {
        let (connector, connections) = connector(vec![]);
        let error = create_fx_mcp_runtime(Some(&servers(json!({ "context7": config }))), None, &connector)
            .await
            .err()
            .unwrap();
        assert!(HarnessError::is_capability_unsupported(error.as_ref()), "{config}");
        assert_eq!(
            error.to_string(),
            "fx MCP server context7 must use an HTTP or SSE URL. Command/stdio servers are not supported."
        );
        assert!(connections.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn identifies_invalid_fields() {
    for (fields, field) in [
        (json!({ "headers": [] }), "headers"),
        (json!({ "headers": { "authorization": 123 } }), "headers.authorization"),
        (json!({ "disabled": "false" }), "disabled"),
    ] {
        let (connector, connections) = connector(vec![]);
        let mut config = json!({ "url": "https://mcp.context7.com/mcp" });
        config.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
        let error = create_fx_mcp_runtime(Some(&servers(json!({ "context7": config }))), None, &connector)
            .await
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .starts_with(&format!("Invalid fx MCP server context7 field {field}:")),
            "{error}"
        );
        assert!(connections.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn closes_initialized_clients_after_a_later_failure() {
    for failure in ["configuration", "initialization", "catalog"] {
        let first = MockClient::new();
        let second = MockClient::with_pages(vec![Err(HarnessError::new("catalog failed").into())]);
        let clients = match failure {
            "initialization" => vec![Some(first.clone()), None],
            "catalog" => vec![Some(first.clone()), Some(second.clone())],
            _ => vec![Some(first.clone())],
        };
        let (connector, _) = connector(clients);
        let second_config = if failure == "configuration" {
            json!({ "headers": [] })
        } else {
            json!({ "url": "https://mcp.context7.com/mcp" })
        };
        let result = create_fx_mcp_runtime(
            Some(&servers(json!({
                "first": { "url": "https://mcp.context7.com/mcp" },
                "second": second_config,
            }))),
            None,
            &connector,
        )
        .await;
        assert!(result.is_err(), "{failure}");
        assert_eq!(first.closes(), 1, "{failure}");
        assert_eq!(second.closes(), usize::from(failure == "catalog"), "{failure}");
    }
}

#[tokio::test]
async fn names_paginates_and_runs_tools() {
    let long = "x".repeat(70);
    let client = MockClient::with_pages(vec![
        Ok(json!({
            "tools": [
                { "name": "resolve.library", "description": "Resolve", "inputSchema": { "type": "object" } },
                { "name": long },
            ],
            "nextCursor": "page-2",
        })),
        Ok(json!({ "tools": [{ "name": format!("{long}y") }] })),
    ]);
    let (connector, _) = connector(vec![Some(client.clone())]);
    let runtime = create_fx_mcp_runtime(
        Some(&servers(json!({ "context 7": { "url": "https://mcp.context7.com/mcp" } }))),
        None,
        &connector,
    )
    .await
    .unwrap();
    assert_eq!(
        client.list_calls.lock().unwrap().clone(),
        vec![None, Some("page-2".into())]
    );
    let names: Vec<&str> = runtime
        .tools
        .iter()
        .map(|tool| tool.descriptor.name.as_str())
        .collect();
    let clipped = format!("mcp_context_7_{long}")[..64].to_string();
    assert_eq!(
        names,
        vec![
            "mcp_context_7_resolve_library".to_string(),
            clipped.clone(),
            format!("{}_2", &clipped[..62]),
        ]
    );
    assert_eq!(runtime.tools[1].descriptor.description, "MCP tool");
    assert_eq!(
        runtime.tools[1].descriptor.input_schema,
        json!({ "type": "object", "properties": {} })
    );

    *client.call_result.lock().unwrap() = Some(json!({
        "content": [
            { "type": "text", "text": "found" },
            { "type": "image", "data": "aW1n", "mimeType": "image/png" },
            { "type": "resource_link", "uri": "https://docs", "name": "Docs" },
        ],
        "structuredContent": { "id": 1 },
    }));
    let result = runtime.tools[0]
        .execute(json!({ "query": "react" }), CancellationToken::new())
        .await;
    assert_eq!(
        client.calls.lock().unwrap().clone(),
        vec![("resolve.library".to_string(), json!({ "query": "react" }))]
    );
    assert_eq!(
        result,
        FxToolResult {
            text: "found\nDocs: https://docs\n{\"id\":1}".into(),
            images: vec![FxImage {
                data: "aW1n".into(),
                mime_type: "image/png".into(),
            }],
            is_error: false,
        }
    );

    *client.call_result.lock().unwrap() = Some(json!({ "content": [], "isError": true }));
    let result = runtime.tools[0].execute(json!({}), CancellationToken::new()).await;
    assert_eq!(result, FxToolResult::error("MCP tool resolve.library failed"));
}

#[tokio::test]
async fn rejects_repeated_cursors_and_duplicate_tools() {
    let client = MockClient::with_pages(vec![
        Ok(json!({ "tools": [], "nextCursor": "again" })),
        Ok(json!({ "tools": [], "nextCursor": "again" })),
    ]);
    let error = create_mcp_adapter(client, "").await.err().unwrap();
    assert_eq!(error.to_string(), "MCP listTools() returned an invalid pagination cursor");
    let client = MockClient::with_pages(vec![Ok(json!({ "tools": [{ "name": "a" }, { "name": "a" }] }))]);
    let error = create_mcp_adapter(client, "").await.err().unwrap();
    assert_eq!(error.to_string(), "MCP tool a is duplicated");
}

/// A one-connection-per-request HTTP server answering MCP over Streamable
/// HTTP; it records each request's method and session header.
async fn serve_streamable_http() -> (String, Arc<Mutex<Vec<(String, Option<String>)>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let seen: Arc<Mutex<Vec<(String, Option<String>)>>> = Arc::default();
    let log = seen.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let log = log.clone();
            tokio::spawn(async move {
                let mut buffer = Vec::new();
                let (head, body) = loop {
                    let mut chunk = [0u8; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    if read == 0 {
                        return;
                    }
                    buffer.extend_from_slice(&chunk[..read]);
                    let text = String::from_utf8_lossy(&buffer).into_owned();
                    let Some(end) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let head = text[..end].to_string();
                    let length = head
                        .lines()
                        .find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(|value| value.trim().parse::<usize>().unwrap()))
                        .unwrap_or(0);
                    if buffer.len() >= end + 4 + length {
                        break (head, String::from_utf8_lossy(&buffer[end + 4..end + 4 + length]).into_owned());
                    }
                };
                let session = head.lines().find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("mcp-session-id:")
                        .map(|value| value.trim().to_string())
                });
                let verb = head.split(' ').next().unwrap().to_string();
                let message: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                let method = message
                    .get("method")
                    .and_then(Value::as_str)
                    .map_or(verb, str::to_string);
                log.lock().unwrap().push((method.clone(), session));
                let response = match method.as_str() {
                    "initialize" => (
                        "application/json",
                        json!({ "jsonrpc": "2.0", "id": message["id"], "result": {
                            "protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": { "name": "test", "version": "1" },
                            "instructions": "Use the docs tool.",
                        }})
                        .to_string(),
                    ),
                    "tools/list" => (
                        "text/event-stream",
                        format!(
                            "event: message\ndata: {}\n\n",
                            json!({ "jsonrpc": "2.0", "id": message["id"], "result": { "tools": [{ "name": "docs" }] } })
                        ),
                    ),
                    "tools/call" => (
                        "text/event-stream",
                        format!(
                            "data: {}\n\ndata: {}\n\n",
                            json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": {} }),
                            json!({ "jsonrpc": "2.0", "id": message["id"], "result": { "content": [{ "type": "text", "text": format!("docs for {}", message["params"]["arguments"]["q"]) }] } })
                        ),
                    ),
                    _ => ("", String::new()),
                };
                let status = if response.0.is_empty() { "202 Accepted" } else { "200 OK" };
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: {}\r\nmcp-session-id: session-1\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    if response.0.is_empty() { "text/plain" } else { response.0 },
                    response.1.len(),
                    response.1
                );
                let _ = socket.write_all(reply.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (url, seen)
}

#[tokio::test]
async fn talks_to_a_streamable_http_server() {
    let (url, seen) = serve_streamable_http().await;
    let connector = default_mcp_connector();
    let runtime = create_fx_mcp_runtime(Some(&servers(json!({ "docs": { "url": url } }))), None, &connector)
        .await
        .unwrap();
    assert_eq!(runtime.instructions, "Use the docs tool.");
    assert_eq!(runtime.tools.len(), 1);
    assert_eq!(runtime.tools[0].descriptor.name, "mcp_docs_docs");
    let result = runtime.tools[0]
        .execute(json!({ "q": "fx" }), CancellationToken::new())
        .await;
    assert_eq!(result, FxToolResult::text("docs for \"fx\""));
    runtime.close().await;
    assert_eq!(
        seen.lock().unwrap().clone(),
        vec![
            ("initialize".to_string(), None),
            ("notifications/initialized".to_string(), Some("session-1".to_string())),
            ("tools/list".to_string(), Some("session-1".to_string())),
            ("tools/call".to_string(), Some("session-1".to_string())),
            ("DELETE".to_string(), Some("session-1".to_string())),
        ]
    );
}
