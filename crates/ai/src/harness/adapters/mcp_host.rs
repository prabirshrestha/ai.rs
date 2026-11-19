//! [rust] An MCP server for the caller's tools, answered in-process. Agents
//! reach it over their own transport: Claude Code's SDK MCP control
//! messages, or the stdio relay the ACP and OpenCode adapters start in the
//! sandbox.

#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::Tool;

pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Runs `tools/call`: the tool name, its arguments and the request's `_meta`,
/// to an MCP `CallToolResult`.
pub type McpCallFn = Arc<
    dyn Fn(String, Value, Option<Value>) -> Pin<Box<dyn Future<Output = Value> + Send>>
        + Send
        + Sync,
>;

/// Per-tool annotations and `_meta`.
#[derive(Debug, Clone, Default)]
pub struct McpToolExtras {
    pub annotations: Option<Value>,
    pub meta: Option<Value>,
}

/// The server.
#[derive(Clone)]
pub struct McpHostServer {
    pub name: String,
    pub tools: Vec<(Tool, McpToolExtras)>,
    pub call: McpCallFn,
}

fn ok(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: &Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// The JSON schema MCP requires for a tool input: an object schema.
fn input_schema(parameters: &Value) -> Value {
    match parameters {
        Value::Object(map) if map.get("type").and_then(Value::as_str) == Some("object") => {
            parameters.clone()
        }
        Value::Object(map) if !map.contains_key("type") => {
            let mut schema = map.clone();
            schema.insert("type".into(), json!("object"));
            Value::Object(schema)
        }
        _ => json!({ "type": "object", "properties": {} }),
    }
}

impl McpHostServer {
    pub fn tools_list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|(tool, extras)| {
                let mut entry = json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": input_schema(&tool.parameters),
                });
                if let Some(annotations) = &extras.annotations {
                    entry["annotations"] = annotations.clone();
                }
                if let Some(meta) = &extras.meta {
                    entry["_meta"] = meta.clone();
                }
                entry
            })
            .collect();
        json!({ "tools": tools })
    }

    /// Answer one JSON-RPC message. Notifications get no answer.
    pub async fn handle(&self, message: &Value) -> Option<Value> {
        let id = message.get("id")?.clone();
        let method = message.get("method").and_then(Value::as_str).unwrap_or("");
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        Some(match method {
            "initialize" => ok(
                &id,
                json!({
                    "protocolVersion": params
                        .get("protocolVersion")
                        .and_then(Value::as_str)
                        .unwrap_or(MCP_PROTOCOL_VERSION),
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": self.name, "version": "1.0.0" },
                }),
            ),
            "ping" => ok(&id, json!({})),
            "tools/list" => ok(&id, self.tools_list()),
            "tools/call" => {
                let name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !self.tools.iter().any(|(tool, _)| tool.name == name) {
                    return Some(error(&id, -32602, &format!("Tool {name} not found")));
                }
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let meta = params.get("_meta").cloned();
                ok(&id, (self.call)(name, arguments, meta).await)
            }
            "resources/list" => ok(&id, json!({ "resources": [] })),
            "prompts/list" => ok(&id, json!({ "prompts": [] })),
            _ => error(&id, -32601, &format!("Method not found: {method}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server() -> McpHostServer {
        McpHostServer {
            name: "harness-tools".into(),
            tools: vec![(
                Tool {
                    name: "echo".into(),
                    description: "Echo".into(),
                    parameters: json!({ "properties": { "x": { "type": "string" } } }),
                    constrained_sampling: None,
                },
                McpToolExtras {
                    annotations: Some(json!({ "readOnlyHint": true })),
                    meta: None,
                },
            )],
            call: Arc::new(|name, args, _| {
                Box::pin(async move {
                    json!({ "content": [{ "type": "text", "text": format!("{name}:{args}") }] })
                })
            }),
        }
    }

    #[tokio::test]
    async fn answers_the_mcp_methods() {
        let server = server();
        let init = server
            .handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2024-11-05" } }))
            .await
            .unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(
            server
                .handle(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))
                .await,
            None
        );
        let list = server
            .handle(&json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }))
            .await
            .unwrap();
        assert_eq!(list["result"]["tools"][0]["inputSchema"]["type"], "object");
        assert_eq!(
            list["result"]["tools"][0]["annotations"]["readOnlyHint"],
            true
        );
        let call = server
            .handle(&json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": "echo", "arguments": { "x": "1" } } }))
            .await
            .unwrap();
        assert_eq!(call["result"]["content"][0]["text"], "echo:{\"x\":\"1\"}");
        let missing = server
            .handle(&json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": { "name": "nope" } }))
            .await
            .unwrap();
        assert_eq!(missing["error"]["code"], -32602);
        let unknown = server
            .handle(&json!({ "jsonrpc": "2.0", "id": 5, "method": "x" }))
            .await
            .unwrap();
        assert_eq!(unknown["error"]["code"], -32601);
    }
}
