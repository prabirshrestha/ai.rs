//! Port of `harness-fx/src/fx-mcp.ts` and libfx's `mcp.js` adapter: remote
//! MCP servers' tools as fx tools.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use super::libfx::engine::{FxImage, FxToolDescriptor, FxToolResult};
use super::mcp_client::{FxMcpClient, FxMcpConnector, FxMcpTransportConfig, FxMcpTransportKind};
use crate::harness::error::{HarnessError, HarnessResult};

const MAX_PAGES: usize = 64;

/// `parseFxMcpServerConfig({ serverName, config })`: `None` for a disabled
/// entry.
pub(crate) fn parse_fx_mcp_server_config(server_name: &str, config: &Value) -> HarnessResult<Option<FxMcpTransportConfig>> {
    if config.get("disabled") == Some(&Value::Bool(true)) {
        return Ok(None);
    }
    let unsupported = || -> HarnessResult<Option<FxMcpTransportConfig>> {
        Err(HarnessError::capability_unsupported(
            format!("fx MCP server {server_name} must use an HTTP or SSE URL. Command/stdio servers are not supported."),
            Some("fx".into()),
        )
        .into())
    };
    let invalid = |field: &str, expected: &str, value: &Value| -> HarnessResult<Option<FxMcpTransportConfig>> {
        Err(HarnessError::new(format!(
            "Invalid fx MCP server {server_name} field {field}: Invalid input: expected {expected}, received {}",
            received(value)
        ))
        .into())
    };
    let Some(object) = config.as_object() else {
        return unsupported();
    };
    let Some(url) = object.get("url").and_then(Value::as_str) else {
        return unsupported();
    };
    let kind = match object.get("type") {
        None => FxMcpTransportKind::Http,
        Some(Value::String(kind)) => match kind.as_str() {
            "http" | "streamableHttp" => FxMcpTransportKind::Http,
            "sse" => FxMcpTransportKind::Sse,
            _ => return unsupported(),
        },
        Some(_) => return unsupported(),
    };
    let headers = match object.get("headers") {
        None => None,
        Some(Value::Object(headers)) => {
            let mut parsed = IndexMap::new();
            for (name, value) in headers {
                let Some(text) = value.as_str() else {
                    return invalid(&format!("headers.{name}"), "string", value);
                };
                parsed.insert(name.clone(), text.to_string());
            }
            Some(parsed)
        }
        Some(value) => return invalid("headers", "record", value),
    };
    if let Some(value) = object.get("disabled").filter(|value| !value.is_boolean()) {
        return invalid("disabled", "boolean", value);
    }
    Ok(Some(FxMcpTransportConfig {
        kind,
        url: url.to_string(),
        headers,
    }))
}

/// Zod's name for a value's type.
fn received(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// One MCP tool, offered to fx under `descriptor.name`.
#[derive(Clone)]
pub(crate) struct FxMcpTool {
    pub descriptor: FxToolDescriptor,
    remote_name: String,
    client: Arc<dyn FxMcpClient>,
}

impl std::fmt::Debug for FxMcpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FxMcpTool")
            .field("descriptor", &self.descriptor)
            .field("remote_name", &self.remote_name)
            .finish_non_exhaustive()
    }
}

impl FxMcpTool {
    /// `execute(input, { signal })`.
    pub async fn execute(&self, input: Value, signal: CancellationToken) -> FxToolResult {
        match self.client.call_tool(&self.remote_name, input, signal).await {
            Ok(result) => mcp_tool_result(&self.remote_name, &result),
            Err(error) => FxToolResult::error(error.to_string()),
        }
    }
}

/// `contentText(content)`.
fn content_text(content: Option<&Value>) -> String {
    let blocks: Vec<&Value> = match content {
        Some(Value::String(text)) => return text.clone(),
        Some(Value::Array(blocks)) => blocks.iter().collect(),
        Some(block @ Value::Object(_)) => vec![block],
        _ => Vec::new(),
    };
    blocks
        .into_iter()
        .filter_map(|item| {
            let text = item.get("text").and_then(Value::as_str);
            if let Some(text) = item
                .get("resource")
                .and_then(|resource| resource.get("text"))
                .and_then(Value::as_str)
                .filter(|_| item.get("type").and_then(Value::as_str) == Some("resource"))
            {
                return Some(text.to_string());
            }
            if let Some(text) = text {
                return Some(text.to_string());
            }
            if item.get("type").and_then(Value::as_str) == Some("resource_link") {
                let uri = item.get("uri").and_then(Value::as_str)?;
                let name = item.get("name").and_then(Value::as_str).unwrap_or("Resource");
                return Some(format!("{name}: {uri}"));
            }
            None
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `resultText(result)`.
fn result_text(result: &Value) -> String {
    let text = content_text(result.get("content"));
    match result.get("structuredContent") {
        Some(structured) => {
            let structured = structured.to_string();
            if !text.is_empty() && text != structured {
                format!("{text}\n{structured}")
            } else {
                structured
            }
        }
        None => text,
    }
}

/// The tool result of a `tools/call` result.
fn mcp_tool_result(name: &str, result: &Value) -> FxToolResult {
    let text = result_text(result);
    let images = result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| match item.get("type").and_then(Value::as_str) {
            Some("image") => Some(FxImage {
                data: item.get("data")?.as_str()?.to_string(),
                mime_type: item.get("mimeType")?.as_str()?.to_string(),
            }),
            Some("resource") => {
                let resource = item.get("resource")?;
                let mime_type = resource.get("mimeType")?.as_str()?;
                mime_type.starts_with("image/").then_some(())?;
                Some(FxImage {
                    data: resource.get("blob")?.as_str()?.to_string(),
                    mime_type: mime_type.to_string(),
                })
            }
            _ => None,
        })
        .collect();
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return FxToolResult {
            text: if text.is_empty() { format!("MCP tool {name} failed") } else { text },
            images,
            is_error: true,
        };
    }
    FxToolResult {
        text,
        images,
        is_error: false,
    }
}

/// `createMcpAdapter(client, { prefix })`: the server's tool catalog.
pub(crate) async fn create_mcp_adapter(client: Arc<dyn FxMcpClient>, prefix: &str) -> HarnessResult<Vec<FxMcpTool>> {
    let mut catalog: Vec<Value> = Vec::new();
    let mut cursors = HashSet::new();
    let mut cursor: Option<String> = None;
    for page in 0.. {
        if page >= MAX_PAGES {
            return Err(HarnessError::new("MCP tool pagination exceeded its page limit").into());
        }
        let listed = client.list_tools(cursor.clone()).await?;
        let (tools, next) = match &listed {
            Value::Array(tools) => (tools.clone(), None),
            listed => (
                listed
                    .get("tools")
                    .and_then(Value::as_array)
                    .cloned()
                    .ok_or_else(|| HarnessError::new("MCP listTools() returned an invalid tool catalog"))?,
                listed.get("nextCursor").filter(|next| !next.is_null()),
            ),
        };
        catalog.extend(tools);
        let Some(next) = next else {
            break;
        };
        let next = next
            .as_str()
            .filter(|next| next.len() <= 4096 && !cursors.contains(*next))
            .ok_or_else(|| HarnessError::new("MCP listTools() returned an invalid pagination cursor"))?;
        cursors.insert(next.to_string());
        cursor = Some(next.to_string());
    }
    let mut names = HashSet::new();
    let mut seen = HashSet::new();
    let mut tools = Vec::new();
    for (index, tool) in catalog.iter().enumerate() {
        let name = tool.get("name").and_then(Value::as_str);
        let description = tool.get("description");
        let Some(name) = name.filter(|name| !name.is_empty() && name.chars().count() <= 256) else {
            return Err(HarnessError::new(format!("MCP tool {index} is invalid")).into());
        };
        if description.is_some_and(|description| !description.is_string()) {
            return Err(HarnessError::new(format!("MCP tool {index} is invalid")).into());
        }
        if !seen.insert(name.to_string()) {
            return Err(HarnessError::new(format!("MCP tool {name} is duplicated")).into());
        }
        let base: String = format!("{prefix}{name}")
            .chars()
            .map(|char| if char.is_ascii_alphanumeric() || char == '_' || char == '-' { char } else { '_' })
            .collect();
        let mut exposed: String = base.chars().take(64).collect();
        let mut suffix = 2;
        while names.contains(&exposed) {
            let tail = format!("_{suffix}");
            exposed = format!("{}{tail}", base.chars().take(64 - tail.len()).collect::<String>());
            suffix += 1;
        }
        names.insert(exposed.clone());
        tools.push(FxMcpTool {
            descriptor: FxToolDescriptor {
                name: exposed,
                description: description
                    .and_then(Value::as_str)
                    .filter(|description| !description.is_empty())
                    .unwrap_or("MCP tool")
                    .to_string(),
                input_schema: tool
                    .get("inputSchema")
                    .filter(|schema| !schema.is_null())
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
            },
            remote_name: name.to_string(),
            client: client.clone(),
        });
    }
    Ok(tools)
}

/// `FxMcpRuntime`.
#[derive(Default)]
pub(crate) struct FxMcpRuntime {
    pub tools: Vec<FxMcpTool>,
    pub instructions: String,
    clients: Vec<Arc<dyn FxMcpClient>>,
}

impl FxMcpRuntime {
    pub async fn close(&self) {
        futures::future::join_all(self.clients.iter().map(|client| client.close())).await;
    }
}

/// `createFxMcpRuntime({ mcpServers, abortSignal })`.
pub(crate) async fn create_fx_mcp_runtime(
    mcp_servers: Option<&Map<String, Value>>,
    signal: Option<CancellationToken>,
    connector: &FxMcpConnector,
) -> HarnessResult<FxMcpRuntime> {
    let mut runtime = FxMcpRuntime::default();
    let mut instructions = Vec::new();
    let result: HarnessResult<()> = async {
        for (server_name, raw_config) in mcp_servers.into_iter().flatten() {
            let Some(config) = parse_fx_mcp_server_config(server_name, raw_config)? else {
                continue;
            };
            let client = connector(config, signal.clone()).await?;
            runtime.clients.push(client.clone());
            if let Some(text) = client.instructions().filter(|text| !text.is_empty()) {
                instructions.push(text);
            }
            let prefix: String = server_name
                .chars()
                .map(|char| if char.is_ascii_alphanumeric() || char == '_' || char == '-' { char } else { '_' })
                .collect();
            runtime
                .tools
                .extend(create_mcp_adapter(client, &format!("mcp_{prefix}_")).await?);
        }
        Ok(())
    }
    .await;
    if let Err(error) = result {
        runtime.close().await;
        return Err(error);
    }
    runtime.instructions = instructions.join("\n\n");
    Ok(runtime)
}

#[cfg(test)]
#[path = "mcp_tests.rs"]
mod tests;
