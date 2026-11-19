//! [rust] The host tool MCP server the ACP agent reaches, in place of the TS
//! bridge's `host-tool-relay.ts`, `host-tool-mcp.ts` and
//! `host-tool-mcp-http.ts`.
//!
//! Over stdio, the agent starts a shell command that copies its MCP stdio
//! to a pair of FIFOs in the sandbox; this process holds the other ends
//! through one long-running relay process. Over HTTP, this process serves
//! an MCP endpoint itself, which the agent must be able to reach.

use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::harness::adapters::host_tools::{HostTools, tool_result_mcp_content};
use crate::harness::adapters::mcp_host::{McpCallFn, McpHostServer, McpToolExtras};
use crate::harness::adapters::stdio_process::JsonLineProcess;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};
use crate::harness::session::NoticeLevel;
use crate::harness::utils::{random_uuid, shell_quote};
use crate::{AgentToolResult, AssistantMessage, Tool, ToolCall};

use super::settings::{AcpHttpRelayOptions, HOST_TOOL_MCP_SERVER_NAME};

/// What the relay asks of the running turn.
pub(super) enum RelayEvent {
    /// The agent called a host tool. The turn records the call and answers
    /// with its id and the assistant message it belongs to.
    Call {
        name: String,
        args: Value,
        reply: oneshot::Sender<(String, AssistantMessage)>,
    },
    Update {
        id: String,
        partial: AgentToolResult,
    },
    End {
        id: String,
        name: String,
        result: AgentToolResult,
        is_error: bool,
    },
    Notice(NoticeLevel, String),
}

/// The turn the relay runs tools for.
#[derive(Clone)]
pub(super) struct TurnBinding {
    pub host: HostTools,
    pub events: mpsc::UnboundedSender<RelayEvent>,
}

enum Transport {
    Stdio {
        process: Arc<JsonLineProcess>,
        directory: String,
    },
    Http {
        url: String,
        token: String,
        task: JoinHandle<()>,
    },
}

/// A running relay.
pub(super) struct HostRelay {
    tools: Arc<StdMutex<Vec<Tool>>>,
    binding: Arc<StdMutex<Option<TurnBinding>>>,
    transport: Transport,
}

/// The FIFO side the agent starts: its stdin goes to `req`, `res` to its
/// stdout.
const AGENT_SIDE_SCRIPT: &str = r#"cat "$1/res" & r=$!; cat > "$1/req"; kill $r 2>/dev/null"#;

/// The side this process holds. Opening both FIFOs read-write keeps them
/// open across agent-side restarts.
const HOST_SIDE_SCRIPT: &str = r#"set -e
rm -f "$1/req" "$1/res"
mkfifo -m 600 "$1/req" "$1/res"
exec 3<>"$1/req" 4<>"$1/res"
echo '{"relay":"ready"}'
cat <&3 & c=$!
cat >&4
kill $c 2>/dev/null"#;

struct Shared {
    tools: Arc<StdMutex<Vec<Tool>>>,
    binding: Arc<StdMutex<Option<TurnBinding>>>,
    images: bool,
}

impl Shared {
    fn server(&self) -> McpHostServer {
        let tools = self.tools.lock().unwrap().clone();
        let binding = self.binding.clone();
        let images = self.images;
        let call: McpCallFn = Arc::new(move |name, args, _meta| {
            let binding = binding.lock().unwrap().clone();
            Box::pin(async move { call_tool(binding, name, args, images).await })
        });
        McpHostServer {
            name: HOST_TOOL_MCP_SERVER_NAME.to_string(),
            tools: tools
                .into_iter()
                .map(|tool| (tool, McpToolExtras::default()))
                .collect(),
            call,
        }
    }
}

fn error_result(message: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": message }], "isError": true })
}

async fn call_tool(binding: Option<TurnBinding>, name: String, args: Value, images: bool) -> Value {
    let Some(binding) = binding else {
        return error_result("No HarnessAgent turn is running.");
    };
    let (reply, recorded) = oneshot::channel();
    if binding
        .events
        .send(RelayEvent::Call {
            name: name.clone(),
            args: args.clone(),
            reply,
        })
        .is_err()
    {
        return error_result("No HarnessAgent turn is running.");
    }
    let Ok((id, assistant)) = recorded.await else {
        return error_result("No HarnessAgent turn is running.");
    };
    let call = ToolCall {
        id: id.clone(),
        name: name.clone(),
        arguments: args,
        thought_signature: None,
    };
    let updates = binding.events.clone();
    let update_id = id.clone();
    let on_update: crate::AgentToolUpdateCallback = Arc::new(move |partial| {
        let _ = updates.send(RelayEvent::Update {
            id: update_id.clone(),
            partial,
        });
        Box::pin(async {})
    });
    let (result, is_error) = binding.host.run(&call, &assistant, Some(on_update)).await;
    let mut content = tool_result_mcp_content(&result);
    if !images && content.iter().any(|part| part["type"] == "image") {
        content.retain(|part| part["type"] != "image");
        let _ = binding.events.send(RelayEvent::Notice(
            NoticeLevel::Warning,
            format!(
                "Tool {name} returned image output, which this ACP agent cannot receive; it was omitted."
            ),
        ));
    }
    if content.is_empty() {
        content.push(json!({ "type": "text", "text": "" }));
    }
    let _ = binding.events.send(RelayEvent::End {
        id,
        name,
        result,
        is_error,
    });
    json!({ "content": content, "isError": is_error })
}

impl HostRelay {
    /// Start the stdio relay with its FIFOs in `directory`.
    pub async fn start_stdio(
        sandbox: &dyn SandboxSession,
        directory: &str,
        images: bool,
    ) -> HarnessResult<Self> {
        let command = format!(
            "mkdir -p {dir} && exec sh -c {script} relay {dir}",
            dir = shell_quote(directory),
            script = shell_quote(HOST_SIDE_SCRIPT),
        );
        let (process, mut requests) = JsonLineProcess::spawn(
            sandbox,
            SandboxProcessOptions::new(command),
            "ACP host tool relay",
        )
        .await?;
        // The FIFOs exist once the relay says so; the agent may open them as
        // soon as it learns the server definition.
        match tokio::time::timeout(std::time::Duration::from_secs(10), requests.recv()).await {
            Ok(Some(message)) if message.get("relay").is_some() => {}
            _ => {
                process.kill().await;
                return Err(HarnessError::new(format!(
                    "The ACP host tool relay did not start: {}",
                    process.stderr_tail()
                ))
                .into());
            }
        }
        let tools: Arc<StdMutex<Vec<Tool>>> = Arc::default();
        let binding: Arc<StdMutex<Option<TurnBinding>>> = Arc::default();
        let shared = Arc::new(Shared {
            tools: tools.clone(),
            binding: binding.clone(),
            images,
        });
        {
            let process = process.clone();
            tokio::spawn(async move {
                while let Some(message) = requests.recv().await {
                    let server = shared.server();
                    let process = process.clone();
                    tokio::spawn(async move {
                        if let Some(response) = server.handle(&message).await {
                            let _ = process.send(&response).await;
                        }
                    });
                }
            });
        }
        Ok(Self {
            tools,
            binding,
            transport: Transport::Stdio {
                process,
                directory: directory.to_string(),
            },
        })
    }

    /// Serve the HTTP endpoint.
    pub async fn start_http(options: &AcpHttpRelayOptions, images: bool) -> HarnessResult<Self> {
        let listener = tokio::net::TcpListener::bind(options.bind).await?;
        let address = listener.local_addr()?;
        let host = options
            .url_host
            .clone()
            .unwrap_or_else(|| address.ip().to_string());
        let url = format!("http://{host}:{}/mcp", address.port());
        let token = random_uuid();
        let tools: Arc<StdMutex<Vec<Tool>>> = Arc::default();
        let binding: Arc<StdMutex<Option<TurnBinding>>> = Arc::default();
        let shared = Arc::new(Shared {
            tools: tools.clone(),
            binding: binding.clone(),
            images,
        });
        let expected = format!("Bearer {token}");
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let shared = shared.clone();
                let expected = expected.clone();
                tokio::spawn(async move {
                    let _ = serve_http_connection(stream, shared, expected).await;
                });
            }
        });
        Ok(Self {
            tools,
            binding,
            transport: Transport::Http { url, token, task },
        })
    }

    /// The `McpServer` entry of `session/new`.
    pub fn mcp_server_definition(&self) -> Value {
        match &self.transport {
            Transport::Stdio { directory, .. } => json!({
                "name": HOST_TOOL_MCP_SERVER_NAME,
                "command": "sh",
                "args": ["-c", AGENT_SIDE_SCRIPT, "relay", directory],
                "env": [],
            }),
            Transport::Http { url, token, .. } => json!({
                "type": "http",
                "name": HOST_TOOL_MCP_SERVER_NAME,
                "url": url,
                "headers": [{ "name": "Authorization", "value": format!("Bearer {token}") }],
            }),
        }
    }

    /// Set the tools; returns whether they changed.
    pub fn set_tools(&self, tools: Vec<Tool>) -> bool {
        let mut current = self.tools.lock().unwrap();
        let changed = *current != tools;
        *current = tools;
        changed
    }

    /// `refreshHostToolCatalog`: tell the agent's MCP client the tools
    /// changed.
    ///
    /// [rust] The TS bridge waits up to ten seconds for the agent to list
    /// the tools again; this only notifies. Over HTTP there is no event
    /// stream to notify on, so the agent sees the new tools when it lists
    /// them again.
    pub async fn notify_tools_changed(&self) {
        if let Transport::Stdio { process, .. } = &self.transport {
            let _ = process
                .send(&json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" }))
                .await;
        }
    }

    pub fn bind(&self, binding: TurnBinding) {
        *self.binding.lock().unwrap() = Some(binding);
    }

    pub fn unbind(&self) {
        *self.binding.lock().unwrap() = None;
    }

    pub async fn close(&self) {
        self.unbind();
        match &self.transport {
            Transport::Stdio { process, .. } => {
                // Closing stdin ends the relay and its reader.
                process.close_stdin().await;
                if tokio::time::timeout(std::time::Duration::from_secs(1), process.wait())
                    .await
                    .is_err()
                {
                    process.kill().await;
                }
            }
            Transport::Http { task, .. } => task.abort(),
        }
    }
}

/// One HTTP/1.1 connection: `POST /mcp` with a JSON-RPC message or batch.
async fn serve_http_connection(
    stream: tokio::net::TcpStream,
    shared: Arc<Shared>,
    expected_authorization: String,
) -> HarnessResult<()> {
    let (read, mut write) = stream.into_split();
    let mut reader = BufReader::new(read);
    loop {
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).await? == 0 {
            return Ok(());
        }
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut content_length = 0usize;
        let mut authorization = None;
        let mut close = false;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await? == 0 {
                return Ok(());
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => content_length = value.parse().unwrap_or(0),
                "authorization" => authorization = Some(value.to_string()),
                "connection" => close = value.eq_ignore_ascii_case("close"),
                _ => {}
            }
        }
        if content_length > 64 * 1024 * 1024 {
            return Err(HarnessError::new("The host tool request is too large.").into());
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).await?;

        let (status, response) = if path.split('?').next() != Some("/mcp") {
            ("404 Not Found", None)
        } else if authorization.as_deref() != Some(expected_authorization.as_str()) {
            ("401 Unauthorized", None)
        } else if method == "DELETE" {
            ("200 OK", None)
        } else if method != "POST" {
            ("405 Method Not Allowed", None)
        } else {
            match serde_json::from_slice::<Value>(&body) {
                Err(_) => (
                    "400 Bad Request",
                    Some(json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": { "code": -32700, "message": "Parse error" },
                    })),
                ),
                Ok(Value::Array(messages)) => {
                    let server = shared.server();
                    let mut responses = Vec::new();
                    for message in &messages {
                        if let Some(response) = server.handle(message).await {
                            responses.push(response);
                        }
                    }
                    if responses.is_empty() {
                        ("202 Accepted", None)
                    } else {
                        ("200 OK", Some(Value::Array(responses)))
                    }
                }
                Ok(message) => match shared.server().handle(&message).await {
                    Some(response) => ("200 OK", Some(response)),
                    None => ("202 Accepted", None),
                },
            }
        };
        let body = response
            .map(|response| serde_json::to_vec(&response).unwrap())
            .unwrap_or_default();
        let mut head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n", body.len());
        if !body.is_empty() {
            head.push_str("Content-Type: application/json\r\n");
        }
        if status.starts_with("405") {
            head.push_str("Allow: POST, DELETE\r\n");
        }
        head.push_str("\r\n");
        write.write_all(head.as_bytes()).await?;
        write.write_all(&body).await?;
        write.flush().await?;
        if close {
            return Ok(());
        }
    }
}
