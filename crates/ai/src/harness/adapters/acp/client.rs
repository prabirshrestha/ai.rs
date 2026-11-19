//! [rust] The ACP client side of the connection: JSON-RPC 2.0 over the
//! agent's stdio, in place of `@agentclientprotocol/sdk`'s `ClientConnection`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::harness::adapters::stdio_process::{JsonLineProcess, process_exit_error};
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};

type Pending = Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// `-32601`.
pub const METHOD_NOT_FOUND: i64 = -32601;

/// A running ACP agent. Its notifications and requests arrive on
/// [`events`](Self::events); responses resolve [`request`](Self::request).
pub struct AcpConnection {
    process: Arc<JsonLineProcess>,
    pending: Pending,
    next_id: AtomicU64,
    closed: Arc<AtomicBool>,
    pub events: Mutex<mpsc::UnboundedReceiver<Value>>,
}

impl AcpConnection {
    pub async fn spawn(
        sandbox: &dyn SandboxSession,
        options: SandboxProcessOptions,
        label: &str,
    ) -> HarnessResult<Arc<Self>> {
        let (process, mut output) = JsonLineProcess::spawn(sandbox, options, label).await?;
        let pending: Pending = Arc::default();
        let (events_sender, events) = mpsc::unbounded_channel();
        let closed: Arc<AtomicBool> = Arc::default();
        {
            let closed = closed.clone();
            let pending = pending.clone();
            let process = process.clone();
            tokio::spawn(async move {
                while let Some(message) = output.recv().await {
                    let is_response = message.get("method").is_none()
                        && (message.get("result").is_some() || message.get("error").is_some());
                    if !is_response {
                        let _ = events_sender.send(message);
                        continue;
                    }
                    let Some(id) = message.get("id").and_then(Value::as_u64) else {
                        continue;
                    };
                    let Some(sender) = pending.lock().unwrap().remove(&id) else {
                        continue;
                    };
                    let result = match message.get("error") {
                        Some(error) => Err(format_error(error)),
                        None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                    };
                    let _ = sender.send(result);
                }
                closed.store(true, Ordering::SeqCst);
                let _ = process.wait().await;
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                let error = process_exit_error(&process, "exited").to_string();
                for (_, sender) in pending.lock().unwrap().drain() {
                    let _ = sender.send(Err(error.clone()));
                }
            });
        }
        Ok(Arc::new(Self {
            process,
            pending,
            next_id: AtomicU64::new(0),
            closed,
            events: Mutex::new(events),
        }))
    }

    /// Send a request and wait for its result.
    pub async fn request(&self, method: &str, params: Value) -> HarnessResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, sender);
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Err(error) = self.process.send(&message).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        match receiver.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => {
                Err(HarnessError::new(format!("ACP {method} failed: {error}")).into())
            }
            Err(_) => Err(HarnessError::new("The ACP agent is not running.").into()),
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> HarnessResult<()> {
        self.process
            .send(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .await
    }

    /// Answer an agent request.
    pub async fn respond(&self, id: &Value, result: Result<Value, (i64, String)>) {
        let message = match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": code, "message": message },
            }),
        };
        let _ = self.process.send(&message).await;
    }

    /// Close stdin, then kill the agent if it does not exit.
    pub async fn close(&self) {
        self.process.close_stdin().await;
        if tokio::time::timeout(std::time::Duration::from_millis(1500), self.process.wait())
            .await
            .is_err()
        {
            self.process.kill().await;
        }
    }

    /// Whether the agent's stdout ended.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub fn stderr_tail(&self) -> String {
        self.process.stderr_tail()
    }
}

fn format_error(error: &Value) -> String {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown error")
        .to_string();
    match error.get("data") {
        Some(Value::String(data)) if !data.is_empty() => format!("{message}: {data}"),
        Some(data) if !data.is_null() => format!("{message}: {data}"),
        _ => message,
    }
}
