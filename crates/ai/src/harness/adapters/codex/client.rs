//! The `codex app-server` JSON-RPC client (`codex-app-server-client.ts`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::harness::adapters::stdio_process::{JsonLineProcess, process_exit_error};
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};

type Pending = Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// A running app server. Notifications and server requests arrive on
/// [`events`](Self::events); responses resolve [`request`](Self::request).
pub struct AppServer {
    process: Arc<JsonLineProcess>,
    pending: Pending,
    next_id: AtomicU64,
    pub events: Mutex<mpsc::UnboundedReceiver<Value>>,
    /// What the process was started with, to restart it when that changes.
    pub config_key: String,
}

impl AppServer {
    pub async fn spawn(
        sandbox: &dyn SandboxSession,
        options: SandboxProcessOptions,
        config_key: String,
    ) -> HarnessResult<Arc<Self>> {
        let (process, mut output) = JsonLineProcess::spawn(sandbox, options, "codex").await?;
        let pending: Pending = Arc::default();
        let (events_sender, events) = mpsc::unbounded_channel();
        {
            let pending = pending.clone();
            let process = process.clone();
            tokio::spawn(async move {
                while let Some(message) = output.recv().await {
                    let is_response = message.get("method").is_none()
                        && (message.get("result").is_some() || message.get("error").is_some());
                    if is_response {
                        let Some(id) = message.get("id").and_then(Value::as_u64) else {
                            continue;
                        };
                        let Some(sender) = pending.lock().unwrap().remove(&id) else {
                            continue;
                        };
                        let result = match message.get("error") {
                            Some(error) => Err(error
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown error")
                                .to_string()),
                            None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                        };
                        let _ = sender.send(result);
                    } else {
                        let _ = events_sender.send(message);
                    }
                }
                // The process ended: fail what is still waiting.
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
            next_id: AtomicU64::new(1),
            events: Mutex::new(events),
            config_key,
        }))
    }

    /// Send a request and wait for its result.
    pub async fn request(&self, method: &str, params: Value) -> HarnessResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, sender);
        let mut message = json!({ "id": id, "method": method });
        if !params.is_null() {
            message["params"] = params;
        }
        if let Err(error) = self.process.send(&message).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        match receiver.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => {
                Err(HarnessError::new(format!("Codex app-server {method} failed: {error}")).into())
            }
            Err(_) => Err(HarnessError::new("Codex app-server is not running.").into()),
        }
    }

    pub async fn notify(&self, method: &str, params: Option<Value>) -> HarnessResult<()> {
        let mut message = json!({ "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.process.send(&message).await
    }

    /// Answer a server request.
    pub async fn respond(&self, id: &Value, result: Result<Value, String>) {
        let message = match result {
            Ok(result) => json!({ "id": id, "result": result }),
            Err(error) => json!({ "id": id, "error": { "code": -32603, "message": error } }),
        };
        let _ = self.process.send(&message).await;
    }

    pub async fn initialize(&self) -> HarnessResult<()> {
        self.request(
            "initialize",
            json!({
                "clientInfo": { "name": "ai-sdk-harness-codex", "version": "1" },
                "capabilities": { "experimentalApi": true },
            }),
        )
        .await?;
        self.notify("initialized", None).await
    }

    /// Close stdin, then kill the process if it does not exit.
    pub async fn close(&self) {
        self.process.close_stdin().await;
        if tokio::time::timeout(std::time::Duration::from_millis(1500), self.process.wait())
            .await
            .is_err()
        {
            self.process.kill().await;
        }
    }

    pub fn stderr_tail(&self) -> String {
        self.process.stderr_tail()
    }
}
