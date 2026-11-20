//! The `codex app-server` JSON-RPC client (`codex-app-server-client.ts`).
//!
//! [rust] One process can serve many sessions: what it says about a thread
//! goes to the session that subscribed to that thread.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};

use serde_json::{Value, json};
use tokio::sync::{OnceCell, mpsc, oneshot};

use crate::harness::adapters::stdio_process::{JsonLineProcess, process_exit_error};
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};

type Pending = Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;
type RouteMap = HashMap<String, (u64, mpsc::UnboundedSender<Value>)>;
type Routes = Arc<StdMutex<RouteMap>>;

static NEXT_SUBSCRIPTION: AtomicU64 = AtomicU64::new(1);

/// A running app server. Notifications and server requests about a thread
/// arrive on that thread's [`subscribe`](Self::subscribe) stream; responses
/// resolve [`request`](Self::request).
pub struct AppServer {
    process: Arc<JsonLineProcess>,
    pending: Pending,
    routes: Routes,
    closed: Arc<AtomicBool>,
    next_id: AtomicU64,
    initialized: OnceCell<()>,
    /// Whether other sessions may use the process too.
    pub shared: bool,
}

/// The thread a message is about: `threadId`, a `thread` object's id, or a
/// legacy `conversationId`.
pub(super) fn message_thread_id(message: &Value) -> Option<&str> {
    let params = message.get("params")?;
    params
        .get("threadId")
        .and_then(Value::as_str)
        .or_else(|| params.pointer("/thread/id").and_then(Value::as_str))
        .or_else(|| params.get("conversationId").and_then(Value::as_str))
}

/// Where a message goes. A process of its own sends everything to its one
/// session, as before sharing. A shared one sends a message to its thread's
/// subscriber, and one about no thread to the only subscriber, if there is
/// just one.
fn route_of(
    routes: &RouteMap,
    message: &Value,
    shared: bool,
) -> Option<mpsc::UnboundedSender<Value>> {
    let only = || {
        (routes.len() == 1)
            .then(|| routes.values().next().map(|(_, sender)| sender.clone()))
            .flatten()
    };
    match message_thread_id(message) {
        Some(thread_id) if shared => routes.get(thread_id).map(|(_, sender)| sender.clone()),
        _ => only(),
    }
}

/// One thread's messages. Dropping it ends the subscription.
pub struct ThreadEvents {
    receiver: mpsc::UnboundedReceiver<Value>,
    routes: Weak<StdMutex<RouteMap>>,
    thread_id: String,
    id: u64,
}

impl ThreadEvents {
    pub async fn recv(&mut self) -> Option<Value> {
        self.receiver.recv().await
    }
}

impl Drop for ThreadEvents {
    fn drop(&mut self) {
        if let Some(routes) = self.routes.upgrade() {
            let mut routes = routes.lock().unwrap();
            if routes
                .get(&self.thread_id)
                .is_some_and(|(id, _)| *id == self.id)
            {
                routes.remove(&self.thread_id);
            }
        }
    }
}

impl AppServer {
    pub async fn spawn(
        sandbox: &dyn SandboxSession,
        options: SandboxProcessOptions,
        shared: bool,
    ) -> HarnessResult<Arc<Self>> {
        let (process, mut output) = JsonLineProcess::spawn(sandbox, options, "codex").await?;
        let pending: Pending = Arc::default();
        let routes: Routes = Arc::default();
        let closed: Arc<AtomicBool> = Arc::default();
        {
            let pending = pending.clone();
            let routes = routes.clone();
            let closed = closed.clone();
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
                        let route = route_of(&routes.lock().unwrap(), &message, shared);
                        match route {
                            Some(sender) => {
                                let _ = sender.send(message);
                            }
                            // A request no session can answer is refused, so
                            // the process does not wait on it.
                            None => {
                                if let Some(id) = message.get("id").cloned() {
                                    let refusal = json!({ "id": id, "error": {
                                        "code": -32603,
                                        "message": "No session is running this thread.",
                                    } });
                                    let _ = process.send(&refusal).await;
                                }
                            }
                        }
                    }
                }
                // The process ended: end every thread's stream and fail what
                // is still waiting.
                closed.store(true, Ordering::SeqCst);
                routes.lock().unwrap().clear();
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
            routes,
            closed,
            next_id: AtomicU64::new(1),
            initialized: OnceCell::new(),
            shared,
        }))
    }

    /// The messages about `thread_id` from now on. A later subscription to
    /// the same thread takes them over.
    pub fn subscribe(&self, thread_id: &str) -> ThreadEvents {
        let (sender, receiver) = mpsc::unbounded_channel();
        let id = NEXT_SUBSCRIPTION.fetch_add(1, Ordering::SeqCst);
        let mut routes = self.routes.lock().unwrap();
        // A stream of an exited process ends at once.
        if !self.is_closed() {
            routes.insert(thread_id.to_string(), (id, sender));
        }
        drop(routes);
        ThreadEvents {
            receiver,
            routes: Arc::downgrade(&self.routes),
            thread_id: thread_id.to_string(),
            id,
        }
    }

    /// Whether the process has exited.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
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

    /// Initialize the process once, however many sessions use it.
    pub async fn initialize(&self) -> HarnessResult<()> {
        self.initialized
            .get_or_try_init(|| async {
                self.request(
                    "initialize",
                    json!({
                        "clientInfo": { "name": "ai-sdk-harness-codex", "version": "1" },
                        "capabilities": { "experimentalApi": true },
                    }),
                )
                .await?;
                self.notify("initialized", None).await
            })
            .await
            .map(|_| ())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn routes(threads: &[&str]) -> (RouteMap, Vec<mpsc::UnboundedReceiver<Value>>) {
        let mut map = RouteMap::new();
        let mut receivers = Vec::new();
        for (index, thread) in threads.iter().enumerate() {
            let (sender, receiver) = mpsc::unbounded_channel();
            map.insert(thread.to_string(), (index as u64, sender));
            receivers.push(receiver);
        }
        (map, receivers)
    }

    #[test]
    fn messages_go_to_their_thread() {
        let (map, mut receivers) = routes(&["a", "b"]);
        let message = json!({ "method": "x", "params": { "threadId": "b" } });
        route_of(&map, &message, true)
            .unwrap()
            .send(message)
            .unwrap();
        assert!(receivers[0].try_recv().is_err());
        assert!(receivers[1].try_recv().is_ok());
        let started = json!({ "method": "thread/started", "params": { "thread": { "id": "a" } } });
        assert_eq!(message_thread_id(&started), Some("a"));
    }

    #[test]
    fn unrouted_messages_reach_a_lone_session_only() {
        let (one, _receivers) = routes(&["a"]);
        let unknown = json!({ "method": "x", "params": { "threadId": "z" } });
        let none = json!({ "method": "x", "params": {} });
        // A process of its own: everything goes to its session.
        assert!(route_of(&one, &unknown, false).is_some());
        assert!(route_of(&one, &none, false).is_some());
        // Shared: another thread's message goes nowhere.
        assert!(route_of(&one, &unknown, true).is_none());
        assert!(route_of(&one, &none, true).is_some());
        let (two, _receivers) = routes(&["a", "b"]);
        assert!(route_of(&two, &unknown, true).is_none());
        assert!(route_of(&two, &none, true).is_none());
    }
}
