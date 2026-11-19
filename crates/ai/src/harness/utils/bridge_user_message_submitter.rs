//! Port of `utils/bridge-user-message-submitter.ts`: steering messages sent
//! to a running bridge turn, resolved by the bridge's acknowledgement.

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::spec::{
    ExperimentalHarnessV1BridgeUserMessageInbound, ExperimentalHarnessV1BridgeUserMessageResponse,
};
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_channel::Unsubscribe;

/// `Experimental_BridgeUserMessageRequest`.
pub type ExperimentalBridgeUserMessageRequest = ExperimentalHarnessV1BridgeUserMessageInbound;

/// `Experimental_BridgeUserMessageResponse`.
pub type ExperimentalBridgeUserMessageResponse = ExperimentalHarnessV1BridgeUserMessageResponse;

/// A response listener given to `on_response`.
pub type BridgeUserMessageResponseListener =
    Arc<dyn Fn(&ExperimentalBridgeUserMessageResponse) + Send + Sync>;

/// A reconnect listener given to `on_reconnect`.
pub type BridgeReconnectListener = Arc<dyn Fn() + Send + Sync>;

/// Sends one request over the bridge.
pub type SendBridgeUserMessage =
    Arc<dyn Fn(&ExperimentalBridgeUserMessageRequest) -> HarnessResult<()> + Send + Sync>;

/// Options of [`experimental_create_bridge_user_message_submitter`].
pub struct ExperimentalBridgeUserMessageSubmitterOptions {
    pub send: SendBridgeUserMessage,
    pub on_response: Box<dyn FnOnce(BridgeUserMessageResponseListener) -> Unsubscribe + Send>,
    pub on_reconnect: Box<dyn FnOnce(BridgeReconnectListener) -> Unsubscribe + Send>,
}

/// One error delivered to every pending submission.
#[derive(Debug, Clone)]
struct SharedError(Arc<dyn StdError + Send + Sync>);

impl fmt::Display for SharedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl StdError for SharedError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.0.as_ref())
    }
}

struct Pending {
    request: ExperimentalBridgeUserMessageRequest,
    resolve: oneshot::Sender<HarnessResult<()>>,
}

struct Inner {
    send: SendBridgeUserMessage,
    pending: Mutex<IndexMap<String, Pending>>,
    closed: Mutex<bool>,
    unsubscribe: Mutex<Vec<Unsubscribe>>,
}

impl Inner {
    fn send(&self, request: &ExperimentalBridgeUserMessageRequest) {
        if let Err(error) = (self.send)(request)
            && let Some(entry) = self.pending.lock().shift_remove(&request.message_id)
        {
            let _ = entry.resolve.send(Err(error));
        }
    }
}

/// `Experimental_BridgeUserMessageSubmitter`.
#[derive(Clone)]
pub struct ExperimentalBridgeUserMessageSubmitter {
    inner: Arc<Inner>,
}

impl ExperimentalBridgeUserMessageSubmitter {
    /// `submit(text)`: sends right away; resolves once the bridge accepts
    /// the message.
    pub fn submit(&self, text: impl Into<String>) -> BoxFuture<'static, HarnessResult<()>> {
        if *self.inner.closed.lock() {
            return futures::future::ready(Err(plain_error(
                "The bridge turn is no longer accepting user messages.",
            )))
            .boxed();
        }
        let request = ExperimentalBridgeUserMessageRequest {
            r#type: Default::default(),
            message_id: crate::harness::utils::random_uuid(),
            text: text.into(),
        };
        let (resolve, result) = oneshot::channel();
        self.inner.pending.lock().insert(
            request.message_id.clone(),
            Pending {
                request: request.clone(),
                resolve,
            },
        );
        self.inner.send(&request);
        async move {
            result.await.unwrap_or_else(|_| {
                Err(plain_error(
                    "The bridge turn ended before accepting the user message.",
                ))
            })
        }
        .boxed()
    }

    /// `close(error?)`: rejects pending and future submissions.
    pub fn close(&self, error: Option<BoxError>) {
        {
            let mut closed = self.inner.closed.lock();
            if *closed {
                return;
            }
            *closed = true;
        }
        for unsubscribe in self.inner.unsubscribe.lock().drain(..) {
            unsubscribe();
        }
        let reason = SharedError(Arc::from(error.unwrap_or_else(|| {
            plain_error("The bridge turn ended before accepting the user message.")
        })));
        let pending: Vec<Pending> = self
            .inner
            .pending
            .lock()
            .drain(..)
            .map(|(_, p)| p)
            .collect();
        for entry in pending {
            let _ = entry.resolve.send(Err(Box::new(reason.clone())));
        }
    }
}

/// `experimental_createBridgeUserMessageSubmitter(options)`.
pub fn experimental_create_bridge_user_message_submitter(
    options: ExperimentalBridgeUserMessageSubmitterOptions,
) -> ExperimentalBridgeUserMessageSubmitter {
    let inner = Arc::new(Inner {
        send: options.send,
        pending: Mutex::default(),
        closed: Mutex::new(false),
        unsubscribe: Mutex::default(),
    });

    let weak = Arc::downgrade(&inner);
    let unsubscribe_response = (options.on_response)(Arc::new(move |response| {
        let Some(inner) = weak.upgrade() else { return };
        let Some(entry) = inner.pending.lock().shift_remove(&response.message_id) else {
            return;
        };
        let result = if response.accepted {
            Ok(())
        } else {
            Err(plain_error(
                response
                    .error
                    .as_ref()
                    .map(|error| error.message.clone())
                    .unwrap_or_else(|| "The runtime rejected the user message.".into()),
            ))
        };
        let _ = entry.resolve.send(result);
    }));
    let weak = Arc::downgrade(&inner);
    let unsubscribe_reconnect = (options.on_reconnect)(Arc::new(move || {
        let Some(inner) = weak.upgrade() else { return };
        let requests: Vec<_> = inner
            .pending
            .lock()
            .values()
            .map(|entry| entry.request.clone())
            .collect();
        for request in requests {
            inner.send(&request);
        }
    }));
    inner
        .unsubscribe
        .lock()
        .extend([unsubscribe_response, unsubscribe_reconnect]);

    ExperimentalBridgeUserMessageSubmitter { inner }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::spec::HarnessV1BridgeErrorMessage;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Setup {
        submitter: ExperimentalBridgeUserMessageSubmitter,
        sent: Arc<Mutex<Vec<ExperimentalBridgeUserMessageRequest>>>,
        response_listener: Arc<Mutex<Option<BridgeUserMessageResponseListener>>>,
        reconnect_listener: Arc<Mutex<Option<BridgeReconnectListener>>>,
        unsubscribe_response: Arc<AtomicUsize>,
        unsubscribe_reconnect: Arc<AtomicUsize>,
    }

    impl Setup {
        fn respond(&self, message_id: &str, accepted: bool, error: Option<&str>) {
            let listener = self.response_listener.lock().clone().unwrap();
            listener(&ExperimentalBridgeUserMessageResponse {
                r#type: Default::default(),
                message_id: message_id.into(),
                accepted,
                error: error.map(|message| HarnessV1BridgeErrorMessage {
                    message: message.into(),
                }),
            });
        }
    }

    fn setup() -> Setup {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let response_listener = Arc::new(Mutex::new(None));
        let reconnect_listener = Arc::new(Mutex::new(None));
        let unsubscribe_response = Arc::new(AtomicUsize::new(0));
        let unsubscribe_reconnect = Arc::new(AtomicUsize::new(0));
        let (s, rl, cl, ur, uc) = (
            sent.clone(),
            response_listener.clone(),
            reconnect_listener.clone(),
            unsubscribe_response.clone(),
            unsubscribe_reconnect.clone(),
        );
        let submitter = experimental_create_bridge_user_message_submitter(
            ExperimentalBridgeUserMessageSubmitterOptions {
                send: Arc::new(move |message| {
                    s.lock().push(message.clone());
                    Ok(())
                }),
                on_response: Box::new(move |listener| {
                    *rl.lock() = Some(listener);
                    Box::new(move || {
                        ur.fetch_add(1, Ordering::SeqCst);
                    })
                }),
                on_reconnect: Box::new(move |listener| {
                    *cl.lock() = Some(listener);
                    Box::new(move || {
                        uc.fetch_add(1, Ordering::SeqCst);
                    })
                }),
            },
        );
        Setup {
            submitter,
            sent,
            response_listener,
            reconnect_listener,
            unsubscribe_response,
            unsubscribe_reconnect,
        }
    }

    #[tokio::test]
    async fn resolves_only_after_the_matching_acceptance_response() {
        let s = setup();
        let steering = tokio::spawn(s.submitter.submit("Change course."));
        tokio::task::yield_now().await;
        assert!(!steering.is_finished());
        let request = s.sent.lock()[0].clone();
        assert_eq!(request.text, "Change course.");
        assert!(!request.message_id.is_empty());

        s.respond(&request.message_id, true, None);
        steering.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn surfaces_a_runtime_rejection() {
        let s = setup();
        let steering = s.submitter.submit("Change course.");
        let message_id = s.sent.lock()[0].message_id.clone();
        s.respond(&message_id, false, Some("Turn already finished."));
        assert_eq!(
            steering.await.unwrap_err().to_string(),
            "Turn already finished."
        );
    }

    #[tokio::test]
    async fn retries_pending_messages_with_the_same_id_after_reconnect() {
        let s = setup();
        let _steering = s.submitter.submit("Change course.");
        let reconnect = s.reconnect_listener.lock().clone().unwrap();
        reconnect();
        let sent = s.sent.lock();
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[1], sent[0]);
    }

    #[tokio::test]
    async fn rejects_pending_and_future_messages_when_closed() {
        let s = setup();
        let steering = s.submitter.submit("Change course.");
        s.submitter.close(Some(plain_error("Turn ended.")));

        assert_eq!(steering.await.unwrap_err().to_string(), "Turn ended.");
        let error = s.submitter.submit("Again.").await.unwrap_err();
        assert!(error.to_string().contains("no longer accepting"));
        assert_eq!(s.unsubscribe_response.load(Ordering::SeqCst), 1);
        assert_eq!(s.unsubscribe_reconnect.load(Ordering::SeqCst), 1);
    }
}
