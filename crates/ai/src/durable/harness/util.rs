//! Port of durable `src/harness/util.ts`.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::chord::{AbortReason, Context, await_with_context};
use crate::durable::errors::{Error, Result};
use crate::durable::types::{Cursor, Page};

type Waiter<T> = oneshot::Sender<Result<T>>;

/// Pending waits by key. Each settles once: through `resolve`, `reject_all`, or cancellation of its context.
pub struct Waiters<K, T> {
    sets: Mutex<HashMap<K, Vec<Waiter<T>>>>,
    order: Mutex<Vec<K>>,
}

impl<K, T> Default for Waiters<K, T> {
    fn default() -> Self {
        Self {
            sets: Mutex::default(),
            order: Mutex::default(),
        }
    }
}

impl<K: Eq + Hash + Clone + Send + 'static, T: Clone + Send + 'static> Waiters<K, T> {
    /// Register a waiter synchronously; the returned future settles with it.
    pub fn add(self: &Arc<Self>, key: K, context: &Context) -> BoxFuture<'static, Result<T>> {
        if let Some(reason) = context.abort_signal().and_then(|signal| signal.reason()) {
            return Box::pin(futures::future::ready(Err(Error::Aborted(reason))));
        }
        let (sender, receiver) = oneshot::channel();
        {
            let mut sets = self.sets.lock();
            let set = sets.entry(key.clone()).or_insert_with(|| {
                self.order.lock().push(key.clone());
                Vec::new()
            });
            // Waiters whose callers stopped waiting are dropped here.
            set.retain(|waiter| !waiter.is_closed());
            set.push(sender);
        }
        let context = context.clone();
        let waiters = Arc::downgrade(self);
        Box::pin(async move {
            match await_with_context(receiver, &context).await {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err(Error::message("Waiter was dropped")),
                Err(reason) => {
                    if let Some(waiters) = waiters.upgrade() {
                        waiters.prune(&key);
                    }
                    Err(Error::Aborted(reason))
                }
            }
        })
    }

    fn prune(&self, key: &K) {
        let mut sets = self.sets.lock();
        if let Some(set) = sets.get_mut(key) {
            set.retain(|waiter| !waiter.is_closed());
            if set.is_empty() {
                sets.remove(key);
                self.order.lock().retain(|other| other != key);
            }
        }
    }

    /// Keys with pending waiters, in registration order.
    pub fn keys(&self) -> Vec<K> {
        let sets = self.sets.lock();
        self.order
            .lock()
            .iter()
            .filter(|key| sets.contains_key(key))
            .cloned()
            .collect()
    }

    pub fn resolve(&self, key: &K, value: T) {
        let set = self.sets.lock().remove(key);
        self.order.lock().retain(|other| other != key);
        for waiter in set.unwrap_or_default() {
            let _ = waiter.send(Ok(value.clone()));
        }
    }

    pub fn reject_all(&self, error: Error) {
        let sets: Vec<_> = self.sets.lock().drain().collect();
        self.order.lock().clear();
        for (_, set) in sets {
            for waiter in set {
                let _ = waiter.send(Err(error.clone()));
            }
        }
    }
}

/// Every item of a paginated scan, in page order.
pub async fn scan_all<T, F, Fut>(mut scan: F) -> Result<Vec<T>>
where
    F: FnMut(Option<Cursor>) -> Fut,
    Fut: Future<Output = Result<Page<T>>>,
{
    let mut items = Vec::new();
    let mut cursor = None;
    loop {
        let page = scan(cursor).await?;
        items.extend(page.items);
        cursor = page.next;
        if cursor.is_none() {
            return Ok(items);
        }
    }
}

pub fn closed_error() -> Error {
    Error::message("Harness is closed")
}

/// An abort reason carrying a durable error (TS `controller.abort(error)`).
pub fn abort_reason(error: Error) -> AbortReason {
    match error {
        Error::Aborted(reason) => reason,
        error => AbortReason::new(error),
    }
}

/// The durable error a cancelled operation rejects with: the signal's reason, unwrapped when it carries one.
pub fn abort_error(reason: AbortReason) -> Error {
    match reason.downcast_ref::<Error>() {
        Some(error) => error.clone(),
        None => Error::Aborted(reason),
    }
}

/// Convert a caught panic payload into an error (TS: an uncaught throw).
pub fn panic_error(payload: Box<dyn std::any::Any + Send>) -> Error {
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
        .unwrap_or_else(|| "panic".to_string());
    Error::message(message)
}
