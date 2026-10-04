//! Port of chord `src/context/index.ts` (and the `Context`/`ContextKey` types
//! from `src/types.ts`).
//!
//! `Context` is an immutable linked list of `(ContextKey, value)` nodes over an
//! empty root; deriving a context never changes its parent.
//!
//! Divergences from Pi:
//! - JS `AbortSignal` is ported as [`AbortSignal`], which wraps the crate's
//!   cancellation primitive ([`CancellationToken`], see `utils::abort`) and adds
//!   what durable needs from the DOM type: `aborted`, `reason`, abort listeners
//!   and `AbortSignal.any`. [`AbortSignal::token`] hands the token to APIs that
//!   take the crate-wide signal (`StreamOptions.signal`).
//! - JS abort reasons may be any value; here they are errors ([`AbortReason`]).
//!   `abort()` without a reason uses [`AbortError`] ("The operation was
//!   aborted"), as `DOMException("AbortError")` does in JS.
//! - `awaitWithContext` rejects only the waiter in Pi and the awaited promise
//!   keeps running. A Rust future stops when dropped, so work that must outlive
//!   a cancelled waiter has to be spawned by the caller and its handle awaited.

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

// ─── Abort signals ───────────────────────────────────────────────────────────

/// The default abort reason, JS's `DOMException("The operation was aborted", "AbortError")`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("The operation was aborted")]
pub struct AbortError;

/// A plain `Error(message)` used as an abort reason.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct AbortMessage(pub String);

/// The reason a signal was aborted (JS `signal.reason`). Cloning shares the
/// same error, so code that rethrows `signal.reason` rethrows the original.
#[derive(Clone)]
pub struct AbortReason(Arc<dyn std::error::Error + Send + Sync>);

impl AbortReason {
    pub fn new(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self(Arc::new(error))
    }

    pub fn from_arc(error: Arc<dyn std::error::Error + Send + Sync>) -> Self {
        Self(error)
    }

    /// `controller.abort("message")`: a reason carrying a message.
    pub fn message(message: impl Into<String>) -> Self {
        Self::new(AbortMessage(message.into()))
    }

    /// `new DOMException("The operation was aborted", "AbortError")`.
    pub fn abort_error() -> Self {
        Self::new(AbortError)
    }

    pub fn error(&self) -> &Arc<dyn std::error::Error + Send + Sync> {
        &self.0
    }

    pub fn downcast_ref<E: std::error::Error + 'static>(&self) -> Option<&E> {
        self.0.downcast_ref::<E>()
    }

    /// Whether this is the default [`AbortError`].
    pub fn is_abort_error(&self) -> bool {
        self.downcast_ref::<AbortError>().is_some()
    }

    /// Whether both reasons are the same shared error (JS `===` on reasons).
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AbortReason").field(&self.0).finish()
    }
}

impl fmt::Display for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for AbortReason {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

enum Listener {
    Callback(u64, Box<dyn FnOnce(&AbortReason) + Send>),
    Forward(Weak<SignalInner>),
}

#[derive(Default)]
struct SignalState {
    reason: Option<AbortReason>,
    listeners: Vec<Listener>,
}

struct SignalInner {
    token: CancellationToken,
    state: Mutex<SignalState>,
}

/// Port of the DOM `AbortSignal` subset chord and durable use.
#[derive(Clone)]
pub struct AbortSignal(Arc<SignalInner>);

/// Handle returned by [`AbortSignal::add_listener`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbortListenerId(u64);

static NEXT_LISTENER_ID: AtomicU64 = AtomicU64::new(1);

impl AbortSignal {
    fn new_inner() -> Self {
        Self(Arc::new(SignalInner {
            token: CancellationToken::new(),
            state: Mutex::new(SignalState::default()),
        }))
    }

    /// `AbortSignal.abort(reason)`: an already aborted signal.
    pub fn aborted_with(reason: Option<AbortReason>) -> Self {
        let signal = Self::new_inner();
        signal.abort(reason);
        signal
    }

    /// `AbortSignal.any(signals)`: aborted by the first source that aborts, with its reason.
    pub fn any(signals: &[AbortSignal]) -> Self {
        let combined = Self::new_inner();
        for signal in signals {
            if let Some(reason) = signal.reason() {
                combined.abort(Some(reason));
                return combined;
            }
        }
        for signal in signals {
            let forwarded = {
                let mut state = signal.0.state.lock();
                match &state.reason {
                    Some(reason) => Some(reason.clone()),
                    None => {
                        state.listeners.retain(|listener| match listener {
                            Listener::Forward(target) => target.strong_count() > 0,
                            Listener::Callback(..) => true,
                        });
                        state
                            .listeners
                            .push(Listener::Forward(Arc::downgrade(&combined.0)));
                        None
                    }
                }
            };
            if let Some(reason) = forwarded {
                combined.abort(Some(reason));
                break;
            }
        }
        combined
    }

    /// `signal.aborted`.
    pub fn aborted(&self) -> bool {
        self.0.state.lock().reason.is_some()
    }

    /// `signal.reason`.
    pub fn reason(&self) -> Option<AbortReason> {
        self.0.state.lock().reason.clone()
    }

    /// `signal.throwIfAborted()`.
    pub fn throw_if_aborted(&self) -> Result<(), AbortReason> {
        match self.reason() {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    /// The crate-wide cancellation token for this signal; cancelled when it aborts.
    pub fn token(&self) -> CancellationToken {
        self.0.token.clone()
    }

    /// Wait until the signal aborts.
    pub async fn cancelled(&self) {
        self.0.token.cancelled().await;
    }

    /// `signal.addEventListener("abort", listener, { once: true })`. The
    /// listener runs synchronously on the aborting thread. Adding a listener to
    /// an aborted signal never runs it, as in JS.
    pub fn add_listener(
        &self,
        listener: impl FnOnce(&AbortReason) + Send + 'static,
    ) -> AbortListenerId {
        let id = NEXT_LISTENER_ID.fetch_add(1, Ordering::Relaxed);
        let mut state = self.0.state.lock();
        if state.reason.is_none() {
            state
                .listeners
                .push(Listener::Callback(id, Box::new(listener)));
        }
        AbortListenerId(id)
    }

    /// `signal.removeEventListener("abort", listener)`.
    pub fn remove_listener(&self, id: AbortListenerId) {
        self.0
            .state
            .lock()
            .listeners
            .retain(|listener| !matches!(listener, Listener::Callback(own, _) if *own == id.0));
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn abort(&self, reason: Option<AbortReason>) {
        let (reason, listeners) = {
            let mut state = self.0.state.lock();
            if state.reason.is_some() {
                return;
            }
            let reason = reason.unwrap_or_else(AbortReason::abort_error);
            state.reason = Some(reason.clone());
            (reason, std::mem::take(&mut state.listeners))
        };
        self.0.token.cancel();
        for listener in listeners {
            match listener {
                Listener::Callback(_, callback) => callback(&reason),
                Listener::Forward(target) => {
                    if let Some(target) = target.upgrade() {
                        AbortSignal(target).abort(Some(reason.clone()));
                    }
                }
            }
        }
    }
}

impl fmt::Debug for AbortSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AbortSignal")
            .field("aborted", &self.aborted())
            .field("reason", &self.reason())
            .finish()
    }
}

/// Port of the DOM `AbortController`.
#[derive(Clone, Debug)]
pub struct AbortController {
    signal: AbortSignal,
}

impl Default for AbortController {
    fn default() -> Self {
        Self::new()
    }
}

impl AbortController {
    pub fn new() -> Self {
        Self {
            signal: AbortSignal::new_inner(),
        }
    }

    pub fn signal(&self) -> &AbortSignal {
        &self.signal
    }

    /// `controller.abort(reason)`; `None` uses [`AbortError`]. Idempotent.
    pub fn abort(&self, reason: Option<AbortReason>) {
        self.signal.abort(reason);
    }
}

// ─── Context ─────────────────────────────────────────────────────────────────

/// Typed identity for one value carried by a [`Context`] (`ContextKey<T>`).
pub struct ContextKey<T> {
    token: u64,
    description: Arc<str>,
    value_type: PhantomData<fn(T) -> T>,
}

impl<T> Clone for ContextKey<T> {
    fn clone(&self) -> Self {
        Self {
            token: self.token,
            description: self.description.clone(),
            value_type: PhantomData,
        }
    }
}

impl<T> fmt::Debug for ContextKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextKey")
            .field("description", &self.description)
            .finish()
    }
}

impl<T> ContextKey<T> {
    pub fn description(&self) -> &str {
        &self.description
    }
}

static NEXT_CONTEXT_KEY: AtomicU64 = AtomicU64::new(1);

/// `createContextKey(description)`: a fresh key distinct from every other key.
pub fn create_context_key<T>(description: &str) -> ContextKey<T> {
    ContextKey {
        token: NEXT_CONTEXT_KEY.fetch_add(1, Ordering::Relaxed),
        description: Arc::from(description),
        value_type: PhantomData,
    }
}

const ABORT_SIGNAL_CONTEXT_KEY: u64 = 0;

enum Node {
    Empty(&'static str),
    Value {
        parent: Context,
        key: u64,
        description: Arc<str>,
        /// `None` is an explicit `undefined` (used by `withoutAbortSignal`).
        value: Option<Arc<dyn Any + Send + Sync>>,
    },
}

/// Immutable invocation-scoped values passed explicitly through operations.
#[derive(Clone)]
pub struct Context(Arc<Node>);

/// `BACKGROUND_CONTEXT`.
pub static BACKGROUND_CONTEXT: LazyLock<Context> =
    LazyLock::new(|| Context(Arc::new(Node::Empty("[Context BACKGROUND_CONTEXT]"))));
/// `TODO_CONTEXT`.
pub static TODO_CONTEXT: LazyLock<Context> =
    LazyLock::new(|| Context(Arc::new(Node::Empty("[Context TODO_CONTEXT]"))));

/// A clone of [`BACKGROUND_CONTEXT`].
pub fn background_context() -> Context {
    BACKGROUND_CONTEXT.clone()
}

/// A clone of [`TODO_CONTEXT`].
pub fn todo_context() -> Context {
    TODO_CONTEXT.clone()
}

impl Default for Context {
    fn default() -> Self {
        background_context()
    }
}

impl Context {
    fn lookup(&self, key: u64) -> Option<&(dyn Any + Send + Sync)> {
        let mut node = self;
        loop {
            match &*node.0 {
                Node::Empty(_) => return None,
                Node::Value {
                    parent,
                    key: own,
                    value,
                    ..
                } => {
                    if *own == key {
                        return value.as_deref();
                    }
                    node = parent;
                }
            }
        }
    }

    /// `context.value(key)`.
    pub fn value<T: Send + Sync + 'static>(&self, key: &ContextKey<T>) -> Option<&T> {
        self.lookup(key.token)?.downcast_ref::<T>()
    }

    /// `context.abortSignal`.
    pub fn abort_signal(&self) -> Option<&AbortSignal> {
        self.lookup(ABORT_SIGNAL_CONTEXT_KEY)?
            .downcast_ref::<AbortSignal>()
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn with(
        &self,
        key: u64,
        description: Arc<str>,
        value: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Context {
        Context(Arc::new(Node::Value {
            parent: self.clone(),
            key,
            description,
            value,
        }))
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            Node::Empty(name) => f.write_str(name),
            Node::Value {
                parent,
                description,
                ..
            } => write!(f, "{parent}.WithValue({description})"),
        }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Context({self})")
    }
}

/// Derive a context containing one additional or replaced value.
pub fn with_context_value<T: Send + Sync + 'static>(
    key: &ContextKey<T>,
    value: T,
    parent: &Context,
) -> Context {
    parent.with(key.token, key.description.clone(), Some(Arc::new(value)))
}

static ABORT_SIGNAL_DESCRIPTION: LazyLock<Arc<str>> =
    LazyLock::new(|| Arc::from("chord.abortSignal"));

/// Derive a context cancelled by either the parent signal or the supplied signal.
/// The parent context remains unchanged.
pub fn with_abort_signal(signal: &AbortSignal, context: &Context) -> Context {
    let combined = match context.abort_signal() {
        None => signal.clone(),
        Some(parent) => AbortSignal::any(&[parent.clone(), signal.clone()]),
    };
    context.with(
        ABORT_SIGNAL_CONTEXT_KEY,
        ABORT_SIGNAL_DESCRIPTION.clone(),
        Some(Arc::new(combined)),
    )
}

/// Derive a context retaining all values except caller cancellation. Intended for mandatory cleanup only.
pub fn without_abort_signal(context: &Context) -> Context {
    context.with(
        ABORT_SIGNAL_CONTEXT_KEY,
        ABORT_SIGNAL_DESCRIPTION.clone(),
        None,
    )
}

/// The result of [`with_cancel`]: `{ context, cancel }`.
#[derive(Clone, Debug)]
pub struct CancelContext {
    pub context: Context,
    controller: AbortController,
}

impl CancelContext {
    /// `cancel(reason?)`.
    pub fn cancel(&self, reason: Option<AbortReason>) {
        self.controller.abort(reason);
    }
}

/// Derive an independently cancellable child context.
pub fn with_cancel(context: &Context) -> CancelContext {
    let controller = AbortController::new();
    CancelContext {
        context: with_abort_signal(controller.signal(), context),
        controller,
    }
}

/// Observe a future until it completes or the invocation is cancelled.
///
/// Cancellation rejects only this waiter with the signal's reason. Unlike Pi,
/// the future is dropped on cancellation (see the module docs).
pub async fn await_with_context<T>(
    future: impl Future<Output = T>,
    context: &Context,
) -> Result<T, AbortReason> {
    let Some(signal) = context.abort_signal() else {
        return Ok(future.await);
    };
    if let Some(reason) = signal.reason() {
        return Err(abort_error(reason));
    }
    tokio::select! {
        biased;
        value = future => Ok(value),
        _ = signal.cancelled() => Err(abort_error(signal.reason().unwrap_or_else(AbortReason::abort_error))),
    }
}

fn abort_error(reason: AbortReason) -> AbortReason {
    // Every Rust reason is an error, so JS's non-Error fallback never applies.
    reason
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn provides_distinct_empty_root_contexts() {
        let key = create_context_key::<String>("value");
        assert!(!TODO_CONTEXT.ptr_eq(&BACKGROUND_CONTEXT));
        assert!(TODO_CONTEXT.abort_signal().is_none());
        assert!(TODO_CONTEXT.value(&key).is_none());
        assert_eq!(
            BACKGROUND_CONTEXT.to_string(),
            "[Context BACKGROUND_CONTEXT]"
        );
        assert_eq!(TODO_CONTEXT.to_string(), "[Context TODO_CONTEXT]");
    }

    #[test]
    fn layers_typed_values_without_modifying_parents() {
        let first_key = create_context_key::<String>("first");
        let second_key = create_context_key::<i32>("second");
        let first = with_context_value(&first_key, "one".to_string(), &BACKGROUND_CONTEXT);
        let second = with_context_value(&second_key, 2, &first);
        let replaced = with_context_value(&first_key, "updated".to_string(), &second);

        assert!(BACKGROUND_CONTEXT.value(&first_key).is_none());
        assert_eq!(first.value(&first_key).unwrap(), "one");
        assert!(first.value(&second_key).is_none());
        assert_eq!(second.value(&first_key).unwrap(), "one");
        assert_eq!(*second.value(&second_key).unwrap(), 2);
        assert_eq!(replaced.value(&first_key).unwrap(), "updated");
        assert_eq!(second.value(&first_key).unwrap(), "one");
        assert_eq!(
            replaced.to_string(),
            "[Context BACKGROUND_CONTEXT].WithValue(first).WithValue(second).WithValue(first)"
        );
    }

    #[test]
    fn inherits_parent_cancellation_and_isolates_child_cancellation() {
        let parent_controller = AbortController::new();
        let parent = with_abort_signal(parent_controller.signal(), &BACKGROUND_CONTEXT);
        let child = with_cancel(&parent);
        let sibling = with_cancel(&parent);
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        child
            .context
            .abort_signal()
            .unwrap()
            .add_listener(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            });

        child.cancel(Some(AbortReason::message("child")));
        let child_signal = child.context.abort_signal().unwrap();
        assert!(child_signal.aborted());
        assert_eq!(child_signal.reason().unwrap().to_string(), "child");
        assert!(!sibling.context.abort_signal().unwrap().aborted());
        assert!(!parent.abort_signal().unwrap().aborted());
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        parent_controller.abort(Some(AbortReason::message("parent")));
        let sibling_signal = sibling.context.abort_signal().unwrap();
        assert!(sibling_signal.aborted());
        assert_eq!(sibling_signal.reason().unwrap().to_string(), "parent");
        assert!(sibling_signal.token().is_cancelled());
    }

    #[test]
    fn masks_caller_cancellation_for_mandatory_cleanup() {
        let controller = AbortController::new();
        let key = create_context_key::<String>("value");
        let context = with_context_value(
            &key,
            "preserved".to_string(),
            &with_abort_signal(controller.signal(), &BACKGROUND_CONTEXT),
        );
        let cleanup = without_abort_signal(&context);

        controller.abort(None);
        assert!(context.abort_signal().unwrap().aborted());
        assert!(
            context
                .abort_signal()
                .unwrap()
                .reason()
                .unwrap()
                .is_abort_error()
        );
        assert!(cleanup.abort_signal().is_none());
        assert_eq!(cleanup.value(&key).unwrap(), "preserved");
    }

    #[tokio::test]
    async fn stops_waiting_when_the_invocation_is_cancelled() {
        let controller = AbortController::new();
        let context = with_abort_signal(controller.signal(), &BACKGROUND_CONTEXT);
        let (resolve, work) = tokio::sync::oneshot::channel::<&'static str>();
        let work = tokio::spawn(work);
        let waiting = {
            let context = context.clone();
            tokio::spawn(
                async move { await_with_context(std::future::pending::<()>(), &context).await },
            )
        };
        tokio::task::yield_now().await;
        let cancellation = AbortReason::message("cancelled");
        controller.abort(Some(cancellation.clone()));
        let rejected = waiting.await.unwrap().unwrap_err();
        assert!(rejected.ptr_eq(&cancellation));
        resolve.send("completed later").unwrap();
        assert_eq!(work.await.unwrap().unwrap(), "completed later");
        assert_eq!(
            await_with_context(async { "completed" }, &BACKGROUND_CONTEXT)
                .await
                .unwrap(),
            "completed"
        );
        // Already aborted: rejects immediately.
        assert!(
            await_with_context(async { 1 }, &context)
                .await
                .unwrap_err()
                .ptr_eq(&cancellation)
        );
    }

    #[test]
    fn any_adopts_an_already_aborted_reason_and_removes_listeners() {
        let first = AbortController::new();
        let second = AbortController::new();
        second.abort(Some(AbortReason::message("early")));
        let combined = AbortSignal::any(&[first.signal().clone(), second.signal().clone()]);
        assert_eq!(combined.reason().unwrap().to_string(), "early");

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let id = first.signal().add_listener(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        first.signal().remove_listener(id);
        first.abort(None);
        first.abort(Some(AbortReason::message("ignored")));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(first.signal().reason().unwrap().is_abort_error());
    }
}
