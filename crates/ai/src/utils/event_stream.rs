//! Port of `utils/event-stream.ts`.
//!
//! `EventStream` is a push-based async queue: producers call `push()` and
//! `end()`, consumers iterate it as a [`futures::Stream`] and await
//! `result()`. As in Pi, every clone is an independent consumer of one shared
//! queue, and consumers that are already waiting receive pushed events in
//! the order they started waiting.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::watch;

use crate::types::{AssistantMessage, AssistantMessageEvent};

struct WaiterSlot<T> {
    /// `Some(Some(event))`: delivered event. `Some(None)`: the stream ended.
    value: Option<Option<T>>,
    waker: Option<Waker>,
}

type Waiter<T> = Arc<Mutex<WaiterSlot<T>>>;

struct State<T> {
    queue: VecDeque<T>,
    waiting: VecDeque<Waiter<T>>,
    done: bool,
}

struct Shared<T, R> {
    state: Mutex<State<T>>,
    result: watch::Sender<Option<R>>,
    is_complete: Box<dyn Fn(&T) -> bool + Send + Sync>,
    extract_result: Box<dyn Fn(&T) -> R + Send + Sync>,
}

/// Generic event stream for async iteration.
pub struct EventStream<T, R = T> {
    shared: Arc<Shared<T, R>>,
    pending: Option<Waiter<T>>,
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            pending: None,
        }
    }
}

impl<T, R> EventStream<T, R>
where
    T: Send + 'static,
    R: Clone + Send + Sync + 'static,
{
    pub fn new(
        is_complete: impl Fn(&T) -> bool + Send + Sync + 'static,
        extract_result: impl Fn(&T) -> R + Send + Sync + 'static,
    ) -> Self {
        let (result, _) = watch::channel(None);
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    queue: VecDeque::new(),
                    waiting: VecDeque::new(),
                    done: false,
                }),
                result,
                is_complete: Box::new(is_complete),
                extract_result: Box::new(extract_result),
            }),
            pending: None,
        }
    }

    pub fn push(&self, event: T) {
        let mut state = self.shared.state.lock();
        if state.done {
            return;
        }

        if (self.shared.is_complete)(&event) {
            state.done = true;
            self.resolve_final_result((self.shared.extract_result)(&event));
        }

        // Deliver to a waiting consumer or queue it.
        if let Some(waiter) = state.waiting.pop_front() {
            drop(state);
            deliver(&waiter, Some(event));
        } else {
            state.queue.push_back(event);
        }
    }

    pub fn end(&self, result: Option<R>) {
        let waiting = {
            let mut state = self.shared.state.lock();
            state.done = true;
            std::mem::take(&mut state.waiting)
        };
        if let Some(result) = result {
            self.resolve_final_result(result);
        }
        // Notify all waiting consumers that we're done.
        for waiter in waiting {
            deliver(&waiter, None);
        }
    }

    /// Resolves with the final result. Like Pi's promise it never resolves
    /// when the stream ends without a completing event or explicit result.
    pub async fn result(&self) -> R {
        let mut receiver = self.shared.result.subscribe();
        let value = receiver
            .wait_for(Option::is_some)
            .await
            .expect("event stream result sender lives as long as the stream");
        value.clone().expect("waited for a result")
    }

    fn resolve_final_result(&self, result: R) {
        // A promise resolves once.
        self.shared.result.send_if_modified(|current| {
            if current.is_some() {
                return false;
            }
            *current = Some(result);
            true
        });
    }
}

fn deliver<T>(waiter: &Waiter<T>, value: Option<T>) {
    let mut slot = waiter.lock();
    slot.value = Some(value);
    if let Some(waker) = slot.waker.take() {
        waker.wake();
    }
}

impl<T, R> Stream for EventStream<T, R>
where
    T: Unpin,
{
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        if let Some(waiter) = self.pending.clone() {
            let mut slot = waiter.lock();
            if let Some(value) = slot.value.take() {
                drop(slot);
                self.pending = None;
                return Poll::Ready(value);
            }
            slot.waker = Some(cx.waker().clone());
            return Poll::Pending;
        }

        let mut state = self.shared.state.lock();
        if let Some(event) = state.queue.pop_front() {
            return Poll::Ready(Some(event));
        }
        if state.done {
            return Poll::Ready(None);
        }
        let waiter = Arc::new(Mutex::new(WaiterSlot {
            value: None,
            waker: Some(cx.waker().clone()),
        }));
        state.waiting.push_back(Arc::clone(&waiter));
        drop(state);
        self.pending = Some(waiter);
        Poll::Pending
    }
}

impl<T, R> Drop for EventStream<T, R> {
    fn drop(&mut self) {
        let Some(waiter) = self.pending.take() else {
            return;
        };
        let mut state = self.shared.state.lock();
        if let Some(index) = state
            .waiting
            .iter()
            .position(|candidate| Arc::ptr_eq(candidate, &waiter))
        {
            state.waiting.remove(index);
        } else if let Some(Some(event)) = waiter.lock().value.take() {
            // An event was handed to this consumer but never observed; keep it
            // for the next consumer instead of dropping it.
            state.queue.push_front(event);
        }
    }
}

/// Event stream of an assistant response. Completes on `done` or `error` and
/// resolves `result()` with the final [`AssistantMessage`].
#[derive(Clone)]
pub struct AssistantMessageEventStream(EventStream<AssistantMessageEvent, AssistantMessage>);

impl AssistantMessageEventStream {
    pub fn new() -> Self {
        Self(EventStream::new(
            |event| {
                matches!(
                    event,
                    AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
                )
            },
            |event| match event {
                AssistantMessageEvent::Done { message, .. } => message.clone(),
                AssistantMessageEvent::Error { error, .. } => error.clone(),
                _ => unreachable!("Unexpected event type for final result"),
            },
        ))
    }

    pub fn push(&self, event: AssistantMessageEvent) {
        self.0.push(event);
    }

    pub fn end(&self, result: Option<AssistantMessage>) {
        self.0.end(result);
    }

    pub async fn result(&self) -> AssistantMessage {
        self.0.result().await
    }
}

impl Default for AssistantMessageEventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.0).poll_next(cx)
    }
}

/// Factory function for [`AssistantMessageEventStream`].
pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
    AssistantMessageEventStream::new()
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    fn numbers(is_complete: fn(&i32) -> bool) -> EventStream<i32, i32> {
        EventStream::new(is_complete, |event| *event)
    }

    #[tokio::test]
    async fn drains_buffered_events_in_order_and_ignores_events_pushed_after_completion() {
        let stream = numbers(|event| *event == 3);
        stream.push(1);
        stream.push(2);
        stream.push(3);
        stream.push(4);

        assert_eq!(stream.result().await, 3);
        assert_eq!(stream.collect::<Vec<_>>().await, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn preserves_order_when_events_arrive_after_buffered_draining_starts() {
        let stream = numbers(|_| false);
        stream.push(1);
        stream.push(2);

        let mut iterator = stream.clone();
        assert_eq!(iterator.next().await, Some(1));
        stream.push(3);
        assert_eq!(iterator.next().await, Some(2));
        assert_eq!(iterator.next().await, Some(3));

        stream.end(Some(3));
        assert_eq!(iterator.next().await, None);
    }

    #[tokio::test]
    async fn delivers_events_to_waiting_consumers_in_registration_order() {
        let stream = numbers(|_| false);
        let mut first = stream.clone();
        let mut second = stream.clone();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut first).poll_next(&mut cx).is_pending());
        assert!(Pin::new(&mut second).poll_next(&mut cx).is_pending());

        stream.push(1);
        stream.push(2);

        assert_eq!(second.next().await, Some(2));
        assert_eq!(first.next().await, Some(1));
    }

    #[tokio::test]
    async fn drains_buffered_events_after_end_and_resolves_the_explicit_result() {
        let stream: EventStream<i32, String> =
            EventStream::new(|_: &i32| false, |e: &i32| e.to_string());
        stream.push(1);
        stream.push(2);
        stream.end(Some("complete".to_string()));

        assert_eq!(stream.result().await, "complete");
        assert_eq!(stream.collect::<Vec<_>>().await, vec![1, 2]);
    }

    #[tokio::test]
    async fn wakes_all_waiting_consumers_when_ended_without_a_result() {
        let stream = numbers(|_| false);
        let first = tokio::spawn({
            let mut first = stream.clone();
            async move { first.next().await }
        });
        let second = tokio::spawn({
            let mut second = stream.clone();
            async move { second.next().await }
        });
        tokio::task::yield_now().await;

        stream.end(None);

        assert_eq!(first.await.unwrap(), None);
        assert_eq!(second.await.unwrap(), None);
    }

    #[tokio::test]
    async fn dropped_waiting_consumer_does_not_lose_events() {
        let stream = numbers(|_| false);
        let mut abandoned = stream.clone();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut abandoned).poll_next(&mut cx).is_pending());
        stream.push(1);
        drop(abandoned);
        stream.end(None);
        assert_eq!(stream.collect::<Vec<_>>().await, vec![1]);
    }
}
