use std::future::Future;

use futures::{StreamExt, stream::BoxStream};
use tokio::sync::mpsc;

use crate::Result;
use crate::types::AssistantMessageEvent;

pub type AssistantEventStream = BoxStream<'static, Result<AssistantMessageEvent>>;

pub struct AssistantMessageEventStreamSender {
    sender: Option<mpsc::UnboundedSender<Result<AssistantMessageEvent>>>,
}

impl AssistantMessageEventStreamSender {
    pub fn channel() -> (AssistantMessageEventStreamSender, AssistantEventStream) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            AssistantMessageEventStreamSender {
                sender: Some(sender),
            },
            tokio_stream_from_unbounded_receiver(receiver),
        )
    }

    pub fn push(&mut self, event: AssistantMessageEvent) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };

        let is_terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        let _ = sender.send(Ok(event));
        if is_terminal {
            self.sender.take();
        }
    }

    pub fn push_error(&mut self, error: crate::Error) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err(error));
        }
    }

    pub fn end(&mut self) {
        self.sender.take();
    }
}

pub fn create_assistant_message_event_stream()
-> (AssistantMessageEventStreamSender, AssistantEventStream) {
    AssistantMessageEventStreamSender::channel()
}

pub fn stream_from_producer<F, Fut, E, H>(producer: F, error_event: H) -> AssistantEventStream
where
    F: FnOnce(AssistantMessageEventStreamSender) -> Fut + Send + 'static,
    Fut: Future<Output = std::result::Result<(), E>> + Send + 'static,
    E: Send + 'static,
    H: FnOnce(E) -> AssistantMessageEvent + Send + 'static,
{
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let event_sender = AssistantMessageEventStreamSender {
        sender: Some(sender),
    };

    async_stream::try_stream! {
        let producer = producer(event_sender);
        futures::pin_mut!(producer);
        let mut error_event = Some(error_event);

        loop {
            tokio::select! {
                biased;

                event = receiver.recv() => {
                    if let Some(event) = event {
                        yield event?;
                    } else {
                        break;
                    }
                }

                result = &mut producer => {
                    while let Ok(event) = receiver.try_recv() {
                        yield event?;
                    }
                    if let Err(error) = result
                        && let Some(error_event) = error_event.take() {
                            yield error_event(error);
                        }
                    break;
                }
            }
        }
    }
    .boxed()
}

pub fn tokio_stream_from_unbounded_receiver<T: Send + 'static>(
    mut receiver: mpsc::UnboundedReceiver<T>,
) -> BoxStream<'static, T> {
    async_stream::stream! {
        while let Some(item) = receiver.recv().await {
            yield item;
        }
    }
    .boxed()
}
