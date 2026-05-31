use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::{mpsc, oneshot};

use crate::types::{AssistantMessage, AssistantMessageEvent};
use crate::{Error, Result};

pub struct AssistantMessageEventStream {
    receiver: mpsc::UnboundedReceiver<AssistantMessageEvent>,
    result_receiver: Option<oneshot::Receiver<AssistantMessage>>,
    result: Option<AssistantMessage>,
}

pub struct AssistantMessageEventStreamSender {
    sender: Option<mpsc::UnboundedSender<AssistantMessageEvent>>,
    result_sender: Option<oneshot::Sender<AssistantMessage>>,
}

impl AssistantMessageEventStream {
    pub fn channel() -> (AssistantMessageEventStreamSender, Self) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = oneshot::channel();
        (
            AssistantMessageEventStreamSender {
                sender: Some(sender),
                result_sender: Some(result_sender),
            },
            Self {
                receiver,
                result_receiver: Some(result_receiver),
                result: None,
            },
        )
    }

    pub async fn result(&mut self) -> Result<AssistantMessage> {
        if let Some(result) = &self.result {
            return Ok(result.clone());
        }
        let receiver = self.result_receiver.take().ok_or(Error::StreamClosed)?;
        let result = receiver.await.map_err(|_| Error::StreamClosed)?;
        self.result = Some(result.clone());
        Ok(result)
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

impl AssistantMessageEventStreamSender {
    pub fn push(&mut self, event: AssistantMessageEvent) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };

        let final_message = match &event {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
            _ => None,
        };
        let is_terminal = final_message.is_some();

        if let Some(message) = final_message
            && let Some(sender) = self.result_sender.take()
        {
            let _ = sender.send(message);
        }

        let _ = sender.send(event);
        if is_terminal {
            self.sender.take();
        }
    }

    pub fn end(&mut self, message: Option<AssistantMessage>) {
        if let Some(message) = message {
            if let Some(sender) = self.result_sender.take() {
                let _ = sender.send(message);
            }
        } else {
            self.result_sender.take();
        }
        self.sender.take();
    }
}

pub fn create_assistant_message_event_stream() -> (
    AssistantMessageEventStreamSender,
    AssistantMessageEventStream,
) {
    AssistantMessageEventStream::channel()
}
