use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::{mpsc, oneshot};

use crate::types::{AssistantMessage, AssistantMessageEvent};
use crate::{Error, Result};

pub struct AssistantMessageEventStream {
    receiver: mpsc::UnboundedReceiver<AssistantMessageEvent>,
    result_receiver: Option<oneshot::Receiver<AssistantMessage>>,
}

pub struct AssistantMessageEventStreamSender {
    sender: mpsc::UnboundedSender<AssistantMessageEvent>,
    result_sender: Option<oneshot::Sender<AssistantMessage>>,
}

impl AssistantMessageEventStream {
    pub fn channel() -> (AssistantMessageEventStreamSender, Self) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = oneshot::channel();
        (
            AssistantMessageEventStreamSender {
                sender,
                result_sender: Some(result_sender),
            },
            Self {
                receiver,
                result_receiver: Some(result_receiver),
            },
        )
    }

    pub async fn result(&mut self) -> Result<AssistantMessage> {
        let receiver = self.result_receiver.take().ok_or(Error::StreamClosed)?;
        receiver.await.map_err(|_| Error::StreamClosed)
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
        let final_message = match &event {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
            _ => None,
        };

        if let Some(message) = final_message {
            if let Some(sender) = self.result_sender.take() {
                let _ = sender.send(message);
            }
        }

        let _ = self.sender.send(event);
    }

    pub fn end(&mut self, message: Option<AssistantMessage>) {
        if let Some(message) = message {
            if let Some(sender) = self.result_sender.take() {
                let _ = sender.send(message);
            }
        }
    }
}
