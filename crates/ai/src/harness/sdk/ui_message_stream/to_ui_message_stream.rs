//! Mirror of `ai` `ui-message-stream/to-ui-message-stream.ts`,
//! `handle-ui-message-stream-finish.ts`, `get-response-ui-message-id.ts`,
//! `ui-message-stream-outcome.ts` and the `onStepEnd` / `onEnd` callback
//! types.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::stream::BoxStream;
use parking_lot::Mutex;

use crate::harness::provider::JsonValue;
use crate::harness::sdk::content::FinishReason;
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::telemetry::Callback;
use crate::harness::sdk::text_stream_part::TextStreamPart;
use crate::harness::sdk::tool::ToolSet;
use crate::harness::sdk::ui_message_stream::process_ui_message_stream::{
    StreamingUIMessageState, UIMessageStreamError, create_streaming_ui_message_state,
};
use crate::harness::sdk::ui_message_stream::to_ui_message_chunk::{
    DEFAULT_ERROR_TEXT, ToUIMessageChunkOptions, UIMessageStreamOnErrorFn,
    get_model_tool_error_text, to_ui_message_chunk,
};
use crate::harness::sdk::ui_message_stream::ui_message_chunks::UIMessageChunk;
use crate::harness::sdk::ui_message_stream::ui_messages::{UIMessage, UIMessageRole};

/// `IdGenerator`.
pub type IdGeneratorFn = Arc<dyn Fn() -> String + Send + Sync>;

/// `messageMetadata: ({ part }) => metadata | undefined`.
pub type MessageMetadataFn = Arc<dyn Fn(&TextStreamPart) -> Option<JsonValue> + Send + Sync>;

/// A UI message stream: chunks, or the processing failure that ends it.
pub type UIMessageStream = BoxStream<'static, Result<UIMessageChunk, UIMessageStreamError>>;

/// `UIMessageStreamOutcome`: the operation-level outcome of a UI message
/// stream.
#[derive(Debug, Clone, PartialEq)]
pub enum UIMessageStreamOutcome {
    Completed,
    Failed { error: Option<ErrorValue> },
    Aborted,
    Unknown,
}

impl UIMessageStreamOutcome {
    /// `outcome.status`.
    pub fn status(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed { .. } => "failed",
            Self::Aborted => "aborted",
            Self::Unknown => "unknown",
        }
    }
}

/// The `UIMessageStreamOnStepEndCallback` event.
#[derive(Debug, Clone, PartialEq)]
pub struct UIMessageStreamOnStepEndEvent {
    /// The updated list of UI messages at the end of this step.
    pub messages: Vec<UIMessage>,
    /// Whether the response message continues the last original message.
    pub is_continuation: bool,
    /// The message sent to the client as a response.
    pub response_message: UIMessage,
}

/// The `UIMessageStreamOnEndCallback` event.
#[derive(Debug, Clone, PartialEq)]
pub struct UIMessageStreamOnEndEvent {
    /// The updated list of UI messages.
    pub messages: Vec<UIMessage>,
    pub is_continuation: bool,
    pub is_aborted: bool,
    /// `Some(true)` when the consumer cancelled the stream before an
    /// outcome was declared (only ever `true` when present).
    pub is_cancelled: Option<bool>,
    pub outcome: UIMessageStreamOutcome,
    pub response_message: UIMessage,
    pub finish_reason: Option<FinishReason>,
}

/// `UIMessageStreamOnStepEndCallback`.
pub type UIMessageStreamOnStepEndCallback = Callback<UIMessageStreamOnStepEndEvent>;
/// `UIMessageStreamOnEndCallback`.
pub type UIMessageStreamOnEndCallback = Callback<UIMessageStreamOnEndEvent>;

/// `UIMessageStreamOptions`.
///
/// [rust] Callbacks return `()`, so the `onStepEnd` failure path (which
/// reports to `onError`) has no counterpart.
#[derive(Clone, Default)]
pub struct UIMessageStreamOptions {
    /// The original messages. When set, the response message id comes from
    /// the last assistant message or `generate_message_id` (persistence
    /// mode).
    pub original_messages: Option<Vec<UIMessage>>,
    pub generate_message_id: Option<IdGeneratorFn>,
    pub on_step_end: Option<UIMessageStreamOnStepEndCallback>,
    /// Deprecated: use `on_step_end`.
    pub on_step_finish: Option<UIMessageStreamOnStepEndCallback>,
    pub on_end: Option<UIMessageStreamOnEndCallback>,
    /// Deprecated: use `on_end`.
    pub on_finish: Option<UIMessageStreamOnEndCallback>,
    pub message_metadata: Option<MessageMetadataFn>,
    pub send_reasoning: Option<bool>,
    pub send_sources: Option<bool>,
    pub send_finish: Option<bool>,
    pub send_start: Option<bool>,
    pub on_error: Option<UIMessageStreamOnErrorFn>,
    pub redact_tool_errors: Option<bool>,
}

/// `getResponseUIMessageId({ originalMessages, responseMessageId })`.
pub fn get_response_ui_message_id(
    original_messages: Option<&[UIMessage]>,
    response_message_id: &IdGeneratorFn,
) -> Option<String> {
    // without original messages (no persistence), the client generates the
    // assistant message id.
    let original_messages = original_messages?;
    Some(match original_messages.last() {
        Some(last) if last.role == UIMessageRole::Assistant => last.id.clone(),
        _ => response_message_id(),
    })
}

#[cfg(test)]
thread_local! {
    /// Captured warnings (`globalThis.AI_SDK_LOG_WARNINGS = spy` in TS).
    pub(crate) static CAPTURED_WARNINGS: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

fn log_warning(message: &str) {
    #[cfg(test)]
    if CAPTURED_WARNINGS.with(|captured| {
        captured
            .borrow_mut()
            .as_mut()
            .map(|warnings| warnings.push(message.to_string()))
            .is_some()
    }) {
        return;
    }
    static LOGGED_BEFORE: AtomicBool = AtomicBool::new(false);
    // [rust] The `AI_SDK_LOG_WARNINGS` global is an environment variable.
    if std::env::var("AI_SDK_LOG_WARNINGS").is_ok_and(|v| v == "false") {
        return;
    }
    if !LOGGED_BEFORE.swap(true, Ordering::SeqCst) {
        eprintln!(
            "AI SDK Warning System: To turn off warning logging, set the AI_SDK_LOG_WARNINGS global to false."
        );
    }
    eprintln!("AI SDK Warning: {message}");
}

/// `toUIMessageStream({ stream, tools, ...options })`: stream parts as UI
/// message chunks, with response message id injection and the step and
/// stream end callbacks.
pub fn to_ui_message_stream(
    stream: BoxStream<'static, TextStreamPart>,
    tools: Option<ToolSet>,
    options: UIMessageStreamOptions,
) -> UIMessageStream {
    let on_error = options.on_error.clone();
    let response_message_id = options.generate_message_id.as_ref().and_then(|generate| {
        get_response_ui_message_id(options.original_messages.as_deref(), generate)
    });
    let outcome = Arc::new(Mutex::new(SourceOutcome::default()));
    let source_outcome = outcome.clone();
    let message_metadata = options.message_metadata.clone();
    let (send_reasoning, send_sources, send_start, send_finish, redact_tool_errors) = (
        options.send_reasoning,
        options.send_sources,
        options.send_start,
        options.send_finish,
        options.redact_tool_errors,
    );
    let chunk_response_id = response_message_id.clone();
    let chunk_on_error = on_error.clone();

    let chunks = async_stream::stream! {
        let mut stream = stream;
        // streamText follows each invalid tool call with a tool-error part;
        // the tool-input-error chunk already reports it, so the tool-error
        // is skipped.
        let mut invalid_tool_call_ids: HashSet<String> = HashSet::new();
        let mut mismatched_in_current_step: Option<String> = None;
        let mut mismatched_in_earlier_step: Option<String> = None;
        let mut warned_about_mismatch = false;
        while let Some(part) = stream.next().await {
            if let TextStreamPart::ToolCall(call) = &part
                && call.is_invalid()
            {
                invalid_tool_call_ids.insert(call.tool_call_id.clone());
            }
            let is_invalid_tool_call_error = matches!(
                &part,
                TextStreamPart::ToolError(error)
                    if error.provider_executed != Some(true)
                        && invalid_tool_call_ids.remove(&error.tool_call_id)
            );
            let metadata_value = message_metadata.as_ref().and_then(|f| f(&part));
            let chunk = if is_invalid_tool_call_error {
                None
            } else {
                to_ui_message_chunk(
                    &part,
                    &ToUIMessageChunkOptions {
                        tools: tools.as_ref(),
                        send_reasoning,
                        send_sources,
                        send_start,
                        send_finish,
                        on_error: chunk_on_error.clone(),
                        redact_tool_errors,
                        message_metadata: metadata_value.clone(),
                        response_message_id: chunk_response_id.clone(),
                    },
                )
            };

            match (&part, &chunk) {
                (TextStreamPart::StartStep { .. }, _) => {
                    // the model receives the tool errors of a step with the
                    // next step
                    if mismatched_in_earlier_step.is_none() {
                        mismatched_in_earlier_step = mismatched_in_current_step.take();
                    }
                    mismatched_in_current_step = None;
                }
                (
                    TextStreamPart::ToolError(_) | TextStreamPart::ToolCall(_),
                    Some(
                        UIMessageChunk::ToolOutputError { error_text, .. }
                        | UIMessageChunk::ToolInputError { error_text, .. },
                    ),
                ) => {
                    let (provider_executed, tool_call_id, error) = match &part {
                        TextStreamPart::ToolError(e) => {
                            (e.provider_executed, &e.tool_call_id, Some(e.error.clone()))
                        }
                        TextStreamPart::ToolCall(c) => {
                            (c.provider_executed, &c.tool_call_id, c.error.clone())
                        }
                        _ => unreachable!(),
                    };
                    if provider_executed != Some(true) {
                        let error = error.unwrap_or(ErrorValue::Value(JsonValue::Null));
                        if let Some(model_error_text) = get_model_tool_error_text(&error)
                            && *error_text != model_error_text
                            && mismatched_in_current_step.is_none()
                        {
                            mismatched_in_current_step = Some(tool_call_id.clone());
                        }
                    }
                }
                (_, Some(UIMessageChunk::ReasoningStart { .. })) => {
                    if let Some(tool_call_id) = &mismatched_in_earlier_step
                        && !warned_about_mismatch
                    {
                        warned_about_mismatch = true;
                        log_warning(&format!(
                            "The error text sent to the UI for tool call \"{tool_call_id}\" differs from the error text the model received, and reasoning from a later step was sent to the UI. Converting these UI messages back to model messages changes the tool result. Models that bind reasoning to the preceding messages (for example Anthropic models with preserved thinking) reject that history. See https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage#tool-error-text-and-message-history"
                        ));
                    }
                }
                _ => {}
            }

            let is_finish_step = matches!(part, TextStreamPart::FinishStep { .. });
            if let Some(chunk) = &chunk
                && !is_finish_step
            {
                yield chunk.clone();
            }
            // start and finish chunks already carry the metadata; other
            // parts get a separate message-metadata chunk
            if let Some(metadata) = metadata_value
                && !matches!(part, TextStreamPart::Start | TextStreamPart::Finish { .. })
            {
                yield UIMessageChunk::MessageMetadata { message_metadata: metadata };
            }
            // Apply step metadata before the boundary triggers a UI snapshot.
            if let Some(chunk) = chunk
                && is_finish_step
            {
                yield chunk;
            }

            let mut outcome = source_outcome.lock();
            match &part {
                TextStreamPart::Finish { .. } => outcome.set(UIMessageStreamOutcome::Completed),
                TextStreamPart::Abort { .. } => outcome.set(UIMessageStreamOutcome::Aborted),
                TextStreamPart::Error { error } => outcome.set(UIMessageStreamOutcome::Failed {
                    error: Some(error.clone()),
                }),
                _ => {}
            }
        }
    };

    let message_id =
        response_message_id.or_else(|| options.generate_message_id.as_ref().map(|f| f()));
    handle_ui_message_stream_finish(
        chunks.boxed(),
        HandleUIMessageStreamFinishOptions {
            message_id,
            original_messages: options.original_messages.unwrap_or_default(),
            on_step_end: options.on_step_end.or(options.on_step_finish),
            on_step_finish: None,
            on_end: options.on_end.or(options.on_finish),
            on_finish: None,
            on_error: on_error.unwrap_or_else(|| Arc::new(|_| DEFAULT_ERROR_TEXT.to_string())),
            get_outcome: Some(Arc::new(move || outcome.lock().outcome.clone())),
        },
    )
}

/// `toUIMessageStream`'s declared source outcome.
#[derive(Default)]
struct SourceOutcome {
    outcome: Option<UIMessageStreamOutcome>,
}

impl SourceOutcome {
    /// `setSourceOutcome(newOutcome)`.
    fn set(&mut self, new_outcome: UIMessageStreamOutcome) {
        let current = self
            .outcome
            .clone()
            .unwrap_or(UIMessageStreamOutcome::Unknown);
        if !matches!(
            current,
            UIMessageStreamOutcome::Completed | UIMessageStreamOutcome::Aborted
        ) && new_outcome != UIMessageStreamOutcome::Unknown
            && (current == UIMessageStreamOutcome::Unknown
                || !matches!(new_outcome, UIMessageStreamOutcome::Failed { .. }))
        {
            self.outcome = Some(new_outcome);
        }
    }
}

/// `getOutcome: () => UIMessageStreamOutcome`.
pub type GetOutcomeFn = Arc<dyn Fn() -> Option<UIMessageStreamOutcome> + Send + Sync>;

/// `handleUIMessageStreamFinish` options.
#[derive(Clone)]
pub struct HandleUIMessageStreamFinishOptions {
    /// The response message id. Without it no id is set.
    pub message_id: Option<String>,
    pub original_messages: Vec<UIMessage>,
    pub on_step_end: Option<UIMessageStreamOnStepEndCallback>,
    /// Deprecated: use `on_step_end`.
    pub on_step_finish: Option<UIMessageStreamOnStepEndCallback>,
    pub on_end: Option<UIMessageStreamOnEndCallback>,
    /// Deprecated: use `on_end`.
    pub on_finish: Option<UIMessageStreamOnEndCallback>,
    pub on_error: UIMessageStreamOnErrorFn,
    /// The operation-level outcome declared by the stream owner (`None`
    /// is `unknown`).
    pub get_outcome: Option<GetOutcomeFn>,
}

/// The state `handleUIMessageStreamFinish` keeps for its callbacks.
struct FinishState {
    state: StreamingUIMessageState,
    last_message_id: Option<String>,
    original_messages: Vec<UIMessage>,
    is_aborted: bool,
    processing_error: Option<UIMessageStreamError>,
    on_end: Option<UIMessageStreamOnEndCallback>,
    get_outcome: Option<GetOutcomeFn>,
}

impl FinishState {
    fn messages(&self) -> Vec<UIMessage> {
        let is_continuation = Some(&self.state.message.id) == self.last_message_id.as_ref();
        let mut messages = if is_continuation {
            self.original_messages[..self.original_messages.len().saturating_sub(1)].to_vec()
        } else {
            self.original_messages.clone()
        };
        messages.push(self.state.message.clone());
        messages
    }

    fn end_event(&self, is_cancelled: bool) -> UIMessageStreamOnEndEvent {
        let is_continuation = Some(&self.state.message.id) == self.last_message_id.as_ref();
        let declared = self
            .get_outcome
            .as_ref()
            .and_then(|f| f())
            .unwrap_or(UIMessageStreamOutcome::Unknown);
        let outcome = if let Some(error) = &self.processing_error {
            UIMessageStreamOutcome::Failed {
                error: Some(ErrorValue::from_box(Box::new(error.clone()))),
            }
        } else if declared == UIMessageStreamOutcome::Unknown && self.is_aborted {
            UIMessageStreamOutcome::Aborted
        } else {
            declared
        };
        let is_consumer_cancellation = is_cancelled && outcome == UIMessageStreamOutcome::Unknown;
        UIMessageStreamOnEndEvent {
            is_aborted: self.is_aborted || outcome == UIMessageStreamOutcome::Aborted,
            is_cancelled: is_consumer_cancellation.then_some(true),
            is_continuation,
            outcome,
            response_message: self.state.message.clone(),
            messages: self.messages(),
            finish_reason: self.state.finish_reason,
        }
    }
}

/// Calls `onEnd` once: at the end of the stream, or (as a cancellation)
/// when the consumer drops the stream first.
struct EndGuard {
    shared: Arc<Mutex<FinishState>>,
    called: Arc<AtomicBool>,
}

impl EndGuard {
    async fn call(&self, is_cancelled: bool) {
        if self.called.swap(true, Ordering::SeqCst) {
            return;
        }
        let (on_end, event) = {
            let shared = self.shared.lock();
            (shared.on_end.clone(), shared.end_event(is_cancelled))
        };
        if let Some(on_end) = on_end {
            on_end(event).await;
        }
    }
}

impl Drop for EndGuard {
    fn drop(&mut self) {
        // `cancel()`: the consumer stopped reading (or processing failed).
        if self.called.swap(true, Ordering::SeqCst) {
            return;
        }
        let (on_end, event) = {
            let shared = self.shared.lock();
            (shared.on_end.clone(), shared.end_event(true))
        };
        if let (Some(on_end), Ok(handle)) = (on_end, tokio::runtime::Handle::try_current()) {
            handle.spawn(on_end(event));
        }
    }
}

/// `handleUIMessageStreamFinish({ stream, messageId, originalMessages,
/// onStepEnd, onStepFinish, onEnd, onFinish, onError, getOutcome })`.
pub fn handle_ui_message_stream_finish(
    stream: BoxStream<'static, UIMessageChunk>,
    options: HandleUIMessageStreamFinishOptions,
) -> UIMessageStream {
    // last message is only relevant for assistant messages
    let mut message_id = options.message_id;
    let last_message = options
        .original_messages
        .last()
        .filter(|m| m.role == UIMessageRole::Assistant)
        .cloned();
    if let Some(last) = &last_message {
        // appending to the last message, so it keeps its id
        message_id = Some(last.id.clone());
    }

    let inject_id = message_id.clone();
    let inject = move |chunk: UIMessageChunk| match chunk {
        // inject the message id into a start chunk without one (persistence)
        UIMessageChunk::Start {
            message_id: None,
            message_metadata,
        } if inject_id.is_some() => UIMessageChunk::Start {
            message_id: inject_id.clone(),
            message_metadata,
        },
        other => other,
    };

    let on_step_end = options.on_step_end.or(options.on_step_finish);
    let on_end = options.on_end.or(options.on_finish);
    if on_end.is_none() && on_step_end.is_none() {
        return stream.map(move |chunk| Ok(inject(chunk))).boxed();
    }

    let shared = Arc::new(Mutex::new(FinishState {
        state: create_streaming_ui_message_state(
            last_message.clone(),
            message_id.as_deref().unwrap_or_default(),
        ),
        last_message_id: last_message.map(|m| m.id),
        original_messages: options.original_messages,
        is_aborted: false,
        processing_error: None,
        on_end,
        get_outcome: options.get_outcome,
    }));
    let on_error = options.on_error;

    async_stream::stream! {
        let guard = EndGuard {
            shared: shared.clone(),
            called: Arc::new(AtomicBool::new(false)),
        };
        let mut stream = stream;
        while let Some(chunk) = stream.next().await {
            let chunk = inject(chunk);
            let applied = {
                let mut finish = shared.lock();
                if matches!(chunk, UIMessageChunk::Abort { .. }) {
                    finish.is_aborted = true;
                }
                let on_error = on_error.clone();
                let result = finish
                    .state
                    .apply_chunk(&chunk, &move |error| {
                        on_error(error);
                    });
                if let Err(error) = &result {
                    finish.processing_error = Some(error.clone());
                }
                result
            };
            if let Err(error) = applied {
                // [rust] The stream errors; `cancel()` reports the failure.
                guard.call(true).await;
                yield Err(error);
                return;
            }
            if matches!(chunk, UIMessageChunk::FinishStep)
                && let Some(on_step_end) = &on_step_end
            {
                let event = {
                    let finish = shared.lock();
                    UIMessageStreamOnStepEndEvent {
                        is_continuation: Some(&finish.state.message.id)
                            == finish.last_message_id.as_ref(),
                        response_message: finish.state.message.clone(),
                        messages: finish.messages(),
                    }
                };
                on_step_end(event).await;
            }
            yield Ok(chunk);
        }
        // `flush()`
        guard.call(false).await;
    }
    .boxed()
}

#[cfg(test)]
#[path = "to_ui_message_stream_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "handle_ui_message_stream_finish_tests.rs"]
mod finish_tests;
