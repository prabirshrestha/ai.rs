//! Port of `agent/internal/harness-stream-text-result.ts`.
//!
//! [rust] The TS class implements `StreamTextResult` over a `ReadableStream`
//! that every accessor tees. Here the driver appends parts to a shared log,
//! and every stream accessor replays the log from the start, which is what a
//! fresh tee of the untouched base stream sees. The `PromiseLike` accessors
//! are async methods backed by [`DelayedPromise`]s.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use futures::StreamExt;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use tokio::sync::watch;

use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::provider::{JsonValue, LanguageModelV4FinishReason, LanguageModelV4Usage};
use crate::harness::sdk::content::{
    CallWarning, ContentPart, Context, FinishReason, LanguageModelRequestMetadata,
    LanguageModelResponseMetadata, ProviderMetadata, ReasoningOutput, StepModel, StepResult,
    StepResultPerformance, TextContent, ToolsContext, TypedToolCall, TypedToolResult,
};
use crate::harness::sdk::delayed_promise::DelayedPromise;
use crate::harness::sdk::error_value::{ErrorValue, SharedError};
use crate::harness::sdk::messages::{ModelMessage, ReasoningPart, ToolResultOutput};
use crate::harness::sdk::output::{OutputContext, SharedOutput};
use crate::harness::sdk::response_messages::to_response_messages;
use crate::harness::sdk::text_stream_part::{FinishStepResponse, TextStreamPart};
use crate::harness::sdk::tool::ToolSet;
use crate::harness::sdk::usage::{
    LanguageModelUsage, add_language_model_usage, as_language_model_usage,
    create_null_language_model_usage,
};
use crate::harness::utils::plain_error;

/// `EnrichedStreamPart<TOOLS, PARTIAL_OUTPUT>`.
#[derive(Debug, Clone)]
struct EnrichedStreamPart {
    part: TextStreamPart,
    partial_output: Option<JsonValue>,
}

/// The replayable part log that stands in for the tee'd base stream.
struct PartLog {
    parts: Mutex<(Vec<EnrichedStreamPart>, bool)>,
    len: watch::Sender<usize>,
}

impl PartLog {
    fn new() -> Self {
        Self {
            parts: Mutex::new((Vec::new(), false)),
            len: watch::channel(0).0,
        }
    }

    fn push(&self, parts: Vec<EnrichedStreamPart>) {
        if parts.is_empty() {
            return;
        }
        let len = {
            let mut log = self.parts.lock();
            if log.1 {
                return;
            }
            log.0.extend(parts);
            log.0.len()
        };
        self.len.send_replace(len);
    }

    fn close(&self) {
        let len = {
            let mut log = self.parts.lock();
            log.1 = true;
            log.0.len()
        };
        // A send always wakes readers, even when the length is unchanged.
        self.len.send_replace(len);
    }

    fn reader(self: &Arc<Self>) -> BoxStream<'static, EnrichedStreamPart> {
        let log = self.clone();
        let rx = self.len.subscribe();
        futures::stream::unfold((log, rx, 0usize), |(log, mut rx, index)| async move {
            loop {
                let (next, closed) = {
                    let parts = log.parts.lock();
                    (parts.0.get(index).cloned(), parts.1)
                };
                if let Some(part) = next {
                    return Some((part, (log, rx, index + 1)));
                }
                if closed {
                    return None;
                }
                rx.borrow_and_update();
                let now = {
                    let parts = log.parts.lock();
                    (parts.0.len() > index, parts.1)
                };
                if now.0 || now.1 {
                    continue;
                }
                if rx.changed().await.is_err() {
                    return None;
                }
            }
        })
        .boxed()
    }
}

/// `createOutputTransformStream(output)`: coalesces the first text block's
/// deltas into chunks that each carry a new partial output.
struct OutputTransform {
    output: SharedOutput,
    first_text_chunk_id: Option<String>,
    text: String,
    text_chunk: String,
    text_provider_metadata: Option<ProviderMetadata>,
    last_published_value: String,
}

impl OutputTransform {
    fn new(output: SharedOutput) -> Self {
        Self {
            output,
            first_text_chunk_id: None,
            text: String::new(),
            text_chunk: String::new(),
            text_provider_metadata: None,
            last_published_value: String::new(),
        }
    }

    fn publish_text_chunk(
        &mut self,
        out: &mut Vec<EnrichedStreamPart>,
        partial_output: Option<JsonValue>,
    ) {
        out.push(EnrichedStreamPart {
            part: TextStreamPart::TextDelta {
                id: self.first_text_chunk_id.clone().unwrap_or_default(),
                text: std::mem::take(&mut self.text_chunk),
                provider_metadata: self.text_provider_metadata.clone(),
            },
            partial_output,
        });
    }

    fn transform(&mut self, chunk: TextStreamPart) -> Vec<EnrichedStreamPart> {
        let mut out = Vec::new();
        let pass = |part: TextStreamPart| EnrichedStreamPart {
            part,
            partial_output: None,
        };
        if matches!(chunk, TextStreamPart::FinishStep { .. }) && !self.text_chunk.is_empty() {
            self.publish_text_chunk(&mut out, None);
        }
        let id = match &chunk {
            TextStreamPart::TextDelta { id, .. }
            | TextStreamPart::TextStart { id, .. }
            | TextStreamPart::TextEnd { id, .. } => id.clone(),
            _ => {
                out.push(pass(chunk));
                return out;
            }
        };
        match &self.first_text_chunk_id {
            None => self.first_text_chunk_id = Some(id),
            Some(first) if *first != id => {
                out.push(pass(chunk));
                return out;
            }
            Some(_) => {}
        }
        match chunk {
            TextStreamPart::TextStart { .. } => {
                out.push(pass(chunk));
            }
            TextStreamPart::TextEnd { .. } => {
                if !self.text_chunk.is_empty() {
                    self.publish_text_chunk(&mut out, None);
                }
                out.push(pass(chunk));
            }
            TextStreamPart::TextDelta {
                ref text,
                ref provider_metadata,
                ..
            } => {
                self.text.push_str(text);
                self.text_chunk.push_str(text);
                if provider_metadata.is_some() {
                    self.text_provider_metadata = provider_metadata.clone();
                }
                if text.is_empty() && provider_metadata.is_some() {
                    out.push(pass(chunk));
                    return out;
                }
                let Some(result) = self.output.parse_partial_output(&self.text) else {
                    return out;
                };
                let current_value = match &result.partial {
                    JsonValue::String(s) => s.clone(),
                    other => other.to_string(),
                };
                if current_value != self.last_published_value {
                    self.publish_text_chunk(&mut out, Some(result.partial));
                    self.last_published_value = current_value;
                }
            }
            _ => unreachable!("only text parts reach here"),
        }
        out
    }
}

/// `createEmptyPerformance()`.
fn create_empty_performance() -> StepResultPerformance {
    StepResultPerformance::default()
}

/// `notSupportedYet(feature)`.
fn not_supported_yet(feature: &str) -> BoxError {
    plain_error(format!(
        "HarnessAgent: {feature} is not implemented yet. Track the foundation review for follow-up."
    ))
}

#[derive(Default)]
struct Promises {
    content: DelayedPromise<Vec<ContentPart>>,
    text: DelayedPromise<String>,
    reasoning: DelayedPromise<Vec<ReasoningPart>>,
    reasoning_text: DelayedPromise<Option<String>>,
    files: DelayedPromise<Vec<JsonValue>>,
    sources: DelayedPromise<Vec<JsonValue>>,
    tool_calls: DelayedPromise<Vec<TypedToolCall>>,
    static_tool_calls: DelayedPromise<Vec<TypedToolCall>>,
    dynamic_tool_calls: DelayedPromise<Vec<TypedToolCall>>,
    tool_results: DelayedPromise<Vec<TypedToolResult>>,
    static_tool_results: DelayedPromise<Vec<TypedToolResult>>,
    dynamic_tool_results: DelayedPromise<Vec<TypedToolResult>>,
    finish_reason: DelayedPromise<FinishReason>,
    raw_finish_reason: DelayedPromise<Option<String>>,
    usage: DelayedPromise<LanguageModelUsage>,
    warnings: DelayedPromise<Option<Vec<CallWarning>>>,
    steps: DelayedPromise<Vec<StepResult>>,
    final_step: DelayedPromise<StepResult>,
    request: DelayedPromise<LanguageModelRequestMetadata>,
    response: DelayedPromise<LanguageModelResponseMetadata>,
    response_messages: DelayedPromise<Vec<ModelMessage>>,
    provider_metadata: DelayedPromise<Option<ProviderMetadata>>,
}

impl Promises {
    fn reject_all(&self, error: &SharedError) {
        self.content.reject(error.clone());
        self.text.reject(error.clone());
        self.reasoning.reject(error.clone());
        self.reasoning_text.reject(error.clone());
        self.files.reject(error.clone());
        self.sources.reject(error.clone());
        self.tool_calls.reject(error.clone());
        self.static_tool_calls.reject(error.clone());
        self.dynamic_tool_calls.reject(error.clone());
        self.tool_results.reject(error.clone());
        self.static_tool_results.reject(error.clone());
        self.dynamic_tool_results.reject(error.clone());
        self.finish_reason.reject(error.clone());
        self.raw_finish_reason.reject(error.clone());
        self.usage.reject(error.clone());
        self.warnings.reject(error.clone());
        self.steps.reject(error.clone());
        self.final_step.reject(error.clone());
        self.request.reject(error.clone());
        self.response.reject(error.clone());
        self.response_messages.reject(error.clone());
        self.provider_metadata.reject(error.clone());
    }
}

struct WriterState {
    output_transform: Option<OutputTransform>,
    steps_buffer: Vec<StepResult>,
    current_step_content: Vec<ContentPart>,
    /// Index into `current_step_content` of each open text block.
    current_step_text_content: HashMap<String, usize>,
    current_step_reasoning_content: HashMap<String, usize>,
    current_step_warnings: Vec<CallWarning>,
    step_number: u32,
    step_started: bool,
    tool_model_outputs: HashMap<String, ToolResultOutput>,
    model_id: String,
    accumulated_usage: LanguageModelUsage,
    final_provider_metadata: Option<ProviderMetadata>,
    final_finish_reason: FinishReason,
    final_raw_finish_reason: Option<String>,
    aggregate_warnings: Vec<CallWarning>,
    settled: bool,
}

struct Inner {
    promises: Promises,
    log: Arc<PartLog>,
    state: Mutex<WriterState>,
    tools: ToolSet,
    runtime_context: Context,
    tools_context: ToolsContext,
    provider_name: String,
    call_id: String,
    output_specification: Option<SharedOutput>,
}

/// `HarnessStreamTextResult` constructor options.
#[derive(Clone)]
pub struct HarnessStreamTextResultOptions {
    pub tools: ToolSet,
    pub runtime_context: Context,
    pub tools_context: ToolsContext,
    pub harness_id: String,
    pub call_id: String,
    pub model_id: String,
    pub output: Option<SharedOutput>,
}

/// Concrete `StreamTextResult` backed by a single harness prompt turn.
///
/// Each `finish-step` boundary in the driver becomes a `StepResult`; the
/// accessors settle when the driver calls [`finish`](Self::finish),
/// [`fail`](Self::fail) or [`abort`](Self::abort).
///
/// The Node.js response helpers (`pipeUIMessageStreamToResponse`,
/// `pipeTextStreamToResponse`, `toTextStreamResponse`) are not implemented
/// yet and return an error.
#[derive(Clone)]
pub struct HarnessStreamTextResult {
    inner: Arc<Inner>,
}

impl HarnessStreamTextResult {
    pub fn new(options: HarnessStreamTextResultOptions) -> Self {
        let log = Arc::new(PartLog::new());
        let inner = Inner {
            promises: Promises::default(),
            log,
            state: Mutex::new(WriterState {
                output_transform: options.output.clone().map(OutputTransform::new),
                steps_buffer: Vec::new(),
                current_step_content: Vec::new(),
                current_step_text_content: HashMap::new(),
                current_step_reasoning_content: HashMap::new(),
                current_step_warnings: Vec::new(),
                step_number: 0,
                step_started: false,
                tool_model_outputs: HashMap::new(),
                model_id: options.model_id,
                accumulated_usage: create_null_language_model_usage(),
                final_provider_metadata: None,
                final_finish_reason: FinishReason::Other,
                final_raw_finish_reason: None,
                aggregate_warnings: Vec::new(),
                settled: false,
            }),
            tools: options.tools,
            runtime_context: options.runtime_context,
            tools_context: options.tools_context,
            provider_name: format!("harness:{}", options.harness_id),
            call_id: options.call_id,
            output_specification: options.output,
        };
        let result = Self {
            inner: Arc::new(inner),
        };
        // The message-level start event comes first, mirroring `streamText`.
        {
            let mut state = result.inner.state.lock();
            result.write(&mut state, TextStreamPart::Start);
        }
        result
    }

    /// Push one part through the output transform into the log.
    fn write(&self, state: &mut WriterState, part: TextStreamPart) {
        let parts = match &mut state.output_transform {
            Some(transform) => transform.transform(part),
            None => vec![EnrichedStreamPart {
                part,
                partial_output: None,
            }],
        };
        self.inner.log.push(parts);
    }

    // ─── Writer-side methods used by the driver ────────────────────────

    /// `setToolModelOutput({ toolCallId, output })`.
    pub fn set_tool_model_output(&self, tool_call_id: &str, output: ToolResultOutput) {
        self.inner
            .state
            .lock()
            .tool_model_outputs
            .insert(tool_call_id.to_string(), output);
    }

    /// Push a translated part into `fullStream` and accumulate it into the
    /// current step's content.
    pub fn enqueue(&self, part: TextStreamPart) {
        let mut state = self.inner.state.lock();
        self.start_step(&mut state);
        Self::append_to_current_step_content(&mut state, &part);
        self.write(&mut state, part);
    }

    /// `setModelId(modelId)`.
    pub fn set_model_id(&self, model_id: impl Into<String>) {
        let mut state = self.inner.state.lock();
        if state.steps_buffer.is_empty() {
            state.model_id = model_id.into();
        }
    }

    /// `addWarnings(warnings)`.
    pub fn add_warnings(&self, warnings: &[CallWarning]) {
        self.inner
            .state
            .lock()
            .current_step_warnings
            .extend_from_slice(warnings);
    }

    /// Push a continuation input without attributing it to a model step.
    pub fn enqueue_continuation(&self, part: TextStreamPart) {
        let mut state = self.inner.state.lock();
        self.write(&mut state, part);
    }

    /// Push adapter-specific raw data without attributing it to a step.
    pub fn enqueue_raw(&self, raw_value: JsonValue) {
        let mut state = self.inner.state.lock();
        self.write(&mut state, TextStreamPart::Raw { raw_value });
    }

    /// Drop content replayed while a suspended host-input pause closes its
    /// already-recorded model step.
    pub fn discard_current_step_content(&self) {
        let mut state = self.inner.state.lock();
        state.current_step_content.clear();
        state.current_step_text_content.clear();
        state.current_step_reasoning_content.clear();
        state.current_step_warnings.clear();
        state.step_started = false;
    }

    #[allow(clippy::too_many_arguments)]
    fn new_step_result(
        &self,
        step_number: u32,
        model_id: &str,
        content: Vec<ContentPart>,
        finish_reason: FinishReason,
        raw_finish_reason: Option<String>,
        usage: LanguageModelUsage,
        warnings: Option<Vec<CallWarning>>,
        provider_metadata: Option<ProviderMetadata>,
    ) -> StepResult {
        StepResult {
            call_id: self.inner.call_id.clone(),
            step_number,
            model: StepModel {
                provider: self.inner.provider_name.clone(),
                model_id: model_id.to_string(),
            },
            tools_context: self.inner.tools_context.clone(),
            runtime_context: self.inner.runtime_context.clone(),
            content,
            finish_reason,
            raw_finish_reason,
            usage,
            performance: create_empty_performance(),
            warnings,
            request: LanguageModelRequestMetadata::default(),
            response: LanguageModelResponseMetadata {
                id: format!("{}-{step_number}", self.inner.call_id),
                timestamp: SystemTime::now(),
                model_id: model_id.to_string(),
                headers: None,
                messages: Vec::new(),
                body: None,
            },
            provider_metadata,
        }
    }

    /// Mark the end of a step and record its `StepResult`.
    pub fn finish_step(
        &self,
        finish_reason: &LanguageModelV4FinishReason,
        usage: &LanguageModelV4Usage,
        provider_metadata: Option<ProviderMetadata>,
    ) -> StepResult {
        let mut state = self.inner.state.lock();
        self.start_step(&mut state);

        let normalized_usage = as_language_model_usage(usage);
        let unified = finish_reason.unified;
        let raw = finish_reason.raw.clone();
        let warnings =
            (!state.current_step_warnings.is_empty()).then(|| state.current_step_warnings.clone());
        let step = self.new_step_result(
            state.step_number,
            &state.model_id.clone(),
            std::mem::take(&mut state.current_step_content),
            unified,
            raw.clone(),
            normalized_usage.clone(),
            warnings,
            provider_metadata.clone(),
        );
        state.steps_buffer.push(step.clone());

        self.write(
            &mut state,
            TextStreamPart::FinishStep {
                response: FinishStepResponse::from(&step.response),
                usage: normalized_usage.clone(),
                performance: create_empty_performance(),
                finish_reason: unified,
                raw_finish_reason: raw.clone(),
                provider_metadata: provider_metadata.clone(),
            },
        );

        state.accumulated_usage =
            add_language_model_usage(&state.accumulated_usage, &normalized_usage);
        state.final_finish_reason = unified;
        state.final_raw_finish_reason = raw;
        state.final_provider_metadata = provider_metadata;
        if !state.current_step_warnings.is_empty() {
            let warnings = state.current_step_warnings.clone();
            state.aggregate_warnings.extend(warnings);
        }

        state.step_number += 1;
        state.current_step_content.clear();
        state.current_step_text_content.clear();
        state.current_step_reasoning_content.clear();
        state.current_step_warnings.clear();
        state.step_started = false;

        step
    }

    fn start_step(&self, state: &mut WriterState) {
        if state.step_started {
            return;
        }
        state.step_started = true;
        let warnings = state.current_step_warnings.clone();
        self.write(
            state,
            TextStreamPart::StartStep {
                request: LanguageModelRequestMetadata::default(),
                warnings,
            },
        );
    }

    /// Resolve every delayed promise and close `fullStream`. Idempotent.
    pub async fn finish(
        &self,
        input: Option<(
            &LanguageModelV4FinishReason,
            &LanguageModelV4Usage,
            Option<ProviderMetadata>,
        )>,
    ) {
        let (steps, final_step, model_outputs) = {
            let mut state = self.inner.state.lock();
            if state.settled {
                return;
            }
            // Trailing content that no `finish-step` captured means an
            // invalid harness stream.
            if !state.current_step_content.is_empty() {
                drop(state);
                self.fail(ErrorValue::from_box(plain_error(
                    "HarnessAgent: received terminal finish with unclosed step content. Harness adapters must emit `finish-step` before `finish`.",
                )));
                return;
            }
            if let Some((finish_reason, total_usage, provider_metadata)) = input {
                state.final_finish_reason = finish_reason.unified;
                state.final_raw_finish_reason = finish_reason.raw.clone();
                state.final_provider_metadata = provider_metadata;
                state.accumulated_usage = as_language_model_usage(total_usage);
            }
            state.settled = true;
            let final_step = match state.steps_buffer.last() {
                Some(step) => step.clone(),
                None => self.new_step_result(
                    0,
                    &state.model_id,
                    Vec::new(),
                    state.final_finish_reason,
                    state.final_raw_finish_reason.clone(),
                    create_null_language_model_usage(),
                    None,
                    None,
                ),
            };
            (
                state.steps_buffer.clone(),
                final_step,
                state.tool_model_outputs.clone(),
            )
        };

        let p = &self.inner.promises;
        p.content
            .resolve(steps.iter().flat_map(|s| s.content.clone()).collect());
        p.text.resolve(final_step.text());
        p.reasoning.resolve(final_step.reasoning());
        p.reasoning_text.resolve(final_step.reasoning_text());
        p.files.resolve(Vec::new());
        p.sources.resolve(Vec::new());
        p.tool_calls
            .resolve(steps.iter().flat_map(|s| s.tool_calls()).collect());
        p.static_tool_calls
            .resolve(steps.iter().flat_map(|s| s.static_tool_calls()).collect());
        p.dynamic_tool_calls
            .resolve(steps.iter().flat_map(|s| s.dynamic_tool_calls()).collect());
        p.tool_results
            .resolve(steps.iter().flat_map(|s| s.tool_results()).collect());
        p.static_tool_results
            .resolve(steps.iter().flat_map(|s| s.static_tool_results()).collect());
        p.dynamic_tool_results.resolve(
            steps
                .iter()
                .flat_map(|s| s.dynamic_tool_results())
                .collect(),
        );
        let (finish_reason, raw_finish_reason, usage, warnings, provider_metadata) = {
            let state = self.inner.state.lock();
            (
                state.final_finish_reason,
                state.final_raw_finish_reason.clone(),
                state.accumulated_usage.clone(),
                (!state.aggregate_warnings.is_empty()).then(|| state.aggregate_warnings.clone()),
                state.final_provider_metadata.clone(),
            )
        };
        p.finish_reason.resolve(finish_reason);
        p.raw_finish_reason.resolve(raw_finish_reason.clone());
        p.usage.resolve(usage.clone());
        p.warnings.resolve(warnings);
        p.steps.resolve(steps.clone());
        p.final_step.resolve(final_step.clone());
        p.request.resolve(final_step.request.clone());
        p.response.resolve(final_step.response.clone());
        p.provider_metadata.resolve(provider_metadata.clone());

        let mut response_messages = Vec::new();
        for step in &steps {
            match to_response_messages(&step.content, Some(&self.inner.tools), Some(&model_outputs))
                .await
            {
                Ok(messages) => response_messages.extend(messages),
                Err(error) => {
                    // [rust] TS leaves the promise to reject asynchronously;
                    // reject it and still close the stream.
                    p.response_messages
                        .reject(ErrorValue::from_box(error).to_shared());
                    break;
                }
            }
        }
        p.response_messages.resolve(response_messages);

        let mut state = self.inner.state.lock();
        self.write(
            &mut state,
            TextStreamPart::Finish {
                finish_reason,
                raw_finish_reason,
                total_usage: usage,
                provider_metadata,
            },
        );
        drop(state);
        self.inner.log.close();
    }

    /// Settle the turn as user-aborted: emit a final `abort` part and close
    /// the stream. The accessors reject with `error`. Idempotent.
    pub fn abort(&self, error: ErrorValue, reason: Option<String>) {
        {
            let mut state = self.inner.state.lock();
            if state.settled {
                return;
            }
            state.settled = true;
            self.write(&mut state, TextStreamPart::Abort { reason });
        }
        self.inner.log.close();
        self.inner.promises.reject_all(&error.to_shared());
    }

    /// Surface a fatal error as an `error` part and reject every accessor.
    /// Idempotent.
    pub fn fail(&self, error: ErrorValue) {
        {
            let mut state = self.inner.state.lock();
            if state.settled {
                return;
            }
            state.settled = true;
            self.write(
                &mut state,
                TextStreamPart::Error {
                    error: error.clone(),
                },
            );
        }
        self.inner.log.close();
        self.inner.promises.reject_all(&error.to_shared());
    }

    fn append_to_current_step_content(state: &mut WriterState, part: &TextStreamPart) {
        match part {
            TextStreamPart::TextStart {
                id,
                provider_metadata,
            } => {
                state
                    .current_step_content
                    .push(ContentPart::Text(TextContent {
                        text: String::new(),
                        citations: None,
                        provider_metadata: provider_metadata.clone(),
                    }));
                let index = state.current_step_content.len() - 1;
                state.current_step_text_content.insert(id.clone(), index);
            }
            TextStreamPart::TextDelta {
                id,
                text,
                provider_metadata,
            } => {
                let index = match state.current_step_text_content.get(id) {
                    Some(index) => *index,
                    None => {
                        state
                            .current_step_content
                            .push(ContentPart::Text(TextContent {
                                text: String::new(),
                                citations: None,
                                provider_metadata: provider_metadata.clone(),
                            }));
                        let index = state.current_step_content.len() - 1;
                        state.current_step_text_content.insert(id.clone(), index);
                        index
                    }
                };
                if let Some(ContentPart::Text(content)) = state.current_step_content.get_mut(index)
                {
                    content.text.push_str(text);
                    if provider_metadata.is_some() {
                        content.provider_metadata = provider_metadata.clone();
                    }
                }
            }
            TextStreamPart::TextEnd {
                id,
                citations,
                provider_metadata,
            } => {
                if let Some(index) = state.current_step_text_content.remove(id)
                    && let Some(ContentPart::Text(content)) =
                        state.current_step_content.get_mut(index)
                {
                    if provider_metadata.is_some() {
                        content.provider_metadata = provider_metadata.clone();
                    }
                    if citations.is_some() {
                        content.citations = citations.clone();
                    }
                }
            }
            TextStreamPart::ReasoningStart {
                id,
                provider_metadata,
            } => {
                state
                    .current_step_content
                    .push(ContentPart::Reasoning(ReasoningOutput {
                        text: String::new(),
                        provider_metadata: provider_metadata.clone(),
                    }));
                let index = state.current_step_content.len() - 1;
                state
                    .current_step_reasoning_content
                    .insert(id.clone(), index);
            }
            TextStreamPart::ReasoningDelta {
                id,
                text,
                provider_metadata,
            } => {
                let index = match state.current_step_reasoning_content.get(id) {
                    Some(index) => *index,
                    None => {
                        state
                            .current_step_content
                            .push(ContentPart::Reasoning(ReasoningOutput {
                                text: String::new(),
                                provider_metadata: provider_metadata.clone(),
                            }));
                        let index = state.current_step_content.len() - 1;
                        state
                            .current_step_reasoning_content
                            .insert(id.clone(), index);
                        index
                    }
                };
                if let Some(ContentPart::Reasoning(content)) =
                    state.current_step_content.get_mut(index)
                {
                    content.text.push_str(text);
                    if provider_metadata.is_some() {
                        content.provider_metadata = provider_metadata.clone();
                    }
                }
            }
            TextStreamPart::ReasoningEnd {
                id,
                provider_metadata,
            } => {
                if let Some(index) = state.current_step_reasoning_content.remove(id)
                    && let Some(ContentPart::Reasoning(content)) =
                        state.current_step_content.get_mut(index)
                    && provider_metadata.is_some()
                {
                    content.provider_metadata = provider_metadata.clone();
                }
            }
            TextStreamPart::ToolCall(c) => {
                state
                    .current_step_content
                    .push(ContentPart::ToolCall(c.clone()));
            }
            TextStreamPart::ToolApprovalRequest(r) => state
                .current_step_content
                .push(ContentPart::ToolApprovalRequest(r.clone())),
            TextStreamPart::ToolApprovalResponse(r) => state
                .current_step_content
                .push(ContentPart::ToolApprovalResponse(r.clone())),
            TextStreamPart::ToolResult(r) => {
                state
                    .current_step_content
                    .push(ContentPart::ToolResult(r.clone()));
            }
            TextStreamPart::ToolError(e) => {
                state
                    .current_step_content
                    .push(ContentPart::ToolError(e.clone()));
            }
            // Boundary, raw, error, and finish parts are not content parts.
            _ => {}
        }
    }

    // ─── Reader-side public surface (StreamTextResult contract) ────────

    /// `stream`: every part, replayed from the start.
    pub fn stream(&self) -> BoxStream<'static, TextStreamPart> {
        self.inner.log.reader().map(|p| p.part).boxed()
    }

    /// `fullStream`.
    pub fn full_stream(&self) -> BoxStream<'static, TextStreamPart> {
        self.stream()
    }

    /// `textStream`.
    pub fn text_stream(&self) -> BoxStream<'static, String> {
        self.stream()
            .filter_map(|part| async move {
                match part {
                    TextStreamPart::TextDelta { text, .. } => Some(text),
                    _ => None,
                }
            })
            .boxed()
    }

    /// `partialOutputStream`.
    pub fn partial_output_stream(&self) -> BoxStream<'static, JsonValue> {
        self.inner
            .log
            .reader()
            .filter_map(|p| async move { p.partial_output })
            .boxed()
    }

    /// `experimental_partialOutputStream`.
    pub fn experimental_partial_output_stream(&self) -> BoxStream<'static, JsonValue> {
        self.partial_output_stream()
    }

    /// `elementStream`. Errors when the output has no element transform.
    pub fn element_stream(&self) -> HarnessResult<BoxStream<'static, HarnessResult<JsonValue>>> {
        let transform = self
            .inner
            .output_specification
            .as_ref()
            .and_then(|o| o.create_element_stream_transform());
        let Some(mut transform) = transform else {
            return Err(not_supported_yet(&format!(
                "element streams in {} mode",
                self.inner
                    .output_specification
                    .as_ref()
                    .map(|o| o.name().to_string())
                    .unwrap_or_else(|| "text".into())
            )));
        };
        let reader = self.inner.log.reader();
        Ok(async_stream::stream! {
            let mut reader = reader;
            while let Some(part) = reader.next().await {
                match transform.transform(part.partial_output.as_ref()) {
                    Ok(elements) => {
                        for element in elements {
                            yield Ok(element);
                        }
                    }
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        }
        .boxed())
    }

    pub async fn content(&self) -> Result<Vec<ContentPart>, SharedError> {
        self.inner.promises.content.promise().await
    }
    pub async fn text(&self) -> Result<String, SharedError> {
        self.inner.promises.text.promise().await
    }
    pub async fn reasoning(&self) -> Result<Vec<ReasoningPart>, SharedError> {
        self.inner.promises.reasoning.promise().await
    }
    pub async fn reasoning_text(&self) -> Result<Option<String>, SharedError> {
        self.inner.promises.reasoning_text.promise().await
    }
    /// `files` ([rust] the harness never produces files: always empty).
    pub async fn files(&self) -> Result<Vec<JsonValue>, SharedError> {
        self.inner.promises.files.promise().await
    }
    /// `sources` ([rust] the harness never produces sources: always empty).
    pub async fn sources(&self) -> Result<Vec<JsonValue>, SharedError> {
        self.inner.promises.sources.promise().await
    }
    pub async fn tool_calls(&self) -> Result<Vec<TypedToolCall>, SharedError> {
        self.inner.promises.tool_calls.promise().await
    }
    pub async fn static_tool_calls(&self) -> Result<Vec<TypedToolCall>, SharedError> {
        self.inner.promises.static_tool_calls.promise().await
    }
    pub async fn dynamic_tool_calls(&self) -> Result<Vec<TypedToolCall>, SharedError> {
        self.inner.promises.dynamic_tool_calls.promise().await
    }
    pub async fn tool_results(&self) -> Result<Vec<TypedToolResult>, SharedError> {
        self.inner.promises.tool_results.promise().await
    }
    pub async fn static_tool_results(&self) -> Result<Vec<TypedToolResult>, SharedError> {
        self.inner.promises.static_tool_results.promise().await
    }
    pub async fn dynamic_tool_results(&self) -> Result<Vec<TypedToolResult>, SharedError> {
        self.inner.promises.dynamic_tool_results.promise().await
    }
    pub async fn finish_reason(&self) -> Result<FinishReason, SharedError> {
        self.inner.promises.finish_reason.promise().await
    }
    pub async fn raw_finish_reason(&self) -> Result<Option<String>, SharedError> {
        self.inner.promises.raw_finish_reason.promise().await
    }
    pub async fn usage(&self) -> Result<LanguageModelUsage, SharedError> {
        self.inner.promises.usage.promise().await
    }
    pub async fn total_usage(&self) -> Result<LanguageModelUsage, SharedError> {
        self.inner.promises.usage.promise().await
    }
    pub async fn warnings(&self) -> Result<Option<Vec<CallWarning>>, SharedError> {
        self.inner.promises.warnings.promise().await
    }
    pub async fn steps(&self) -> Result<Vec<StepResult>, SharedError> {
        self.inner.promises.steps.promise().await
    }
    pub async fn final_step(&self) -> Result<StepResult, SharedError> {
        self.inner.promises.final_step.promise().await
    }
    pub async fn request(&self) -> Result<LanguageModelRequestMetadata, SharedError> {
        self.inner.promises.request.promise().await
    }
    pub async fn response(&self) -> Result<LanguageModelResponseMetadata, SharedError> {
        self.inner.promises.response.promise().await
    }
    pub async fn response_messages(&self) -> Result<Vec<ModelMessage>, SharedError> {
        self.inner.promises.response_messages.promise().await
    }
    pub async fn provider_metadata(&self) -> Result<Option<ProviderMetadata>, SharedError> {
        self.inner.promises.provider_metadata.promise().await
    }

    /// `output`: the final step's text parsed by the output specification.
    pub async fn output(&self) -> Result<JsonValue, SharedError> {
        let step = self.final_step().await?;
        let Some(output) = &self.inner.output_specification else {
            return Err(ErrorValue::from_box(not_supported_yet("structured output")).to_shared());
        };
        output
            .parse_complete_output(
                &step.text(),
                OutputContext {
                    response: step.response.clone(),
                    usage: step.usage.clone(),
                    finish_reason: step.finish_reason,
                },
            )
            .map_err(|e| ErrorValue::from_box(e).to_shared())
    }

    /// `consumeStream()`.
    pub async fn consume_stream(&self) {
        let mut stream = self.full_stream();
        while stream.next().await.is_some() {}
    }

    /// `pipeUIMessageStreamToResponse()`: not implemented yet.
    pub fn pipe_ui_message_stream_to_response(&self) -> HarnessResult<()> {
        Err(not_supported_yet("pipeUIMessageStreamToResponse"))
    }

    /// `pipeTextStreamToResponse()`: not implemented yet.
    pub fn pipe_text_stream_to_response(&self) -> HarnessResult<()> {
        Err(not_supported_yet("pipeTextStreamToResponse"))
    }

    /// `toTextStreamResponse()`: not implemented yet.
    pub fn to_text_stream_response(&self) -> HarnessResult<()> {
        Err(not_supported_yet("toTextStreamResponse"))
    }

    /// The tools the turn ran with (`toUIMessageStream` needs them).
    pub fn tools(&self) -> &ToolSet {
        &self.inner.tools
    }
}
