//! Port of `providers/faux.ts`: a scripted provider for tests.
//!
//! Differences from Pi:
//! - Response factories receive owned copies of the context, options, a
//!   snapshot of the provider state and the model, and return a
//!   `Result`; an `Err` becomes an error event like a thrown error in Pi.
//! - `getModel()` / `getModel(id)` are [`FauxCore::get_model`] and
//!   [`FauxCore::get_model_by_id`].
//! - Text is chunked by `char`, not by UTF-16 code unit, so a chunk never
//!   splits a surrogate pair. Token estimates still count UTF-16 code units.
//! - Producers run on a spawned Tokio task; an unpaced chunk yields to the
//!   runtime where Pi awaits a microtask.

use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use parking_lot::Mutex;
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::lazy::panic_message;
use crate::auth::{ApiKeyAuth, ApiKeyAuthInput, AuthResult, ProviderAuth};
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::types::{
    AnyModel, AssistantContent, AssistantMessage, AssistantMessageEvent, BoxFuture, CacheRetention,
    DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, Message, Model, ModelCost,
    ModelInput, ModelInputLimits, ProviderResponse, ProviderStreams, ResponseHook,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, ToolCall,
    ToolResultMessage, TranscriptContext, Usage, UserContent, UserMessageContent,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::text::get_system_message_text;
use crate::utils::time::now_millis;
use crate::{Error, Result};

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

/// `FauxModelDefinition`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: Option<bool>,
    pub input: Option<Vec<ModelInput>>,
    pub input_limits: Option<ModelInputLimits>,
    pub cost: Option<ModelCost>,
    pub context_window: Option<u32>,
    pub max_tokens: Option<u32>,
}

impl FauxModelDefinition {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }
}

/// `FauxContentBlock`: text, thinking or tool call.
pub type FauxContentBlock = AssistantContent;

pub fn faux_text(text: impl Into<String>) -> FauxContentBlock {
    AssistantContent::Text(TextContent::new(text))
}

pub fn faux_thinking(thinking: impl Into<String>) -> FauxContentBlock {
    AssistantContent::Thinking(ThinkingContent {
        thinking: thinking.into(),
        thinking_signature: None,
        redacted: None,
    })
}

/// `fauxToolCall(name, arguments, { id })`. Without an id, a random one is used.
pub fn faux_tool_call(
    name: impl Into<String>,
    arguments: Value,
    id: Option<&str>,
) -> FauxContentBlock {
    AssistantContent::ToolCall(ToolCall {
        id: id.map_or_else(|| random_id("tool"), str::to_string),
        name: name.into(),
        arguments,
        thought_signature: None,
        namespace: None,
    })
}

/// `string | FauxContentBlock | FauxContentBlock[]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FauxContent(pub Vec<FauxContentBlock>);

impl From<&str> for FauxContent {
    fn from(text: &str) -> Self {
        Self(vec![faux_text(text)])
    }
}

impl From<String> for FauxContent {
    fn from(text: String) -> Self {
        Self(vec![faux_text(text)])
    }
}

impl From<FauxContentBlock> for FauxContent {
    fn from(block: FauxContentBlock) -> Self {
        Self(vec![block])
    }
}

impl From<Vec<FauxContentBlock>> for FauxContent {
    fn from(blocks: Vec<FauxContentBlock>) -> Self {
        Self(blocks)
    }
}

impl<const N: usize> From<[FauxContentBlock; N]> for FauxContent {
    fn from(blocks: [FauxContentBlock; N]) -> Self {
        Self(blocks.into())
    }
}

/// Options of `fauxAssistantMessage`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FauxMessageOptions {
    pub stop_reason: Option<StopReason>,
    pub deferred: Option<DeferredHandle>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<u64>,
}

fn message_for(
    content: Vec<AssistantContent>,
    api: &str,
    provider: &str,
    model_id: &str,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: api.to_string(),
        provider: provider.to_string(),
        model: model_id.to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_millis(),
    }
}

/// `fauxAssistantMessage(content, options)`.
pub fn faux_assistant_message(
    content: impl Into<FauxContent>,
    options: FauxMessageOptions,
) -> AssistantMessage {
    let mut message = message_for(
        content.into().0,
        DEFAULT_API,
        DEFAULT_PROVIDER,
        DEFAULT_MODEL_ID,
        options.stop_reason.unwrap_or(StopReason::Stop),
    );
    message.deferred = options.deferred;
    message.error_message = options.error_message;
    message.response_id = options.response_id;
    if let Some(timestamp) = options.timestamp {
        message.timestamp = timestamp;
    }
    message
}

/// `FauxProviderState`. Read it with [`FauxCore::state`], which returns a snapshot.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FauxProviderState {
    pub call_count: usize,
    pub deferred_fetch_count: usize,
    pub cancelled_deferred: Vec<DeferredHandle>,
}

/// `FauxResponseFactory`: `(context, options, state, model) => AssistantMessage`.
pub type FauxResponseFactory = Arc<
    dyn Fn(
            TranscriptContext,
            SimpleStreamOptions,
            FauxProviderState,
            Model,
        ) -> BoxFuture<Result<AssistantMessage>>
        + Send
        + Sync,
>;

/// `FauxResponseStep`: a scripted message or a factory.
#[derive(Clone)]
pub enum FauxResponseStep {
    Message(Box<AssistantMessage>),
    Factory(FauxResponseFactory),
}

impl FauxResponseStep {
    /// A synchronous factory.
    pub fn factory(
        factory: impl Fn(
            TranscriptContext,
            SimpleStreamOptions,
            FauxProviderState,
            Model,
        ) -> Result<AssistantMessage>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::Factory(Arc::new(move |context, options, state, model| {
            let result = factory(context, options, state, model);
            Box::pin(async move { result })
        }))
    }

    /// An asynchronous factory.
    pub fn async_factory<F>(
        factory: impl Fn(TranscriptContext, SimpleStreamOptions, FauxProviderState, Model) -> F
        + Send
        + Sync
        + 'static,
    ) -> Self
    where
        F: Future<Output = Result<AssistantMessage>> + Send + 'static,
    {
        Self::Factory(Arc::new(move |context, options, state, model| {
            Box::pin(factory(context, options, state, model))
        }))
    }
}

impl From<AssistantMessage> for FauxResponseStep {
    fn from(message: AssistantMessage) -> Self {
        Self::Message(Box::new(message))
    }
}

impl std::fmt::Debug for FauxResponseStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(message) => f.debug_tuple("Message").field(message).finish(),
            Self::Factory(_) => f.write_str("Factory(..)"),
        }
    }
}

/// `deferred` of [`RegisterFauxProviderOptions`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FauxDeferredOptions {
    /// Number of fetches that return the original handle before the scripted
    /// response becomes ready.
    pub pending_fetches: Option<u32>,
    pub poll_after_ms: Option<u64>,
}

/// `tokenSize` of [`RegisterFauxProviderOptions`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FauxTokenSize {
    pub min: Option<usize>,
    pub max: Option<usize>,
}

/// `RegisterFauxProviderOptions`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegisterFauxProviderOptions {
    pub api: Option<String>,
    pub provider: Option<String>,
    pub models: Vec<FauxModelDefinition>,
    pub deferred: Option<FauxDeferredOptions>,
    pub tokens_per_second: Option<f64>,
    pub token_size: Option<FauxTokenSize>,
}

fn estimate_tokens(text: &str) -> u32 {
    utf16_len(text).div_ceil(4) as u32
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn random_u64() -> u64 {
    let mut bytes = [0u8; 8];
    SystemRandom::new()
        .fill(&mut bytes)
        .expect("system randomness is available");
    u64::from_le_bytes(bytes)
}

pub(crate) fn random_base36(mut value: u64) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = Vec::new();
    loop {
        out.push(DIGITS[(value % 36) as usize]);
        value /= 36;
        if value == 0 {
            break;
        }
    }
    out.reverse();
    String::from_utf8(out).expect("base36 digits are ASCII")
}

fn random_id(prefix: &str) -> String {
    format!("{prefix}:{}:{}", now_millis(), random_base36(random_u64()))
}

/// Random suffix for registration source ids (`Math.random().toString(36).slice(2, 10)`).
pub(crate) fn random_suffix() -> String {
    let mut suffix = random_base36(random_u64());
    suffix.truncate(8);
    suffix
}

fn content_to_text(content: &[UserContent]) -> String {
    content
        .iter()
        .map(|block| match block {
            UserContent::Text(text) => text.text.clone(),
            UserContent::Image(image) => {
                format!("[image:{}:{}]", image.mime_type, utf16_len(&image.data))
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_content_to_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantContent::Text(text) => text.text.clone(),
            AssistantContent::Thinking(thinking) => thinking.thinking.clone(),
            AssistantContent::ToolCall(tool_call) => format!(
                "{}:{}",
                tool_call.name,
                serde_json::to_string(&tool_call.arguments).unwrap_or_default()
            ),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_to_text(message: &ToolResultMessage) -> String {
    std::iter::once(message.tool_name.clone())
        .chain(
            message
                .content
                .iter()
                .map(|block| content_to_text(std::slice::from_ref(block))),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

fn to_json(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn message_to_text(message: &Message) -> String {
    match message {
        Message::System(message) => std::iter::once(get_system_message_text(message))
            .chain(
                message
                    .tools_removed
                    .iter()
                    .flatten()
                    .map(|tool| format!("tool-:{}", to_json(tool))),
            )
            .chain(
                message
                    .tools_added
                    .iter()
                    .flatten()
                    .map(|tool| format!("tool+:{}", to_json(tool))),
            )
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Message::User(message) => match &message.content {
            UserMessageContent::Text(text) => text.clone(),
            UserMessageContent::Parts(parts) => content_to_text(parts),
        },
        Message::Assistant(message) => assistant_content_to_text(&message.content),
        Message::ToolResult(message) => tool_result_to_text(message),
    }
}

fn serialize_context(context: &TranscriptContext) -> String {
    context
        .messages
        .iter()
        .map(|message| format!("{}:{}", message.role(), message_to_text(message)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Common prefix length in UTF-16 code units.
fn common_prefix_length(a: &str, b: &str) -> usize {
    a.encode_utf16()
        .zip(b.encode_utf16())
        .take_while(|(a, b)| a == b)
        .count()
}

fn with_usage_estimate(
    mut message: AssistantMessage,
    context: &TranscriptContext,
    options: &StreamOptions,
    prompt_cache: &mut HashMap<String, String>,
) -> AssistantMessage {
    let prompt_text = serialize_context(context);
    let prompt_tokens = estimate_tokens(&prompt_text);
    let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0;
    let mut cache_write = 0;

    if let Some(session_id) = options.session_id.as_deref().filter(|id| !id.is_empty())
        && options.cache_retention != Some(CacheRetention::None)
    {
        match prompt_cache
            .get(session_id)
            .filter(|prompt| !prompt.is_empty())
        {
            Some(previous_prompt) => {
                let cached_units = common_prefix_length(previous_prompt, &prompt_text);
                cache_read = cached_units.div_ceil(4) as u32;
                cache_write = (utf16_len(&prompt_text) - cached_units).div_ceil(4) as u32;
                input = prompt_tokens.saturating_sub(cache_read);
            }
            None => cache_write = prompt_tokens,
        }
        prompt_cache.insert(session_id.to_string(), prompt_text);
    }

    message.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens: input + output_tokens + cache_read + cache_write,
        ..Default::default()
    };
    message
}

fn split_string_by_token_size(
    text: &str,
    min_token_size: usize,
    max_token_size: usize,
) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let span = (max_token_size - min_token_size + 1) as u64;
        let token_size = min_token_size + (random_u64() % span) as usize;
        let char_size = (token_size * 4).max(1);
        let end = (index + char_size).min(chars.len());
        chunks.push(chars[index..end].iter().collect());
        index += char_size;
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

fn clone_message(
    mut message: AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    message.api = api.to_string();
    message.provider = provider.to_string();
    message.model = model_id.to_string();
    message
}

fn create_deferred_message(model: &Model, handle: DeferredHandle) -> AssistantMessage {
    let mut message = message_for(
        Vec::new(),
        &model.api,
        &model.provider,
        &model.id,
        StopReason::Deferred,
    );
    message.deferred = Some(handle);
    message
}

fn create_error_message(
    error: &Error,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    let mut message = message_for(Vec::new(), api, provider, model_id, StopReason::Error);
    message.error_message = Some(error.to_string());
    message
}

fn create_aborted_message(partial: &AssistantMessage) -> AssistantMessage {
    AssistantMessage {
        stop_reason: StopReason::Aborted,
        error_message: Some("Request was aborted".to_string()),
        timestamp: now_millis(),
        ..partial.clone()
    }
}

async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
    match tokens_per_second.filter(|rate| *rate > 0.0) {
        None => tokio::task::yield_now().await,
        Some(rate) => {
            let delay = f64::from(estimate_tokens(chunk)) / rate;
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        }
    }
}

fn is_aborted(signal: Option<&CancellationToken>) -> bool {
    signal.is_some_and(CancellationToken::is_cancelled)
}

fn push_aborted(stream: &AssistantMessageEventStream, partial: &AssistantMessage) {
    let aborted = create_aborted_message(partial);
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Aborted,
        error: aborted.clone(),
    });
    stream.end(Some(aborted));
}

struct FauxShared {
    api: String,
    provider: String,
    models: Vec<Model>,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    deferred: Option<FauxDeferredOptions>,
    inner: Mutex<FauxInner>,
}

#[derive(Default)]
struct FauxInner {
    pending_responses: VecDeque<FauxResponseStep>,
    state: FauxProviderState,
    prompt_cache: HashMap<String, String>,
    deferred_responses: HashMap<String, DeferredEntry>,
}

struct DeferredEntry {
    handle: DeferredHandle,
    step: FauxResponseStep,
    context: TranscriptContext,
    options: SimpleStreamOptions,
    model: Model,
    pending_fetches: u32,
    cancelled: bool,
    final_message: Option<AssistantMessage>,
}

/// `createFauxCore(options)`: the scripted stream implementation shared by
/// [`faux_provider`] and `compat::register_faux_provider`. Cheap to clone.
#[derive(Clone)]
pub struct FauxCore {
    shared: Arc<FauxShared>,
}

impl std::fmt::Debug for FauxCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxCore")
            .field("api", &self.shared.api)
            .field("provider", &self.shared.provider)
            .finish_non_exhaustive()
    }
}

/// `createFauxCore(options)`.
pub fn create_faux_core(options: RegisterFauxProviderOptions) -> FauxCore {
    let api = options.api.unwrap_or_else(|| random_id(DEFAULT_API));
    let provider = options
        .provider
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
    let token_size = options.token_size.unwrap_or_default();
    let max = token_size.max.unwrap_or(DEFAULT_MAX_TOKEN_SIZE);
    let min_token_size = token_size
        .min
        .unwrap_or(DEFAULT_MIN_TOKEN_SIZE)
        .min(max)
        .max(1);
    let max_token_size = min_token_size.max(max);

    let definitions = if options.models.is_empty() {
        vec![FauxModelDefinition {
            id: DEFAULT_MODEL_ID.to_string(),
            name: Some(DEFAULT_MODEL_NAME.to_string()),
            reasoning: Some(false),
            input: Some(vec![ModelInput::Text, ModelInput::Image]),
            input_limits: None,
            cost: Some(ModelCost::default()),
            context_window: Some(128_000),
            max_tokens: Some(16_384),
        }]
    } else {
        options.models
    };
    let models = definitions
        .into_iter()
        .map(|definition| Model {
            name: definition.name.unwrap_or_else(|| definition.id.clone()),
            id: definition.id,
            api: api.clone(),
            provider: provider.clone(),
            base_url: DEFAULT_BASE_URL.to_string(),
            reasoning: definition.reasoning.unwrap_or(false),
            input: definition
                .input
                .unwrap_or_else(|| vec![ModelInput::Text, ModelInput::Image]),
            input_limits: definition.input_limits,
            cost: definition.cost.unwrap_or_default(),
            context_window: definition.context_window.unwrap_or(128_000),
            max_tokens: definition.max_tokens.unwrap_or(16_384),
            ..Default::default()
        })
        .collect();

    FauxCore {
        shared: Arc::new(FauxShared {
            api,
            provider,
            models,
            min_token_size,
            max_token_size,
            tokens_per_second: options.tokens_per_second,
            deferred: options.deferred,
            inner: Mutex::default(),
        }),
    }
}

impl FauxCore {
    pub fn api(&self) -> &str {
        &self.shared.api
    }

    pub fn provider(&self) -> &str {
        &self.shared.provider
    }

    /// The registered models; never empty.
    pub fn models(&self) -> &[Model] {
        &self.shared.models
    }

    /// `getModel()`: the first model.
    pub fn get_model(&self) -> Model {
        self.shared.models[0].clone()
    }

    /// `getModel(modelId)`.
    pub fn get_model_by_id(&self, model_id: &str) -> Option<Model> {
        if model_id.is_empty() {
            return Some(self.get_model());
        }
        self.shared
            .models
            .iter()
            .find(|candidate| candidate.id == model_id)
            .cloned()
    }

    /// Snapshot of `state`.
    pub fn state(&self) -> FauxProviderState {
        self.shared.inner.lock().state.clone()
    }

    pub fn set_responses(&self, responses: impl IntoIterator<Item = FauxResponseStep>) {
        self.shared.inner.lock().pending_responses = responses.into_iter().collect();
    }

    pub fn append_responses(&self, responses: impl IntoIterator<Item = FauxResponseStep>) {
        self.shared.inner.lock().pending_responses.extend(responses);
    }

    pub fn get_pending_response_count(&self) -> usize {
        self.shared.inner.lock().pending_responses.len()
    }

    fn error_message(&self, error: &Error, model_id: &str) -> AssistantMessage {
        create_error_message(error, &self.shared.api, &self.shared.provider, model_id)
    }

    fn push_error(&self, stream: &AssistantMessageEventStream, error: &Error, model_id: &str) {
        let message = self.error_message(error, model_id);
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: message.clone(),
        });
        stream.end(Some(message));
    }

    async fn resolve_response(
        &self,
        step: FauxResponseStep,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
        request_model: &Model,
    ) -> Result<AssistantMessage> {
        let resolved = match step {
            FauxResponseStep::Message(message) => *message,
            FauxResponseStep::Factory(factory) => {
                let state = self.state();
                factory(
                    context.clone(),
                    options.clone(),
                    state,
                    request_model.clone(),
                )
                .await?
            }
        };
        let cloned = clone_message(
            resolved,
            &self.shared.api,
            &self.shared.provider,
            &request_model.id,
        );
        let mut inner = self.shared.inner.lock();
        Ok(with_usage_estimate(
            cloned,
            context,
            &options.stream,
            &mut inner.prompt_cache,
        ))
    }

    async fn stream_with_deltas(
        &self,
        stream: &AssistantMessageEventStream,
        message: AssistantMessage,
        signal: Option<&CancellationToken>,
    ) -> Result<()> {
        let (min, max, rate) = (
            self.shared.min_token_size,
            self.shared.max_token_size,
            self.shared.tokens_per_second,
        );
        let mut partial = AssistantMessage {
            content: Vec::new(),
            stop_reason: StopReason::Pending,
            ..message.clone()
        };
        if is_aborted(signal) {
            push_aborted(stream, &partial);
            return Ok(());
        }

        stream.push(AssistantMessageEvent::Start {
            partial: partial.clone(),
        });

        for (index, block) in message.content.iter().enumerate() {
            if is_aborted(signal) {
                push_aborted(stream, &partial);
                return Ok(());
            }

            match block {
                AssistantContent::Thinking(thinking) => {
                    partial.content.push(faux_thinking(""));
                    stream.push(AssistantMessageEvent::ThinkingStart {
                        content_index: index,
                        partial: partial.clone(),
                    });
                    for chunk in split_string_by_token_size(&thinking.thinking, min, max) {
                        schedule_chunk(&chunk, rate).await;
                        if is_aborted(signal) {
                            push_aborted(stream, &partial);
                            return Ok(());
                        }
                        if let Some(AssistantContent::Thinking(current)) =
                            partial.content.get_mut(index)
                        {
                            current.thinking.push_str(&chunk);
                        }
                        stream.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: index,
                            delta: chunk,
                            partial: partial.clone(),
                        });
                    }
                    stream.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: index,
                        content: thinking.thinking.clone(),
                        partial: partial.clone(),
                    });
                }
                AssistantContent::Text(text) => {
                    partial.content.push(faux_text(""));
                    stream.push(AssistantMessageEvent::TextStart {
                        content_index: index,
                        partial: partial.clone(),
                    });
                    for chunk in split_string_by_token_size(&text.text, min, max) {
                        schedule_chunk(&chunk, rate).await;
                        if is_aborted(signal) {
                            push_aborted(stream, &partial);
                            return Ok(());
                        }
                        if let Some(AssistantContent::Text(current)) =
                            partial.content.get_mut(index)
                        {
                            current.text.push_str(&chunk);
                        }
                        stream.push(AssistantMessageEvent::TextDelta {
                            content_index: index,
                            delta: chunk,
                            partial: partial.clone(),
                        });
                    }
                    stream.push(AssistantMessageEvent::TextEnd {
                        content_index: index,
                        content: text.text.clone(),
                        partial: partial.clone(),
                    });
                }
                AssistantContent::ToolCall(tool_call) => {
                    partial.content.push(AssistantContent::ToolCall(ToolCall {
                        id: tool_call.id.clone(),
                        name: tool_call.name.clone(),
                        arguments: Value::Object(Default::default()),
                        thought_signature: None,
                        namespace: None,
                    }));
                    stream.push(AssistantMessageEvent::ToolCallStart {
                        content_index: index,
                        partial: partial.clone(),
                    });
                    for chunk in
                        split_string_by_token_size(&to_json(&tool_call.arguments), min, max)
                    {
                        schedule_chunk(&chunk, rate).await;
                        if is_aborted(signal) {
                            push_aborted(stream, &partial);
                            return Ok(());
                        }
                        stream.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: index,
                            delta: chunk,
                            partial: partial.clone(),
                        });
                    }
                    if let Some(AssistantContent::ToolCall(current)) =
                        partial.content.get_mut(index)
                    {
                        current.arguments = tool_call.arguments.clone();
                    }
                    stream.push(AssistantMessageEvent::ToolCallEnd {
                        content_index: index,
                        tool_call: tool_call.clone(),
                        partial: partial.clone(),
                    });
                }
            }
        }

        match message.stop_reason {
            StopReason::Pending => Err(Error::message("Faux response ended without a stop reason")),
            StopReason::Error | StopReason::Aborted => {
                stream.push(AssistantMessageEvent::Error {
                    reason: message.stop_reason,
                    error: message.clone(),
                });
                stream.end(Some(message));
                Ok(())
            }
            reason => {
                stream.push(AssistantMessageEvent::Done {
                    reason,
                    message: message.clone(),
                });
                stream.end(Some(message));
                Ok(())
            }
        }
    }

    async fn run_stream(
        &self,
        outer: &AssistantMessageEventStream,
        step: Option<FauxResponseStep>,
        request_model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> Result<()> {
        call_response_hook(options.stream.on_response.as_ref(), &request_model).await?;
        let Some(step) = step else {
            let error = Error::message("No more faux responses queued");
            let message = self.error_message(&error, &request_model.id);
            let message = {
                let mut inner = self.shared.inner.lock();
                with_usage_estimate(message, &context, &options.stream, &mut inner.prompt_cache)
            };
            outer.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: message.clone(),
            });
            outer.end(Some(message));
            return Ok(());
        };

        if options
            .deferred
            .is_some_and(|deferred| deferred.is_enabled())
        {
            let deferred = self.shared.deferred.unwrap_or_default();
            let handle = DeferredHandle {
                provider: request_model.provider.clone(),
                model_id: request_model.id.clone(),
                api: request_model.api.clone(),
                id: random_id("deferred"),
                expires_at: None,
                poll_after_ms: deferred.poll_after_ms,
                data: None,
            };
            let signal = options.stream.signal.clone();
            self.shared.inner.lock().deferred_responses.insert(
                handle.id.clone(),
                DeferredEntry {
                    handle: handle.clone(),
                    step,
                    context,
                    options,
                    model: request_model.clone(),
                    pending_fetches: deferred.pending_fetches.unwrap_or(0),
                    cancelled: false,
                    final_message: None,
                },
            );
            return self
                .stream_with_deltas(
                    outer,
                    create_deferred_message(&request_model, handle),
                    signal.as_ref(),
                )
                .await;
        }

        let message = self
            .resolve_response(step, &context, &options, &request_model)
            .await?;
        self.stream_with_deltas(outer, message, options.stream.signal.as_ref())
            .await
    }

    async fn run_fetch_deferred(
        &self,
        outer: &AssistantMessageEventStream,
        request_model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions,
    ) -> Result<()> {
        call_response_hook(options.request.on_response.as_ref(), &request_model).await?;
        let signal = options.request.signal.as_ref();
        let pending = {
            let mut inner = self.shared.inner.lock();
            let entry = inner
                .deferred_responses
                .get_mut(&handle.id)
                .filter(|entry| {
                    entry.handle.provider == handle.provider
                        && entry.handle.model_id == handle.model_id
                        && entry.handle.api == handle.api
                })
                .ok_or_else(|| {
                    Error::message(format!("Unknown faux deferred response: {}", handle.id))
                })?;
            if entry.cancelled {
                return Err(Error::message(format!(
                    "Faux deferred response was cancelled: {}",
                    handle.id
                )));
            }
            if entry.pending_fetches > 0 {
                entry.pending_fetches -= 1;
                Some(entry.handle.clone())
            } else {
                None
            }
        };
        if let Some(entry_handle) = pending {
            return self
                .stream_with_deltas(
                    outer,
                    create_deferred_message(&request_model, entry_handle),
                    signal,
                )
                .await;
        }

        let existing = {
            let inner = self.shared.inner.lock();
            let entry = &inner.deferred_responses[&handle.id];
            match &entry.final_message {
                Some(message) => Ok(message.clone()),
                None => Err((
                    entry.step.clone(),
                    entry.context.clone(),
                    entry.options.clone(),
                    entry.model.clone(),
                )),
            }
        };
        let final_message = match existing {
            Ok(message) => message,
            Err((step, context, mut submission_options, model)) => {
                submission_options.deferred = None;
                submission_options.stream.signal = None;
                submission_options.stream.on_response = None;
                let message = match self
                    .resolve_response(step, &context, &submission_options, &model)
                    .await
                {
                    Ok(message) => message,
                    Err(error) => self.error_message(&error, &model.id),
                };
                if let Some(entry) = self
                    .shared
                    .inner
                    .lock()
                    .deferred_responses
                    .get_mut(&handle.id)
                {
                    entry.final_message = Some(message.clone());
                }
                message
            }
        };
        self.stream_with_deltas(outer, final_message, signal).await
    }
}

async fn call_response_hook(hook: Option<&ResponseHook>, model: &Model) -> Result<()> {
    match hook {
        Some(hook) => {
            hook(
                ProviderResponse {
                    status: 200,
                    headers: Default::default(),
                },
                model,
            )
            .await
        }
        None => Ok(()),
    }
}

#[async_trait]
impl ProviderStreams for FauxCore {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream_simple(model, context, SimpleStreamOptions::from(options))
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let outer = AssistantMessageEventStream::new();
        let step = {
            let mut inner = self.shared.inner.lock();
            inner.state.call_count += 1;
            inner.pending_responses.pop_front()
        };
        let core = self.clone();
        let producer = outer.clone();
        tokio::spawn(async move {
            let model_id = model.id.clone();
            // A panicking response factory ends the stream like a throwing one (TS `catch`).
            let run = AssertUnwindSafe(core.run_stream(&producer, step, model, context, options));
            let error = match run.catch_unwind().await {
                Ok(result) => result.err(),
                Err(payload) => Some(Error::message(panic_message(payload.as_ref()))),
            };
            if let Some(error) = error {
                core.push_error(&producer, &error, &model_id);
            }
        });
        outer
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn fetch_deferred(
        &self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        let outer = AssistantMessageEventStream::new();
        self.shared.inner.lock().state.deferred_fetch_count += 1;
        let core = self.clone();
        let producer = outer.clone();
        tokio::spawn(async move {
            let model_id = model.id.clone();
            let run = AssertUnwindSafe(core.run_fetch_deferred(&producer, model, handle, options));
            let error = match run.catch_unwind().await {
                Ok(result) => result.err(),
                Err(payload) => Some(Error::message(panic_message(payload.as_ref()))),
            };
            if let Some(error) = error {
                core.push_error(&producer, &error, &model_id);
            }
        });
        outer
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }

    async fn cancel_deferred(
        &self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions,
    ) -> Result<()> {
        {
            let mut inner = self.shared.inner.lock();
            inner.state.cancelled_deferred.push(handle.clone());
            if let Some(entry) = inner.deferred_responses.get_mut(&handle.id) {
                entry.cancelled = true;
            }
        }
        call_response_hook(options.on_response.as_ref(), &model).await
    }
}

struct FauxAuth;

#[async_trait]
impl ApiKeyAuth for FauxAuth {
    fn name(&self) -> &str {
        "Faux"
    }

    async fn resolve(&self, _input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
        Ok(Some(AuthResult::default()))
    }
}

/// `FauxProviderHandle`. Derefs to [`FauxCore`] for `api`, `models`,
/// `get_model`, `state`, `set_responses`, `append_responses` and
/// `get_pending_response_count`.
#[derive(Clone)]
pub struct FauxProviderHandle {
    pub provider: Arc<dyn Provider>,
    core: FauxCore,
}

impl Deref for FauxProviderHandle {
    type Target = FauxCore;

    fn deref(&self) -> &FauxCore {
        &self.core
    }
}

/// Faux provider for tests built on explicit `Models` collections:
///
/// ```no_run
/// # async fn demo() {
/// use ai::{FauxMessageOptions, create_models, faux_assistant_message, faux_provider};
///
/// let faux = faux_provider(Default::default());
/// let models = create_models(Default::default());
/// models.set_provider(faux.provider.clone());
/// faux.set_responses([faux_assistant_message("hi", FauxMessageOptions::default()).into()]);
/// # }
/// ```
pub fn faux_provider(options: RegisterFauxProviderOptions) -> FauxProviderHandle {
    let core = create_faux_core(options);
    let provider = create_provider(CreateProviderOptions {
        id: core.provider().to_string(),
        auth: ProviderAuth {
            api_key: Some(Arc::new(FauxAuth)),
            oauth: None,
        },
        models: core.models().iter().cloned().map(AnyModel::Chat).collect(),
        api: Some(ProviderApi::Single(Arc::new(core.clone()))),
        ..Default::default()
    })
    .expect("the faux provider has an API implementation");
    FauxProviderHandle { provider, core }
}

/// `FauxProviderRegistration` returned by `compat::register_faux_provider`.
/// Derefs to [`FauxCore`].
#[derive(Debug, Clone)]
pub struct FauxProviderRegistration {
    core: FauxCore,
    source_id: String,
}

impl FauxProviderRegistration {
    pub(crate) fn new(core: FauxCore, source_id: String) -> Self {
        Self { core, source_id }
    }

    pub fn unregister(&self) {
        crate::compat::unregister_api_providers(&self.source_id);
    }
}

impl Deref for FauxProviderRegistration {
    type Target = FauxCore;

    fn deref(&self) -> &FauxCore {
        &self.core
    }
}

#[cfg(test)]
mod tests {
    //! Port of `test/faux-provider.test.ts` and the `fauxProvider` block of
    //! `test/providers.test.ts`.

    use futures::StreamExt;
    use serde_json::json;
    use tokio::sync::MutexGuard;

    use super::*;
    use crate::compat::{REGISTRY_TEST_LOCK, complete, register_faux_provider, stream};
    use crate::models::create_models;
    use crate::types::{Context, DeferredRequest, DeferredWindow, ImageContent, Tool, UserMessage};

    /// Holds the registry lock and unregisters on drop (`afterEach`).
    struct Registered {
        registration: FauxProviderRegistration,
        _lock: MutexGuard<'static, ()>,
    }

    impl Deref for Registered {
        type Target = FauxProviderRegistration;

        fn deref(&self) -> &FauxProviderRegistration {
            &self.registration
        }
    }

    impl Drop for Registered {
        fn drop(&mut self) {
            self.registration.unregister();
        }
    }

    async fn register(options: RegisterFauxProviderOptions) -> Registered {
        let lock = REGISTRY_TEST_LOCK.lock().await;
        Registered {
            registration: register_faux_provider(options),
            _lock: lock,
        }
    }

    fn message(text: &str) -> FauxResponseStep {
        faux_assistant_message(text, FauxMessageOptions::default()).into()
    }

    fn hi() -> Context {
        Context::builder().message(Message::user_text("hi")).build()
    }

    async fn run(
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> AssistantMessage {
        complete(model, context, options).await.unwrap()
    }

    async fn collect_events(
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> Vec<AssistantMessageEvent> {
        stream(model, context, options).unwrap().collect().await
    }

    fn event_types(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
        events
            .iter()
            .map(AssistantMessageEvent::event_type)
            .collect()
    }

    fn terminal_error(event: &AssistantMessageEvent) -> (StopReason, &AssistantMessage) {
        match event {
            AssistantMessageEvent::Error { reason, error } => (*reason, error),
            other => panic!("expected an error event, got {}", other.event_type()),
        }
    }

    #[tokio::test]
    async fn registers_a_custom_provider_and_estimates_usage() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("hello world")]);

        let context = Context::builder()
            .system_prompt("Be concise.")
            .message(Message::user_text("hi there"))
            .build();
        let response = run(registration.get_model(), context, None).await;
        assert_eq!(response.content, vec![faux_text("hello world")]);
        assert!(response.usage.input > 0);
        assert!(response.usage.output > 0);
        assert_eq!(
            response.usage.total_tokens,
            response.usage.input + response.usage.output
        );
        assert_eq!(registration.state().call_count, 1);
    }

    #[tokio::test]
    async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
        let registration = register(Default::default()).await;
        registration.set_responses([faux_assistant_message(
            [
                faux_thinking("think"),
                faux_tool_call("echo", json!({ "text": "hi" }), None),
                faux_text("done"),
            ],
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into()]);

        let response = run(registration.get_model(), hi(), None).await;
        assert_eq!(response.content.len(), 3);
        assert_eq!(response.content[0], faux_thinking("think"));
        let AssistantContent::ToolCall(tool_call) = &response.content[1] else {
            panic!("expected a tool call");
        };
        assert!(!tool_call.id.is_empty());
        assert_eq!(
            (tool_call.name.as_str(), &tool_call.arguments),
            ("echo", &json!({ "text": "hi" }))
        );
        assert_eq!(response.content[2], faux_text("done"));
        assert_eq!(response.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn supports_multiple_models_with_per_model_reasoning_and_model_aware_factories() {
        let registration = register(RegisterFauxProviderOptions {
            models: vec![
                FauxModelDefinition {
                    name: Some("Faux Fast".to_string()),
                    reasoning: Some(false),
                    ..FauxModelDefinition::new("faux-fast")
                },
                FauxModelDefinition {
                    name: Some("Faux Thinker".to_string()),
                    reasoning: Some(true),
                    ..FauxModelDefinition::new("faux-thinker")
                },
            ],
            ..Default::default()
        })
        .await;
        let factory = || {
            FauxResponseStep::factory(|_, _, _, model| {
                Ok(faux_assistant_message(
                    format!("{}:{}", model.id, model.reasoning),
                    FauxMessageOptions::default(),
                ))
            })
        };
        registration.set_responses([factory(), factory()]);

        let ids: Vec<_> = registration
            .models()
            .iter()
            .map(|model| model.id.as_str())
            .collect();
        assert_eq!(ids, vec!["faux-fast", "faux-thinker"]);
        assert_eq!(registration.get_model(), registration.models()[0]);
        assert!(!registration.get_model_by_id("faux-fast").unwrap().reasoning);
        assert!(
            registration
                .get_model_by_id("faux-thinker")
                .unwrap()
                .reasoning
        );

        let fast = run(
            registration.get_model_by_id("faux-fast").unwrap(),
            hi(),
            None,
        )
        .await;
        let thinker = run(
            registration.get_model_by_id("faux-thinker").unwrap(),
            hi(),
            None,
        )
        .await;
        assert_eq!(fast.content, vec![faux_text("faux-fast:false")]);
        assert_eq!(thinker.content, vec![faux_text("faux-thinker:true")]);
    }

    #[tokio::test]
    async fn rewrites_api_provider_and_model_on_returned_messages() {
        let registration = register(RegisterFauxProviderOptions {
            api: Some("faux:test".to_string()),
            provider: Some("faux-provider".to_string()),
            models: vec![FauxModelDefinition::new("faux-model")],
            ..Default::default()
        })
        .await;
        registration.set_responses([message("hello")]);

        let response = run(registration.get_model(), hi(), None).await;
        assert_eq!(response.api, "faux:test");
        assert_eq!(response.provider, "faux-provider");
        assert_eq!(response.model, "faux-model");
    }

    #[tokio::test]
    async fn consumes_queued_responses_in_order_and_errors_when_exhausted() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("first"), message("second")]);

        let first = run(registration.get_model(), hi(), None).await;
        let second = run(registration.get_model(), hi(), None).await;
        let exhausted = run(registration.get_model(), hi(), None).await;

        assert_eq!(first.content, vec![faux_text("first")]);
        assert_eq!(second.content, vec![faux_text("second")]);
        assert_eq!(exhausted.stop_reason, StopReason::Error);
        assert_eq!(
            exhausted.error_message.as_deref(),
            Some("No more faux responses queued")
        );
        assert_eq!(registration.get_pending_response_count(), 0);
        assert_eq!(registration.state().call_count, 3);
    }

    #[tokio::test]
    async fn can_replace_and_append_queued_responses() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("first")]);
        let text = |response: AssistantMessage| response.content;

        assert_eq!(
            text(run(registration.get_model(), hi(), None).await),
            vec![faux_text("first")]
        );
        assert_eq!(registration.get_pending_response_count(), 0);

        registration.set_responses([message("second")]);
        assert_eq!(registration.get_pending_response_count(), 1);
        assert_eq!(
            text(run(registration.get_model(), hi(), None).await),
            vec![faux_text("second")]
        );

        registration.append_responses([message("third"), message("fourth")]);
        assert_eq!(registration.get_pending_response_count(), 2);
        assert_eq!(
            text(run(registration.get_model(), hi(), None).await),
            vec![faux_text("third")]
        );
        assert_eq!(
            text(run(registration.get_model(), hi(), None).await),
            vec![faux_text("fourth")]
        );
        assert_eq!(registration.get_pending_response_count(), 0);
    }

    #[tokio::test]
    async fn supports_async_response_factories() {
        let registration = register(Default::default()).await;
        registration.set_responses([FauxResponseStep::async_factory(
            |context, _, state, _| async move {
                Ok(faux_assistant_message(
                    format!("{}:{}", context.messages.len(), state.call_count),
                    FauxMessageOptions::default(),
                ))
            },
        )]);

        let response = run(registration.get_model(), hi(), None).await;
        assert_eq!(response.content, vec![faux_text("1:1")]);
    }

    #[tokio::test]
    async fn emits_an_error_when_a_response_factory_throws() {
        let registration = register(Default::default()).await;
        registration.set_responses([FauxResponseStep::factory(|_, _, _, _| {
            Err(Error::message("boom"))
        })]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        assert_eq!(events.len(), 1);
        let (_, error) = terminal_error(&events[0]);
        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(error.error_message.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn emits_an_error_when_a_response_factory_panics() {
        // A panic is the Rust form of a throwing factory: the stream ends with an error event.
        let registration = register(Default::default()).await;
        registration.set_responses([FauxResponseStep::factory(|_, _, _, _| {
            panic!("factory panicked")
        })]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        assert_eq!(events.len(), 1);
        let (_, error) = terminal_error(&events[0]);
        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(error.error_message.as_deref(), Some("factory panicked"));
    }

    #[tokio::test]
    async fn rejects_a_queued_response_without_a_terminal_stop_reason() {
        let registration = register(Default::default()).await;
        registration.set_responses([faux_assistant_message(
            "partial",
            FauxMessageOptions {
                stop_reason: Some(StopReason::Pending),
                ..Default::default()
            },
        )
        .into()]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        assert!(!event_types(&events).contains(&"done"));
        let (_, error) = terminal_error(events.last().unwrap());
        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(
            error.error_message.as_deref(),
            Some("Faux response ended without a stop reason")
        );
    }

    #[tokio::test]
    async fn estimates_prompt_and_output_tokens_from_serialized_context() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("done")]);

        let tool = Tool {
            name: "echo".to_string(),
            description: "Echo back text".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            }),
            constrained_sampling: None,
        };
        let context = Context::builder()
            .system_prompt("sys")
            .message(Message::User(UserMessage {
                content: UserMessageContent::Parts(vec![
                    UserContent::text("hello"),
                    UserContent::Image(ImageContent {
                        data: "abcd".to_string(),
                        mime_type: "image/png".to_string(),
                    }),
                ]),
                timestamp: 1,
            }))
            .message(Message::Assistant(faux_assistant_message(
                "prior",
                FauxMessageOptions::default(),
            )))
            .message(Message::ToolResult(ToolResultMessage {
                tool_call_id: "tool-1".to_string(),
                tool_name: "echo".to_string(),
                content: vec![UserContent::text("tool out")],
                details: None,
                usage: None,
                nested_calls: None,
                is_error: false,
                timestamp: 2,
            }))
            .tool(tool.clone())
            .build();

        let response = run(registration.get_model(), context, None).await;
        // Pi's test lists the tools as a trailing `tools:` entry; the
        // implementation serializes them on the leading system message.
        let prompt_text = [
            format!(
                "system:sys\ntool+:{}",
                serde_json::to_string(&tool).unwrap()
            ),
            "user:hello\n[image:image/png:4]".to_string(),
            "assistant:prior".to_string(),
            "toolResult:echo\ntool out".to_string(),
        ]
        .join("\n\n");
        let expected_prompt_tokens = prompt_text.len().div_ceil(4) as u32;
        let expected_output_tokens = "done".len().div_ceil(4) as u32;

        assert_eq!(response.usage.input, expected_prompt_tokens);
        assert_eq!(response.usage.output, expected_output_tokens);
        assert_eq!(response.usage.cache_read, 0);
        assert_eq!(response.usage.cache_write, 0);
        assert_eq!(
            response.usage.total_tokens,
            expected_prompt_tokens + expected_output_tokens
        );
    }

    fn session(id: &str, retention: CacheRetention) -> Option<StreamOptions> {
        Some(StreamOptions {
            session_id: Some(id.to_string()),
            cache_retention: Some(retention),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn does_not_share_cache_across_sessions_or_requests_without_session_id() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("first"), message("second"), message("third")]);

        let mut context = Context::builder()
            .message(Message::user_text("hello"))
            .build();
        let first = run(
            registration.get_model(),
            context.clone(),
            session("session-1", CacheRetention::Short),
        )
        .await;
        assert!(first.usage.cache_write > 0);
        context.messages.push(Message::Assistant(first));
        context.messages.push(Message::user_text("follow up"));

        let second = run(
            registration.get_model(),
            context.clone(),
            session("session-2", CacheRetention::Short),
        )
        .await;
        assert_eq!(second.usage.cache_read, 0);
        assert!(second.usage.cache_write > 0);

        let third = run(registration.get_model(), context, None).await;
        assert_eq!(third.usage.cache_read, 0);
        assert_eq!(third.usage.cache_write, 0);
    }

    #[tokio::test]
    async fn simulates_prompt_caching_per_session_id() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("first"), message("second")]);

        let mut context = Context::builder()
            .system_prompt("Be concise.")
            .message(Message::user_text("hello"))
            .build();
        let first = run(
            registration.get_model(),
            context.clone(),
            session("session-1", CacheRetention::Short),
        )
        .await;
        assert_eq!(first.usage.cache_read, 0);
        assert!(first.usage.cache_write > 0);

        context.messages.push(Message::Assistant(first));
        context.messages.push(Message::user_text("follow up"));
        let second = run(
            registration.get_model(),
            context,
            session("session-1", CacheRetention::Short),
        )
        .await;
        assert!(second.usage.cache_read > 0);
    }

    #[tokio::test]
    async fn does_not_simulate_caching_when_cache_retention_is_none() {
        let registration = register(Default::default()).await;
        registration.set_responses([message("first"), message("second")]);

        let mut context = Context::builder()
            .message(Message::user_text("hello"))
            .build();
        run(
            registration.get_model(),
            context.clone(),
            session("session-1", CacheRetention::None),
        )
        .await;
        context
            .messages
            .push(Message::Assistant(faux_assistant_message(
                "first",
                FauxMessageOptions::default(),
            )));
        context.messages.push(Message::user_text("follow up"));
        let second = run(
            registration.get_model(),
            context,
            session("session-1", CacheRetention::None),
        )
        .await;
        assert_eq!(second.usage.cache_read, 0);
        assert_eq!(second.usage.cache_write, 0);
    }

    fn tool_use(blocks: Vec<FauxContentBlock>) -> FauxResponseStep {
        faux_assistant_message(
            blocks,
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into()
    }

    #[tokio::test]
    async fn streams_thinking_text_and_partial_tool_call_deltas() {
        let registration = register(Default::default()).await;
        registration.set_responses([tool_use(vec![
            faux_thinking("thinking text"),
            faux_text("answer text"),
            faux_tool_call("echo", json!({ "text": "hi", "count": 12 }), Some("tool-1")),
        ])]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        let types = event_types(&events);
        for expected in [
            "thinking_start",
            "thinking_delta",
            "text_start",
            "text_delta",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ] {
            assert!(types.contains(&expected), "{expected}");
        }
        let tool_call_deltas: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert!(tool_call_deltas.len() > 1);
        let arguments: Value = serde_json::from_str(&tool_call_deltas.concat()).unwrap();
        assert_eq!(arguments, json!({ "text": "hi", "count": 12 }));
    }

    #[tokio::test]
    async fn streams_an_exact_event_order_for_fixed_size_chunks() {
        let registration = register(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..Default::default()
        })
        .await;
        registration.set_responses([tool_use(vec![
            faux_thinking("go"),
            faux_text("ok"),
            faux_tool_call("echo", json!({}), Some("tool-1")),
        ])]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        let AssistantMessageEvent::Start { partial } = &events[0] else {
            panic!("expected start");
        };
        assert_eq!(partial.stop_reason, StopReason::Pending);
        assert_eq!(
            event_types(&events),
            vec![
                "start",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "text_start",
                "text_delta",
                "text_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
                "done",
            ]
        );
    }

    #[tokio::test]
    async fn streams_multiple_tool_calls_in_one_message() {
        let registration = register(Default::default()).await;
        registration.set_responses([tool_use(vec![
            faux_tool_call("echo", json!({ "text": "one" }), Some("tool-1")),
            faux_tool_call("echo", json!({ "text": "two" }), Some("tool-2")),
        ])]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        let types = event_types(&events);
        assert_eq!(
            types
                .iter()
                .filter(|kind| **kind == "toolcall_start")
                .count(),
            2
        );
        assert_eq!(
            types.iter().filter(|kind| **kind == "toolcall_end").count(),
            2
        );
    }

    async fn explicit_terminal(stop_reason: StopReason, error_message: &str) {
        let registration = register(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        })
        .await;
        registration.set_responses([faux_assistant_message(
            "partial",
            FauxMessageOptions {
                stop_reason: Some(stop_reason),
                error_message: Some(error_message.to_string()),
                ..Default::default()
            },
        )
        .into()]);

        let events = collect_events(registration.get_model(), hi(), None).await;
        assert_eq!(
            event_types(&events),
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );
        let (reason, error) = terminal_error(events.last().unwrap());
        assert_eq!(reason, stop_reason);
        assert_eq!(error.stop_reason, stop_reason);
        assert_eq!(error.error_message.as_deref(), Some(error_message));
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_error_message_as_a_terminal_error() {
        explicit_terminal(StopReason::Error, "upstream failed").await;
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_aborted_message_as_a_terminal_error() {
        explicit_terminal(StopReason::Aborted, "Request was aborted").await;
    }

    fn paced(tokens_per_second: f64) -> RegisterFauxProviderOptions {
        RegisterFauxProviderOptions {
            tokens_per_second: Some(tokens_per_second),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        }
    }

    fn with_signal(signal: &CancellationToken) -> Option<StreamOptions> {
        Some(StreamOptions {
            signal: Some(signal.clone()),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn supports_aborting_before_the_first_chunk() {
        let registration = register(paced(50.0)).await;
        registration.set_responses([message("abcdefghijklmnopqrstuvwxyz")]);

        let controller = CancellationToken::new();
        controller.cancel();
        let events = collect_events(registration.get_model(), hi(), with_signal(&controller)).await;
        assert_eq!(events.len(), 1);
        let (reason, error) = terminal_error(&events[0]);
        assert_eq!(reason, StopReason::Aborted);
        assert_eq!(error.stop_reason, StopReason::Aborted);
    }

    /// Abort on the first delta of `delta_type` and check the stream stops there.
    async fn abort_mid_stream(
        response: AssistantMessage,
        delta_type: &str,
        start_type: &str,
        end_type: &str,
    ) {
        let registration = register(paced(100.0)).await;
        registration.set_responses([response.into()]);

        let controller = CancellationToken::new();
        let mut events = Vec::new();
        let mut delta_count = 0;
        let mut s = stream(registration.get_model(), hi(), with_signal(&controller)).unwrap();
        while let Some(event) = s.next().await {
            events.push(event.event_type());
            if event.event_type() == delta_type {
                delta_count += 1;
                controller.cancel();
            }
        }

        assert_eq!(delta_count, 1);
        assert!(events.contains(&start_type));
        assert!(events.contains(&delta_type));
        assert!(events.contains(&"error"));
        assert!(!events.contains(&end_type));
    }

    #[tokio::test]
    async fn supports_aborting_mid_text_stream_when_paced() {
        abort_mid_stream(
            faux_assistant_message("abcdefghijklmnopqrstuvwxyz", FauxMessageOptions::default()),
            "text_delta",
            "text_start",
            "text_end",
        )
        .await;
    }

    #[tokio::test]
    async fn supports_aborting_mid_thinking_stream_when_paced() {
        abort_mid_stream(
            faux_assistant_message(
                faux_thinking("abcdefghijklmnopqrstuvwxyz"),
                FauxMessageOptions::default(),
            ),
            "thinking_delta",
            "thinking_start",
            "thinking_end",
        )
        .await;
    }

    #[tokio::test]
    async fn supports_aborting_mid_toolcall_stream_when_paced() {
        abort_mid_stream(
            faux_assistant_message(
                faux_tool_call(
                    "echo",
                    json!({ "text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789 }),
                    Some("tool-1"),
                ),
                FauxMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            ),
            "toolcall_delta",
            "toolcall_start",
            "toolcall_end",
        )
        .await;
    }

    #[tokio::test]
    async fn unregisters_the_provider() {
        let _lock = REGISTRY_TEST_LOCK.lock().await;
        let registration = register_faux_provider(Default::default());
        registration.set_responses([message("hello")]);
        registration.unregister();

        let error = stream(registration.get_model(), hi(), None).err().unwrap();
        assert_eq!(
            error.to_string(),
            format!("No API provider registered for api: {}", registration.api())
        );
    }

    // providers.test.ts: fauxProvider

    #[tokio::test]
    async fn streams_queued_responses_through_a_models_collection() {
        let faux = faux_provider(Default::default());
        let models = create_models(Default::default());
        models.set_provider(faux.provider.clone());
        faux.set_responses([message("hello from faux")]);

        let model = models.get_models(Some(faux.provider.id()))[0].clone();
        let result = models
            .complete_simple(&model, &hi(), SimpleStreamOptions::default())
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.content, vec![faux_text("hello from faux")]);
        assert_eq!(faux.state().call_count, 1);
    }

    fn deferred_options(request: DeferredRequest) -> SimpleStreamOptions {
        SimpleStreamOptions {
            deferred: Some(request),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn submits_polls_and_redeems_deferred_responses() {
        let faux = faux_provider(RegisterFauxProviderOptions {
            deferred: Some(FauxDeferredOptions {
                pending_fetches: Some(1),
                poll_after_ms: Some(25),
            }),
            ..Default::default()
        });
        let models = create_models(Default::default());
        models.set_provider(faux.provider.clone());
        faux.set_responses([message("ready")]);
        let model = faux.get_model();

        let submission = models.stream_simple(
            &model,
            &hi(),
            deferred_options(DeferredRequest::Window(Some(DeferredWindow::Hour1))),
        );
        let types: Vec<_> = submission
            .clone()
            .map(|event| event.event_type())
            .collect()
            .await;
        let deferred = submission.result().await;
        assert_eq!(types, vec!["start", "done"]);
        assert_eq!(deferred.stop_reason, StopReason::Deferred);
        assert!(deferred.content.is_empty());
        let handle = deferred.deferred.clone().unwrap();
        assert_eq!(
            (
                handle.provider.as_str(),
                handle.model_id.as_str(),
                handle.api.as_str()
            ),
            (
                model.provider.as_str(),
                model.id.as_str(),
                model.api.as_str()
            )
        );
        assert!(!handle.id.is_empty());
        assert_eq!(handle.poll_after_ms, Some(25));
        assert_eq!((handle.expires_at, handle.data.as_ref()), (None, None));

        let pending = models
            .fetch_deferred(&model, &handle, DeferredFetchOptions::default())
            .await;
        assert_eq!(pending.stop_reason, StopReason::Deferred);
        assert_eq!(pending.deferred.as_ref(), Some(&handle));

        let ready = models
            .fetch_deferred(
                &model,
                &handle,
                DeferredFetchOptions {
                    wait: Some(0),
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(ready.stop_reason, StopReason::Stop);
        assert_eq!(ready.content, vec![faux_text("ready")]);
        assert!(ready.usage.total_tokens > 0);
        let state = faux.state();
        assert_eq!((state.call_count, state.deferred_fetch_count), (1, 2));
    }

    #[tokio::test]
    async fn records_cancellation_and_returns_deferred_fetch_failures_in_band() {
        let faux = faux_provider(Default::default());
        let models = create_models(Default::default());
        models.set_provider(faux.provider.clone());
        faux.set_responses([
            FauxResponseStep::async_factory(|_, _, _, _| async {
                Err(Error::message("deferred failed"))
            }),
            message("cancelled"),
        ]);
        let model = faux.get_model();

        let failed_submission = models
            .complete_simple(&model, &hi(), deferred_options(DeferredRequest::Flag(true)))
            .await;
        let failed = models
            .fetch_deferred(
                &model,
                failed_submission.deferred.as_ref().unwrap(),
                DeferredFetchOptions::default(),
            )
            .await;
        assert_eq!(failed.stop_reason, StopReason::Error);
        assert_eq!(failed.error_message.as_deref(), Some("deferred failed"));

        let cancelled_submission = models
            .complete_simple(&model, &hi(), deferred_options(DeferredRequest::Flag(true)))
            .await;
        let handle = cancelled_submission.deferred.unwrap();
        models
            .cancel_deferred(&model, &handle, DeferredCancelOptions::default())
            .await
            .unwrap();
        assert_eq!(faux.state().cancelled_deferred, vec![handle.clone()]);
        let cancelled = models
            .fetch_deferred(&model, &handle, DeferredFetchOptions::default())
            .await;
        assert_eq!(cancelled.stop_reason, StopReason::Error);
        assert!(cancelled.error_message.unwrap().contains("was cancelled"));
    }

    // models-entry.test.ts

    #[tokio::test]
    async fn runs_a_faux_completion_through_an_explicit_models_collection() {
        let models = create_models(Default::default());
        let faux = faux_provider(Default::default());
        models.set_provider(faux.provider.clone());
        faux.set_responses([message("OK")]);
        let response = models
            .complete_simple(
                &faux.get_model(),
                &Context::default(),
                SimpleStreamOptions::default(),
            )
            .await;
        assert_eq!(response.content, vec![faux_text("OK")]);
    }

    #[test]
    fn splits_text_into_token_sized_chunks_and_estimates_utf16_units() {
        assert_eq!(split_string_by_token_size("", 1, 1), vec![String::new()]);
        assert_eq!(
            split_string_by_token_size("abcdefghij", 1, 1),
            vec!["abcd", "efgh", "ij"]
        );
        assert_eq!(estimate_tokens("😀😀"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(common_prefix_length("abc", "abd"), 2);
        assert_eq!(random_base36(0), "0");
        assert_eq!(random_base36(35), "z");
    }
}
