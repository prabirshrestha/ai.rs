use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

use crate::api_registry::{self, ApiProvider};
use crate::event_stream::{AssistantMessageEventStream, AssistantMessageEventStreamSender};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
    ImageContent, Message, Model, ModelCost, ModelInput, ProviderResponse, SimpleStreamOptions,
    StopReason, StreamOptions, TextContent, ThinkingContent, ToolCall, ToolResultContent,
    ToolResultMessage, Usage, UsageCost, UserContent, UserMessageContent,
};
use crate::{Error, Result};

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

type FauxResponseFuture = Pin<Box<dyn Future<Output = Result<AssistantMessage>> + Send>>;
type FauxResponseFactory =
    dyn Fn(Context, StreamOptions, FauxProviderState, Model) -> FauxResponseFuture + Send + Sync;

#[derive(Debug, Clone, Default)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: Option<bool>,
    pub input: Option<Vec<ModelInput>>,
    pub cost: Option<ModelCost>,
    pub context_window: Option<u32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct FauxTokenSize {
    pub min: Option<usize>,
    pub max: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct RegisterFauxProviderOptions {
    pub api: Option<String>,
    pub provider: Option<String>,
    pub models: Vec<FauxModelDefinition>,
    pub tokens_per_second: Option<f64>,
    pub token_size: Option<FauxTokenSize>,
}

#[derive(Clone, Default)]
pub struct FauxProviderState {
    call_count: Arc<AtomicUsize>,
}

impl FauxProviderState {
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    fn increment_call_count(&self) {
        self.call_count.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
pub enum FauxResponseStep {
    Message(AssistantMessage),
    Factory(Arc<FauxResponseFactory>),
}

impl FauxResponseStep {
    pub fn factory<F, Fut>(factory: F) -> Self
    where
        F: Fn(Context, StreamOptions, FauxProviderState, Model) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AssistantMessage>> + Send + 'static,
    {
        Self::Factory(Arc::new(move |context, options, state, model| {
            Box::pin(factory(context, options, state, model))
        }))
    }
}

impl From<AssistantMessage> for FauxResponseStep {
    fn from(value: AssistantMessage) -> Self {
        Self::Message(value)
    }
}

pub enum FauxAssistantContent {
    Text(String),
    Block(AssistantContent),
    Blocks(Vec<AssistantContent>),
}

impl From<&str> for FauxAssistantContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for FauxAssistantContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<AssistantContent> for FauxAssistantContent {
    fn from(value: AssistantContent) -> Self {
        Self::Block(value)
    }
}

impl From<Vec<AssistantContent>> for FauxAssistantContent {
    fn from(value: Vec<AssistantContent>) -> Self {
        Self::Blocks(value)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FauxAssistantMessageOptions {
    pub stop_reason: Option<StopReason>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<u64>,
}

pub struct FauxProviderRegistration {
    pub api: String,
    pub models: Vec<Model>,
    pub state: FauxProviderState,
    pending_responses: Arc<Mutex<VecDeque<FauxResponseStep>>>,
    source_id: String,
}

impl FauxProviderRegistration {
    pub fn get_model(&self) -> Model {
        self.models[0].clone()
    }

    pub fn get_model_by_id(&self, model_id: &str) -> Option<Model> {
        self.models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
    }

    pub fn set_responses<I, S>(&self, responses: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<FauxResponseStep>,
    {
        let mut pending = self
            .pending_responses
            .lock()
            .expect("faux response queue poisoned");
        *pending = responses
            .into_iter()
            .map(Into::into)
            .collect::<VecDeque<_>>();
    }

    pub fn append_responses<I, S>(&self, responses: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<FauxResponseStep>,
    {
        let mut pending = self
            .pending_responses
            .lock()
            .expect("faux response queue poisoned");
        pending.extend(responses.into_iter().map(Into::into));
    }

    pub fn get_pending_response_count(&self) -> usize {
        self.pending_responses
            .lock()
            .expect("faux response queue poisoned")
            .len()
    }

    pub fn unregister(&self) {
        api_registry::unregister_api_providers(&self.source_id);
    }
}

pub fn faux_text<T: Into<String>>(text: T) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: text.into(),
        text_signature: None,
    })
}

pub fn faux_thinking<T: Into<String>>(thinking: T) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: thinking.into(),
        thinking_signature: None,
        redacted: None,
    })
}

pub fn faux_tool_call<T: Into<String>>(
    name: T,
    arguments: Value,
    id: Option<String>,
) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.unwrap_or_else(|| random_id("tool")),
        name: name.into(),
        arguments,
        thought_signature: None,
    })
}

pub fn faux_assistant_message(
    content: impl Into<FauxAssistantContent>,
    options: Option<FauxAssistantMessageOptions>,
) -> AssistantMessage {
    let options = options.unwrap_or_default();
    AssistantMessage {
        content: normalize_faux_assistant_content(content.into()),
        api: DEFAULT_API.to_string(),
        provider: DEFAULT_PROVIDER.to_string(),
        model: DEFAULT_MODEL_ID.to_string(),
        response_model: None,
        response_id: options.response_id,
        diagnostics: Vec::new(),
        usage: default_usage(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        error_message: options.error_message,
        timestamp: options
            .timestamp
            .unwrap_or_else(crate::utils::time::now_millis),
    }
}

pub fn register_faux_provider(
    options: Option<RegisterFauxProviderOptions>,
) -> FauxProviderRegistration {
    let options = options.unwrap_or_default();
    let api = options.api.unwrap_or_else(|| random_id(DEFAULT_API));
    let provider = options
        .provider
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
    let source_id = random_id("faux-provider");
    let token_size = options.token_size.unwrap_or_default();
    let min_token_size = std::cmp::max(
        1,
        std::cmp::min(
            token_size.min.unwrap_or(DEFAULT_MIN_TOKEN_SIZE),
            token_size.max.unwrap_or(DEFAULT_MAX_TOKEN_SIZE),
        ),
    );
    let max_token_size = std::cmp::max(
        min_token_size,
        token_size.max.unwrap_or(DEFAULT_MAX_TOKEN_SIZE),
    );
    let pending_responses = Arc::new(Mutex::new(VecDeque::new()));
    let state = FauxProviderState::default();
    let prompt_cache = Arc::new(Mutex::new(HashMap::new()));

    let model_definitions = if options.models.is_empty() {
        vec![FauxModelDefinition {
            id: DEFAULT_MODEL_ID.to_string(),
            name: Some(DEFAULT_MODEL_NAME.to_string()),
            reasoning: Some(false),
            input: Some(vec![ModelInput::Text, ModelInput::Image]),
            cost: Some(ModelCost::default()),
            context_window: Some(128_000),
            max_tokens: Some(16_384),
        }]
    } else {
        options.models
    };
    let models = model_definitions
        .into_iter()
        .map(|definition| {
            let id = definition.id;
            Model {
                name: definition.name.unwrap_or_else(|| id.clone()),
                id,
                api: api.clone(),
                provider: provider.clone(),
                base_url: DEFAULT_BASE_URL.to_string(),
                reasoning: definition.reasoning.unwrap_or(false),
                input: definition
                    .input
                    .unwrap_or_else(|| vec![ModelInput::Text, ModelInput::Image]),
                cost: definition.cost.unwrap_or_default(),
                context_window: definition.context_window.unwrap_or(128_000),
                max_tokens: definition.max_tokens.unwrap_or(16_384),
                ..Model::default()
            }
        })
        .collect::<Vec<_>>();

    let stream_pending_responses = pending_responses.clone();
    let stream_prompt_cache = prompt_cache.clone();
    let stream_state = state.clone();
    let stream_api = api.clone();
    let stream_provider = provider.clone();
    let stream = Arc::new(
        move |request_model: Model,
              context: Context,
              stream_options: StreamOptions|
              -> Result<AssistantMessageEventStream> {
            if request_model.api != stream_api {
                return Err(Error::UnsupportedApi(format!(
                    "Mismatched api: {} expected {}",
                    request_model.api, stream_api
                )));
            }

            let (sender, output_stream) = AssistantMessageEventStream::channel();
            let step = stream_pending_responses
                .lock()
                .expect("faux response queue poisoned")
                .pop_front();
            stream_state.increment_call_count();

            let context = context.clone();
            let request_model_for_task = request_model.clone();
            let stream_api = stream_api.clone();
            let stream_provider = stream_provider.clone();
            let stream_state = stream_state.clone();
            let prompt_cache = stream_prompt_cache.clone();
            tokio::spawn(async move {
                stream_faux_response(
                    sender,
                    step,
                    request_model_for_task,
                    context,
                    stream_options,
                    stream_state,
                    stream_api,
                    stream_provider,
                    min_token_size,
                    max_token_size,
                    options.tokens_per_second,
                    prompt_cache,
                )
                .await;
            });

            Ok(output_stream)
        },
    );

    let simple_stream = {
        let stream = stream.clone();
        Arc::new(
            move |model: Model,
                  context: Context,
                  options: SimpleStreamOptions|
                  -> Result<AssistantMessageEventStream> {
                stream(model, context, options.stream)
            },
        )
    };

    api_registry::register_api_provider(
        ApiProvider {
            api: api.clone(),
            stream,
            stream_simple: simple_stream,
        },
        Some(source_id.clone()),
    );

    FauxProviderRegistration {
        api,
        models,
        state,
        pending_responses,
        source_id,
    }
}

async fn stream_faux_response(
    mut sender: AssistantMessageEventStreamSender,
    step: Option<FauxResponseStep>,
    request_model: Model,
    context: Context,
    stream_options: StreamOptions,
    state: FauxProviderState,
    api: String,
    provider: String,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    prompt_cache: Arc<Mutex<HashMap<String, String>>>,
) {
    let result: Result<TerminalFauxMessage> = async {
        if let Some(on_response) = stream_options.on_response.clone() {
            on_response(
                ProviderResponse {
                    status: 200,
                    headers: HashMap::new(),
                },
                &request_model,
            )
            .await?;
        }

        let Some(step) = step else {
            let message = create_error_message(
                "No more faux responses queued",
                &api,
                &provider,
                &request_model.id,
            );
            return Ok(TerminalFauxMessage::ImmediateError(with_usage_estimate(
                message,
                &context,
                &stream_options,
                &prompt_cache,
            )));
        };

        let resolved = match step {
            FauxResponseStep::Message(message) => Ok(message),
            FauxResponseStep::Factory(factory) => {
                factory(
                    context.clone(),
                    stream_options.clone(),
                    state,
                    request_model.clone(),
                )
                .await
            }
        }?;
        let message = clone_message(resolved, &api, &provider, &request_model.id);
        Ok(TerminalFauxMessage::Stream(with_usage_estimate(
            message,
            &context,
            &stream_options,
            &prompt_cache,
        )))
    }
    .await;

    match result {
        Ok(TerminalFauxMessage::ImmediateError(message)) => {
            sender.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: message,
            });
        }
        Ok(TerminalFauxMessage::Stream(message)) => {
            stream_with_deltas(
                &mut sender,
                message,
                min_token_size,
                max_token_size,
                tokens_per_second,
                stream_options,
            )
            .await;
        }
        Err(error) => {
            let message =
                create_error_message(error.to_string(), &api, &provider, &request_model.id);
            sender.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: message,
            });
        }
    }
}

enum TerminalFauxMessage {
    ImmediateError(AssistantMessage),
    Stream(AssistantMessage),
}

async fn stream_with_deltas(
    sender: &mut AssistantMessageEventStreamSender,
    message: AssistantMessage,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    stream_options: StreamOptions,
) {
    let mut partial = AssistantMessage {
        content: Vec::new(),
        ..message.clone()
    };

    if is_cancelled(&stream_options) {
        let aborted = create_aborted_message(partial);
        sender.push(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: aborted,
        });
        return;
    }

    sender.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    for (index, block) in message.content.iter().enumerate() {
        if is_cancelled(&stream_options) {
            let aborted = create_aborted_message(partial);
            sender.push(AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                error: aborted,
            });
            return;
        }

        match block {
            AssistantContent::Thinking(thinking) => {
                partial
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: thinking.thinking_signature.clone(),
                        redacted: thinking.redacted,
                    }));
                sender.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: partial.clone(),
                });
                for chunk in
                    split_string_by_token_size(&thinking.thinking, min_token_size, max_token_size)
                {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if is_cancelled(&stream_options) {
                        let aborted = create_aborted_message(partial);
                        sender.push(AssistantMessageEvent::Error {
                            reason: StopReason::Aborted,
                            error: aborted,
                        });
                        return;
                    }
                    if let Some(AssistantContent::Thinking(partial_thinking)) =
                        partial.content.get_mut(index)
                    {
                        partial_thinking.thinking.push_str(&chunk);
                    }
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: index,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content: thinking.thinking.clone(),
                    partial: partial.clone(),
                });
            }
            AssistantContent::Text(text) => {
                partial.content.push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: text.text_signature.clone(),
                }));
                sender.push(AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: partial.clone(),
                });
                for chunk in split_string_by_token_size(&text.text, min_token_size, max_token_size)
                {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if is_cancelled(&stream_options) {
                        let aborted = create_aborted_message(partial);
                        sender.push(AssistantMessageEvent::Error {
                            reason: StopReason::Aborted,
                            error: aborted,
                        });
                        return;
                    }
                    if let Some(AssistantContent::Text(partial_text)) =
                        partial.content.get_mut(index)
                    {
                        partial_text.text.push_str(&chunk);
                    }
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: index,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content: text.text.clone(),
                    partial: partial.clone(),
                });
            }
            AssistantContent::ToolCall(tool_call) => {
                partial.content.push(AssistantContent::ToolCall(ToolCall {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: json!({}),
                    thought_signature: tool_call.thought_signature.clone(),
                }));
                sender.push(AssistantMessageEvent::ToolCallStart {
                    content_index: index,
                    partial: partial.clone(),
                });
                let serialized_args = serde_json::to_string(&tool_call.arguments)
                    .unwrap_or_else(|_| "null".to_string());
                for chunk in
                    split_string_by_token_size(&serialized_args, min_token_size, max_token_size)
                {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if is_cancelled(&stream_options) {
                        let aborted = create_aborted_message(partial);
                        sender.push(AssistantMessageEvent::Error {
                            reason: StopReason::Aborted,
                            error: aborted,
                        });
                        return;
                    }
                    sender.push(AssistantMessageEvent::ToolCallDelta {
                        content_index: index,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                if let Some(AssistantContent::ToolCall(partial_tool_call)) =
                    partial.content.get_mut(index)
                {
                    partial_tool_call.arguments = tool_call.arguments.clone();
                }
                sender.push(AssistantMessageEvent::ToolCallEnd {
                    content_index: index,
                    tool_call: tool_call.clone(),
                    partial: partial.clone(),
                });
            }
        }
    }

    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
        sender.push(AssistantMessageEvent::Error {
            reason: message.stop_reason,
            error: message,
        });
        return;
    }

    sender.push(AssistantMessageEvent::Done {
        reason: message.stop_reason,
        message,
    });
}

fn default_usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 0,
        cost: UsageCost::default(),
    }
}

fn normalize_faux_assistant_content(content: FauxAssistantContent) -> Vec<AssistantContent> {
    match content {
        FauxAssistantContent::Text(text) => vec![faux_text(text)],
        FauxAssistantContent::Block(block) => vec![block],
        FauxAssistantContent::Blocks(blocks) => blocks,
    }
}

fn estimate_tokens(text: &str) -> u32 {
    text.chars().count().div_ceil(4) as u32
}

fn random_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    format!(
        "{}:{}:{}",
        prefix,
        crate::utils::time::now_millis(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn content_to_text(content: &UserMessageContent) -> String {
    match content {
        UserMessageContent::Text(text) => text.clone(),
        UserMessageContent::Parts(parts) => parts
            .iter()
            .map(user_content_block_to_text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn user_content_block_to_text(block: &UserContent) -> String {
    match block {
        UserContent::Text(text) => text.text.clone(),
        UserContent::Image(image) => image_to_text(image),
    }
}

fn tool_result_block_to_text(block: &ToolResultContent) -> String {
    match block {
        ToolResultContent::Text(text) => text.text.clone(),
        ToolResultContent::Image(image) => image_to_text(image),
    }
}

fn image_to_text(image: &ImageContent) -> String {
    format!("[image:{}:{}]", image.mime_type, image.data.len())
}

fn assistant_content_to_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantContent::Text(text) => text.text.clone(),
            AssistantContent::Thinking(thinking) => thinking.thinking.clone(),
            AssistantContent::ToolCall(tool_call) => {
                let arguments = serde_json::to_string(&tool_call.arguments)
                    .unwrap_or_else(|_| "null".to_string());
                format!("{}:{}", tool_call.name, arguments)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_to_text(message: &ToolResultMessage) -> String {
    std::iter::once(message.tool_name.clone())
        .chain(message.content.iter().map(tool_result_block_to_text))
        .collect::<Vec<_>>()
        .join("\n")
}

fn message_to_text(message: &Message) -> String {
    match message {
        Message::User(message) => content_to_text(&message.content),
        Message::Assistant(message) => assistant_content_to_text(&message.content),
        Message::ToolResult(message) => tool_result_to_text(message),
    }
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn serialize_context(context: &Context) -> String {
    let mut parts = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        parts.push(format!("system:{system_prompt}"));
    }
    for message in &context.messages {
        parts.push(format!(
            "{}:{}",
            message_role(message),
            message_to_text(message)
        ));
    }
    if !context.tools.is_empty() {
        let tools = serde_json::to_string(&context.tools).unwrap_or_else(|_| "[]".to_string());
        parts.push(format!("tools:{tools}"));
    }
    parts.join("\n\n")
}

fn common_prefix_length(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(left, right)| left == right)
        .count()
}

fn prefix_by_chars(text: &str, char_count: usize) -> String {
    text.chars().take(char_count).collect()
}

fn suffix_from_chars(text: &str, char_count: usize) -> String {
    text.chars().skip(char_count).collect()
}

fn with_usage_estimate(
    mut message: AssistantMessage,
    context: &Context,
    options: &StreamOptions,
    prompt_cache: &Arc<Mutex<HashMap<String, String>>>,
) -> AssistantMessage {
    let prompt_text = serialize_context(context);
    let prompt_tokens = estimate_tokens(&prompt_text);
    let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0;
    let mut cache_write = 0;

    if let Some(session_id) = &options.session_id {
        if !matches!(options.cache_retention, Some(CacheRetention::None)) {
            let mut cache = prompt_cache.lock().expect("faux prompt cache poisoned");
            if let Some(previous_prompt) = cache.get(session_id) {
                let cached_chars = common_prefix_length(previous_prompt, &prompt_text);
                cache_read = estimate_tokens(&prefix_by_chars(previous_prompt, cached_chars));
                cache_write = estimate_tokens(&suffix_from_chars(&prompt_text, cached_chars));
                input = prompt_tokens.saturating_sub(cache_read);
            } else {
                cache_write = prompt_tokens;
            }
            cache.insert(session_id.clone(), prompt_text);
        }
    }

    message.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens: input + output_tokens + cache_read + cache_write,
        cost: UsageCost::default(),
    };
    message
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

fn create_error_message(
    error: impl Into<String>,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: api.to_string(),
        provider: provider.to_string(),
        model: model_id.to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: default_usage(),
        stop_reason: StopReason::Error,
        error_message: Some(error.into()),
        timestamp: crate::utils::time::now_millis(),
    }
}

fn create_aborted_message(mut partial: AssistantMessage) -> AssistantMessage {
    partial.stop_reason = StopReason::Aborted;
    partial.error_message = Some("Request was aborted".to_string());
    partial.timestamp = crate::utils::time::now_millis();
    partial
}

async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
    let Some(tokens_per_second) = tokens_per_second else {
        tokio::task::yield_now().await;
        return;
    };
    if tokens_per_second <= 0.0 {
        tokio::task::yield_now().await;
        return;
    }
    let delay = (estimate_tokens(chunk) as f64 / tokens_per_second).max(0.0);
    tokio::time::sleep(Duration::from_secs_f64(delay)).await;
}

fn split_string_by_token_size(
    text: &str,
    min_token_size: usize,
    max_token_size: usize,
) -> Vec<String> {
    let token_size = std::cmp::max(1, std::cmp::min(min_token_size, max_token_size));
    let char_size = std::cmp::max(1, token_size * 4);
    let chars = text.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return vec![String::new()];
    }
    chars
        .chunks(char_size)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}

fn is_cancelled(options: &StreamOptions) -> bool {
    options
        .cancellation_token
        .as_ref()
        .is_some_and(|token| token.is_cancelled())
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::api_registry::get_api_provider;
    use crate::stream::{complete, stream};
    use crate::types::{
        AssistantContent, AssistantMessageEvent, Context, Message, StreamOptions, TextContent,
        Tool, ToolResultContent, ToolResultMessage, UserContent, UserMessage,
    };

    use super::*;

    async fn collect_events(mut stream: AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    #[tokio::test]
    async fn registers_custom_provider_and_estimates_usage() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message("hello world", None)]);

        let context = Context {
            system_prompt: Some("Be concise.".to_string()),
            messages: vec![Message::user_text("hi there")],
            tools: Vec::new(),
        };

        let response = complete(registration.get_model(), context, None)
            .await
            .expect("faux response");
        assert_eq!(response.content, vec![faux_text("hello world")]);
        assert!(response.usage.input > 0);
        assert!(response.usage.output > 0);
        assert_eq!(
            response.usage.total_tokens,
            response.usage.input + response.usage.output
        );
        assert_eq!(registration.state.call_count(), 1);

        registration.unregister();
    }

    #[tokio::test]
    async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message(
            vec![
                faux_thinking("think"),
                faux_tool_call("echo", json!({ "text": "hi" }), Some("tool-1".to_string())),
                faux_text("done"),
            ],
            Some(FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            }),
        )]);

        let response = complete(
            registration.get_model(),
            Context {
                messages: vec![Message::user_text("hi")],
                ..Context::default()
            },
            None,
        )
        .await
        .expect("faux response");

        assert_eq!(
            response.content,
            vec![
                faux_thinking("think"),
                faux_tool_call("echo", json!({ "text": "hi" }), Some("tool-1".to_string())),
                faux_text("done"),
            ]
        );
        assert_eq!(response.stop_reason, StopReason::ToolUse);

        registration.unregister();
    }

    #[tokio::test]
    async fn supports_multiple_models_and_model_aware_factories() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            models: vec![
                FauxModelDefinition {
                    id: "faux-fast".to_string(),
                    name: Some("Faux Fast".to_string()),
                    reasoning: Some(false),
                    ..Default::default()
                },
                FauxModelDefinition {
                    id: "faux-thinker".to_string(),
                    name: Some("Faux Thinker".to_string()),
                    reasoning: Some(true),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }));
        registration.set_responses([
            FauxResponseStep::factory(|_context, _options, _state, model| async move {
                Ok(faux_assistant_message(
                    format!("{}:{}", model.id, model.reasoning),
                    None,
                ))
            }),
            FauxResponseStep::factory(|_context, _options, _state, model| async move {
                Ok(faux_assistant_message(
                    format!("{}:{}", model.id, model.reasoning),
                    None,
                ))
            }),
        ]);

        assert_eq!(
            registration
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["faux-fast", "faux-thinker"]
        );
        assert!(!registration.get_model_by_id("faux-fast").unwrap().reasoning);
        assert!(
            registration
                .get_model_by_id("faux-thinker")
                .unwrap()
                .reasoning
        );

        let fast = complete(
            registration.get_model_by_id("faux-fast").unwrap(),
            Context {
                messages: vec![Message::user_text("hi")],
                ..Context::default()
            },
            None,
        )
        .await
        .expect("fast response");
        let thinker = complete(
            registration.get_model_by_id("faux-thinker").unwrap(),
            Context {
                messages: vec![Message::user_text("hi")],
                ..Context::default()
            },
            None,
        )
        .await
        .expect("thinker response");

        assert_eq!(fast.content, vec![faux_text("faux-fast:false")]);
        assert_eq!(thinker.content, vec![faux_text("faux-thinker:true")]);

        registration.unregister();
    }

    #[tokio::test]
    async fn consumes_queued_responses_and_errors_when_exhausted() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message("first", None),
            faux_assistant_message("second", None),
        ]);

        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Context::default()
        };
        let first = complete(registration.get_model(), context.clone(), None)
            .await
            .expect("first response");
        let second = complete(registration.get_model(), context.clone(), None)
            .await
            .expect("second response");
        let exhausted = complete(registration.get_model(), context, None)
            .await
            .expect("exhausted response");

        assert_eq!(first.content, vec![faux_text("first")]);
        assert_eq!(second.content, vec![faux_text("second")]);
        assert_eq!(exhausted.stop_reason, StopReason::Error);
        assert_eq!(
            exhausted.error_message.as_deref(),
            Some("No more faux responses queued")
        );
        assert_eq!(registration.get_pending_response_count(), 0);
        assert_eq!(registration.state.call_count(), 3);

        registration.unregister();
    }

    #[tokio::test]
    async fn estimates_prompt_and_output_tokens_from_serialized_context() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message("done", None)]);

        let tool = Tool {
            name: "echo".to_string(),
            description: "Echo back text".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            }),
        };
        let context = Context {
            system_prompt: Some("sys".to_string()),
            messages: vec![
                Message::User(UserMessage {
                    content: UserMessageContent::Parts(vec![
                        UserContent::Text(TextContent {
                            text: "hello".to_string(),
                            text_signature: None,
                        }),
                        UserContent::Image(ImageContent {
                            mime_type: "image/png".to_string(),
                            data: "abcd".to_string(),
                        }),
                    ]),
                    timestamp: 1,
                }),
                Message::Assistant(faux_assistant_message("prior", None)),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "tool-1".to_string(),
                    tool_name: "echo".to_string(),
                    content: vec![ToolResultContent::text("tool out")],
                    details: None,
                    is_error: false,
                    timestamp: 2,
                }),
            ],
            tools: vec![tool],
        };

        let response = complete(registration.get_model(), context.clone(), None)
            .await
            .expect("faux response");
        let expected_prompt_tokens = estimate_tokens(&serialize_context(&context));
        let expected_output_tokens = estimate_tokens("done");

        assert_eq!(response.usage.input, expected_prompt_tokens);
        assert_eq!(response.usage.output, expected_output_tokens);
        assert_eq!(response.usage.cache_read, 0);
        assert_eq!(response.usage.cache_write, 0);
        assert_eq!(
            response.usage.total_tokens,
            expected_prompt_tokens + expected_output_tokens
        );

        registration.unregister();
    }

    #[tokio::test]
    async fn simulates_prompt_caching_per_session_id() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message("first", None),
            faux_assistant_message("second", None),
            faux_assistant_message("third", None),
        ]);

        let mut context = Context {
            system_prompt: Some("Be concise.".to_string()),
            messages: vec![Message::user_text("hello")],
            ..Context::default()
        };

        let first = complete(
            registration.get_model(),
            context.clone(),
            Some(StreamOptions {
                session_id: Some("session-1".to_string()),
                cache_retention: Some(CacheRetention::Short),
                ..StreamOptions::default()
            }),
        )
        .await
        .expect("first response");
        assert_eq!(first.usage.cache_read, 0);
        assert!(first.usage.cache_write > 0);

        context.messages.push(Message::Assistant(first));
        context.messages.push(Message::user_text("follow up"));

        let second = complete(
            registration.get_model(),
            context.clone(),
            Some(StreamOptions {
                session_id: Some("session-1".to_string()),
                cache_retention: Some(CacheRetention::Short),
                ..StreamOptions::default()
            }),
        )
        .await
        .expect("second response");
        assert!(second.usage.cache_read > 0);

        let third = complete(
            registration.get_model(),
            context,
            Some(StreamOptions {
                session_id: Some("session-2".to_string()),
                cache_retention: Some(CacheRetention::Short),
                ..StreamOptions::default()
            }),
        )
        .await
        .expect("third response");
        assert_eq!(third.usage.cache_read, 0);
        assert!(third.usage.cache_write > 0);

        registration.unregister();
    }

    #[tokio::test]
    async fn streams_exact_event_order_for_fixed_size_chunks() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..Default::default()
        }));
        registration.set_responses([faux_assistant_message(
            vec![
                faux_thinking("go"),
                faux_text("ok"),
                faux_tool_call("echo", json!({}), Some("tool-1".to_string())),
            ],
            Some(FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            }),
        )]);

        let events = collect_events(
            stream(
                registration.get_model(),
                Context {
                    messages: vec![Message::user_text("hi")],
                    ..Context::default()
                },
                None,
            )
            .expect("faux stream"),
        )
        .await;

        assert_eq!(
            events
                .iter()
                .map(|event| match event {
                    AssistantMessageEvent::Start { .. } => "start",
                    AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
                    AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
                    AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
                    AssistantMessageEvent::TextStart { .. } => "text_start",
                    AssistantMessageEvent::TextDelta { .. } => "text_delta",
                    AssistantMessageEvent::TextEnd { .. } => "text_end",
                    AssistantMessageEvent::ToolCallStart { .. } => "toolcall_start",
                    AssistantMessageEvent::ToolCallDelta { .. } => "toolcall_delta",
                    AssistantMessageEvent::ToolCallEnd { .. } => "toolcall_end",
                    AssistantMessageEvent::Done { .. } => "done",
                    AssistantMessageEvent::Error { .. } => "error",
                })
                .collect::<Vec<_>>(),
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

        registration.unregister();
    }

    #[tokio::test]
    async fn supports_aborting_mid_text_stream_when_paced() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        }));
        registration.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz", None)]);

        let cancellation_token = CancellationToken::new();
        let mut stream = stream(
            registration.get_model(),
            Context {
                messages: vec![Message::user_text("hi")],
                ..Context::default()
            },
            Some(StreamOptions {
                cancellation_token: Some(cancellation_token.clone()),
                ..StreamOptions::default()
            }),
        )
        .expect("faux stream");
        let mut text_delta_count = 0;
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
                text_delta_count += 1;
                cancellation_token.cancel();
            }
            events.push(event);
        }

        assert_eq!(text_delta_count, 1);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AssistantMessageEvent::TextStart { .. }))
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                ..
            }
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
        );

        registration.unregister();
    }

    #[tokio::test]
    async fn unregisters_provider() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message("hello", None)]);
        let api = registration.api.clone();
        registration.unregister();

        assert!(get_api_provider(&api).is_none());
        let error = complete(
            registration.get_model(),
            Context {
                messages: vec![Message::user_text("hi")],
                ..Context::default()
            },
            None,
        )
        .await
        .expect_err("provider should be unregistered");
        assert!(matches!(error, Error::UnsupportedApi(_)));
    }

    #[test]
    fn split_empty_text_into_one_chunk() {
        assert_eq!(split_string_by_token_size("", 1, 1), vec![""]);
    }

    #[test]
    fn content_helpers_return_assistant_content_variants() {
        assert!(matches!(faux_text("x"), AssistantContent::Text(_)));
        assert!(matches!(faux_thinking("x"), AssistantContent::Thinking(_)));
        assert!(matches!(
            faux_tool_call("echo", json!({}), None),
            AssistantContent::ToolCall(_)
        ));
    }
}
