//! Port of `packages/ai/src/types.ts`.
//!
//! JSON shapes match Pi's wire format (camelCase field names, `role`/`type`
//! discriminators). JavaScript object key order is significant in Pi (system
//! prompt sections, header merge order, tool schemas), so maps use
//! [`IndexMap`] and `serde_json` is built with `preserve_order`.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Result;
use crate::utils::diagnostics::AssistantMessageDiagnostic;

pub use crate::utils::event_stream::AssistantMessageEventStream;

pub type Api = String;
pub type ImageApi = String;
pub type ProviderId = String;

/// Image APIs with a built-in implementation (`KnownImageApi`).
/// `OpenaiImages` is an ai.rs extra (OpenAI-compatible `/images/generations`),
/// not part of Pi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownImageApi {
    OpenrouterImages,
    OpenaiImages,
}

impl KnownImageApi {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenrouterImages => "openrouter-images",
            Self::OpenaiImages => "openai-images",
        }
    }
}

/// APIs with a built-in implementation in Pi (`KnownApi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownApi {
    OpenaiCompletions,
    MistralConversations,
    OpenaiResponses,
    AzureOpenaiResponses,
    OpenaiCodexResponses,
    AnthropicMessages,
    BedrockConverseStream,
    GoogleGenerativeAi,
    GoogleVertex,
    PiMessages,
}

impl KnownApi {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenaiCompletions => "openai-completions",
            Self::MistralConversations => "mistral-conversations",
            Self::OpenaiResponses => "openai-responses",
            Self::AzureOpenaiResponses => "azure-openai-responses",
            Self::OpenaiCodexResponses => "openai-codex-responses",
            Self::AnthropicMessages => "anthropic-messages",
            Self::BedrockConverseStream => "bedrock-converse-stream",
            Self::GoogleGenerativeAi => "google-generative-ai",
            Self::GoogleVertex => "google-vertex",
            Self::PiMessages => "pi-messages",
        }
    }
}

impl From<KnownApi> for String {
    fn from(value: KnownApi) -> Self {
        value.as_str().to_string()
    }
}

impl fmt::Display for KnownApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ThinkingLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ModelThinkingLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "off" => Some(Self::Off),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::Xhigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// The level as a [`ThinkingLevel`], or `None` for `off`.
    pub const fn thinking_level(self) -> Option<ThinkingLevel> {
        match self {
            Self::Off => None,
            Self::Minimal => Some(ThinkingLevel::Minimal),
            Self::Low => Some(ThinkingLevel::Low),
            Self::Medium => Some(ThinkingLevel::Medium),
            Self::High => Some(ThinkingLevel::High),
            Self::Xhigh => Some(ThinkingLevel::Xhigh),
            Self::Max => Some(ThinkingLevel::Max),
        }
    }
}

impl From<ThinkingLevel> for ModelThinkingLevel {
    fn from(value: ThinkingLevel) -> Self {
        match value {
            ThinkingLevel::Minimal => Self::Minimal,
            ThinkingLevel::Low => Self::Low,
            ThinkingLevel::Medium => Self::Medium,
            ThinkingLevel::High => Self::High,
            ThinkingLevel::Xhigh => Self::Xhigh,
            ThinkingLevel::Max => Self::Max,
        }
    }
}

impl fmt::Display for ModelThinkingLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maps pi thinking levels to provider/model-specific values. Missing keys
/// use provider defaults; `None` (JSON `null`) marks a level as unsupported.
pub type ThinkingLevelMap = IndexMap<ModelThinkingLevel, Option<String>>;
pub type SamplingParams = serde_json::Map<String, Value>;
pub type SamplingParamsByThinkingLevel = IndexMap<ModelThinkingLevel, SamplingParams>;

/// `ChatTemplateKwargValue`: a literal or a `{ "$var": ... }` reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateKwargValue {
    String(String),
    Number(serde_json::Number),
    Boolean(bool),
    Null(()),
    Variable(ChatTemplateKwargVariable),
}

impl ChatTemplateKwargValue {
    pub fn from_f64(value: f64) -> Option<Self> {
        serde_json::Number::from_f64(value).map(Self::Number)
    }

    pub fn variable(variable: ChatTemplateVariable, omit_when_off: bool) -> Self {
        Self::Variable(ChatTemplateKwargVariable {
            variable,
            omit_when_off,
        })
    }
}

impl From<String> for ChatTemplateKwargValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for ChatTemplateKwargValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<bool> for ChatTemplateKwargValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

macro_rules! impl_chat_template_kwarg_number {
    ($($type:ty),+ $(,)?) => {
        $(
            impl From<$type> for ChatTemplateKwargValue {
                fn from(value: $type) -> Self {
                    Self::Number(value.into())
                }
            }
        )+
    };
}

impl_chat_template_kwarg_number!(i8, i16, i32, i64, u8, u16, u32, u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTemplateKwargVariable {
    #[serde(rename = "$var")]
    pub variable: ChatTemplateVariable,
    #[serde(default, skip_serializing_if = "is_false")]
    pub omit_when_off: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatTemplateVariable {
    #[serde(rename = "thinking.enabled")]
    ThinkingEnabled,
    #[serde(rename = "thinking.effort")]
    ThinkingEffort,
    #[serde(rename = "thinking.budget")]
    ThinkingBudget,
}

/// Top-level request field used to cap reasoning tokens on OpenAI-compatible servers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingTokenBudgetField {
    ThinkingTokenBudget,
    ThinkingBudget,
    ThinkingBudgetTokens,
}

/// Token budgets for each thinking level (token-based providers only).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

/// Best-effort prompt cache lifetime in seconds for each retention tier.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPromptCache {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    Sse,
    Websocket,
    WebsocketCached,
    Auto,
}

/// Provider-scoped environment overrides. Values take precedence over the
/// process environment.
pub type ProviderEnv = HashMap<String, String>;

/// Ordered header overrides. A `None` value suppresses a default header with
/// the same name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderHeaders(IndexMap<String, Option<String>>);

impl ProviderHeaders {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        name: impl Into<String>,
        value: impl Into<Option<String>>,
    ) -> Option<Option<String>> {
        self.0.insert(name.into(), value.into())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Option<String>)> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, name: &str) -> Option<&Option<String>> {
        self.0.get(name)
    }

    pub fn shift_remove(&mut self, name: &str) -> Option<Option<String>> {
        self.0.shift_remove(name)
    }
}

impl<K, V> FromIterator<(K, V)> for ProviderHeaders
where
    K: Into<String>,
    V: Into<Option<String>>,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut headers = Self::default();
        for (name, value) in iter {
            headers.insert(name, value);
        }
        headers
    }
}

impl<'a> IntoIterator for &'a ProviderHeaders {
    type Item = (&'a String, &'a Option<String>);
    type IntoIter = indexmap::map::Iter<'a, String, Option<String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl From<IndexMap<String, String>> for ProviderHeaders {
    fn from(headers: IndexMap<String, String>) -> Self {
        headers.into_iter().collect()
    }
}

impl From<HashMap<String, String>> for ProviderHeaders {
    fn from(headers: HashMap<String, String>) -> Self {
        headers.into_iter().collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionAffinityFormat {
    Openai,
    OpenaiNosession,
    Openrouter,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: IndexMap<String, String>,
}

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// `onPayload`: inspect or replace a provider payload before sending.
/// Resolve to `None` to keep the payload unchanged.
pub type PayloadHook = Arc<dyn Fn(Value, &Model) -> BoxFuture<Result<Option<Value>>> + Send + Sync>;
/// `onResponse`: invoked after an HTTP response is received.
pub type ResponseHook =
    Arc<dyn Fn(ProviderResponse, &Model) -> BoxFuture<Result<()>> + Send + Sync>;
/// `onProviderStreamEvent`: observe each parsed provider stream event.
pub type ProviderStreamEventHook = Arc<dyn Fn(&Value, &Model) -> BoxFuture<()> + Send + Sync>;

/// Authentication, HTTP transport, and lifecycle callbacks shared by provider
/// requests (`ProviderRequestOptions`).
///
/// `fetch` becomes `http_client`; `telemetryContext` is not ported.
#[derive(Clone, Default)]
pub struct ProviderRequestOptions {
    pub signal: Option<CancellationToken>,
    pub api_key: Option<String>,
    /// Optional HTTP client for provider requests (Pi's `fetch` option).
    pub http_client: Option<reqwest::Client>,
    pub env: Option<ProviderEnv>,
    pub on_payload: Option<PayloadHook>,
    pub on_response: Option<ResponseHook>,
    pub headers: Option<ProviderHeaders>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    /// Maximum delay to wait when the server requests a long retry delay.
    /// Default 60000; zero disables the cap.
    pub max_retry_delay_ms: Option<u64>,
}

/// Options for `stream()` / `complete()` (`StreamOptions & Record<string, unknown>`).
///
/// Pi's provider-specific option bags (`ApiStreamOptions<TApi>`) become
/// `provider_options`, an ordered map that API modules read by Pi's field
/// names.
#[derive(Clone, Default)]
pub struct StreamOptions {
    pub signal: Option<CancellationToken>,
    pub api_key: Option<String>,
    /// Optional HTTP client for provider requests (Pi's `fetch` option).
    pub http_client: Option<reqwest::Client>,
    pub env: Option<ProviderEnv>,
    pub on_payload: Option<PayloadHook>,
    pub on_response: Option<ResponseHook>,
    pub on_provider_stream_event: Option<ProviderStreamEventHook>,
    pub headers: Option<ProviderHeaders>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub temperature: Option<f64>,
    pub sampling_params: Option<SamplingParams>,
    pub max_tokens: Option<u32>,
    pub transport: Option<Transport>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub websocket_connect_timeout_ms: Option<u64>,
    pub metadata: Option<serde_json::Map<String, Value>>,
    /// API-specific options (`ProviderStreamOptions` record entries).
    pub provider_options: serde_json::Map<String, Value>,
}

impl StreamOptions {
    /// The `ProviderRequestOptions` subset of these options.
    pub fn request_options(&self) -> ProviderRequestOptions {
        ProviderRequestOptions {
            signal: self.signal.clone(),
            api_key: self.api_key.clone(),
            http_client: self.http_client.clone(),
            env: self.env.clone(),
            on_payload: self.on_payload.clone(),
            on_response: self.on_response.clone(),
            headers: self.headers.clone(),
            timeout_ms: self.timeout_ms,
            max_retries: self.max_retries,
            max_retry_delay_ms: self.max_retry_delay_ms,
        }
    }
}

impl fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamOptions")
            .field("signal", &self.signal)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("transport", &self.transport)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id)
            .field("provider_options", &self.provider_options)
            .finish_non_exhaustive()
    }
}

/// Options for fetching a deferred response.
#[derive(Clone, Default)]
pub struct DeferredFetchOptions {
    pub request: ProviderRequestOptions,
    /// Maximum provider long-poll duration in milliseconds. Defaults to 0,
    /// which performs one status check.
    pub wait: Option<u64>,
}

/// Request options for best-effort deferred-response cancellation.
pub type DeferredCancelOptions = ProviderRequestOptions;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicAllowedFallbackModel {
    pub provider: ProviderId,
    pub model: String,
    pub cost: ModelCost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferredWindow {
    #[serde(rename = "15m")]
    Minutes15,
    #[serde(rename = "1h")]
    Hour1,
    #[serde(rename = "24h")]
    Hours24,
}

/// `deferred?: boolean | { window?: "15m" | "1h" | "24h" }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredRequest {
    Flag(bool),
    Window(Option<DeferredWindow>),
}

impl DeferredRequest {
    /// JavaScript truthiness of the option.
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Flag(enabled) => *enabled,
            Self::Window(_) => true,
        }
    }

    pub fn window(&self) -> Option<DeferredWindow> {
        match self {
            Self::Flag(_) => None,
            Self::Window(window) => *window,
        }
    }
}

/// Unified options with reasoning passed to `stream_simple()` and
/// `complete_simple()`. Derefs to the inherited [`StreamOptions`].
#[derive(Clone, Default, Debug)]
pub struct SimpleStreamOptions {
    pub stream: StreamOptions,
    /// Provider-neutral tool selection. When omitted, adapters use
    /// provider-specific behavior.
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: Option<ThinkingLevel>,
    /// Ask a capable provider to return a durable handle and continue the
    /// request asynchronously.
    pub deferred: Option<DeferredRequest>,
    /// Custom token budgets for thinking levels (token-based providers only).
    pub thinking_budgets: Option<ThinkingBudgets>,
}

impl std::ops::Deref for SimpleStreamOptions {
    type Target = StreamOptions;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl std::ops::DerefMut for SimpleStreamOptions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stream
    }
}

impl From<StreamOptions> for SimpleStreamOptions {
    fn from(stream: StreamOptions) -> Self {
        Self {
            stream,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSignatureV1 {
    pub v: u8,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<TextPhase>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextPhase {
    Commentary,
    FinalAnswer,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    /// e.g. OpenAI Responses message metadata (legacy id string or
    /// `TextSignatureV1` JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_signature: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    /// Provider-specific opaque or serialized reasoning replay data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// When true, the thinking content was redacted by safety filters and the
    /// encrypted payload is stored in `thinking_signature`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64 encoded image data.
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON object arguments (`JsonObject`).
    pub arguments: Value,
    /// Google-specific opaque signature for reusing thought context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// OpenAI Responses namespace for calls to dynamically loaded or
    /// namespaced tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// `TextContent | ImageContent` in user messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UserContent {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "image")]
    Image(ImageContent),
}

impl UserContent {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text(TextContent::new(text))
    }
}

/// `TextContent | ImageContent` in tool results.
pub type ToolResultContent = UserContent;

/// `TextContent | ThinkingContent | ToolCall`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantContent {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
}

impl AssistantContent {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text(TextContent::new(text))
    }
}

/// `TextContent` inside a system message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SystemContent {
    #[serde(rename = "text")]
    Text(TextContent),
}

/// `string | TextContent[]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemMessageContent {
    Text(String),
    Parts(Vec<SystemContent>),
}

impl Default for SystemMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for SystemMessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for SystemMessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

/// `string | (TextContent | ImageContent)[]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<UserContent>),
}

impl Default for UserMessageContent {
    fn default() -> Self {
        Self::Parts(Vec::new())
    }
}

impl From<String> for UserMessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for UserMessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u32,
    pub output: u32,
    pub cache_read: u32,
    pub cache_write: u32,
    /// Subset of `cache_write` written with 1h retention. Only Anthropic
    /// reports this split.
    #[serde(rename = "cacheWrite1h", skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u32>,
    /// Reasoning/thinking tokens, a subset of `output`, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u32>,
    pub total_tokens: u32,
    pub cost: UsageCost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

impl StopReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolUse => "toolUse",
            Self::Error => "error",
            Self::Aborted => "aborted",
            Self::Deferred => "deferred",
        }
    }
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
    pub provider: String,
    pub model_id: String,
    pub api: String,
    /// Provider token, such as a response id or batch id plus row id.
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    /// Provider conversion data required to reconstruct the final message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// System instructions and tool declarations at one point in the transcript.
///
/// The leading system message is the system prompt. Later system messages
/// change it: `content` adds instructions from that point on, `sections`
/// replace or remove named prompt sections, and `tools_added`/`tools_removed`
/// change the tool set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "system", rename_all = "camelCase")]
pub struct SystemMessage {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub content: SystemMessageContent,
    /// Named, ordered prompt sections rendered verbatim after `content`; a
    /// `None` value removes a section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<IndexMap<String, Option<String>>>,
    /// Complete definitions of tools that become available at this point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_added: Option<Vec<Tool>>,
    /// Tools that stop being available at this point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_removed: Option<Vec<ToolReference>>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "user", rename_all = "camelCase")]
pub struct UserMessage {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub content: UserMessageContent,
    pub timestamp: u64,
}

impl UserMessage {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self {
            content: UserMessageContent::Text(text.into()),
            timestamp: crate::utils::time::now_millis(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "assistant", rename_all = "camelCase")]
pub struct AssistantMessage {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub content: Vec<AssistantContent>,
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    /// Concrete model reported by the provider when different from `model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-specific response/message identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Exact provider-native effort level used for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    /// Pi thinking level the agent loop requested for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ModelThinkingLevel>,
    /// Redacted provider/runtime diagnostics for failures and recoveries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Provider indication of whether the model explicitly ended its turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    pub timestamp: u64,
}

impl AssistantMessage {
    /// An empty message attributed to `model`, the usual starting `output`
    /// of API implementations.
    pub fn empty_for(model: &Model) -> Self {
        Self {
            content: Vec::new(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NestedToolCallStatus {
    Ok,
    Error,
    Unfinished,
}

/// A tool call that another tool made while it ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedToolCallRecord {
    pub id: String,
    pub name: String,
    /// Omitted when over the size limits; `arguments_bytes` then gives the size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_bytes: Option<u64>,
    pub status: NestedToolCallStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Bounded record of the nested calls a tool made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NestedToolCalls {
    pub calls: Vec<NestedToolCallRecord>,
    /// False when calls were dropped, arguments omitted, or calls had not finished.
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "toolResult", rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub content: Vec<ToolResultContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage from the tool execution itself. Not part of main LLM context accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Calls this tool made to other tools. Kept for the session record; not
    /// sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_calls: Option<NestedToolCalls>,
    pub is_error: bool,
    pub timestamp: u64,
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Option::<T>::deserialize(deserializer).map(Option::unwrap_or_default)
}

fn is_false(value: &bool) -> bool {
    !value
}

/// `SystemMessage | UserMessage | AssistantMessage | ToolResultMessage`.
#[allow(clippy::large_enum_variant)] // mirrors Pi's plain unions
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn user_text<T: Into<String>>(text: T) -> Self {
        Self::User(UserMessage::text(text))
    }

    pub fn role(&self) -> &'static str {
        match self {
            Self::System(_) => "system",
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::ToolResult(_) => "toolResult",
        }
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            Self::System(message) => message.timestamp,
            Self::User(message) => message.timestamp,
            Self::Assistant(message) => message.timestamp,
            Self::ToolResult(message) => message.timestamp,
        }
    }
}

impl From<SystemMessage> for Message {
    fn from(value: SystemMessage) -> Self {
        Self::System(value)
    }
}

impl From<UserMessage> for Message {
    fn from(value: UserMessage) -> Self {
        Self::User(value)
    }
}

impl From<AssistantMessage> for Message {
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant(value)
    }
}

impl From<ToolResultMessage> for Message {
    fn from(value: ToolResultMessage) -> Self {
        Self::ToolResult(value)
    }
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::System(message) => message.serialize(serializer),
            Self::User(message) => message.serialize(serializer),
            Self::Assistant(message) => message.serialize(serializer),
            Self::ToolResult(message) => message.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let role = value
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| de::Error::custom("missing message role"))?;
        match role {
            "system" => serde_json::from_value(value)
                .map(Self::System)
                .map_err(de::Error::custom),
            "user" => serde_json::from_value(value)
                .map(Self::User)
                .map_err(de::Error::custom),
            "assistant" => serde_json::from_value(value)
                .map(Self::Assistant)
                .map_err(de::Error::custom),
            "toolResult" => serde_json::from_value(value)
                .map(Self::ToolResult)
                .map_err(de::Error::custom),
            other => Err(de::Error::custom(format!("unknown message role: {other}"))),
        }
    }
}

/// OpenAI grammar variants for constrained sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarFormat {
    OpenaiLark,
    OpenaiRegex,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarVariants {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConstrainedSamplingStrict {
    Prefer,
    Require,
}

/// Optional provider-side constrained sampling config for a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSamplingConfig {
    JsonSchema { strict: ConstrainedSamplingStrict },
    Grammar { variants: GrammarVariants },
}

/// `false | ConstrainedSamplingConfig`.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstrainedSampling {
    Disabled,
    Config(ConstrainedSamplingConfig),
}

impl From<ConstrainedSamplingConfig> for ConstrainedSampling {
    fn from(config: ConstrainedSamplingConfig) -> Self {
        Self::Config(config)
    }
}

impl Serialize for ConstrainedSampling {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Disabled => serializer.serialize_bool(false),
            Self::Config(config) => config.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ConstrainedSampling {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value == Value::Bool(false) {
            return Ok(Self::Disabled);
        }
        if value.is_boolean() {
            return Err(de::Error::custom(
                "constrainedSampling only accepts false or a configuration",
            ));
        }
        serde_json::from_value(value)
            .map(Self::Config)
            .map_err(de::Error::custom)
    }
}

/// A tool declaration. `parameters` is a JSON Schema object (Pi uses
/// TypeBox schemas, which are plain JSON Schema at runtime).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSampling>,
}

impl Tool {
    pub fn builder(name: impl Into<String>) -> ToolBuilder {
        ToolBuilder {
            name: name.into(),
            description: None,
            parameters: None,
            constrained_sampling: None,
        }
    }
}

/// Rust convenience builder for [`Tool`] (not in Pi).
#[derive(Debug, Clone)]
pub struct ToolBuilder {
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    constrained_sampling: Option<ConstrainedSampling>,
}

impl ToolBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn parameters(mut self, parameters: Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn constrained_sampling(mut self, config: impl Into<ConstrainedSampling>) -> Self {
        self.constrained_sampling = Some(config.into());
        self
    }

    pub fn build(self) -> Result<Tool> {
        let name = self.name.trim().to_string();
        if name.is_empty() {
            return Err(crate::Error::Validation(
                "tool name must not be empty".to_string(),
            ));
        }

        let description = self
            .description
            .map(|description| description.trim().to_string())
            .filter(|description| !description.is_empty())
            .ok_or_else(|| {
                crate::Error::Validation("tool description must not be empty".to_string())
            })?;

        let parameters = self
            .parameters
            .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} }));
        if !parameters.is_object() {
            return Err(crate::Error::Validation(
                "tool parameters must be a JSON object".to_string(),
            ));
        }

        Ok(Tool {
            name,
            description,
            parameters,
            constrained_sampling: self.constrained_sampling,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReference {
    pub name: String,
}

/// Request input accepted by the public stream entry points. `system_prompt`
/// and `tools` are shorthand for a leading system message;
/// `normalize_context()` folds them into one before the request reaches a
/// provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl Context {
    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }
}

/// Rust convenience builder for [`Context`] (not in Pi).
#[derive(Debug, Clone, Default)]
pub struct ContextBuilder {
    context: Context,
}

impl ContextBuilder {
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.context.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.context.messages.push(message.into());
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.context.messages.extend(messages);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.context.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.context
            .tools
            .get_or_insert_with(Vec::new)
            .extend(tools);
        self
    }

    pub fn build(self) -> Context {
        self.context
    }
}

/// Normalized request context passed to providers and API implementations.
/// The prompt and tool declarations are carried by the transcript's system
/// messages. Only `normalize_context()` (and the transcript helpers) produce
/// this type, so a raw [`Context`] cannot reach provider code by accident.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct TranscriptContext {
    pub messages: Vec<Message>,
}

impl TranscriptContext {
    pub(crate) fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

/// The uniform stream contract of an API implementation module
/// (`ProviderStreams`): every module under `api/` provides `stream` and
/// `stream_simple`; capable modules may also provide deferred-response
/// methods (`supports_*` reports whether Pi's optional method exists).
///
/// Implementations return immediately and produce events from a spawned
/// Tokio task, so they must be called inside a Tokio runtime.
#[async_trait::async_trait]
pub trait ProviderStreams: Send + Sync {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> crate::utils::event_stream::AssistantMessageEventStream;

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> crate::utils::event_stream::AssistantMessageEventStream;

    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    fn fetch_deferred(
        &self,
        model: Model,
        _handle: DeferredHandle,
        _options: DeferredFetchOptions,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        crate::api::lazy::error_stream(&model, "API does not support deferred responses")
    }

    fn supports_cancel_deferred(&self) -> bool {
        false
    }

    async fn cancel_deferred(
        &self,
        _model: Model,
        _handle: DeferredHandle,
        _options: DeferredCancelOptions,
    ) -> Result<()> {
        Err(crate::Error::message(
            "API cannot cancel deferred responses",
        ))
    }
}

/// Event protocol for [`AssistantMessageEventStream`].
///
/// Successful streams emit `Start` before partial updates and terminate with
/// `Done`. A stream may terminate directly with `Error` when request setup
/// fails before generation starts. `partial` is a snapshot of the response so
/// far at the time of the event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
    #[serde(rename = "start")]
    Start { partial: AssistantMessage },
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    /// `reason` is one of `Stop`, `Length`, `ToolUse` or `Deferred`.
    #[serde(rename = "done")]
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    /// `reason` is `Aborted` or `Error`.
    #[serde(rename = "error")]
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// The Pi event `type` string.
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::Start { .. } => "start",
            Self::TextStart { .. } => "text_start",
            Self::TextDelta { .. } => "text_delta",
            Self::TextEnd { .. } => "text_end",
            Self::ThinkingStart { .. } => "thinking_start",
            Self::ThinkingDelta { .. } => "thinking_delta",
            Self::ThinkingEnd { .. } => "thinking_end",
            Self::ToolCallStart { .. } => "toolcall_start",
            Self::ToolCallDelta { .. } => "toolcall_delta",
            Self::ToolCallEnd { .. } => "toolcall_end",
            Self::Done { .. } => "done",
            Self::Error { .. } => "error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    MaxCompletionTokens,
    MaxTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAIThinkingFormat {
    Openai,
    Openrouter,
    Deepseek,
    Together,
    Baseten,
    Zai,
    Qwen,
    ChatTemplate,
    QwenChatTemplate,
    StringThinking,
    AntLing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlFormat {
    Anthropic,
}

/// OpenRouter provider routing preferences, sent as the `provider` request field.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterRouting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// `"deny" | "allow"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    /// A string or `{ by, partition }`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<Value>,
}

/// Vercel AI Gateway routing preferences.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VercelGatewayRouting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
}

/// Compatibility overrides for a model.
///
/// Pi types `Model.compat` per API (`OpenAICompletionsCompat`,
/// `OpenAIResponsesCompat`, `AnthropicMessagesCompat`, `BedrockCompat`,
/// `MistralConversationsCompat`). Those interfaces share field names, so the
/// Rust port uses one struct holding the union of their fields; each API
/// reads the fields its TypeScript interface declares. The per-API names are
/// kept as aliases.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCompat {
    // OpenAICompletionsCompat
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_finish_reason: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<OpenAIThinkingFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<IndexMap<String, ChatTemplateKwargValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_args: Option<IndexMap<String, ChatTemplateKwargValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<OpenRouterRouting>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_thinking_token_budget: Option<bool>,
    #[serde(
        default,
        rename = "supportsOpenAIGrammarTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_system_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_tool_additions: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<CacheControlFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm_priority: Option<i64>,
    // OpenAIResponsesCompat
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_additional_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_max_output_tokens: Option<bool>,
    // AnthropicMessagesCompat
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_tool_changes: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
}

pub type OpenAICompletionsCompat = ModelCompat;
pub type OpenAIResponsesCompat = ModelCompat;
pub type AnthropicMessagesCompat = ModelCompat;
pub type BedrockCompat = ModelCompat;
pub type MistralConversationsCompat = ModelCompat;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    /// Use this tier for requests whose total input usage exceeds this token count.
    pub input_tokens_above: u32,
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    /// Request-wide pricing tiers. The highest matching input threshold
    /// applies to the full request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageResizeOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height: Option<u32>,
    /// Maximum base64-encoded payload size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jpeg_quality: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageInputLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resize: Option<ModelImageResizeOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_message: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_request: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInputLimits {
    /// Maximum serialized provider request size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<ModelImageInputLimits>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelInput {
    Text,
    Image,
}

pub type ModelOutput = ModelInput;

/// What a catalog entry is for (`ModelType`). Classifier models are not
/// ported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Chat,
    Image,
}

/// Chat model: usable with `Models::stream()` and friends (`Model<Api>`).
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    pub base_url: String,
    /// Optional: chat is the default model type.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<ModelType>,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<ModelPromptCache>,
    #[serde(default)]
    pub input: Vec<ModelInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    pub cost: ModelCost,
    pub context_window: u32,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params_by_thinking_level: Option<SamplingParamsByThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("api", &self.api)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("model_type", &self.model_type)
            .field("reasoning", &self.reasoning)
            .field("thinking_level_map", &self.thinking_level_map)
            .field("prompt_cache", &self.prompt_cache)
            .field("input", &self.input)
            .field("input_limits", &self.input_limits)
            .field("cost", &self.cost)
            .field("context_window", &self.context_window)
            .field("max_tokens", &self.max_tokens)
            .field("headers", &self.headers.as_ref().map(|_| "<redacted>"))
            .field("sampling_params", &self.sampling_params)
            .field(
                "sampling_params_by_thinking_level",
                &self.sampling_params_by_thinking_level,
            )
            .field("compat", &self.compat)
            .finish()
    }
}

impl Model {
    pub fn compat(&self) -> ModelCompat {
        self.compat.clone().unwrap_or_default()
    }
}

/// Image-generation model: usable with `Models::generate_images()` only.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModel {
    pub id: String,
    pub name: String,
    pub api: ImageApi,
    pub provider: ProviderId,
    pub base_url: String,
    #[serde(rename = "type")]
    pub model_type: ImageModelType,
    #[serde(default)]
    pub input: Vec<ModelInput>,
    /// Output modalities. Always includes `Image`.
    #[serde(default)]
    pub output: Vec<ModelOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    pub cost: ModelCost,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
}

/// The `"image"` discriminator of [`ImageModel`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageModelType {
    #[default]
    Image,
}

/// Anything a provider can list (`AnyModel`). Narrow with
/// `is_model_type()`.
#[allow(clippy::large_enum_variant)] // mirrors Pi's plain unions
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum AnyModel {
    Chat(Model),
    Image(ImageModel),
}

impl<'de> Deserialize<'de> for AnyModel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            Some("image") => serde_json::from_value(value)
                .map(Self::Image)
                .map_err(de::Error::custom),
            _ => serde_json::from_value(value)
                .map(Self::Chat)
                .map_err(de::Error::custom),
        }
    }
}

/// Pi's `hasKnownModelType()` on a raw model: a model without a `type` (or
/// with `type: null`) is a chat model; `chat` and `image` are known.
///
/// Divergence: Pi also knows `classifier`. Classifier models are not ported,
/// so they are treated like any other type this version does not know.
pub fn has_known_model_type(model: &Value) -> bool {
    match model.get("type") {
        None | Some(Value::Null) => true,
        Some(Value::String(model_type)) => matches!(model_type.as_str(), "chat" | "image"),
        Some(_) => false,
    }
}

/// Deserialize raw models of every type, dropping those whose type this
/// version does not know (Pi's `models.filter(hasKnownModelType)`, applied to
/// stored catalogs and to fetched model lists). A `fetch_models`
/// implementation that reads a remote JSON list should use this, so a newer
/// model type does not fail the refresh.
pub fn known_models_from_values(
    models: impl IntoIterator<Item = Value>,
) -> serde_json::Result<Vec<AnyModel>> {
    models
        .into_iter()
        .filter(has_known_model_type)
        .map(serde_json::from_value)
        .collect()
}

/// `deserialize_with` helper for model lists: see [`known_models_from_values`].
pub(crate) fn deserialize_known_models<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<AnyModel>, D::Error>
where
    D: Deserializer<'de>,
{
    let models = Vec::<Value>::deserialize(deserializer)?;
    known_models_from_values(models).map_err(de::Error::custom)
}

impl AnyModel {
    pub fn id(&self) -> &str {
        match self {
            Self::Chat(model) => &model.id,
            Self::Image(model) => &model.id,
        }
    }

    pub fn provider(&self) -> &str {
        match self {
            Self::Chat(model) => &model.provider,
            Self::Image(model) => &model.provider,
        }
    }

    pub fn api(&self) -> &str {
        match self {
            Self::Chat(model) => &model.api,
            Self::Image(model) => &model.api,
        }
    }

    pub fn base_url(&self) -> &str {
        match self {
            Self::Chat(model) => &model.base_url,
            Self::Image(model) => &model.base_url,
        }
    }

    pub fn as_chat(&self) -> Option<&Model> {
        match self {
            Self::Chat(model) => Some(model),
            Self::Image(_) => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageModel> {
        match self {
            Self::Image(model) => Some(model),
            Self::Chat(_) => None,
        }
    }
}

impl From<Model> for AnyModel {
    fn from(value: Model) -> Self {
        Self::Chat(value)
    }
}

impl From<ImageModel> for AnyModel {
    fn from(value: ImageModel) -> Self {
        Self::Image(value)
    }
}

/// `TextContent | ImageContent` accepted by image generation.
pub type ImagesInputContent = UserContent;
/// `TextContent | ImageContent` returned by image generation.
pub type ImagesOutputContent = UserContent;

/// `ImagesContext`: the prompt for an image-generation request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesContext {
    pub input: Vec<ImagesInputContent>,
}

impl ImagesContext {
    /// Rust addition kept from the pre-1.0 API.
    pub fn builder() -> ImagesContextBuilder {
        ImagesContextBuilder::default()
    }
}

/// Builder for [`ImagesContext`] (pre-1.0 API).
#[derive(Debug, Clone, Default)]
pub struct ImagesContextBuilder {
    context: ImagesContext,
}

impl ImagesContextBuilder {
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.context.input.push(UserContent::text(text));
        self
    }

    pub fn image(mut self, image: ImageContent) -> Self {
        self.context.input.push(UserContent::Image(image));
        self
    }

    pub fn input(mut self, input: impl IntoIterator<Item = ImagesInputContent>) -> Self {
        self.context.input.extend(input);
        self
    }

    pub fn build(self) -> ImagesContext {
        self.context
    }
}

/// `ImagesStopReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ImagesStopReason {
    Stop,
    Error,
    Aborted,
}

/// `AssistantImages`: the result of an image-generation request. Failures
/// are reported in-band (`stop_reason` error/aborted plus `error_message`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    pub api: ImageApi,
    pub provider: ProviderId,
    pub model: String,
    pub output: Vec<ImagesOutputContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub stop_reason: ImagesStopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

impl AssistantImages {
    /// An empty `stop` result for `model`, stamped now.
    pub fn empty_for(model: &ImageModel) -> Self {
        Self {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            output: Vec::new(),
            response_id: None,
            usage: None,
            stop_reason: ImagesStopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }
}

/// `onPayload` for image requests (`ProviderRequestOptions<ImageModel>`).
pub type ImagesPayloadHook =
    Arc<dyn Fn(Value, &ImageModel) -> BoxFuture<Result<Option<Value>>> + Send + Sync>;
/// `onResponse` for image requests.
pub type ImagesResponseHook =
    Arc<dyn Fn(ProviderResponse, &ImageModel) -> BoxFuture<Result<()>> + Send + Sync>;

/// `ImagesOptions` (`ProviderImagesOptions`): request options for image
/// generation. Pi's open `Record<string, unknown>` extras become
/// `provider_options`.
#[derive(Clone, Default)]
pub struct ImagesOptions {
    pub signal: Option<CancellationToken>,
    pub api_key: Option<String>,
    /// Optional HTTP client for provider requests (Pi's `fetch` option).
    pub http_client: Option<reqwest::Client>,
    pub env: Option<ProviderEnv>,
    pub on_payload: Option<ImagesPayloadHook>,
    pub on_response: Option<ImagesResponseHook>,
    pub headers: Option<ProviderHeaders>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    /// Optional metadata to include in API requests. Providers extract the
    /// fields they understand and ignore the rest.
    pub metadata: Option<serde_json::Map<String, Value>>,
    /// API-specific options (`ProviderImagesOptions` record entries).
    pub provider_options: serde_json::Map<String, Value>,
}

impl fmt::Debug for ImagesOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImagesOptions")
            .field("signal", &self.signal)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("headers", &self.headers)
            .field("timeout_ms", &self.timeout_ms)
            .field("provider_options", &self.provider_options)
            .finish_non_exhaustive()
    }
}

/// `ProviderImages`: the uniform contract of an image-generation API
/// implementation. Never fails: errors are reported in the result.
#[async_trait::async_trait]
pub trait ProviderImages: Send + Sync {
    async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn usage_json() -> Value {
        json!({
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
        })
    }

    fn assistant_message() -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::text("hello")],
            timestamp: 123,
            ..AssistantMessage::empty_for(&Model {
                id: "gpt-5.5".to_string(),
                api: "openai-responses".to_string(),
                provider: "openai".to_string(),
                ..Default::default()
            })
        }
    }

    // message-types.test.ts
    #[test]
    fn system_messages_carry_named_sections_and_tool_changes() {
        let message: Message = serde_json::from_value(json!({
            "role": "system",
            "content": "base",
            "sections": { "skills": "<skills/>", "old": null },
            "toolsAdded": [{ "name": "read", "description": "Read", "parameters": { "type": "object" } }],
            "toolsRemoved": [{ "name": "write" }],
            "timestamp": 1
        }))
        .unwrap();
        let Message::System(system) = &message else {
            panic!("expected system message");
        };
        let sections = system.sections.as_ref().unwrap();
        assert_eq!(
            sections.keys().collect::<Vec<_>>(),
            vec!["skills", "old"],
            "section order is preserved"
        );
        assert_eq!(sections["old"], None);
        assert_eq!(system.tools_removed.as_ref().unwrap()[0].name, "write");
        assert_eq!(serde_json::to_value(&message).unwrap()["role"], "system");
    }

    #[test]
    fn bare_messages_serialize_with_upstream_roles() {
        assert_eq!(
            serde_json::to_value(UserMessage {
                content: UserMessageContent::Text("hi".to_string()),
                timestamp: 1,
            })
            .unwrap()["role"],
            json!("user")
        );
        assert_eq!(
            serde_json::to_value(assistant_message()).unwrap()["role"],
            json!("assistant")
        );
        let tool_result = ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ToolResultContent::text("done")],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: 2,
        };
        assert_eq!(
            serde_json::to_value(tool_result).unwrap()["role"],
            json!("toolResult")
        );
    }

    #[test]
    fn assistant_events_include_role_in_nested_messages() {
        let event = AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: assistant_message(),
        };
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["type"], json!("done"));
        assert_eq!(value["message"]["role"], json!("assistant"));
    }

    #[test]
    fn context_builder_collects_messages_and_tools() {
        let tool = Tool::builder("lookup")
            .description("Lookup a value.")
            .build()
            .unwrap();
        let context = Context::builder()
            .system_prompt("You are concise.")
            .message(Message::user_text("hi"))
            .tool(tool.clone())
            .build();

        assert_eq!(context.system_prompt.as_deref(), Some("You are concise."));
        assert_eq!(context.messages.len(), 1);
        assert_eq!(context.tools, Some(vec![tool]));
    }

    #[test]
    fn tool_builder_validates_and_defaults_schema() {
        let tool = Tool::builder("ping")
            .description("Ping the tool.")
            .build()
            .unwrap();
        assert_eq!(
            tool.parameters,
            json!({ "type": "object", "properties": {} })
        );
        assert!(Tool::builder("").description("desc").build().is_err());
        assert!(Tool::builder("lookup").build().is_err());
        assert!(
            Tool::builder("lookup")
                .description("desc")
                .parameters(json!("not an object"))
                .build()
                .is_err()
        );
    }

    #[test]
    fn constrained_sampling_matches_pi_wire_format() {
        let disabled: ConstrainedSampling = serde_json::from_value(json!(false)).unwrap();
        assert_eq!(disabled, ConstrainedSampling::Disabled);
        assert!(serde_json::from_value::<ConstrainedSampling>(json!(true)).is_err());

        let tool = Tool::builder("apply_patch")
            .description("Apply a patch.")
            .constrained_sampling(ConstrainedSamplingConfig::Grammar {
                variants: GrammarVariants {
                    openai_lark: Some("start: /.+/s".to_string()),
                    openai_regex: None,
                },
            })
            .build()
            .unwrap();
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(
            value["constrainedSampling"],
            json!({ "type": "grammar", "variants": { "openai_lark": "start: /.+/s" } })
        );
        assert_eq!(serde_json::from_value::<Tool>(value).unwrap(), tool);
    }

    #[test]
    fn context_messages_round_trip_with_roles() {
        let context = Context {
            messages: vec![
                Message::System(SystemMessage {
                    content: "sys".into(),
                    timestamp: 0,
                    ..Default::default()
                }),
                Message::user_text("hi"),
                Message::Assistant(assistant_message()),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ToolResultContent::text("done")],
                    details: None,
                    usage: None,
                    nested_calls: Some(NestedToolCalls {
                        calls: vec![NestedToolCallRecord {
                            id: "n1".to_string(),
                            name: "grep".to_string(),
                            arguments: None,
                            arguments_bytes: Some(10),
                            status: NestedToolCallStatus::Unfinished,
                            duration_ms: None,
                            error: None,
                        }],
                        complete: false,
                    }),
                    is_error: false,
                    timestamp: 2,
                }),
            ],
            ..Default::default()
        };
        let value = serde_json::to_value(&context).unwrap();
        let restored: Context = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(value["messages"][0]["role"], json!("system"));
        assert_eq!(value["messages"][1]["role"], json!("user"));
        assert_eq!(value["messages"][2]["role"], json!("assistant"));
        assert_eq!(value["messages"][3]["role"], json!("toolResult"));
        assert_eq!(
            value["messages"][3]["nestedCalls"]["calls"][0]["argumentsBytes"],
            json!(10)
        );
        assert_eq!(restored, context);
    }

    #[test]
    fn deserializes_null_or_missing_message_content_as_empty() {
        let messages: Vec<Message> = serde_json::from_value(json!([
            { "role": "user", "content": null, "timestamp": 1 },
            {
                "role": "assistant", "content": null, "api": "openai-completions",
                "provider": "openai", "model": "gpt-4o-mini", "usage": usage_json(),
                "stopReason": "stop", "timestamp": 2
            },
            { "role": "toolResult", "toolCallId": "call_1", "toolName": "web_search", "isError": false, "timestamp": 3 }
        ]))
        .expect("lax messages deserialize");

        let Message::User(user) = &messages[0] else {
            panic!()
        };
        assert_eq!(user.content, UserMessageContent::Parts(Vec::new()));
        let Message::Assistant(assistant) = &messages[1] else {
            panic!()
        };
        assert!(assistant.content.is_empty());
        let Message::ToolResult(result) = &messages[2] else {
            panic!()
        };
        assert!(result.content.is_empty());
    }

    #[test]
    fn assistant_message_optional_fields_round_trip() {
        let message = AssistantMessage {
            stop_reason: StopReason::Deferred,
            deferred: Some(DeferredHandle {
                provider: "openai".to_string(),
                model_id: "gpt-5.5".to_string(),
                api: "openai-responses".to_string(),
                id: "resp_1".to_string(),
                expires_at: Some(5),
                poll_after_ms: None,
                data: Some(json!({ "k": 1 })),
            }),
            provider_thinking_level: Some("xhigh".to_string()),
            thinking_level: Some(ModelThinkingLevel::High),
            raw_stop_reason: Some("end_turn".to_string()),
            end_turn: Some(true),
            ..assistant_message()
        };
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["stopReason"], json!("deferred"));
        assert_eq!(value["deferred"]["modelId"], json!("gpt-5.5"));
        assert_eq!(value["thinkingLevel"], json!("high"));
        assert!(value.get("diagnostics").is_none());
        assert_eq!(
            serde_json::from_value::<AssistantMessage>(value).unwrap(),
            message
        );
    }

    // max-thinking.test.ts (type-level parts)
    #[test]
    fn max_thinking_level_round_trips() {
        assert_eq!(
            serde_json::to_value(ThinkingLevel::Max).unwrap(),
            json!("max")
        );
        assert_eq!(
            serde_json::from_value::<ModelThinkingLevel>(json!("max")).unwrap(),
            ModelThinkingLevel::Max
        );
        assert_eq!(
            ModelThinkingLevel::parse("max"),
            Some(ModelThinkingLevel::Max)
        );
    }

    #[test]
    fn usage_optional_breakdowns_and_cost_tiers_round_trip() {
        let usage = Usage {
            output: 20,
            cache_write: 10,
            cache_write_1h: Some(4),
            reasoning: Some(7),
            ..Default::default()
        };
        let serialized = serde_json::to_value(&usage).unwrap();
        assert_eq!(serialized["cacheWrite1h"], json!(4));
        assert_eq!(serialized["reasoning"], json!(7));
        assert_eq!(serde_json::from_value::<Usage>(serialized).unwrap(), usage);
        let empty = serde_json::to_value(Usage::default()).unwrap();
        assert!(empty.get("cacheWrite1h").is_none());
        assert!(empty.get("reasoning").is_none());

        let cost = ModelCost {
            input: 5.0,
            output: 30.0,
            cache_read: 0.5,
            cache_write: 6.25,
            tiers: Some(vec![ModelCostTier {
                input_tokens_above: 272_000,
                input: 10.0,
                output: 45.0,
                cache_read: 1.0,
                cache_write: 12.5,
            }]),
        };
        let serialized = serde_json::to_value(&cost).unwrap();
        assert_eq!(serialized["tiers"][0]["inputTokensAbove"], 272_000);
        assert_eq!(
            serde_json::from_value::<ModelCost>(serialized).unwrap(),
            cost
        );
    }

    #[test]
    fn model_compat_round_trips_pi_field_names() {
        let compat: ModelCompat = serde_json::from_value(json!({
            "supportsDeveloperRole": false,
            "supportsOpenAIGrammarTools": true,
            "supportsMidConvoSystemMessages": true,
            "sessionAffinityFormat": "openai-nosession",
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
                "enabled": { "$var": "thinking.enabled" },
                "effort": { "$var": "thinking.effort", "omitWhenOff": true },
                "budget": { "$var": "thinking.budget" },
                "preserve": true,
                "temperature": 0.5,
                "sentinel": null
            },
            "thinkingTokenBudgetField": "thinking_budget_tokens",
            "supportsTemperature": false,
            "allowedFallbackModels": [{
                "provider": "anthropic", "model": "claude-x",
                "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 }
            }]
        }))
        .unwrap();
        assert_eq!(compat.supports_developer_role, Some(false));
        assert_eq!(compat.supports_openai_grammar_tools, Some(true));
        assert_eq!(
            compat.session_affinity_format,
            Some(SessionAffinityFormat::OpenaiNosession)
        );
        let kwargs = compat.chat_template_kwargs.as_ref().unwrap();
        assert_eq!(
            kwargs["budget"],
            ChatTemplateKwargValue::variable(ChatTemplateVariable::ThinkingBudget, false)
        );
        assert_eq!(kwargs["sentinel"], ChatTemplateKwargValue::Null(()));
        let value = serde_json::to_value(&compat).unwrap();
        assert_eq!(value["supportsOpenAIGrammarTools"], json!(true));
        assert_eq!(
            value["chatTemplateKwargs"]["effort"],
            json!({ "$var": "thinking.effort", "omitWhenOff": true })
        );
        assert_eq!(
            serde_json::from_value::<ModelCompat>(value).unwrap(),
            compat
        );
        assert!(ChatTemplateKwargValue::from_f64(f64::NAN).is_none());
    }

    #[test]
    fn provider_headers_keep_insertion_order_and_null_suppression() {
        let headers: ProviderHeaders = [("x-b", Some("1".to_string())), ("x-a", None)]
            .into_iter()
            .collect();
        assert_eq!(
            serde_json::to_string(&headers).unwrap(),
            r#"{"x-b":"1","x-a":null}"#
        );
    }

    #[test]
    fn any_model_dispatches_on_type() {
        let chat: AnyModel = serde_json::from_value(json!({
            "id": "m", "name": "M", "api": "openai-responses", "provider": "openai",
            "baseUrl": "https://x", "type": "chat", "reasoning": true, "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 10, "maxTokens": 5,
            "thinkingLevelMap": { "off": null, "xhigh": "xhigh" }
        }))
        .unwrap();
        let model = chat.as_chat().unwrap();
        assert_eq!(model.model_type, Some(ModelType::Chat));
        assert_eq!(
            model.thinking_level_map.as_ref().unwrap()[&ModelThinkingLevel::Off],
            None
        );
        let image: AnyModel = serde_json::from_value(json!({
            "id": "i", "name": "I", "api": "openrouter-images", "provider": "openrouter",
            "baseUrl": "https://x", "type": "image", "input": ["text"], "output": ["image"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }))
        .unwrap();
        assert!(image.as_image().is_some());
    }
}
