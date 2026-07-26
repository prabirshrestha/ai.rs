use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Result;
use crate::provider::LanguageModelApi;

pub type Api = String;
pub type ProviderId = String;
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderHeaders(HashMap<String, Option<String>>);

impl ProviderHeaders {
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

    pub fn get(&self, name: &str) -> Option<&Option<String>> {
        self.0.get(name)
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
    type IntoIter = std::collections::hash_map::Iter<'a, String, Option<String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl From<HashMap<String, String>> for ProviderHeaders {
    fn from(headers: HashMap<String, String>) -> Self {
        headers.into_iter().collect()
    }
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownApi {
    OpenaiCompletions,
    OpenaiResponses,
    OpenaiImages,
    AnthropicMessages,
    OpenrouterImages,
}

impl KnownApi {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenaiCompletions => "openai-completions",
            Self::OpenaiResponses => "openai-responses",
            Self::OpenaiImages => "openai-images",
            Self::AnthropicMessages => "anthropic-messages",
            Self::OpenrouterImages => "openrouter-images",
        }
    }
}

impl From<KnownApi> for String {
    fn from(value: KnownApi) -> Self {
        value.as_str().to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    Sse,
    Websocket,
    WebsocketCached,
    #[default]
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum SessionAffinityFormat {
    #[default]
    Openai,
    OpenaiNosession,
    Openrouter,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

pub type PayloadHook = Arc<
    dyn Fn(Value, &Model) -> Pin<Box<dyn Future<Output = Result<Option<Value>>> + Send>>
        + Send
        + Sync,
>;
pub type ResponseHook = Arc<
    dyn Fn(ProviderResponse, &Model) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone, Default)]
pub struct StreamOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub cancellation_token: Option<CancellationToken>,
    pub api_key: Option<String>,
    pub transport: Option<Transport>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub on_payload: Option<PayloadHook>,
    pub on_response: Option<ResponseHook>,
    pub headers: ProviderHeaders,
    pub timeout_ms: Option<u64>,
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Maximum retry attempts for providers that support client-side retries.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait when the server requests a long
    /// retry delay. If the requested delay exceeds this value, the request
    /// fails immediately. Defaults to 60 seconds; set to zero to disable the
    /// cap.
    pub max_retry_delay_ms: Option<u64>,
    pub http_client: Option<reqwest::Client>,
    pub metadata: Option<Value>,
    pub provider_options: HashMap<String, Value>,
}

#[derive(Clone, Default)]
pub struct ImageGenerationOptions {
    pub base: StreamOptions,
}

#[derive(Clone, Default)]
pub struct SimpleStreamOptions {
    pub stream: StreamOptions,
    pub reasoning: Option<ModelThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

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
        Self::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResultContent {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "image")]
    Image(ImageContent),
}

impl ToolResultContent {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })
    }
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<UserContent>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    pub total: f64,
}

impl Default for UsageCost {
    fn default() -> Self {
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u32,
    pub output: u32,
    pub cache_read: u32,
    pub cache_write: u32,
    /// Subset of `cache_write` written with one-hour retention. Only
    /// Anthropic reports this split.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u32>,
    /// Reasoning/thinking tokens when the provider reports them. This is a
    /// subset of `output`, which already includes these tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u32>,
    pub total_tokens: u32,
    pub cost: UsageCost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ImagesStopReason {
    Stop,
    Error,
    Aborted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserMessage {
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

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    pub response_model: Option<String>,
    pub response_id: Option<String>,
    pub diagnostics: Vec<Value>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    pub error_message: Option<String>,
    pub timestamp: u64,
}

impl AssistantMessage {
    pub fn empty_for(model: &Model) -> Self {
        Self {
            content: Vec::new(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ToolResultContent>,
    pub details: Option<Value>,
    /// Names from `Context::tools` that became available after this result.
    pub added_tool_names: Vec<String>,
    pub is_error: bool,
    pub timestamp: u64,
}

fn validate_role(role: Option<&str>, expected: &str) -> std::result::Result<(), String> {
    match role {
        Some(actual) if actual != expected => {
            Err(format!("expected role {expected}, got {actual}"))
        }
        _ => Ok(()),
    }
}

impl Serialize for UserMessage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("UserMessage", 3)?;
        state.serialize_field("role", "user")?;
        state.serialize_field("content", &self.content)?;
        state.serialize_field("timestamp", &self.timestamp)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for UserMessage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            role: Option<String>,
            content: UserMessageContent,
            timestamp: u64,
        }

        let helper = Helper::deserialize(deserializer)?;
        validate_role(helper.role.as_deref(), "user").map_err(serde::de::Error::custom)?;
        Ok(Self {
            content: helper.content,
            timestamp: helper.timestamp,
        })
    }
}

impl Serialize for AssistantMessage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut field_count = 8;
        if self.response_model.is_some() {
            field_count += 1;
        }
        if self.response_id.is_some() {
            field_count += 1;
        }
        if !self.diagnostics.is_empty() {
            field_count += 1;
        }
        if self.error_message.is_some() {
            field_count += 1;
        }

        let mut state = serializer.serialize_struct("AssistantMessage", field_count)?;
        state.serialize_field("role", "assistant")?;
        state.serialize_field("content", &self.content)?;
        state.serialize_field("api", &self.api)?;
        state.serialize_field("provider", &self.provider)?;
        state.serialize_field("model", &self.model)?;
        if let Some(response_model) = &self.response_model {
            state.serialize_field("responseModel", response_model)?;
        }
        if let Some(response_id) = &self.response_id {
            state.serialize_field("responseId", response_id)?;
        }
        if !self.diagnostics.is_empty() {
            state.serialize_field("diagnostics", &self.diagnostics)?;
        }
        state.serialize_field("usage", &self.usage)?;
        state.serialize_field("stopReason", &self.stop_reason)?;
        if let Some(error_message) = &self.error_message {
            state.serialize_field("errorMessage", error_message)?;
        }
        state.serialize_field("timestamp", &self.timestamp)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for AssistantMessage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Helper {
            role: Option<String>,
            content: Vec<AssistantContent>,
            api: Api,
            provider: ProviderId,
            model: String,
            response_model: Option<String>,
            response_id: Option<String>,
            #[serde(default)]
            diagnostics: Vec<Value>,
            usage: Usage,
            stop_reason: StopReason,
            error_message: Option<String>,
            timestamp: u64,
        }

        let helper = Helper::deserialize(deserializer)?;
        validate_role(helper.role.as_deref(), "assistant").map_err(serde::de::Error::custom)?;
        Ok(Self {
            content: helper.content,
            api: helper.api,
            provider: helper.provider,
            model: helper.model,
            response_model: helper.response_model,
            response_id: helper.response_id,
            diagnostics: helper.diagnostics,
            usage: helper.usage,
            stop_reason: helper.stop_reason,
            error_message: helper.error_message,
            timestamp: helper.timestamp,
        })
    }
}

impl Serialize for ToolResultMessage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut field_count = 6;
        if self.details.is_some() {
            field_count += 1;
        }
        if !self.added_tool_names.is_empty() {
            field_count += 1;
        }
        let mut state = serializer.serialize_struct("ToolResultMessage", field_count)?;
        state.serialize_field("role", "toolResult")?;
        state.serialize_field("toolCallId", &self.tool_call_id)?;
        state.serialize_field("toolName", &self.tool_name)?;
        state.serialize_field("content", &self.content)?;
        if let Some(details) = &self.details {
            state.serialize_field("details", details)?;
        }
        if !self.added_tool_names.is_empty() {
            state.serialize_field("addedToolNames", &self.added_tool_names)?;
        }
        state.serialize_field("isError", &self.is_error)?;
        state.serialize_field("timestamp", &self.timestamp)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ToolResultMessage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Helper {
            role: Option<String>,
            tool_call_id: String,
            tool_name: String,
            content: Vec<ToolResultContent>,
            details: Option<Value>,
            #[serde(default)]
            added_tool_names: Vec<String>,
            is_error: bool,
            timestamp: u64,
        }

        let helper = Helper::deserialize(deserializer)?;
        validate_role(helper.role.as_deref(), "toolResult").map_err(serde::de::Error::custom)?;
        Ok(Self {
            tool_call_id: helper.tool_call_id,
            tool_name: helper.tool_name,
            content: helper.content,
            details: helper.details,
            added_tool_names: helper.added_tool_names,
            is_error: helper.is_error,
            timestamp: helper.timestamp,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    Custom(Value),
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::User(message) => message.serialize(serializer),
            Self::Assistant(message) => message.serialize(serializer),
            Self::ToolResult(message) => message.serialize(serializer),
            Self::Custom(value) => {
                let mut value = value.clone();
                match &mut value {
                    Value::Object(object) => {
                        object
                            .entry("role".to_string())
                            .or_insert_with(|| Value::String("custom".to_string()));
                        value.serialize(serializer)
                    }
                    _ => {
                        let mut state = serializer.serialize_struct("CustomMessage", 2)?;
                        state.serialize_field("role", "custom")?;
                        state.serialize_field("value", &value)?;
                        state.end()
                    }
                }
            }
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
            .ok_or_else(|| serde::de::Error::custom("missing message role"))?;
        match role {
            "user" => serde_json::from_value(value)
                .map(Self::User)
                .map_err(serde::de::Error::custom),
            "assistant" => serde_json::from_value(value)
                .map(Self::Assistant)
                .map_err(serde::de::Error::custom),
            "toolResult" => serde_json::from_value(value)
                .map(Self::ToolResult)
                .map_err(serde::de::Error::custom),
            "custom" => Ok(Self::Custom(value)),
            _ => Ok(Self::Custom(value)),
        }
    }
}

impl Message {
    pub fn user_text<T: Into<String>>(text: T) -> Self {
        Self::User(UserMessage::text(text))
    }

    pub fn custom(value: Value) -> Self {
        Self::Custom(value)
    }

    pub fn is_llm_message(&self) -> bool {
        matches!(
            self,
            Self::User(_) | Self::Assistant(_) | Self::ToolResult(_)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSamplingConfig {
    JsonSchema { strict: ConstrainedSamplingStrict },
    Grammar { variants: GrammarVariants },
}

/// A tool's constrained-sampling setting. Pi permits either an explicit
/// `false` opt-out or a constrained-sampling configuration.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConstrainedSamplingStrict {
    Prefer,
    Require,
}

/// OpenAI grammar encodings supported by Pi constrained-sampling configs.
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(
        rename = "constrainedSampling",
        skip_serializing_if = "Option::is_none"
    )]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesContext {
    #[serde(default)]
    pub input: Vec<UserContent>,
}

impl ImagesContext {
    pub fn builder() -> ImagesContextBuilder {
        ImagesContextBuilder::default()
    }
}

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

    pub fn input(mut self, input: impl IntoIterator<Item = UserContent>) -> Self {
        self.context.input.extend(input);
        self
    }

    pub fn build(self) -> ImagesContext {
        self.context
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ImageOutput {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "image")]
    Image(ImageContent),
}

impl ImageOutput {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default)]
    pub output: Vec<ImageOutput>,
    pub usage: Usage,
    pub stop_reason: ImagesStopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub timestamp: u64,
}

impl AssistantImages {
    pub fn empty_for(model: &Model) -> Self {
        Self {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_id: None,
            output: Vec::new(),
            usage: Usage::default(),
            stop_reason: ImagesStopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }
}

impl Context {
    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ContextBuilder {
    context: Context,
}

impl ContextBuilder {
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.context.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.context.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.context.messages.extend(messages);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.context.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.context.tools.extend(tools);
        self
    }

    pub fn build(self) -> Context {
        self.context
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelInput {
    Text,
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelOutput {
    Text,
    Image,
}

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
    /// Use this tier for requests whose total input usage exceeds this token
    /// count.
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    pub base_url: String,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub thinking_level_map: HashMap<String, Option<String>>,
    #[serde(default)]
    pub input: Vec<ModelInput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output: Vec<ModelOutput>,
    pub cost: ModelCost,
    pub context_window: u32,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub compat: ModelCompat,
    #[serde(skip)]
    pub(crate) language_api: Option<Arc<dyn LanguageModelApi>>,
    #[serde(skip)]
    pub(crate) image_api: Option<Arc<dyn crate::provider::ImageModelApi>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider_id: ProviderId,
    pub api_id: Api,
    pub id: String,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("api", &self.api)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("reasoning", &self.reasoning)
            .field("thinking_level_map", &self.thinking_level_map)
            .field("input", &self.input)
            .field("output", &self.output)
            .field("cost", &self.cost)
            .field("context_window", &self.context_window)
            .field("max_tokens", &self.max_tokens)
            .field(
                "headers",
                &(!self.headers.is_empty()).then_some("<redacted>"),
            )
            .field("compat", &self.compat)
            .finish()
    }
}

impl PartialEq for Model {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.name == other.name
            && self.api == other.api
            && self.provider == other.provider
            && self.base_url == other.base_url
            && self.reasoning == other.reasoning
            && self.thinking_level_map == other.thinking_level_map
            && self.input == other.input
            && self.output == other.output
            && self.cost == other.cost
            && self.context_window == other.context_window
            && self.max_tokens == other.max_tokens
            && self.headers == other.headers
            && self.compat == other.compat
    }
}

impl Model {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn provider_id(&self) -> &str {
        &self.provider
    }

    pub fn api_id(&self) -> &str {
        &self.api
    }

    pub fn model_ref(&self) -> ModelRef {
        ModelRef::from(self)
    }

    pub fn language_api(&self) -> Option<Arc<dyn LanguageModelApi>> {
        self.language_api.clone()
    }

    pub fn image_api(&self) -> Option<Arc<dyn crate::provider::ImageModelApi>> {
        self.image_api.clone()
    }
}

impl From<&Model> for ModelRef {
    fn from(model: &Model) -> Self {
        Self {
            provider_id: model.provider.clone(),
            api_id: model.api.clone(),
            id: model.id.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCompat {
    #[serde(flatten)]
    pub openai_completions: OpenAICompletionsCompat,
    #[serde(flatten)]
    pub openai_responses: OpenAIResponsesCompat,
    #[serde(flatten)]
    pub anthropic_messages: AnthropicMessagesCompat,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<OpenAIThinkingFormat>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub chat_template_kwargs: HashMap<String, ChatTemplateKwargValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<CacheControlFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred_tools_mode: Option<DeferredToolsMode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeferredToolsMode {
    Kimi,
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
    Zai,
    Qwen,
    QwenChatTemplate,
    ChatTemplate,
    StringThinking,
    AntLing,
}

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

    pub fn thinking_enabled() -> Self {
        Self::Variable(ChatTemplateKwargVariable {
            variable: ChatTemplateVariable::ThinkingEnabled,
            omit_when_off: false,
        })
    }

    pub fn thinking_effort(omit_when_off: bool) -> Self {
        Self::Variable(ChatTemplateKwargVariable {
            variable: ChatTemplateVariable::ThinkingEffort,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlFormat {
    Anthropic,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAIResponsesCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Legacy compatibility setting. `false` maps to `OpenaiNosession` when
    /// `session_affinity_format` is not explicitly configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_id_header: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicMessagesCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_references: Option<bool>,
}

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
    #[serde(rename = "done")]
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    #[serde(rename = "error")]
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn assistant_message() -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: "hello".to_string(),
                text_signature: None,
            })],
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            model: "gpt-5.5".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 123,
        }
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
        assert_eq!(
            serde_json::to_value(ToolResultMessage {
                tool_call_id: "call_1".to_string(),
                tool_name: "read".to_string(),
                content: vec![ToolResultContent::text("done")],
                details: None,
                added_tool_names: Vec::new(),
                is_error: false,
                timestamp: 2,
            })
            .unwrap()["role"],
            json!("toolResult")
        );
    }

    #[test]
    fn assistant_events_include_role_in_nested_messages() {
        let message = assistant_message();
        let event = AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message,
        };
        let value = serde_json::to_value(event).unwrap();

        assert_eq!(value["type"], json!("done"));
        assert_eq!(value["message"]["role"], json!("assistant"));
    }

    #[test]
    fn context_builder_collects_system_messages_and_tools() {
        let tool = Tool {
            name: "lookup".to_string(),
            description: "Lookup a value.".to_string(),
            parameters: json!({ "type": "object" }),
            constrained_sampling: None,
        };
        let context = Context::builder()
            .system_prompt("You are concise.")
            .message(Message::user_text("hi"))
            .tool(tool.clone())
            .build();

        assert_eq!(context.system_prompt.as_deref(), Some("You are concise."));
        assert_eq!(context.messages, vec![Message::user_text("hi")]);
        assert_eq!(context.tools, vec![tool]);
    }

    #[test]
    fn tool_builder_creates_tool_with_parameters() {
        let tool = Tool::builder("lookup")
            .description("Lookup a value.")
            .parameters(json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string" }
                },
                "required": ["key"]
            }))
            .build()
            .expect("tool");

        assert_eq!(tool.name, "lookup");
        assert_eq!(tool.description, "Lookup a value.");
        assert_eq!(tool.parameters["type"], json!("object"));
    }

    #[test]
    fn tool_builder_defaults_to_empty_object_schema() {
        let tool = Tool::builder("ping")
            .description("Ping the tool.")
            .build()
            .expect("tool");

        assert_eq!(
            tool.parameters,
            json!({ "type": "object", "properties": {} })
        );
    }

    #[test]
    fn constrained_sampling_matches_pi_wire_format() {
        let disabled: ConstrainedSampling = serde_json::from_value(json!(false)).unwrap();
        assert_eq!(disabled, ConstrainedSampling::Disabled);

        let tool = Tool::builder("apply_patch")
            .description("Apply a patch.")
            .parameters(json!({
                "type": "object",
                "properties": { "input": { "type": "string" } },
                "required": ["input"]
            }))
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
            json!({
                "type": "grammar",
                "variants": { "openai_lark": "start: /.+/s" }
            })
        );
        assert_eq!(serde_json::from_value::<Tool>(value).unwrap(), tool);
    }

    #[test]
    fn tool_builder_validates_required_fields() {
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
    fn context_messages_round_trip_with_roles() {
        let context = Context {
            messages: vec![
                Message::user_text("hi"),
                Message::Assistant(assistant_message()),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ToolResultContent::text("done")],
                    details: None,
                    added_tool_names: Vec::new(),
                    is_error: false,
                    timestamp: 2,
                }),
            ],
            ..Default::default()
        };
        let value = serde_json::to_value(&context).unwrap();
        let restored: Context = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(value["messages"][0]["role"], json!("user"));
        assert_eq!(value["messages"][1]["role"], json!("assistant"));
        assert_eq!(value["messages"][2]["role"], json!("toolResult"));
        assert_eq!(restored, context);
    }

    #[test]
    fn model_ref_serializes_provider_api_and_model_id() {
        let model = Model {
            id: "gpt-5.5".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            ..Model::default()
        };

        assert_eq!(
            serde_json::to_value(model.model_ref()).unwrap(),
            json!({
                "providerId": "openai",
                "apiId": "openai-responses",
                "id": "gpt-5.5"
            })
        );
    }

    #[test]
    fn max_thinking_level_round_trips() {
        assert_eq!(
            serde_json::to_value(ThinkingLevel::Max).unwrap(),
            json!("max")
        );
        assert_eq!(
            serde_json::from_value::<ThinkingLevel>(json!("max")).unwrap(),
            ThinkingLevel::Max
        );
        assert_eq!(
            serde_json::to_value(ModelThinkingLevel::Max).unwrap(),
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
    fn session_affinity_formats_and_legacy_responses_setting_round_trip() {
        for (format, serialized) in [
            (SessionAffinityFormat::Openai, "openai"),
            (SessionAffinityFormat::OpenaiNosession, "openai-nosession"),
            (SessionAffinityFormat::Openrouter, "openrouter"),
        ] {
            assert_eq!(serde_json::to_value(format).unwrap(), json!(serialized));
            assert_eq!(
                serde_json::from_value::<SessionAffinityFormat>(json!(serialized)).unwrap(),
                format
            );
        }

        let completions = OpenAICompletionsCompat {
            session_affinity_format: Some(SessionAffinityFormat::OpenaiNosession),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(completions).unwrap()["sessionAffinityFormat"],
            json!("openai-nosession")
        );

        let legacy: OpenAIResponsesCompat =
            serde_json::from_value(json!({ "sendSessionIdHeader": false })).unwrap();
        assert_eq!(legacy.send_session_id_header, Some(false));
        assert_eq!(
            serde_json::to_value(legacy).unwrap(),
            json!({ "sendSessionIdHeader": false })
        );

        let responses = OpenAIResponsesCompat {
            supports_developer_role: Some(false),
            session_affinity_format: Some(SessionAffinityFormat::Openrouter),
            supports_long_cache_retention: Some(false),
            supports_explicit_prompt_cache_mode: Some(true),
            ..Default::default()
        };
        let serialized = serde_json::to_value(&responses).unwrap();
        assert_eq!(serialized["supportsDeveloperRole"], json!(false));
        assert_eq!(serialized["sessionAffinityFormat"], json!("openrouter"));
        assert_eq!(serialized["supportsLongCacheRetention"], json!(false));
        assert_eq!(serialized["supportsExplicitPromptCacheMode"], json!(true));
        assert_eq!(
            serde_json::from_value::<OpenAIResponsesCompat>(serialized).unwrap(),
            responses
        );

        let headers: ProviderHeaders = [
            ("x-keep".to_string(), Some("value".to_string())),
            ("x-remove".to_string(), None),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            serde_json::to_value(headers).unwrap(),
            json!({ "x-keep": "value", "x-remove": null })
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
        let serialized_usage = serde_json::to_value(&usage).unwrap();
        assert_eq!(serialized_usage["cacheWrite1h"], json!(4));
        assert_eq!(serialized_usage["reasoning"], json!(7));
        assert_eq!(
            serde_json::from_value::<Usage>(serialized_usage).unwrap(),
            usage
        );
        let empty_usage = serde_json::to_value(Usage::default()).unwrap();
        assert!(empty_usage.get("cacheWrite1h").is_none());
        assert!(empty_usage.get("reasoning").is_none());

        let cost = ModelCost {
            input: 5.0,
            output: 30.0,
            cache_read: 0.5,
            cache_write: 6.25,
            tiers: vec![ModelCostTier {
                input_tokens_above: 272_000,
                input: 10.0,
                output: 45.0,
                cache_read: 1.0,
                cache_write: 12.5,
            }],
        };
        let serialized_cost = serde_json::to_value(&cost).unwrap();
        assert_eq!(serialized_cost["tiers"][0]["inputTokensAbove"], 272_000);
        assert_eq!(
            serde_json::from_value::<ModelCost>(serialized_cost).unwrap(),
            cost
        );
    }

    #[test]
    fn chat_template_compat_round_trips() {
        let compat = OpenAICompletionsCompat {
            thinking_format: Some(OpenAIThinkingFormat::ChatTemplate),
            chat_template_kwargs: [
                (
                    "enabled".to_string(),
                    ChatTemplateKwargValue::thinking_enabled(),
                ),
                (
                    "effort".to_string(),
                    ChatTemplateKwargValue::thinking_effort(true),
                ),
                ("preserve".to_string(), true.into()),
                (
                    "temperature".to_string(),
                    ChatTemplateKwargValue::from_f64(0.5).unwrap(),
                ),
                ("sentinel".to_string(), ChatTemplateKwargValue::Null(())),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let value = serde_json::to_value(&compat).unwrap();
        let restored: OpenAICompletionsCompat = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(value["thinkingFormat"], json!("chat-template"));
        assert_eq!(
            value["chatTemplateKwargs"]["enabled"],
            json!({ "$var": "thinking.enabled" })
        );
        assert_eq!(
            value["chatTemplateKwargs"]["effort"],
            json!({ "$var": "thinking.effort", "omitWhenOff": true })
        );
        assert_eq!(value["chatTemplateKwargs"]["temperature"], json!(0.5));
        assert_eq!(value["chatTemplateKwargs"]["sentinel"], Value::Null);
        assert_eq!(restored, compat);
    }

    #[test]
    fn anthropic_temperature_compat_round_trips() {
        let compat = AnthropicMessagesCompat {
            supports_temperature: Some(false),
            ..Default::default()
        };

        let value = serde_json::to_value(&compat).unwrap();
        let restored: AnthropicMessagesCompat = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(value["supportsTemperature"], json!(false));
        assert_eq!(restored, compat);
    }

    #[test]
    fn deferred_tool_metadata_round_trips() {
        let message = ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "base_tool".to_string(),
            content: vec![ToolResultContent::text("done")],
            details: None,
            added_tool_names: vec!["late_tool".to_string()],
            is_error: false,
            timestamp: 3,
        };
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["addedToolNames"], json!(["late_tool"]));
        assert_eq!(
            serde_json::from_value::<ToolResultMessage>(value).unwrap(),
            message
        );

        let compat = ModelCompat {
            openai_completions: OpenAICompletionsCompat {
                deferred_tools_mode: Some(DeferredToolsMode::Kimi),
                ..Default::default()
            },
            openai_responses: OpenAIResponsesCompat {
                supports_tool_search: Some(true),
                ..Default::default()
            },
            anthropic_messages: AnthropicMessagesCompat {
                supports_tool_references: Some(true),
                ..Default::default()
            },
        };
        let value = serde_json::to_value(&compat).unwrap();
        assert_eq!(value["deferredToolsMode"], json!("kimi"));
        assert_eq!(value["supportsToolSearch"], json!(true));
        assert_eq!(value["supportsToolReferences"], json!(true));
        assert_eq!(
            serde_json::from_value::<ModelCompat>(value).unwrap(),
            compat
        );
    }

    #[test]
    fn chat_template_float_values_reject_non_finite_numbers() {
        assert!(ChatTemplateKwargValue::from_f64(f64::NAN).is_none());
        assert!(ChatTemplateKwargValue::from_f64(f64::INFINITY).is_none());
    }
}
