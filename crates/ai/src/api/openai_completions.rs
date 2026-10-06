//! Port of `api/openai-completions.ts`: the OpenAI Chat Completions API and
//! OpenAI-compatible servers.
//!
//! The `openai` SDK client is replaced by [`OpenAIClient`]. Requests and
//! stream chunks are `serde_json::Value`s; streaming scratch state
//! (`partialArgs`, `customInput`, `streamIndex`) lives in side tables keyed
//! by content index instead of on the tool call blocks.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::constrained_sampling::{
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties, get_grammar_tool_input, get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
};
use super::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use super::lazy::error_stream;
use super::openai_client::{
    OpenAIClient, OpenAIRequestOptions, js_template_string, js_truthy, sse_json_events,
};
use super::openai_prompt_cache::clamp_openai_prompt_cache_key;
use super::openai_responses::{
    get_client_api_key, model_thinking_level_to_effort, resolve_cache_retention, tool_choice_value,
};
use super::openai_responses_shared::GrammarToolInputProperties;
use super::simple_options::{
    build_base_options, clamp_thinking_budget_to_answer_room, resolve_sampling_params,
    thinking_budget_for_level,
};
use super::transform_messages::transform_messages;
use crate::models::{calculate_cost, clamp_thinking_level};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheControlFormat, CacheRetention,
    ChatTemplateKwargValue, ChatTemplateVariable, MaxTokensField, Message, Model, ModelInput,
    ModelThinkingLevel, OpenAIThinkingFormat, OpenRouterRouting, ProviderHeaders, ProviderResponse,
    ProviderStreams, SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions,
    ThinkingBudgets, ThinkingContent, ThinkingLevel, ThinkingTokenBudgetField, Tool, ToolCall,
    TranscriptContext, Usage, UsageCost, UserContent, UserMessageContent, VercelGatewayRouting,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::headers::headers_to_record;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::time::now_millis;
use crate::utils::transcript::{get_declared_tools, resolve_transcript, resolve_transcript_tools};
use crate::{Error, Result};

/// Check if conversation messages contain tool calls or tool results.
/// This is needed because Anthropic (via proxy) requires the tools param
/// to be present when messages include tool_calls or tool role messages.
fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::ToolResult(_) => true,
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|block| matches!(block, AssistantContent::ToolCall(_))),
        _ => false,
    })
}

fn has_valid_common_reasoning_detail_fields(candidate: &Map<String, Value>) -> bool {
    matches!(
        candidate.get("id"),
        None | Some(Value::Null | Value::String(_))
    ) && matches!(candidate.get("format"), None | Some(Value::String(_)))
        && matches!(candidate.get("index"), None | Some(Value::Number(_)))
}

fn is_openai_reasoning_detail(detail: &Value) -> bool {
    let Some(detail) = detail.as_object() else {
        return false;
    };
    if !has_valid_common_reasoning_detail_fields(detail) {
        return false;
    }
    match detail.get("type").and_then(Value::as_str) {
        Some("reasoning.summary") => matches!(detail.get("summary"), Some(Value::String(_))),
        Some("reasoning.encrypted") => matches!(detail.get("data"), Some(Value::String(_))),
        Some("reasoning.text") => {
            matches!(detail.get("text"), Some(Value::String(_)))
                && matches!(
                    detail.get("signature"),
                    None | Some(Value::Null | Value::String(_))
                )
        }
        _ => false,
    }
}

/// OpenAI Chat Completions-specific options (`OpenAICompletionsOptions extends StreamOptions`).
#[derive(Debug, Clone, Default)]
pub struct OpenAICompletionsOptions {
    pub stream: StreamOptions,
    /// `ChatCompletionToolChoiceOption`.
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<ThinkingLevel>,
    /// Token budgets per thinking level. Used when `compat.thinkingTokenBudgetField` or `compat.supportsThinkingTokenBudget` is set, or by `{ "$var": "thinking.budget" }`.
    pub thinking_budgets: Option<ThinkingBudgets>,
}

impl std::ops::Deref for OpenAICompletionsOptions {
    type Target = StreamOptions;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl std::ops::DerefMut for OpenAICompletionsOptions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stream
    }
}

impl From<StreamOptions> for OpenAICompletionsOptions {
    /// Reads the Chat Completions options from `provider_options` under their
    /// Pi names (`toolChoice`, `reasoningEffort`, `thinkingBudgets`).
    fn from(stream: StreamOptions) -> Self {
        let provider_options = &stream.provider_options;
        let tool_choice = provider_options.get("toolChoice").cloned();
        let reasoning_effort = provider_options
            .get("reasoningEffort")
            .and_then(|value| serde_json::from_value(value.clone()).ok());
        let thinking_budgets = provider_options
            .get("thinkingBudgets")
            .and_then(|value| serde_json::from_value(value.clone()).ok());
        Self {
            tool_choice,
            reasoning_effort,
            thinking_budgets,
            stream,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConvertCompletionsMessagesOptions {
    pub grammar_tool_input_properties: Option<GrammarToolInputProperties>,
}

/// `ResolvedOpenAICompletionsCompat`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOpenAICompletionsCompat {
    pub supports_store: bool,
    pub supports_developer_role: bool,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    pub supports_finish_reason: bool,
    pub max_tokens_field: MaxTokensField,
    pub requires_tool_result_name: bool,
    pub requires_assistant_after_tool_result: bool,
    pub requires_thinking_as_text: bool,
    pub requires_reasoning_content_on_assistant_messages: bool,
    pub thinking_format: OpenAIThinkingFormat,
    pub open_router_routing: OpenRouterRouting,
    pub vercel_gateway_routing: VercelGatewayRouting,
    pub chat_template_kwargs: IndexMap<String, ChatTemplateKwargValue>,
    pub chat_template_args: IndexMap<String, ChatTemplateKwargValue>,
    pub zai_tool_stream: bool,
    pub supports_thinking_token_budget: Option<bool>,
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    pub supports_strict_mode: bool,
    pub supports_openai_grammar_tools: bool,
    pub supports_mid_convo_system_messages: Option<bool>,
    pub supports_mid_convo_tool_additions: Option<bool>,
    pub cache_control_format: Option<CacheControlFormat>,
    pub send_session_affinity_headers: bool,
    pub session_affinity_format: SessionAffinityFormat,
    pub supports_long_cache_retention: bool,
    pub vllm_priority: Option<i64>,
}

const OPENAI_COMPLETIONS_REASONING_FIELDS: [&str; 3] =
    ["reasoning", "reasoning_content", "reasoning_text"];

fn parse_openai_reasoning_details(signature: Option<&str>) -> Option<Vec<Value>> {
    let signature = signature.filter(|signature| !signature.is_empty())?;
    let parsed: Value = serde_json::from_str(signature).ok()?;
    let details = parsed.as_array()?;
    (!details.is_empty() && details.iter().all(is_openai_reasoning_detail)).then(|| details.clone())
}

fn parse_legacy_encrypted_reasoning_detail(signature: Option<&str>) -> Option<Value> {
    let signature = signature.filter(|signature| !signature.is_empty())?;
    let parsed: Value = serde_json::from_str(signature).ok()?;
    (is_openai_reasoning_detail(&parsed)
        && parsed.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
        && parsed
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty())
        && parsed
            .get("data")
            .and_then(Value::as_str)
            .is_some_and(|data| !data.is_empty()))
    .then_some(parsed)
}

fn fill_missing_common_reasoning_detail_fields(target: &mut Map<String, Value>, source: &Value) {
    // target.id ??= source.id
    if target.get("id").is_none_or(Value::is_null)
        && let Some(id) = source.get("id")
    {
        target.insert("id".to_string(), id.clone());
    }
    // target.format ||= source.format
    if !js_truthy(target.get("format"))
        && let Some(format) = source.get("format")
    {
        target.insert("format".to_string(), format.clone());
    }
    // target.index ??= source.index
    if target.get("index").is_none_or(Value::is_null)
        && let Some(index) = source.get("index")
    {
        target.insert("index".to_string(), index.clone());
    }
}

fn append_string_field(target: &mut Map<String, Value>, key: &str, source: &Value) {
    let combined = format!(
        "{}{}",
        target.get(key).and_then(Value::as_str).unwrap_or_default(),
        source.get(key).and_then(Value::as_str).unwrap_or_default()
    );
    target.insert(key.to_string(), Value::String(combined));
}

fn append_openai_reasoning_detail(details: &mut Vec<Value>, detail: &Value) {
    let detail_type = detail.get("type").and_then(Value::as_str);
    if let Some(Value::Object(last)) = details.last_mut() {
        let last_type = last.get("type").and_then(Value::as_str).map(str::to_string);
        if detail_type == Some("reasoning.text") && last_type.as_deref() == Some("reasoning.text") {
            append_string_field(last, "text", detail);
            // lastDetail.signature ||= detail.signature
            if !js_truthy(last.get("signature"))
                && let Some(signature) = detail.get("signature")
            {
                last.insert("signature".to_string(), signature.clone());
            }
            fill_missing_common_reasoning_detail_fields(last, detail);
            return;
        }
        if detail_type == Some("reasoning.summary")
            && last_type.as_deref() == Some("reasoning.summary")
        {
            append_string_field(last, "summary", detail);
            fill_missing_common_reasoning_detail_fields(last, detail);
            return;
        }
    }
    details.push(detail.clone());
}

/// Pi's `StreamingToolCallBlock.customInput`.
struct CustomInput {
    property: String,
    json_buffer: GrammarToolInputJsonBuffer,
}

/// Generate function for the OpenAI Chat Completions API.
pub fn stream_openai_completions(
    model: Model,
    context: TranscriptContext,
    options: OpenAICompletionsOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let normalized_context = resolve_transcript(
        &context,
        get_compat(&model).supports_mid_convo_system_messages,
    );

    let events = stream.clone();
    tokio::spawn(async move {
        let mut run = CompletionsRun {
            model: &model,
            options: &options,
            stream: &events,
            output: AssistantMessage::empty_for(&model),
            streamed_reasoning_details: None,
            text_block: None,
            thinking_block: None,
            tool_call_blocks_by_index: HashMap::new(),
            tool_call_blocks_by_id: HashMap::new(),
            partial_args: HashMap::new(),
            custom_inputs: HashMap::new(),
            stream_indexes: HashMap::new(),
            grammar_tool_input_properties: GrammarToolInputProperties::new(),
        };
        run.output.stop_reason = StopReason::Pending;
        run.output.timestamp = now_millis();

        match run.run(&normalized_context).await {
            Ok(()) => {
                events.push(AssistantMessageEvent::Done {
                    reason: run.output.stop_reason,
                    message: run.output.clone(),
                });
                events.end(None);
            }
            Err(error) => {
                for index in 0..run.output.content.len() {
                    if matches!(run.output.content[index], AssistantContent::Thinking(_)) {
                        run.apply_streamed_reasoning_details(index);
                    }
                }
                // Streaming scratch buffers are only used during parsing; never persist them.
                run.partial_args.clear();
                run.custom_inputs.clear();
                run.stream_indexes.clear();
                let aborted = options
                    .signal
                    .as_ref()
                    .is_some_and(|signal| signal.is_cancelled());
                let output = &mut run.output;
                output.stop_reason = if aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                let mut error_message =
                    format_provider_error(&normalize_provider_error(&error), None);
                // Some providers via OpenRouter give additional information in this field.
                // normalizeProviderError already stringifies the parsed body (error.error)
                // into errorMessage, so only append the raw metadata when it is not already
                // present to avoid double-printing it.
                if let Some(raw_metadata) = error_raw_metadata(&error) {
                    let raw_metadata = js_template_string(Some(&raw_metadata));
                    if !error_message.contains(&raw_metadata) {
                        error_message.push('\n');
                        error_message.push_str(&raw_metadata);
                    }
                }
                output.error_message = Some(error_message);
                events.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason,
                    error: output.clone(),
                });
                events.end(None);
            }
        }
    });

    stream
}

/// `(error as any)?.error?.metadata?.raw`: the SDK's `APIError.error` is the
/// `error` field of the parsed response body.
fn error_raw_metadata(error: &Error) -> Option<Value> {
    let Error::ProviderHttp(http) = error else {
        return None;
    };
    let body: Value = serde_json::from_str(http.body.as_deref()?).ok()?;
    body.get("error")?
        .get("metadata")?
        .get("raw")
        .filter(|raw| js_truthy(Some(raw)))
        .cloned()
}

/// State of one streamed Chat Completions request.
struct CompletionsRun<'a> {
    model: &'a Model,
    options: &'a OpenAICompletionsOptions,
    stream: &'a AssistantMessageEventStream,
    output: AssistantMessage,
    // `reasoning_details` are replay metadata, not user-visible stream deltas.
    // Keep them in memory during streaming and serialize once when the block is finalized.
    streamed_reasoning_details: Option<Vec<Value>>,
    text_block: Option<usize>,
    thinking_block: Option<usize>,
    tool_call_blocks_by_index: HashMap<u64, usize>,
    tool_call_blocks_by_id: HashMap<String, usize>,
    /// `StreamingToolCallBlock.partialArgs`, keyed by content index.
    partial_args: HashMap<usize, String>,
    /// `StreamingToolCallBlock.customInput`, keyed by content index.
    custom_inputs: HashMap<usize, CustomInput>,
    /// `StreamingToolCallBlock.streamIndex`, keyed by content index.
    stream_indexes: HashMap<usize, u64>,
    grammar_tool_input_properties: GrammarToolInputProperties,
}

/// `typeof index === "number"`, as a map key.
fn stream_index(tool_call: &Value) -> Option<u64> {
    tool_call
        .get("index")
        .and_then(Value::as_f64)
        .map(f64::to_bits)
}

impl CompletionsRun<'_> {
    fn apply_streamed_reasoning_details(&mut self, content_index: usize) {
        if let Some(details) = &self.streamed_reasoning_details {
            let signature = Value::Array(details.clone()).to_string();
            if let AssistantContent::Thinking(block) = &mut self.output.content[content_index] {
                block.thinking_signature = Some(signature);
            }
        }
    }

    fn tool_call_mut(&mut self, content_index: usize) -> &mut ToolCall {
        match &mut self.output.content[content_index] {
            AssistantContent::ToolCall(block) => block,
            _ => unreachable!("tool call index points at a tool call block"),
        }
    }

    fn push(&self, event: AssistantMessageEvent) {
        self.stream.push(event);
    }

    fn get_custom_tool_call_input(&self, content_index: usize) -> String {
        let Some(custom_input) = self.custom_inputs.get(&content_index) else {
            return String::new();
        };
        match &self.output.content[content_index] {
            AssistantContent::ToolCall(block) => block
                .arguments
                .get(&custom_input.property)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            _ => String::new(),
        }
    }

    fn append_custom_tool_call_input(
        &mut self,
        content_index: usize,
        next_input: &str,
        close: bool,
    ) -> Result<Option<String>> {
        let Some(custom_input) = self.custom_inputs.get_mut(&content_index) else {
            return Ok(None);
        };
        let delta = append_grammar_tool_input_json_delta(
            &mut custom_input.json_buffer,
            &custom_input.property,
            next_input,
            close,
        )?;
        let mut arguments = Map::new();
        arguments.insert(custom_input.property.clone(), json!(next_input));
        self.tool_call_mut(content_index).arguments = Value::Object(arguments);
        Ok(delta)
    }

    fn finish_block(&mut self, content_index: usize) -> Result<()> {
        match &self.output.content[content_index] {
            AssistantContent::Text(block) => {
                let content = block.text.clone();
                self.push(AssistantMessageEvent::TextEnd {
                    content_index,
                    content,
                    partial: self.output.clone(),
                });
            }
            AssistantContent::Thinking(_) => {
                self.apply_streamed_reasoning_details(content_index);
                let AssistantContent::Thinking(block) = &self.output.content[content_index] else {
                    unreachable!();
                };
                let content = block.thinking.clone();
                self.push(AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content,
                    partial: self.output.clone(),
                });
            }
            AssistantContent::ToolCall(_) => {
                if self.custom_inputs.contains_key(&content_index) {
                    let input = self.get_custom_tool_call_input(content_index);
                    if let Some(delta) =
                        self.append_custom_tool_call_input(content_index, &input, true)?
                    {
                        self.push(AssistantMessageEvent::ToolCallDelta {
                            content_index,
                            delta,
                            partial: self.output.clone(),
                        });
                    }
                } else {
                    let arguments = parse_streaming_json(
                        self.partial_args.get(&content_index).map(String::as_str),
                    );
                    self.tool_call_mut(content_index).arguments = arguments;
                }
                // Finalize in-place and strip the scratch buffers so replay only
                // carries parsed arguments.
                self.partial_args.remove(&content_index);
                self.custom_inputs.remove(&content_index);
                self.stream_indexes.remove(&content_index);
                let tool_call = self.tool_call_mut(content_index).clone();
                self.push(AssistantMessageEvent::ToolCallEnd {
                    content_index,
                    tool_call,
                    partial: self.output.clone(),
                });
            }
        }
        Ok(())
    }

    fn ensure_text_block(&mut self) -> usize {
        if let Some(index) = self.text_block {
            return index;
        }
        self.output.content.push(AssistantContent::text(""));
        let index = self.output.content.len() - 1;
        self.text_block = Some(index);
        self.push(AssistantMessageEvent::TextStart {
            content_index: index,
            partial: self.output.clone(),
        });
        index
    }

    fn ensure_thinking_block(&mut self, thinking_signature: &str) -> usize {
        if let Some(index) = self.thinking_block {
            return index;
        }
        self.output
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(thinking_signature.to_string()),
                redacted: None,
            }));
        let index = self.output.content.len() - 1;
        self.thinking_block = Some(index);
        self.push(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            partial: self.output.clone(),
        });
        index
    }

    fn start_custom_input(&mut self, content_index: usize, property: String) {
        let mut arguments = Map::new();
        arguments.insert(property.clone(), json!(""));
        self.tool_call_mut(content_index).arguments = Value::Object(arguments);
        self.custom_inputs.insert(
            content_index,
            CustomInput {
                property,
                json_buffer: GrammarToolInputJsonBuffer::default(),
            },
        );
    }

    fn ensure_tool_call_block(&mut self, tool_call: &Value) -> usize {
        let stream_index = stream_index(tool_call);
        let function = tool_call
            .get("function")
            .filter(|value| js_truthy(Some(value)));
        let custom = tool_call
            .get("custom")
            .filter(|value| js_truthy(Some(value)));
        let name = function
            .and_then(|function| function.get("name"))
            .filter(|name| !name.is_null())
            .or_else(|| {
                custom
                    .and_then(|custom| custom.get("name"))
                    .filter(|name| !name.is_null())
            })
            .map(|name| js_template_string(Some(name)))
            .unwrap_or_default();
        let id = tool_call
            .get("id")
            .filter(|id| js_truthy(Some(id)))
            .map(|id| js_template_string(Some(id)));

        let mut block =
            stream_index.and_then(|index| self.tool_call_blocks_by_index.get(&index).copied());
        if block.is_none()
            && let Some(id) = &id
        {
            block = self.tool_call_blocks_by_id.get(id).copied();
        }
        let block = match block {
            Some(block) => block,
            None => {
                // Note: the "input" fallback here should/must not be taken.  in case the LLM makes up
                // a tool we don't knwo about, we at least have a place to stash our stuff.
                let custom_input_property = (custom.is_some() && function.is_none()).then(|| {
                    self.grammar_tool_input_properties
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| "input".to_string())
                });
                self.output
                    .content
                    .push(AssistantContent::ToolCall(ToolCall {
                        id: id.clone().unwrap_or_default(),
                        name: name.clone(),
                        arguments: json!({}),
                        thought_signature: None,
                        namespace: None,
                    }));
                let content_index = self.output.content.len() - 1;
                match custom_input_property {
                    Some(property) => self.start_custom_input(content_index, property),
                    None => {
                        self.partial_args.insert(content_index, String::new());
                    }
                }
                if let Some(stream_index) = stream_index {
                    self.stream_indexes.insert(content_index, stream_index);
                    self.tool_call_blocks_by_index
                        .insert(stream_index, content_index);
                }
                if let Some(id) = &id {
                    self.tool_call_blocks_by_id
                        .insert(id.clone(), content_index);
                }
                self.push(AssistantMessageEvent::ToolCallStart {
                    content_index,
                    partial: self.output.clone(),
                });
                content_index
            }
        };
        if let Some(stream_index) = stream_index
            && !self.stream_indexes.contains_key(&block)
        {
            self.stream_indexes.insert(block, stream_index);
            self.tool_call_blocks_by_index.insert(stream_index, block);
        }
        if let Some(id) = &id {
            self.tool_call_blocks_by_id.insert(id.clone(), block);
        }
        if self.tool_call_mut(block).name.is_empty() && !name.is_empty() {
            self.tool_call_mut(block).name = name;
        }
        if custom.is_some() && function.is_none() && !self.custom_inputs.contains_key(&block) {
            let block_name = self.tool_call_mut(block).name.clone();
            let property = self
                .grammar_tool_input_properties
                .get(&block_name)
                .cloned()
                .unwrap_or_else(|| "input".to_string());
            self.start_custom_input(block, property);
            self.partial_args.remove(&block);
        }
        block
    }

    async fn run(&mut self, context: &TranscriptContext) -> Result<()> {
        let model = self.model;
        let options = self.options;
        let api_key = get_client_api_key(
            &model.provider,
            options.api_key.as_deref(),
            options.headers.as_ref(),
        )?;
        let compat = get_compat(model);
        self.grammar_tool_input_properties = create_grammar_tool_input_properties(
            Some(&get_declared_tools(&context.messages)),
            compat.supports_openai_grammar_tools,
        )?;
        let cache_retention =
            resolve_cache_retention(options.cache_retention, options.env.as_ref());
        let cache_session_id = if cache_retention == CacheRetention::None {
            None
        } else {
            options.session_id.as_deref()
        };
        let client = create_client(
            model,
            context,
            api_key,
            options.headers.as_ref(),
            options.http_client.as_ref(),
            cache_session_id,
            &compat,
        );
        let mut params = build_params(
            model,
            context,
            Some(options),
            &compat,
            cache_retention,
            &self.grammar_tool_input_properties,
        )?;
        if let Some(on_payload) = &options.on_payload
            && let Some(next_params) = on_payload(params.clone(), model).await?
        {
            params = next_params;
        }
        let request_options = OpenAIRequestOptions {
            signal: options.signal.clone(),
            timeout_ms: options.timeout_ms,
        };
        let response = retry_provider_request(
            || client.post("/chat/completions", &params, &request_options),
            &ProviderRetryOptions {
                max_retries: options.max_retries,
                max_retry_delay_ms: options.max_retry_delay_ms,
                signal: options.signal.clone(),
            },
        )
        .await?;
        if let Some(on_response) = &options.on_response {
            on_response(
                ProviderResponse {
                    status: response.status().as_u16(),
                    headers: headers_to_record(response.headers()),
                },
                model,
            )
            .await?;
        }
        self.push(AssistantMessageEvent::Start {
            partial: self.output.clone(),
        });

        let mut has_finish_reason = false;
        let openai_stream = sse_json_events(response, options.signal.clone());
        futures::pin_mut!(openai_stream);
        while let Some(chunk) = openai_stream.next().await {
            let chunk = chunk?;
            if let Some(hook) = &options.on_provider_stream_event {
                hook(&chunk, model).await;
            }
            if !chunk.is_object() {
                continue;
            }

            // OpenAI documents ChatCompletionChunk.id as the unique chat completion identifier,
            // and each chunk in a streamed completion carries the same id.
            if self.output.response_id.as_deref().is_none_or(str::is_empty)
                && let Some(id) = chunk.get("id").filter(|id| !id.is_null())
            {
                self.output.response_id = Some(js_template_string(Some(id)));
            }
            if let Some(chunk_model) = chunk.get("model").and_then(Value::as_str)
                && !chunk_model.is_empty()
                && chunk_model != model.id
                && self
                    .output
                    .response_model
                    .as_deref()
                    .is_none_or(str::is_empty)
            {
                self.output.response_model = Some(chunk_model.to_string());
            }
            let chunk_usage = chunk.get("usage").filter(|usage| js_truthy(Some(usage)));
            if let Some(usage) = chunk_usage {
                self.output.usage = parse_chunk_usage(usage, model);
            }

            let Some(choice) = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .filter(|choice| js_truthy(Some(choice)))
            else {
                continue;
            };

            // Fallback: some providers (e.g., Moonshot) return usage
            // in choice.usage instead of the standard chunk.usage
            if chunk_usage.is_none()
                && let Some(usage) = choice.get("usage").filter(|usage| js_truthy(Some(usage)))
            {
                self.output.usage = parse_chunk_usage(usage, model);
            }

            if let Some(finish_reason) = choice
                .get("finish_reason")
                .filter(|reason| js_truthy(Some(reason)))
            {
                let finish_reason = js_template_string(Some(finish_reason));
                let finish_reason_result = map_stop_reason(Some(&finish_reason));
                self.output.raw_stop_reason = Some(finish_reason);
                self.output.stop_reason = finish_reason_result.stop_reason;
                if let Some(error_message) = finish_reason_result.error_message {
                    self.output.error_message = Some(error_message);
                }
                has_finish_reason = true;
            }

            let Some(delta) = choice.get("delta").filter(|delta| js_truthy(Some(delta))) else {
                continue;
            };
            if let Some(content) = delta.get("content").and_then(Value::as_str)
                && !content.is_empty()
            {
                let index = self.ensure_text_block();
                if let AssistantContent::Text(block) = &mut self.output.content[index] {
                    block.text.push_str(content);
                }
                self.push(AssistantMessageEvent::TextDelta {
                    content_index: index,
                    delta: content.to_string(),
                    partial: self.output.clone(),
                });
            }

            // Some endpoints return reasoning in reasoning_content (llama.cpp),
            // or reasoning (other openai compatible endpoints)
            // Use the first non-empty reasoning field to avoid duplication
            // (e.g., chutes.ai returns both reasoning_content and reasoning with same content)
            let reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"];
            let found_reasoning_field = reasoning_fields.into_iter().find(|field| {
                delta
                    .get(*field)
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty())
            });

            if let Some(field) = found_reasoning_field
                && let Some(reasoning_delta) = delta.get(field).and_then(Value::as_str)
            {
                let thinking_signature = if model.provider == "opencode-go" && field == "reasoning"
                {
                    "reasoning_content"
                } else {
                    field
                };
                let index = self.ensure_thinking_block(thinking_signature);
                if let AssistantContent::Thinking(block) = &mut self.output.content[index] {
                    block.thinking.push_str(reasoning_delta);
                }
                self.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: index,
                    delta: reasoning_delta.to_string(),
                    partial: self.output.clone(),
                });
            }

            if let Some(tool_calls) = delta
                .get("tool_calls")
                .filter(|tool_calls| js_truthy(Some(tool_calls)))
                .and_then(Value::as_array)
            {
                for tool_call in tool_calls {
                    let block = self.ensure_tool_call_block(tool_call);
                    let id = tool_call
                        .get("id")
                        .filter(|id| js_truthy(Some(id)))
                        .map(|id| js_template_string(Some(id)));
                    if self.tool_call_mut(block).id.is_empty()
                        && let Some(id) = id
                    {
                        self.tool_call_mut(block).id = id.clone();
                        self.tool_call_blocks_by_id.insert(id, block);
                    }
                    let function = tool_call
                        .get("function")
                        .filter(|value| js_truthy(Some(value)));
                    let custom = tool_call
                        .get("custom")
                        .filter(|value| js_truthy(Some(value)));
                    let name = function
                        .and_then(|function| function.get("name"))
                        .filter(|name| !name.is_null())
                        .or_else(|| {
                            custom
                                .and_then(|custom| custom.get("name"))
                                .filter(|name| !name.is_null())
                        })
                        .map(|name| js_template_string(Some(name)));
                    if self.tool_call_mut(block).name.is_empty()
                        && let Some(name) = name.filter(|name| !name.is_empty())
                    {
                        self.tool_call_mut(block).name = name;
                    }

                    let mut tool_delta = String::new();
                    let function_arguments = function
                        .and_then(|function| function.get("arguments"))
                        .filter(|arguments| js_truthy(Some(arguments)));
                    let custom_input = custom
                        .and_then(|custom| custom.get("input"))
                        .filter(|input| js_truthy(Some(input)));
                    if let Some(arguments) = function_arguments {
                        tool_delta = js_template_string(Some(arguments));
                        let partial_args = self.partial_args.entry(block).or_default();
                        partial_args.push_str(&tool_delta);
                        let parsed = parse_streaming_json(Some(partial_args));
                        self.tool_call_mut(block).arguments = parsed;
                    } else if let Some(input) = custom_input {
                        let next_input = self.get_custom_tool_call_input(block)
                            + &js_template_string(Some(input));
                        tool_delta = self
                            .append_custom_tool_call_input(block, &next_input, false)?
                            .unwrap_or_default();
                    }
                    self.push(AssistantMessageEvent::ToolCallDelta {
                        content_index: block,
                        delta: tool_delta,
                        partial: self.output.clone(),
                    });
                }
            }

            if let Some(reasoning_details) =
                delta.get("reasoning_details").and_then(Value::as_array)
            {
                for detail in reasoning_details {
                    if !is_openai_reasoning_detail(detail) {
                        continue;
                    }
                    self.ensure_thinking_block("");
                    // Keep provider replay data in the existing signature slot. OpenRouter streams
                    // reasoning_details as deltas: consecutive text/summary deltas are merged into
                    // logical entries, while encrypted entries remain opaque and discrete.
                    append_openai_reasoning_detail(
                        self.streamed_reasoning_details.get_or_insert_with(Vec::new),
                        detail,
                    );
                }
            }
        }

        for content_index in 0..self.output.content.len() {
            self.finish_block(content_index)?;
        }
        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.is_cancelled())
        {
            return Err(Error::message("Request was aborted"));
        }

        if self.output.stop_reason == StopReason::Aborted {
            return Err(Error::message("Request was aborted"));
        }
        if !has_finish_reason && !compat.supports_finish_reason {
            self.output.stop_reason = if self
                .output
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_)))
            {
                StopReason::ToolUse
            } else {
                StopReason::Stop
            };
        }
        if self.output.stop_reason == StopReason::Error {
            return Err(Error::message(
                self.output
                    .error_message
                    .clone()
                    .filter(|message| !message.is_empty())
                    .unwrap_or_else(|| "Provider returned an error stop reason".to_string()),
            ));
        }
        if (compat.supports_finish_reason && !has_finish_reason)
            || self.output.stop_reason == StopReason::Pending
        {
            return Err(Error::message("Stream ended without finish_reason"));
        }
        Ok(())
    }
}

pub fn stream_simple_openai_completions(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions,
) -> Result<AssistantMessageEventStream> {
    get_client_api_key(
        &model.provider,
        options.api_key.as_deref(),
        options.headers.as_ref(),
    )?;

    let base = build_base_options(&model, &context, Some(&options), options.api_key.as_deref());
    let clamped_reasoning = options
        .reasoning
        .map(|reasoning| clamp_thinking_level(&model, reasoning.into()));
    let reasoning_effort = clamped_reasoning.and_then(model_thinking_level_to_effort);

    Ok(stream_openai_completions(
        model,
        context,
        OpenAICompletionsOptions {
            stream: base,
            tool_choice: tool_choice_value(options.tool_choice),
            reasoning_effort,
            thinking_budgets: options.thinking_budgets.clone(),
        },
    ))
}

fn create_client(
    model: &Model,
    context: &TranscriptContext,
    api_key: String,
    options_headers: Option<&ProviderHeaders>,
    http_client: Option<&reqwest::Client>,
    session_id: Option<&str>,
    compat: &ResolvedOpenAICompletionsCompat,
) -> OpenAIClient {
    let mut headers = ProviderHeaders::new();
    headers.insert("User-Agent", Some(get_pi_user_agent()));
    for (name, value) in model.headers.iter().flatten() {
        headers.insert(name.clone(), Some(value.clone()));
    }
    if model.provider == "github-copilot" {
        let has_images = has_copilot_vision_input(&context.messages);
        let copilot_headers = build_copilot_dynamic_headers(&context.messages, has_images);
        for (name, value) in copilot_headers.iter() {
            headers.insert(name.clone(), value.clone());
        }
    }

    if let Some(session_id) = session_id.filter(|session_id| !session_id.is_empty())
        && compat.send_session_affinity_headers
    {
        if compat.session_affinity_format == SessionAffinityFormat::Openrouter {
            headers.insert("x-session-id", Some(session_id.to_string()));
        } else {
            if compat.session_affinity_format == SessionAffinityFormat::Openai {
                headers.insert("session_id", Some(session_id.to_string()));
            }
            headers.insert("x-client-request-id", Some(session_id.to_string()));
            headers.insert("x-session-affinity", Some(session_id.to_string()));
        }
    }

    // Merge options headers last so they can override defaults
    if let Some(options_headers) = options_headers {
        for (name, value) in options_headers.iter() {
            headers.insert(name.clone(), value.clone());
        }
    }

    OpenAIClient::new(api_key, &model.base_url, http_client, headers)
}

/// The value of `model.thinkingLevelMap?.[level]`: `None` is undefined,
/// `Some(None)` is null.
fn thinking_level_map_entry(model: &Model, level: ModelThinkingLevel) -> Option<Option<String>> {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level))
        .cloned()
}

/// `model.thinkingLevelMap?.[effort] ?? effort`.
fn mapped_effort_or_requested(model: &Model, effort: ThinkingLevel) -> String {
    thinking_level_map_entry(model, effort.into())
        .flatten()
        .unwrap_or_else(|| effort.as_str().to_string())
}

pub fn build_params(
    model: &Model,
    context: &TranscriptContext,
    options: Option<&OpenAICompletionsOptions>,
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
    grammar_tool_input_properties: &GrammarToolInputProperties,
) -> Result<Value> {
    let transcript_tools = resolve_transcript_tools(
        &context.messages,
        compat.supports_mid_convo_system_messages == Some(true)
            && compat.supports_mid_convo_tool_additions == Some(true),
    );
    let mut messages = convert_messages(
        model,
        context,
        compat,
        Some(&ConvertCompletionsMessagesOptions {
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
        }),
    )?;
    let cache_control = get_compat_cache_control(compat, cache_retention);
    let session_id = options.and_then(|options| options.session_id.as_deref());
    let reasoning_effort = options.and_then(|options| options.reasoning_effort);

    let mut params = Map::new();
    params.insert("model".to_string(), json!(model.id));
    params.insert("messages".to_string(), Value::Null);
    params.insert("stream".to_string(), json!(true));
    if ((model.base_url.contains("api.openai.com") && cache_retention != CacheRetention::None)
        || (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention))
        && let Some(key) = clamp_openai_prompt_cache_key(session_id)
    {
        params.insert("prompt_cache_key".to_string(), json!(key));
    }
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        params.insert("prompt_cache_retention".to_string(), json!("24h"));
    }

    if compat.supports_usage_in_streaming {
        params.insert(
            "stream_options".to_string(),
            json!({ "include_usage": true }),
        );
    }

    if compat.supports_store {
        params.insert("store".to_string(), json!(false));
    }

    let max_tokens = options
        .and_then(|options| options.max_tokens)
        .filter(|max_tokens| *max_tokens > 0);
    if let Some(max_tokens) = max_tokens {
        if compat.max_tokens_field == MaxTokensField::MaxTokens {
            // Deprecated by OpenAI, but some OpenAI-compatible providers only accept max_tokens.
            params.insert("max_tokens".to_string(), json!(max_tokens));
        } else {
            params.insert("max_completion_tokens".to_string(), json!(max_tokens));
        }
    }

    if let Some(temperature) = options.and_then(|options| options.temperature) {
        params.insert("temperature".to_string(), json!(temperature));
    }

    let mut tools: Option<Vec<Value>> = None;
    if !transcript_tools.request_tools.is_empty() {
        tools = Some(convert_tools(&transcript_tools.request_tools, compat)?);
    } else if has_tool_history(&context.messages) {
        // Anthropic (via LiteLLM/proxy) requires tools param when conversation has tool_calls/tool_results
        tools = Some(Vec::new());
    }

    if let Some(cache_control) = &cache_control {
        apply_anthropic_cache_control(&mut messages, tools.as_mut(), cache_control);
    }
    params.insert("messages".to_string(), Value::Array(messages));
    if let Some(tools) = tools {
        let has_request_tools = !transcript_tools.request_tools.is_empty();
        params.insert("tools".to_string(), Value::Array(tools));
        if has_request_tools && compat.zai_tool_stream {
            params.insert("tool_stream".to_string(), json!(true));
        }
    }

    if let Some(tool_choice) = options
        .and_then(|options| options.tool_choice.as_ref())
        .filter(|tool_choice| js_truthy(Some(tool_choice)))
    {
        params.insert("tool_choice".to_string(), tool_choice.clone());
    }

    if let Some(vllm_priority) = compat.vllm_priority {
        params.insert("priority".to_string(), json!(vllm_priority));
    }

    let thinking_token_budget_field = resolve_thinking_token_budget_field(compat);
    let thinking_budget = resolve_clamped_thinking_budget(model, options, &params);

    match compat.thinking_format {
        OpenAIThinkingFormat::Zai if model.reasoning => {
            params.insert(
                "thinking".to_string(),
                if reasoning_effort.is_some() {
                    json!({ "type": "enabled", "clear_thinking": false })
                } else {
                    json!({ "type": "disabled" })
                },
            );
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                let effort = match thinking_level_map_entry(model, effort.into()) {
                    None => Some(effort.as_str().to_string()),
                    Some(mapped) => mapped,
                };
                if let Some(effort) = effort {
                    params.insert("reasoning_effort".to_string(), json!(effort));
                }
            }
        }
        OpenAIThinkingFormat::Qwen if model.reasoning => {
            params.insert(
                "enable_thinking".to_string(),
                json!(reasoning_effort.is_some()),
            );
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                params.insert(
                    "reasoning_effort".to_string(),
                    json!(mapped_effort_or_requested(model, effort)),
                );
            }
        }
        OpenAIThinkingFormat::QwenChatTemplate if model.reasoning => {
            params.insert(
                "chat_template_kwargs".to_string(),
                json!({
                    "enable_thinking": reasoning_effort.is_some(),
                    "preserve_thinking": true,
                }),
            );
        }
        OpenAIThinkingFormat::ChatTemplate if model.reasoning => {
            if let Some(chat_template_kwargs) = build_chat_template_values(
                model,
                reasoning_effort,
                &compat.chat_template_kwargs,
                thinking_budget,
            ) {
                params.insert("chat_template_kwargs".to_string(), chat_template_kwargs);
            }
        }
        OpenAIThinkingFormat::Baseten if model.reasoning => {
            if let Some(chat_template_args) = build_chat_template_values(
                model,
                reasoning_effort,
                &compat.chat_template_args,
                thinking_budget,
            ) {
                params.insert("chat_template_args".to_string(), chat_template_args);
            }
            if compat.supports_reasoning_effort {
                let mapped_effort = match reasoning_effort {
                    Some(effort) => thinking_level_map_entry(model, effort.into()),
                    None => thinking_level_map_entry(model, ModelThinkingLevel::Off),
                };
                let effort = match mapped_effort {
                    None => reasoning_effort.map(|effort| effort.as_str().to_string()),
                    Some(mapped) => mapped,
                };
                if let Some(effort) = effort {
                    params.insert("reasoning_effort".to_string(), json!(effort));
                }
            }
        }
        OpenAIThinkingFormat::Deepseek if model.reasoning => {
            if reasoning_effort.is_some() {
                params.insert("thinking".to_string(), json!({ "type": "enabled" }));
            } else if thinking_level_map_entry(model, ModelThinkingLevel::Off) != Some(None) {
                params.insert("thinking".to_string(), json!({ "type": "disabled" }));
            }
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                params.insert(
                    "reasoning_effort".to_string(),
                    json!(mapped_effort_or_requested(model, effort)),
                );
            }
        }
        OpenAIThinkingFormat::Openrouter if model.reasoning => {
            // OpenRouter normalizes reasoning across providers via a nested reasoning object.
            if let Some(effort) = reasoning_effort {
                params.insert(
                    "reasoning".to_string(),
                    json!({ "effort": mapped_effort_or_requested(model, effort) }),
                );
            } else {
                let off = thinking_level_map_entry(model, ModelThinkingLevel::Off);
                if off != Some(None) {
                    params.insert(
                        "reasoning".to_string(),
                        json!({ "effort": off.flatten().unwrap_or_else(|| "none".to_string()) }),
                    );
                }
            }
        }
        OpenAIThinkingFormat::AntLing if model.reasoning && reasoning_effort.is_some() => {
            if let Some(effort) = reasoning_effort
                && let Some(Some(effort)) = thinking_level_map_entry(model, effort.into())
            {
                params.insert("reasoning".to_string(), json!({ "effort": effort }));
            }
        }
        OpenAIThinkingFormat::Together if model.reasoning => {
            params.insert(
                "reasoning".to_string(),
                json!({ "enabled": reasoning_effort.is_some() }),
            );
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                params.insert(
                    "reasoning_effort".to_string(),
                    json!(mapped_effort_or_requested(model, effort)),
                );
            }
        }
        OpenAIThinkingFormat::StringThinking if model.reasoning => {
            if let Some(effort) = reasoning_effort {
                params.insert(
                    "thinking".to_string(),
                    json!(mapped_effort_or_requested(model, effort)),
                );
            } else {
                let off = thinking_level_map_entry(model, ModelThinkingLevel::Off);
                if off != Some(None) {
                    params.insert(
                        "thinking".to_string(),
                        json!(off.flatten().unwrap_or_else(|| "none".to_string())),
                    );
                }
            }
        }
        _ => {
            if let Some(effort) = reasoning_effort
                && model.reasoning
                && compat.supports_reasoning_effort
            {
                // OpenAI-style reasoning_effort
                params.insert(
                    "reasoning_effort".to_string(),
                    json!(mapped_effort_or_requested(model, effort)),
                );
            } else if reasoning_effort.is_none()
                && model.reasoning
                && compat.supports_reasoning_effort
                && let Some(Some(off_value)) =
                    thinking_level_map_entry(model, ModelThinkingLevel::Off)
            {
                params.insert("reasoning_effort".to_string(), json!(off_value));
            }
        }
    }

    // Cap reasoning with a top-level budget field. Independent of thinkingFormat: the
    // same server can serve zai, qwen or chat-template models. Reasoning and the answer
    // share max_tokens here, so an uncapped reasoning phase can consume the whole
    // response and leave no answer and no tool call.
    if let Some(field) = thinking_token_budget_field
        && let Some(thinking_budget) = thinking_budget
    {
        let field = serde_json::to_value(field)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_default();
        params.insert(field, json!(thinking_budget));
    }

    // OpenRouter provider routing preferences
    if let Some(routing) = model
        .compat
        .as_ref()
        .and_then(|compat| compat.open_router_routing.as_ref())
    {
        params.insert("provider".to_string(), serde_json::to_value(routing)?);
    }

    // Vercel AI Gateway provider routing preferences
    if let Some(routing) = model
        .compat
        .as_ref()
        .and_then(|compat| compat.vercel_gateway_routing.as_ref())
        && (routing.only.is_some() || routing.order.is_some())
    {
        let mut gateway_options = Map::new();
        if let Some(only) = &routing.only {
            gateway_options.insert("only".to_string(), json!(only));
        }
        if let Some(order) = &routing.order {
            gateway_options.insert("order".to_string(), json!(order));
        }
        params.insert(
            "providerOptions".to_string(),
            json!({ "gateway": gateway_options }),
        );
    }

    // Last so model and request sampling parameters override named request fields.
    if let Some(sampling_params) = resolve_sampling_params(
        model,
        reasoning_effort
            .map(ModelThinkingLevel::from)
            .unwrap_or(ModelThinkingLevel::Off),
        options.and_then(|options| options.sampling_params.as_ref()),
    ) {
        for (key, value) in sampling_params {
            params.insert(key, value);
        }
    }

    Ok(Value::Object(params))
}

fn resolve_thinking_token_budget_field(
    compat: &ResolvedOpenAICompletionsCompat,
) -> Option<ThinkingTokenBudgetField> {
    if let Some(field) = compat.thinking_token_budget_field {
        return Some(field);
    }
    if compat.supports_thinking_token_budget == Some(true) {
        return Some(ThinkingTokenBudgetField::ThinkingTokenBudget);
    }
    None
}

fn resolve_clamped_thinking_budget(
    model: &Model,
    options: Option<&OpenAICompletionsOptions>,
    params: &Map<String, Value>,
) -> Option<u32> {
    let options = options?;
    let effort = options.reasoning_effort?;
    if !model.reasoning {
        return None;
    }
    let ceiling = params
        .get("max_tokens")
        .filter(|value| !value.is_null())
        .or_else(|| {
            params
                .get("max_completion_tokens")
                .filter(|value| !value.is_null())
        })
        .and_then(Value::as_u64)
        .map(|value| value.min(u64::from(u32::MAX)) as u32)
        .unwrap_or(model.max_tokens);
    let budget = clamp_thinking_budget_to_answer_room(
        thinking_budget_for_level(effort, options.thinking_budgets.as_ref()),
        ceiling,
    );
    (budget > 0).then_some(budget)
}

fn build_chat_template_values(
    model: &Model,
    reasoning_effort: Option<ThinkingLevel>,
    values: &IndexMap<String, ChatTemplateKwargValue>,
    thinking_budget: Option<u32>,
) -> Option<Value> {
    let mut resolved_values = Map::new();

    for (key, value) in values {
        if let Some(resolved) =
            resolve_chat_template_kwarg_value(model, reasoning_effort, value, thinking_budget)
        {
            resolved_values.insert(key.clone(), resolved);
        }
    }

    (!resolved_values.is_empty()).then_some(Value::Object(resolved_values))
}

fn resolve_chat_template_kwarg_value(
    model: &Model,
    reasoning_effort: Option<ThinkingLevel>,
    value: &ChatTemplateKwargValue,
    thinking_budget: Option<u32>,
) -> Option<Value> {
    let variable = match value {
        ChatTemplateKwargValue::String(text) => return Some(json!(text)),
        ChatTemplateKwargValue::Number(number) => return Some(Value::Number(number.clone())),
        ChatTemplateKwargValue::Boolean(flag) => return Some(json!(flag)),
        ChatTemplateKwargValue::Null(()) => return Some(Value::Null),
        ChatTemplateKwargValue::Variable(variable) => variable,
    };

    if reasoning_effort.is_none() && variable.omit_when_off {
        return None;
    }
    match variable.variable {
        ChatTemplateVariable::ThinkingEnabled => return Some(json!(reasoning_effort.is_some())),
        ChatTemplateVariable::ThinkingBudget => return thinking_budget.map(|budget| json!(budget)),
        ChatTemplateVariable::ThinkingEffort => {}
    }

    let mapped_value = match reasoning_effort {
        Some(effort) => thinking_level_map_entry(model, effort.into()),
        None => thinking_level_map_entry(model, ModelThinkingLevel::Off),
    };
    match mapped_value {
        None => reasoning_effort.map(|effort| json!(effort.as_str())),
        Some(Some(mapped)) => Some(json!(mapped)),
        Some(None) => None,
    }
}

fn get_compat_cache_control(
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
) -> Option<Value> {
    if compat.cache_control_format != Some(CacheControlFormat::Anthropic)
        || cache_retention == CacheRetention::None
    {
        return None;
    }

    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        Some(json!({ "type": "ephemeral", "ttl": "1h" }))
    } else {
        Some(json!({ "type": "ephemeral" }))
    }
}

fn apply_anthropic_cache_control(
    messages: &mut [Value],
    tools: Option<&mut Vec<Value>>,
    cache_control: &Value,
) {
    add_cache_control_to_system_prompt(messages, cache_control);
    add_cache_control_to_last_tool(tools, cache_control);
    add_cache_control_to_last_conversation_message(messages, cache_control);
}

fn message_role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn add_cache_control_to_system_prompt(messages: &mut [Value], cache_control: &Value) {
    for message in messages.iter_mut() {
        if matches!(message_role(message), Some("system" | "developer")) {
            add_cache_control_to_text_content(message, cache_control);
            return;
        }
    }
}

fn add_cache_control_to_last_conversation_message(messages: &mut [Value], cache_control: &Value) {
    for message in messages.iter_mut().rev() {
        if matches!(message_role(message), Some("user" | "assistant" | "tool"))
            && add_cache_control_to_text_content(message, cache_control)
        {
            return;
        }
    }
}

fn add_cache_control_to_last_tool(tools: Option<&mut Vec<Value>>, cache_control: &Value) {
    let Some(last_tool) = tools.and_then(|tools| tools.last_mut()) else {
        return;
    };
    if let Some(last_tool) = last_tool.as_object_mut() {
        last_tool.insert("cache_control".to_string(), cache_control.clone());
    }
}

fn add_cache_control_to_text_content(message: &mut Value, cache_control: &Value) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    if let Some(text) = content.as_str() {
        if text.is_empty() {
            return false;
        }
        *content = json!([{
            "type": "text",
            "text": text,
            "cache_control": cache_control,
        }]);
        return true;
    }

    let Some(parts) = content.as_array_mut() else {
        return false;
    };

    for part in parts.iter_mut().rev() {
        if part.get("type").and_then(Value::as_str) == Some("text")
            && let Some(part) = part.as_object_mut()
        {
            part.insert("cache_control".to_string(), cache_control.clone());
            return true;
        }
    }

    false
}

/// `id.replace(/[^a-zA-Z0-9_-]/g, "_")` (per UTF-16 code unit).
fn sanitize_tool_call_id_part(part: &str) -> String {
    let mut sanitized = String::new();
    for character in part.chars() {
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            sanitized.push(character);
        } else {
            for _ in 0..character.len_utf16() {
                sanitized.push('_');
            }
        }
    }
    sanitized
}

/// `text.slice(0, n)` in UTF-16 code units.
fn slice_utf16(text: &str, length: usize) -> String {
    let units: Vec<u16> = text.encode_utf16().take(length).collect();
    String::from_utf16_lossy(&units)
}

fn image_url_part(mime_type: &str, data: &str) -> Value {
    json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{mime_type};base64,{data}") },
    })
}

pub fn convert_messages(
    model: &Model,
    context: &TranscriptContext,
    compat: &ResolvedOpenAICompletionsCompat,
    options: Option<&ConvertCompletionsMessagesOptions>,
) -> Result<Vec<Value>> {
    let normalized_context = resolve_transcript(context, compat.supports_mid_convo_system_messages);
    let mut params: Vec<Value> = Vec::new();

    let normalize_tool_call_id =
        |id: &str, _target: &Model, _source: &AssistantMessage| -> String {
            // Handle pipe-separated IDs from OpenAI Responses API
            // Format: {call_id}|{id} where {id} can be 400+ chars with special chars (+, /, =)
            // These come from providers like github-copilot, openai-codex, opencode
            // Extract just the call_id part and normalize it
            // Multiple tool calls in the same turn can share call_id but differ by item_id.
            // Preserve item-level uniqueness when replaying into Chat Completions, which
            // requires distinct tool call ids.
            if let Some(separator_index) = id.find('|') {
                // Sanitize to allowed chars and truncate to 40 chars (OpenAI limit)
                let call_id = sanitize_tool_call_id_part(&id[..separator_index]);
                let item_id = sanitize_tool_call_id_part(&id[separator_index + 1..]);
                let combined_id = if item_id.is_empty() {
                    call_id.clone()
                } else {
                    format!("{call_id}_{item_id}")
                };
                if combined_id.len() <= 40 {
                    return combined_id;
                }
                let hash = slice_utf16(&short_hash(id), 8);
                let prefix = slice_utf16(&call_id, 1usize.max(40 - hash.len() - 1));
                return format!("{prefix}_{hash}");
            }

            if model.provider == "openai" && id.encode_utf16().count() > 40 {
                return slice_utf16(id, 40);
            }
            id.to_string()
        };

    let transformed_messages = transform_messages(
        &normalized_context.messages,
        model,
        Some(&normalize_tool_call_id),
    );
    let transcript_tools = resolve_transcript_tools(
        &normalized_context.messages,
        compat.supports_mid_convo_system_messages == Some(true)
            && compat.supports_mid_convo_tool_additions == Some(true),
    );
    let instruction_role = if model.reasoning && compat.supports_developer_role {
        "developer"
    } else {
        "system"
    };
    let grammar_properties =
        options.and_then(|options| options.grammar_tool_input_properties.as_ref());

    let mut last_role: Option<&str> = None;

    let mut i = 0;
    while i < transformed_messages.len() {
        let msg = &transformed_messages[i];
        // Some providers don't allow user messages directly after tool results
        // Insert a synthetic assistant message to bridge the gap
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(msg, Message::User(_))
        {
            params.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results.",
            }));
        }

        match msg {
            Message::System(system) => {
                let added_tools: &[Tool] = if i > 0 && transcript_tools.anchors_additions {
                    system.tools_added.as_deref().unwrap_or_default()
                } else {
                    &[]
                };
                if !added_tools.is_empty() {
                    params.push(json!({
                        "role": "system",
                        "tools": convert_tools(added_tools, compat)?,
                    }));
                }
                let text = if i == 0 {
                    get_system_message_text(system)
                } else {
                    render_system_message_update(system)
                };
                if !text.is_empty() {
                    params.push(json!({
                        "role": instruction_role,
                        "content": sanitize_surrogates(&text),
                    }));
                }
            }
            Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => {
                    params.push(json!({ "role": "user", "content": sanitize_surrogates(text) }));
                }
                UserMessageContent::Parts(parts) => {
                    let content: Vec<Value> = parts
                        .iter()
                        .filter(|item| match item {
                            UserContent::Text(text) => !text.text.is_empty(),
                            UserContent::Image(_) => true,
                        })
                        .map(|item| match item {
                            UserContent::Text(text) => json!({
                                "type": "text",
                                "text": sanitize_surrogates(&text.text),
                            }),
                            UserContent::Image(image) => {
                                image_url_part(&image.mime_type, &image.data)
                            }
                        })
                        .collect();
                    if content.is_empty() {
                        i += 1;
                        continue;
                    }
                    params.push(json!({ "role": "user", "content": content }));
                }
            },
            Message::Assistant(assistant) => {
                // Some providers don't accept null content, use empty string instead
                let mut assistant_msg = Map::new();
                assistant_msg.insert("role".to_string(), json!("assistant"));
                assistant_msg.insert(
                    "content".to_string(),
                    if compat.requires_assistant_after_tool_result {
                        json!("")
                    } else {
                        Value::Null
                    },
                );

                let assistant_text_parts: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                            Some(json!({
                                "type": "text",
                                "text": sanitize_surrogates(&text.text),
                            }))
                        }
                        _ => None,
                    })
                    .collect();
                let assistant_text: String = assistant_text_parts
                    .iter()
                    .map(|part| part["text"].as_str().unwrap_or_default())
                    .collect();
                let thinking_blocks: Vec<&ThinkingContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking) => Some(thinking),
                        _ => None,
                    })
                    .collect();
                let tool_calls: Vec<&ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .collect();
                let signed_reasoning_details = thinking_blocks.iter().find_map(|block| {
                    parse_openai_reasoning_details(block.thinking_signature.as_deref())
                });
                let legacy_reasoning_details: Vec<Value> = tool_calls
                    .iter()
                    .filter_map(|tool_call| {
                        parse_legacy_encrypted_reasoning_detail(
                            tool_call.thought_signature.as_deref(),
                        )
                    })
                    .collect();
                let preserved_reasoning_details = signed_reasoning_details
                    .or((!legacy_reasoning_details.is_empty()).then_some(legacy_reasoning_details));

                let non_empty_thinking_blocks: Vec<&&ThinkingContent> = thinking_blocks
                    .iter()
                    .filter(|block| !block.thinking.trim().is_empty())
                    .collect();
                if !non_empty_thinking_blocks.is_empty() {
                    if compat.requires_thinking_as_text {
                        // Convert thinking blocks to plain text (no tags to avoid model mimicking them)
                        let thinking_text = non_empty_thinking_blocks
                            .iter()
                            .map(|block| sanitize_surrogates(&block.thinking))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        let mut content = vec![json!({ "type": "text", "text": thinking_text })];
                        content.extend(assistant_text_parts.iter().cloned());
                        assistant_msg.insert("content".to_string(), Value::Array(content));
                    } else {
                        // Always send assistant content as a plain string (OpenAI Chat Completions
                        // API standard format). Sending as an array of {type:"text", text:"..."}
                        // objects is non-standard and causes some models (e.g. DeepSeek V3.2 via
                        // NVIDIA NIM) to mirror the content-block structure literally in their
                        // output, producing recursive nesting like [{'type':'text','text':'[{...}]'}].
                        if !assistant_text.is_empty() {
                            assistant_msg.insert("content".to_string(), json!(assistant_text));
                        }

                        // reasoning_details is the structured alternative to a raw reasoning field.
                        if preserved_reasoning_details.is_none() {
                            // Use the signature from the first thinking block if available (for llama.cpp server + gpt-oss)
                            let mut signature =
                                non_empty_thinking_blocks[0].thinking_signature.as_deref();
                            if model.provider == "opencode-go" && signature == Some("reasoning") {
                                signature = Some("reasoning_content");
                            }
                            if let Some(signature) = signature
                                && OPENAI_COMPLETIONS_REASONING_FIELDS.contains(&signature)
                            {
                                assistant_msg.insert(
                                    signature.to_string(),
                                    json!(
                                        non_empty_thinking_blocks
                                            .iter()
                                            .map(|block| block.thinking.as_str())
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    ),
                                );
                            }
                        }
                    }
                } else if !assistant_text.is_empty() {
                    // Always send assistant content as a plain string (OpenAI Chat Completions
                    // API standard format). Sending as an array of {type:"text", text:"..."}
                    // objects is non-standard and causes some models (e.g. DeepSeek V3.2 via
                    // NVIDIA NIM) to mirror the content-block structure literally in their
                    // output, producing recursive nesting like [{'type':'text','text':'[{...}]'}].
                    assistant_msg.insert("content".to_string(), json!(assistant_text));
                }

                let has_tool_calls = !tool_calls.is_empty();
                if has_tool_calls {
                    let converted = tool_calls
                        .iter()
                        .map(|tool_call| {
                            if let Some(property) = grammar_properties
                                .and_then(|properties| properties.get(&tool_call.name))
                            {
                                return Ok(json!({
                                    "id": tool_call.id,
                                    "type": "custom",
                                    "custom": {
                                        "name": tool_call.name,
                                        "input": sanitize_surrogates(get_grammar_tool_input(
                                            &tool_call.name,
                                            &tool_call.arguments,
                                            property,
                                        )?),
                                    },
                                }));
                            }
                            Ok(json!({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments.to_string(),
                                },
                            }))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    assistant_msg.insert("tool_calls".to_string(), Value::Array(converted));
                }
                if let Some(details) = preserved_reasoning_details {
                    assistant_msg.insert("reasoning_details".to_string(), Value::Array(details));
                }
                if compat.requires_reasoning_content_on_assistant_messages
                    && model.reasoning
                    && !assistant_msg.contains_key("reasoning_content")
                {
                    assistant_msg.insert("reasoning_content".to_string(), json!(""));
                }
                // Skip assistant messages that have no content and no tool calls.
                // Some providers require "either content or tool_calls, but not none".
                // Other providers also don't accept empty assistant messages.
                // This handles aborted assistant responses that got no content.
                let has_content = match assistant_msg.get("content") {
                    Some(Value::String(text)) => !text.is_empty(),
                    Some(Value::Array(parts)) => !parts.is_empty(),
                    _ => false,
                };
                if !has_content && !has_tool_calls {
                    i += 1;
                    continue;
                }
                params.push(Value::Object(assistant_msg));
            }
            Message::ToolResult(_) => {
                let mut image_blocks: Vec<Value> = Vec::new();
                let mut j = i;

                while j < transformed_messages.len() {
                    let Message::ToolResult(tool_msg) = &transformed_messages[j] else {
                        break;
                    };

                    // Extract text and image content
                    let text_result = tool_msg
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            UserContent::Text(text) => Some(text.text.as_str()),
                            UserContent::Image(_) => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let has_images = tool_msg
                        .content
                        .iter()
                        .any(|block| matches!(block, UserContent::Image(_)));

                    // Always send tool result with text (or placeholder if only images)
                    let has_text = !text_result.is_empty();
                    let tool_result_text = if has_text {
                        text_result.as_str()
                    } else if has_images {
                        "(see attached image)"
                    } else {
                        "(no tool output)"
                    };
                    // Some providers require the 'name' field in tool results
                    let mut tool_result_msg = json!({
                        "role": "tool",
                        "content": sanitize_surrogates(tool_result_text),
                        "tool_call_id": tool_msg.tool_call_id,
                    });
                    if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                        tool_result_msg["name"] = json!(tool_msg.tool_name);
                    }
                    params.push(tool_result_msg);

                    if has_images && model.input.contains(&ModelInput::Image) {
                        for block in &tool_msg.content {
                            if let UserContent::Image(image) = block {
                                image_blocks.push(image_url_part(&image.mime_type, &image.data));
                            }
                        }
                    }
                    j += 1;
                }

                i = j;

                if !image_blocks.is_empty() {
                    if compat.requires_assistant_after_tool_result {
                        params.push(json!({
                            "role": "assistant",
                            "content": "I have processed the tool results.",
                        }));
                    }

                    let mut content = vec![json!({
                        "type": "text",
                        "text": "Attached image(s) from tool result:",
                    })];
                    content.extend(image_blocks);
                    params.push(json!({ "role": "user", "content": content }));
                    last_role = Some("user");
                } else {
                    last_role = Some("toolResult");
                }

                continue;
            }
        }

        last_role = Some(msg.role());
        i += 1;
    }

    Ok(params)
}

fn convert_tools(tools: &[Tool], compat: &ResolvedOpenAICompletionsCompat) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) =
                resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)?
            {
                return Ok(json!({
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": {
                                "syntax": grammar.format.as_str(),
                                "definition": grammar.definition,
                            },
                        },
                    },
                }));
            }

            let strict =
                resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode, None)?;
            let mut function = json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": get_json_schema_tool_parameters(tool, strict)?,
            });
            // Only include strict if provider supports it. Some reject unknown fields.
            if compat.supports_strict_mode {
                function["strict"] = json!(strict.unwrap_or(false));
            }
            Ok(json!({ "type": "function", "function": function }))
        })
        .collect()
}

/// `value || 0` / `value ?? 0` for a token count.
fn token_count(value: Option<&Value>) -> u32 {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .map(|number| number as u32)
        .unwrap_or(0)
}

/// `a ?? b ?? c ?? 0`: the first present (non-null) count.
fn first_present_count(values: &[Option<&Value>]) -> u32 {
    values
        .iter()
        .flatten()
        .find(|value| !value.is_null())
        .map(|value| token_count(Some(value)))
        .unwrap_or(0)
}

fn parse_chunk_usage(raw_usage: &Value, model: &Model) -> Usage {
    let prompt_tokens = token_count(raw_usage.get("prompt_tokens"));
    let prompt_details = raw_usage.get("prompt_tokens_details");
    let cache_read_tokens = first_present_count(&[
        prompt_details.and_then(|details| details.get("cached_tokens")),
        raw_usage.get("prompt_cache_hit_tokens"),
        raw_usage.get("cached_tokens"),
    ]);
    let cache_write_tokens =
        token_count(prompt_details.and_then(|details| details.get("cache_write_tokens")));

    // Follow documented OpenAI/OpenRouter semantics: cached_tokens is cache-read
    // tokens (hits). Providers disagree on placement: OpenAI/OpenRouter use
    // prompt_tokens_details.cached_tokens, DeepSeek uses prompt_cache_hit_tokens,
    // and Kimi documents top-level usage.cached_tokens on the final usage chunk.
    // OpenAI does not document or emit cache_write_tokens, but
    // OpenRouter-compatible providers can include it as a separate write count.
    // OpenRouter's own provider/tests affirm the separate mapping:
    // https://github.com/OpenRouterTeam/ai-sdk-provider/pull/409
    // Do not subtract writes from cached_tokens, otherwise spec-compliant
    // providers are under-reported. DS4 mirrors this contract too:
    // https://github.com/antirez/ds4/pull/29
    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    // OpenAI completion_tokens already includes reasoning_tokens.
    let output_tokens = token_count(raw_usage.get("completion_tokens"));
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        cache_write_1h: None,
        reasoning: Some(token_count(
            raw_usage
                .get("completion_tokens_details")
                .and_then(|details| details.get("reasoning_tokens")),
        )),
        total_tokens: input + output_tokens + cache_read_tokens + cache_write_tokens,
        cost: UsageCost::default(),
    };
    calculate_cost(model, &mut usage);
    usage
}

struct MappedStop {
    stop_reason: StopReason,
    error_message: Option<String>,
}

fn map_stop_reason(reason: Option<&str>) -> MappedStop {
    let stop = |stop_reason| MappedStop {
        stop_reason,
        error_message: None,
    };
    let Some(reason) = reason else {
        return stop(StopReason::Stop);
    };
    match reason {
        "stop" | "end" => stop(StopReason::Stop),
        "length" => stop(StopReason::Length),
        "function_call" | "tool_calls" => stop(StopReason::ToolUse),
        _ => MappedStop {
            stop_reason: StopReason::Error,
            error_message: Some(format!("Provider finish_reason: {reason}")),
        },
    }
}

/// Auto-detect compatibility settings from provider name and baseUrl.
/// Used as the base when model.compat is not set; explicit model.compat
/// entries override these detected values.
fn detect_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let provider = model.provider.as_str();
    let base_url = model.base_url.as_str();

    let is_zai = provider == "zai"
        || provider == "zai-coding-cn"
        || base_url.contains("api.z.ai")
        || base_url.contains("open.bigmodel.cn");
    let is_together = provider == "together"
        || base_url.contains("api.together.ai")
        || base_url.contains("api.together.xyz");
    let is_moonshot = provider == "moonshotai"
        || provider == "moonshotai-cn"
        || base_url.contains("api.moonshot.");
    let is_open_router = provider == "openrouter" || base_url.contains("openrouter.ai");
    let is_cloudflare_workers_ai =
        provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
    let is_cloudflare_ai_gateway =
        provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
    let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
    let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
    let is_cerebras = provider == "cerebras" || base_url.contains("cerebras.ai");
    let is_deep_seek = provider == "deepseek" || base_url.to_lowercase().contains("deepseek.com");

    let is_non_standard = is_nvidia
        || is_cerebras
        || provider == "xai"
        || base_url.contains("api.x.ai")
        || is_together
        || base_url.contains("chutes.ai")
        || is_deep_seek
        || is_zai
        || is_moonshot
        || provider == "opencode"
        || base_url.contains("opencode.ai")
        || is_cloudflare_workers_ai
        || is_cloudflare_ai_gateway
        || is_ant_ling;

    let use_max_tokens = base_url.contains("chutes.ai")
        || is_deep_seek
        || is_moonshot
        || is_cloudflare_ai_gateway
        || is_together
        || is_nvidia
        || is_ant_ling
        || is_zai;

    let is_grok = provider == "xai" || base_url.contains("api.x.ai");
    let is_open_router_developer_role_model =
        is_open_router && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
    let cache_control_format = (provider == "openrouter" && model.id.starts_with("anthropic/"))
        .then_some(CacheControlFormat::Anthropic);

    ResolvedOpenAICompletionsCompat {
        supports_store: !is_non_standard,
        supports_developer_role: is_open_router_developer_role_model
            || (!is_non_standard && !is_open_router),
        supports_reasoning_effort: !is_grok
            && !is_zai
            && !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia
            && !is_ant_ling,
        supports_usage_in_streaming: true,
        supports_finish_reason: true,
        max_tokens_field: if use_max_tokens {
            MaxTokensField::MaxTokens
        } else {
            MaxTokensField::MaxCompletionTokens
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: is_deep_seek,
        thinking_format: if is_deep_seek {
            OpenAIThinkingFormat::Deepseek
        } else if is_zai {
            OpenAIThinkingFormat::Zai
        } else if is_together {
            OpenAIThinkingFormat::Together
        } else if is_ant_ling {
            OpenAIThinkingFormat::AntLing
        } else if is_open_router {
            OpenAIThinkingFormat::Openrouter
        } else {
            OpenAIThinkingFormat::Openai
        },
        open_router_routing: OpenRouterRouting::default(),
        vercel_gateway_routing: VercelGatewayRouting::default(),
        chat_template_kwargs: IndexMap::new(),
        chat_template_args: IndexMap::new(),
        zai_tool_stream: false,
        supports_thinking_token_budget: Some(false),
        thinking_token_budget_field: None,
        // OpenAI compatibility alone does not imply strict JSON-schema tool support.
        supports_strict_mode: false,
        supports_openai_grammar_tools: false,
        supports_mid_convo_system_messages: Some(false),
        supports_mid_convo_tool_additions: Some(false),
        cache_control_format,
        send_session_affinity_headers: is_open_router,
        session_affinity_format: if is_open_router {
            SessionAffinityFormat::Openrouter
        } else {
            SessionAffinityFormat::Openai
        },
        supports_long_cache_retention: !(is_together
            || is_cloudflare_workers_ai
            || is_cloudflare_ai_gateway
            || is_nvidia
            || is_ant_ling),
        vllm_priority: None,
    }
}

/// Get resolved compatibility settings for a model.
/// Auto-detects from provider/URL then overrides with explicit model.compat.
pub fn get_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let detected = detect_compat(model);
    let Some(compat) = &model.compat else {
        return detected;
    };

    ResolvedOpenAICompletionsCompat {
        supports_store: compat.supports_store.unwrap_or(detected.supports_store),
        supports_developer_role: compat
            .supports_developer_role
            .unwrap_or(detected.supports_developer_role),
        supports_reasoning_effort: compat
            .supports_reasoning_effort
            .unwrap_or(detected.supports_reasoning_effort),
        supports_usage_in_streaming: compat
            .supports_usage_in_streaming
            .unwrap_or(detected.supports_usage_in_streaming),
        supports_finish_reason: compat
            .supports_finish_reason
            .unwrap_or(detected.supports_finish_reason),
        max_tokens_field: compat.max_tokens_field.unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: compat
            .requires_tool_result_name
            .unwrap_or(detected.requires_tool_result_name),
        requires_assistant_after_tool_result: compat
            .requires_assistant_after_tool_result
            .unwrap_or(detected.requires_assistant_after_tool_result),
        requires_thinking_as_text: compat
            .requires_thinking_as_text
            .unwrap_or(detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages: compat
            .requires_reasoning_content_on_assistant_messages
            .unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
        thinking_format: compat.thinking_format.unwrap_or(detected.thinking_format),
        open_router_routing: compat.open_router_routing.clone().unwrap_or_default(),
        vercel_gateway_routing: compat
            .vercel_gateway_routing
            .clone()
            .unwrap_or(detected.vercel_gateway_routing),
        chat_template_kwargs: compat
            .chat_template_kwargs
            .clone()
            .unwrap_or(detected.chat_template_kwargs),
        chat_template_args: compat
            .chat_template_args
            .clone()
            .unwrap_or(detected.chat_template_args),
        zai_tool_stream: compat.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
        supports_thinking_token_budget: compat
            .supports_thinking_token_budget
            .or(detected.supports_thinking_token_budget),
        thinking_token_budget_field: compat
            .thinking_token_budget_field
            .or(detected.thinking_token_budget_field),
        supports_strict_mode: compat
            .supports_strict_mode
            .unwrap_or(detected.supports_strict_mode),
        supports_openai_grammar_tools: compat
            .supports_openai_grammar_tools
            .unwrap_or(detected.supports_openai_grammar_tools),
        supports_mid_convo_system_messages: compat
            .supports_mid_convo_system_messages
            .or(detected.supports_mid_convo_system_messages),
        supports_mid_convo_tool_additions: compat
            .supports_mid_convo_tool_additions
            .or(detected.supports_mid_convo_tool_additions),
        cache_control_format: compat
            .cache_control_format
            .or(detected.cache_control_format),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(detected.send_session_affinity_headers),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or(detected.session_affinity_format),
        supports_long_cache_retention: compat
            .supports_long_cache_retention
            .unwrap_or(detected.supports_long_cache_retention),
        vllm_priority: compat.vllm_priority,
    }
}

struct OpenAICompletionsApi;

impl ProviderStreams for OpenAICompletionsApi {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        stream_openai_completions(model, context, options.into())
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let error_model = model.clone();
        stream_simple_openai_completions(model, context, options)
            .unwrap_or_else(|error| error_stream(&error_model, error))
    }
}

/// The `openai-completions` implementation as `ProviderStreams`.
pub fn openai_completions_api() -> Arc<dyn ProviderStreams> {
    Arc::new(OpenAICompletionsApi)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::api::openai_client::test_support::{
        CapturedRequest, MockResponse, MockServer, capture_payload, collect, collect_aborting,
        context, model, names, openai_model, serve_stalled_sse, stream_event_hook,
        tool_addition_context, tool_change_context, tool_change_model,
    };
    use crate::types::ToolChoice;

    fn hi() -> TranscriptContext {
        context(json!({ "messages": [{ "role": "user", "content": "hi", "timestamp": 1 }] }))
    }

    fn sys_hi() -> TranscriptContext {
        context(json!({
            "systemPrompt": "Follow instructions.",
            "messages": [{ "role": "user", "content": "hi", "timestamp": 1 }],
        }))
    }

    fn with_key(api_key: &str) -> OpenAICompletionsOptions {
        let mut options = OpenAICompletionsOptions::default();
        options.api_key = Some(api_key.to_string());
        options
    }

    fn simple(reasoning: Option<ThinkingLevel>) -> SimpleStreamOptions {
        let mut options = SimpleStreamOptions::default();
        options.api_key = Some("test".to_string());
        options.reasoning = reasoning;
        options
    }

    /// A completions model from a Pi literal, with Pi's zero cost default.
    fn completions_model(overrides: Value) -> Model {
        let mut base = json!({
            "id": "test-model",
            "name": "Test Model",
            "api": "openai-completions",
            "provider": "openai",
            "baseUrl": "https://api.openai.com/v1",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 128000,
            "maxTokens": 4096,
        });
        for (key, value) in overrides.as_object().unwrap() {
            base[key] = value.clone();
        }
        model(base)
    }

    /// Pi's `localOpenAICompletionsModel`.
    fn local_model(overrides: Value) -> Model {
        let mut base = json!({
            "id": "local-model",
            "name": "Local Model",
            "provider": "local",
            "baseUrl": "http://localhost:8000/v1",
            "reasoning": true,
            "contextWindow": 128000,
            "maxTokens": 4096,
        });
        for (key, value) in overrides.as_object().unwrap() {
            base[key] = value.clone();
        }
        completions_model(base)
    }

    /// `getModel("openai", id)` without its compat, as `openai-completions`.
    fn openai_as_completions(id: &str) -> Model {
        let mut model = openai_model(id);
        model.compat = None;
        model.api = "openai-completions".into();
        model
    }

    fn openrouter_auto() -> Model {
        completions_model(json!({
            "id": "openrouter/auto",
            "name": "OpenRouter Auto",
            "provider": "openrouter",
            "baseUrl": "https://openrouter.ai/api/v1",
            "contextWindow": 200000,
            "maxTokens": 8192,
        }))
    }

    fn read_tool() -> Value {
        json!({
            "name": "read",
            "description": "Read a file",
            "parameters": { "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] },
        })
    }

    fn stop_chunk() -> Value {
        json!({
            "choices": [{ "delta": {}, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 0 } },
        })
    }

    async fn payload_for(
        model: &Model,
        ctx: TranscriptContext,
        mut base: OpenAICompletionsOptions,
    ) -> Value {
        let model = model.clone();
        if base.api_key.is_none() {
            base.api_key = Some("test".to_string());
        }
        capture_payload(move |hook| {
            base.on_payload = Some(hook);
            stream_openai_completions(model, ctx, base)
        })
        .await
    }

    async fn simple_payload(
        model: &Model,
        ctx: TranscriptContext,
        mut options: SimpleStreamOptions,
    ) -> Value {
        let model = model.clone();
        capture_payload(move |hook| {
            options.on_payload = Some(hook);
            stream_simple_openai_completions(model, ctx, options).unwrap()
        })
        .await
    }

    /// Stream against a mock server serving `chunks` and return the request
    /// and the final message.
    async fn run_chunks(
        mut model: Model,
        ctx: TranscriptContext,
        options: OpenAICompletionsOptions,
        chunks: &[Value],
    ) -> (CapturedRequest, AssistantMessage) {
        let server = MockServer::start(vec![MockResponse::sse(chunks)]).await;
        model.base_url = server.url.clone();
        let (_, result) = collect(stream_openai_completions(model, ctx, options)).await;
        (server.last(), result)
    }

    async fn run_simple_chunks(
        mut model: Model,
        ctx: TranscriptContext,
        options: SimpleStreamOptions,
        chunks: &[Value],
    ) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
        let server = MockServer::start(vec![MockResponse::sse(chunks)]).await;
        model.base_url = server.url.clone();
        collect(stream_simple_openai_completions(model, ctx, options).unwrap()).await
    }

    // transcript-tool-changes.test.ts (Kimi and OpenAI-compatible)

    async fn tool_change_payload(model: Model, ctx: TranscriptContext) -> Value {
        simple_payload(&model, ctx, simple(None)).await
    }

    fn system_contents(payload: &Value) -> Vec<Value> {
        payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "system")
            .map(|message| message.get("content").cloned().unwrap_or(Value::Null))
            .collect()
    }

    #[tokio::test]
    async fn anchors_kimi_additions_in_tool_bearing_system_messages() {
        let model = tool_change_model(
            "kimi-k3",
            "openai-completions",
            "moonshotai",
            json!({ "supportsMidConvoSystemMessages": true, "supportsMidConvoToolAdditions": true }),
        );
        let payload = tool_change_payload(model, tool_addition_context()).await;
        assert_eq!(names(&payload["tools"], "/function/name"), ["base_tool"]);
        let with_tools = payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message.get("tools").is_some())
            .unwrap();
        assert_eq!(names(&with_tools["tools"], "/function/name"), ["late_tool"]);
        assert!(with_tools.get("content").is_none());
        assert_eq!(
            system_contents(&payload),
            [json!("base prompt"), Value::Null, json!("updated guidance")]
        );
    }

    #[tokio::test]
    async fn keeps_kimi_k2_system_text_inline_without_dynamic_tool_messages() {
        let model = tool_change_model(
            "kimi-k2.7-code",
            "openai-completions",
            "moonshotai",
            json!({ "supportsMidConvoSystemMessages": true }),
        );
        let payload = tool_change_payload(model, tool_addition_context()).await;
        assert_eq!(
            names(&payload["tools"], "/function/name"),
            ["base_tool", "late_tool"]
        );
        assert!(
            payload["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|message| message.get("tools").is_none())
        );
        assert_eq!(
            system_contents(&payload),
            [json!("base prompt"), json!("updated guidance")]
        );
    }

    #[tokio::test]
    async fn folds_openai_compatible_updates_into_the_system_prompt_without_native_support() {
        let mut model = tool_change_model(
            "custom-model",
            "openai-completions",
            "custom-provider",
            Value::Null,
        );
        model.reasoning = false;
        let payload = tool_change_payload(model, tool_change_context()).await;
        assert_eq!(names(&payload["tools"], "/function/name"), ["late_tool"]);
        let roles: Vec<_> = payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].clone())
            .collect();
        assert_eq!(roles, [json!("system"), json!("user")]);
        assert_eq!(
            payload["messages"][0]["content"],
            "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"
        );
    }

    #[tokio::test]
    async fn abort_mid_stream_ends_open_blocks_then_fails_as_aborted() {
        // The SDK swallows the abort and ends the stream, so Pi finishes the
        // open blocks before "Request was aborted".
        let head = format!(
            "data: {}\n\n",
            json!({ "id": "chatcmpl-1", "choices": [{ "delta": { "content": "Hel" } }] })
        );
        let mut model = completions_model(json!({}));
        model.base_url = serve_stalled_sse(head).await;
        let signal = tokio_util::sync::CancellationToken::new();
        let mut options = with_key("test");
        options.signal = Some(signal.clone());
        let (events, result) = collect_aborting(
            stream_openai_completions(model, hi(), options),
            signal,
            "text_delta",
        )
        .await;
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type())
                .collect::<Vec<_>>(),
            ["start", "text_start", "text_delta", "text_end", "error"]
        );
        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
        assert_eq!(
            content_json(&result),
            json!([{ "type": "text", "text": "Hel" }])
        );
    }

    fn content_json(message: &AssistantMessage) -> Value {
        serde_json::to_value(&message.content).unwrap()
    }

    // openai-completions-raw-stop-reason.test.ts
    #[tokio::test]
    async fn preserves_raw_finish_reasons_for_successful_stops() {
        let chunk = json!({ "id": "chatcmpl-1", "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }] });
        let (_, message) = run_chunks(
            completions_model(json!({})),
            hi(),
            with_key("test"),
            &[chunk],
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(message.raw_stop_reason.as_deref(), Some("stop"));
        assert_eq!(message.error_message, None);
    }

    #[tokio::test]
    async fn preserves_raw_finish_reasons_for_provider_error_stops() {
        let chunk = json!({ "id": "chatcmpl-2", "choices": [{ "index": 0, "delta": {}, "finish_reason": "content_filter" }] });
        let (_, message) = run_chunks(
            completions_model(json!({})),
            hi(),
            with_key("test"),
            &[chunk],
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(message.raw_stop_reason.as_deref(), Some("content_filter"));
        assert_eq!(
            message.error_message.as_deref(),
            Some("Provider finish_reason: content_filter")
        );
    }

    // openai-completions-provider-stream-event.test.ts
    #[tokio::test]
    async fn exposes_provider_chunks_including_openrouter_metadata() {
        let first = json!({
            "id": "chatcmpl-1", "model": "anthropic/claude-sonnet-4.6",
            "choices": [{ "index": 0, "delta": { "content": "hello" } }],
        });
        let last = json!({
            "id": "chatcmpl-1", "model": "anthropic/claude-sonnet-4.6",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0012, "is_byok": false },
            "openrouter_metadata": { "strategy": "direct", "region": "iad" },
        });
        let (hook, events) = stream_event_hook();
        let mut options = simple(None);
        options.on_provider_stream_event = Some(hook);
        let (_, message) = run_simple_chunks(
            openrouter_auto(),
            hi(),
            options,
            &[first.clone(), last.clone()],
        )
        .await;
        assert_eq!(
            content_json(&message),
            json!([{ "type": "text", "text": "hello" }])
        );
        let events: Vec<Value> = events
            .lock()
            .iter()
            .map(|(event, _)| event.clone())
            .collect();
        assert_eq!(events, vec![first, last]);
    }

    // openai-completions-response-model.test.ts
    fn usage_chunk(id: &str, model: Option<&str>, prompt: u32, completion: u32) -> Value {
        let mut chunk = json!({
            "id": id,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": prompt, "completion_tokens": completion,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 0 } },
        });
        if let Some(model) = model {
            chunk["model"] = json!(model);
        }
        chunk
    }

    #[tokio::test]
    async fn surfaces_routed_chunk_model_on_response_model() {
        let chunks = [
            json!({ "id": "chatcmpl-1", "model": "anthropic/claude-opus-4.8", "choices": [{ "index": 0, "delta": { "content": "hi" } }] }),
            usage_chunk("chatcmpl-1", Some("anthropic/claude-opus-4.8"), 10, 5),
        ];
        let (_, message) = run_chunks(openrouter_auto(), hi(), with_key("test"), &chunks).await;
        assert_eq!(message.model, "openrouter/auto");
        assert_eq!(
            message.response_model.as_deref(),
            Some("anthropic/claude-opus-4.8")
        );
        assert_eq!(message.provider, "openrouter");
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn leaves_response_model_unset_when_chunks_echo_requested_id() {
        let chunks = [
            json!({ "id": "chatcmpl-2", "model": "openrouter/auto", "choices": [{ "index": 0, "delta": { "content": "hi" } }] }),
            usage_chunk("chatcmpl-2", Some("openrouter/auto"), 1, 1),
        ];
        let (_, message) = run_chunks(openrouter_auto(), hi(), with_key("test"), &chunks).await;
        assert_eq!(message.model, "openrouter/auto");
        assert_eq!(message.response_model, None);
    }

    #[tokio::test]
    async fn ignores_empty_or_missing_chunk_model() {
        let chunks = [
            json!({ "id": "chatcmpl-3", "choices": [{ "index": 0, "delta": { "content": "hi" } }] }),
            json!({ "id": "chatcmpl-3", "model": "", "choices": [{ "index": 0, "delta": { "content": "!" } }] }),
            usage_chunk("chatcmpl-3", None, 1, 2),
        ];
        let (_, message) = run_chunks(openrouter_auto(), hi(), with_key("test"), &chunks).await;
        assert_eq!(message.model, "openrouter/auto");
        assert_eq!(message.response_model, None);
    }

    // openai-completions-retry.test.ts
    fn retry_model() -> Model {
        completions_model(json!({
            "provider": "opencode-go",
            "baseUrl": "https://opencode.ai/zen/go/v1",
            "contextWindow": 1000,
            "maxTokens": 100,
        }))
    }

    fn ok_chunks() -> MockResponse {
        MockResponse::sse(&[
            json!({ "id": "chatcmpl-test", "choices": [{ "index": 0, "delta": { "content": "ok" } }] }),
            json!({ "id": "chatcmpl-test", "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }] }),
        ])
    }

    async fn retry_run(
        responses: Vec<MockResponse>,
        max_retries: Option<u32>,
        max_retry_delay_ms: Option<u64>,
    ) -> (usize, AssistantMessage) {
        let server = MockServer::start(responses).await;
        let mut model = retry_model();
        model.base_url = server.url.clone();
        let mut options = with_key("test");
        options.max_retries = max_retries;
        options.max_retry_delay_ms = max_retry_delay_ms;
        let (_, result) = collect(stream_openai_completions(model, hi(), options)).await;
        (server.requests().len(), result)
    }

    #[tokio::test]
    async fn does_not_retry_by_default() {
        let (requests, result) = retry_run(
            vec![
                MockResponse::status(500, &[("retry-after-ms", "1")], "server error"),
                ok_chunks(),
            ],
            None,
            None,
        )
        .await;
        assert_eq!(requests, 1);
        assert_eq!(result.stop_reason, StopReason::Error);
    }

    #[tokio::test]
    async fn honors_provider_retries() {
        let (requests, result) = retry_run(
            vec![
                MockResponse::status(429, &[("retry-after-ms", "100")], "rate limited"),
                MockResponse::status(500, &[("retry-after-ms", "100")], "server error"),
                ok_chunks(),
            ],
            Some(2),
            Some(100),
        )
        .await;
        assert_eq!(requests, 3);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(
            content_json(&result),
            json!([{ "type": "text", "text": "ok" }])
        );
    }

    #[tokio::test]
    async fn fails_immediately_when_retry_delay_exceeds_the_limit() {
        let (requests, result) = retry_run(
            vec![
                MockResponse::status(429, &[("retry-after", "277403")], "rate limited"),
                ok_chunks(),
            ],
            Some(2),
            Some(1000),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        let message = result.error_message.unwrap();
        assert!(
            message.contains("Server requested 277403s retry delay (max: 1s)"),
            "{message}"
        );
        assert!(message.contains("rate limited"), "{message}");
        assert_eq!(requests, 1);
    }

    // openai-completions-reasoning-details.test.ts
    fn reasoning_detail() -> Value {
        json!({ "type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature" })
    }

    fn signed_reasoning_text_detail() -> Value {
        json!({
            "type": "reasoning.text", "text": "I should call the read tool.",
            "signature": "sha256:signed-text", "id": "reasoning-text-1",
            "format": "anthropic-claude-v1", "index": 0,
        })
    }

    fn reasoning_summary_detail() -> Value {
        json!({
            "type": "reasoning.summary", "summary": "Decided to inspect the requested file.",
            "id": "reasoning-summary-1", "format": "anthropic-claude-v1", "index": 1,
        })
    }

    fn gemini_model() -> Model {
        completions_model(json!({
            "id": "google/gemini-test",
            "name": "Gemini Test",
            "provider": "openrouter",
            "baseUrl": "https://openrouter.ai/api/v1",
            "reasoning": true,
            "contextWindow": 100000,
        }))
    }

    fn delta_chunk(delta: Value, finish_reason: Option<&str>) -> Value {
        json!({
            "id": "chatcmpl-test", "model": "google/gemini-test",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        })
    }

    fn tool_call_chunk() -> Value {
        delta_chunk(
            json!({ "tool_calls": [{ "index": 0, "id": "call_1", "type": "function",
                "function": { "name": "read", "arguments": "{\"path\":\"README.md\"}" } }] }),
            None,
        )
    }

    fn reasoning_context(messages: &[AssistantMessage]) -> TranscriptContext {
        let messages: Vec<Value> = messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        context(json!({ "messages": messages, "tools": [read_tool()] }))
    }

    async fn run_reasoning(
        messages: &[AssistantMessage],
        chunks: &[Value],
    ) -> (Value, AssistantMessage) {
        let (request, result) = run_chunks(
            gemini_model(),
            reasoning_context(messages),
            with_key("test"),
            chunks,
        )
        .await;
        (request.body, result)
    }

    fn assistant_payload(payload: &Value) -> Value {
        payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "assistant")
            .cloned()
            .unwrap()
    }

    fn find_block(message: &AssistantMessage, kind: &str) -> Value {
        content_json(message)
            .as_array()
            .unwrap()
            .iter()
            .find(|block| block["type"] == kind)
            .cloned()
            .unwrap_or(Value::Null)
    }

    fn replay_chunks() -> Vec<Value> {
        vec![
            delta_chunk(json!({ "content": "ok" }), None),
            delta_chunk(json!({}), Some("stop")),
        ]
    }

    #[tokio::test]
    async fn preserves_reasoning_details_in_the_thinking_signature() {
        let (_, assistant) = run_reasoning(
            &[],
            &[
                delta_chunk(json!({ "reasoning_details": [reasoning_detail()] }), None),
                tool_call_chunk(),
                delta_chunk(json!({}), Some("tool_calls")),
            ],
        )
        .await;
        assert_eq!(
            find_block(&assistant, "thinking"),
            json!({ "type": "thinking", "thinking": "",
                "thinkingSignature": serde_json::to_string(&json!([reasoning_detail()])).unwrap() })
        );
        assert_eq!(
            find_block(&assistant, "toolCall"),
            json!({ "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "README.md" } })
        );
        let (payload, _) = run_reasoning(&[assistant], &replay_chunks()).await;
        assert_eq!(
            assistant_payload(&payload)["reasoning_details"],
            json!([reasoning_detail()])
        );
    }

    #[tokio::test]
    async fn falls_back_to_encrypted_tool_call_signatures() {
        let (_, mut assistant) = run_reasoning(
            &[],
            &[
                delta_chunk(json!({ "reasoning_details": [reasoning_detail()] }), None),
                tool_call_chunk(),
                delta_chunk(json!({}), Some("tool_calls")),
            ],
        )
        .await;
        assistant
            .content
            .retain(|block| !matches!(block, AssistantContent::Thinking(_)));
        for block in &mut assistant.content {
            if let AssistantContent::ToolCall(tool_call) = block {
                tool_call.thought_signature = Some(reasoning_detail().to_string());
            }
        }
        let (payload, _) = run_reasoning(&[assistant], &replay_chunks()).await;
        assert_eq!(
            assistant_payload(&payload)["reasoning_details"],
            json!([reasoning_detail()])
        );
    }

    #[tokio::test]
    async fn preserves_signed_text_and_summary_reasoning_details_in_sequence() {
        let signed = signed_reasoning_text_detail();
        let (_, assistant) = run_reasoning(
            &[],
            &[
                delta_chunk(json!({ "reasoning": signed["text"], "reasoning_details": [signed] }), None),
                delta_chunk(json!({ "reasoning_details": [reasoning_detail(), reasoning_summary_detail()] }), None),
                tool_call_chunk(),
                delta_chunk(json!({}), Some("tool_calls")),
            ],
        )
        .await;
        let expected = json!([signed, reasoning_detail(), reasoning_summary_detail()]);
        assert_eq!(
            find_block(&assistant, "thinking"),
            json!({ "type": "thinking", "thinking": signed["text"],
                "thinkingSignature": serde_json::to_string(&expected).unwrap() })
        );
        let (payload, _) = run_reasoning(&[assistant], &replay_chunks()).await;
        let replayed = assistant_payload(&payload);
        assert_eq!(replayed["reasoning_details"], expected);
        assert!(replayed.get("reasoning").is_none());
    }

    #[tokio::test]
    async fn merges_consecutive_text_and_summary_reasoning_details_deltas() {
        let text_delta = json!({ "type": "reasoning.text", "text": "The", "index": 0 });
        let text_delta_with_signature = json!({
            "type": "reasoning.text", "text": " user wants the time.",
            "signature": "sha256:text-signature", "format": "openai-responses-v1", "index": 0,
        });
        let summary_delta = json!({ "type": "reasoning.summary", "summary": "Looked", "index": 0 });
        let summary_delta_with_format = json!({
            "type": "reasoning.summary", "summary": " up time.", "format": "openai-responses-v1", "index": 0,
        });
        let later_summary_delta = json!({
            "type": "reasoning.summary", "summary": "After encrypted block.", "format": "openai-responses-v1", "index": 0,
        });
        let expected = json!([
            { "type": "reasoning.text", "text": "The user wants the time.", "index": 0,
                "signature": "sha256:text-signature", "format": "openai-responses-v1" },
            { "type": "reasoning.summary", "summary": "Looked up time.", "index": 0, "format": "openai-responses-v1" },
            reasoning_detail(),
            later_summary_delta,
        ]);
        let (_, assistant) = run_reasoning(
            &[],
            &[
                delta_chunk(json!({ "reasoning_details": [text_delta] }), None),
                delta_chunk(
                    json!({ "reasoning_details": [text_delta_with_signature] }),
                    None,
                ),
                delta_chunk(json!({ "reasoning_details": [summary_delta] }), None),
                delta_chunk(
                    json!({ "reasoning_details": [summary_delta_with_format] }),
                    None,
                ),
                delta_chunk(json!({ "reasoning_details": [reasoning_detail()] }), None),
                delta_chunk(json!({ "reasoning_details": [later_summary_delta] }), None),
                tool_call_chunk(),
                delta_chunk(json!({}), Some("tool_calls")),
            ],
        )
        .await;
        let thinking = find_block(&assistant, "thinking");
        assert_eq!(thinking["thinking"], "");
        let signature: Value =
            serde_json::from_str(thinking["thinkingSignature"].as_str().unwrap()).unwrap();
        assert_eq!(signature, expected);
        assert_eq!(
            thinking["thinkingSignature"],
            serde_json::to_string(&expected).unwrap()
        );
        let (payload, _) = run_reasoning(&[assistant], &replay_chunks()).await;
        assert_eq!(assistant_payload(&payload)["reasoning_details"], expected);
    }

    // openai-completions-thinking-as-text.test.ts
    fn thinking_as_text_model() -> Model {
        completions_model(json!({
            "id": "repro-model",
            "name": "Repro Model",
            "provider": "repro-provider",
            "baseUrl": "http://127.0.0.1:1",
            "reasoning": true,
            "compat": {
                "supportsStore": true, "supportsDeveloperRole": true, "supportsReasoningEffort": true,
                "supportsUsageInStreaming": true, "supportsFinishReason": true,
                "maxTokensField": "max_completion_tokens", "requiresToolResultName": false,
                "requiresAssistantAfterToolResult": false, "requiresThinkingAsText": true,
                "requiresReasoningContentOnAssistantMessages": false, "thinkingFormat": "openai",
                "openRouterRouting": {}, "vercelGatewayRouting": {}, "chatTemplateKwargs": {},
                "chatTemplateArgs": {}, "zaiToolStream": false, "supportsThinkingTokenBudget": false,
                "supportsStrictMode": true, "supportsOpenAIGrammarTools": false,
                "supportsMidConvoSystemMessages": false, "supportsMidConvoToolAdditions": false,
                "sendSessionAffinityHeaders": false, "sessionAffinityFormat": "openai",
                "supportsLongCacheRetention": true,
            },
        }))
    }

    fn thinking_as_text_context(content: Value) -> TranscriptContext {
        context(json!({
            "messages": [
                { "role": "user", "content": "hello", "timestamp": 1 },
                {
                    "role": "assistant", "content": content, "api": "openai-completions",
                    "provider": "repro-provider", "model": "repro-model",
                    "usage": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 } },
                    "stopReason": "stop", "timestamp": 2,
                },
                { "role": "user", "content": "continue", "timestamp": 3 },
            ],
        }))
    }

    #[test]
    fn serializes_same_model_thinking_plus_text_replay_as_text_parts() {
        let model = thinking_as_text_model();
        let ctx = thinking_as_text_context(json!([
            { "type": "thinking", "thinking": "internal reasoning" },
            { "type": "text", "text": "visible answer" },
        ]));
        let messages = convert_messages(&model, &ctx, &get_compat(&model), None).unwrap();
        assert_eq!(
            messages[1],
            json!({ "role": "assistant", "content": [
                { "type": "text", "text": "internal reasoning" },
                { "type": "text", "text": "visible answer" },
            ] })
        );
    }

    #[test]
    fn serializes_same_model_thinking_only_replay_as_text_parts() {
        let model = thinking_as_text_model();
        let ctx = thinking_as_text_context(
            json!([{ "type": "thinking", "thinking": "internal reasoning" }]),
        );
        let messages = convert_messages(&model, &ctx, &get_compat(&model), None).unwrap();
        assert_eq!(
            messages[1],
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "internal reasoning" }] })
        );
    }

    #[tokio::test]
    async fn thinking_as_text_replay_reaches_the_endpoint() {
        let ctx = thinking_as_text_context(json!([
            { "type": "thinking", "thinking": "internal reasoning" },
            { "type": "text", "text": "visible answer" },
        ]));
        let server = MockServer::start(vec![MockResponse::sse(&[
            json!({ "id": "chatcmpl-repro", "object": "chat.completion.chunk", "created": 0, "model": "repro-model",
                "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "ok" }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-repro", "object": "chat.completion.chunk", "created": 0, "model": "repro-model",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1 } }),
        ])])
        .await;
        let mut model = thinking_as_text_model();
        model.base_url = server.url.trim_end_matches("/v1").to_string();
        let (events, _) =
            collect(stream_openai_completions(model, ctx, with_key("test-key"))).await;
        let requests = server.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].path, "/chat/completions");
        assert_eq!(
            requests[0].body["messages"][1],
            json!({ "role": "assistant", "content": [
                { "type": "text", "text": "internal reasoning" },
                { "type": "text", "text": "visible answer" },
            ] })
        );
        assert!(matches!(
            events.last(),
            Some(AssistantMessageEvent::Done { .. })
        ));
    }

    // openai-completions-thinking-token-budget.test.ts
    fn vllm_model(compat: Value) -> Model {
        completions_model(json!({
            "id": "zai-org/glm-5.2",
            "name": "GLM 5.2 (local vLLM)",
            "provider": "local-vllm",
            "baseUrl": "http://localhost:8000/v1",
            "reasoning": true,
            "contextWindow": 262144,
            "maxTokens": 16384,
            "compat": compat,
        }))
    }

    fn default_vllm() -> Model {
        vllm_model(json!({ "thinkingFormat": "zai", "supportsThinkingTokenBudget": true }))
    }

    async fn capture_budget(
        model: &Model,
        reasoning: Option<ThinkingLevel>,
        budgets: Option<Value>,
        max_tokens: Option<u32>,
    ) -> Value {
        let ctx =
            context(json!({ "messages": [{ "role": "user", "content": "Hi", "timestamp": 1 }] }));
        let mut options = simple(reasoning);
        options.thinking_budgets = budgets.map(|budgets| serde_json::from_value(budgets).unwrap());
        options.max_tokens = max_tokens;
        simple_payload(model, ctx, options).await
    }

    #[tokio::test]
    async fn sends_the_configured_budget_for_the_requested_level() {
        let params = capture_budget(
            &default_vllm(),
            Some(ThinkingLevel::Medium),
            Some(json!({ "medium": 4096 })),
            None,
        )
        .await;
        assert_eq!(params["thinking_token_budget"], 4096);
    }

    #[tokio::test]
    async fn omits_the_budget_when_neither_field_nor_alias_is_set() {
        let model = vllm_model(json!({ "thinkingFormat": "zai" }));
        let params = capture_budget(
            &model,
            Some(ThinkingLevel::Medium),
            Some(json!({ "medium": 4096 })),
            None,
        )
        .await;
        assert!(params.get("thinking_token_budget").is_none());
        assert!(params.get("thinking_budget").is_none());
        assert!(params.get("thinking_budget_tokens").is_none());
    }

    #[tokio::test]
    async fn omits_the_budget_when_thinking_is_off() {
        let params =
            capture_budget(&default_vllm(), None, Some(json!({ "high": 8192 })), None).await;
        assert!(params.get("thinking_token_budget").is_none());
    }

    #[tokio::test]
    async fn clamps_xhigh_and_max_to_the_high_budget() {
        let xhigh = capture_budget(
            &default_vllm(),
            Some(ThinkingLevel::Xhigh),
            Some(json!({ "high": 8192 })),
            None,
        )
        .await;
        let max = capture_budget(
            &default_vllm(),
            Some(ThinkingLevel::Max),
            Some(json!({ "high": 8192 })),
            None,
        )
        .await;
        assert_eq!(xhigh["thinking_token_budget"], 8192);
        assert_eq!(max["thinking_token_budget"], 8192);
    }

    #[tokio::test]
    async fn leaves_room_for_the_answer_at_the_response_ceiling() {
        let params = capture_budget(&default_vllm(), Some(ThinkingLevel::High), None, None).await;
        assert_eq!(params["thinking_token_budget"], 16384 - 1024);
    }

    #[tokio::test]
    async fn uses_the_caller_max_tokens_as_the_ceiling() {
        let params = capture_budget(
            &default_vllm(),
            Some(ThinkingLevel::High),
            Some(json!({ "high": 8192 })),
            Some(4096),
        )
        .await;
        assert_eq!(params["thinking_token_budget"], 4096 - 1024);
    }

    #[tokio::test]
    async fn sends_the_configured_thinking_token_budget_field() {
        for field in ["thinking_budget", "thinking_budget_tokens"] {
            let model =
                vllm_model(json!({ "thinkingFormat": "qwen", "thinkingTokenBudgetField": field }));
            let params = capture_budget(
                &model,
                Some(ThinkingLevel::Medium),
                Some(json!({ "medium": 4096 })),
                None,
            )
            .await;
            assert_eq!(params[field], 4096, "{field}");
            assert!(params.get("thinking_token_budget").is_none());
        }
    }

    #[tokio::test]
    async fn thinking_token_budget_field_wins_over_the_boolean_alias() {
        let model = vllm_model(json!({
            "thinkingFormat": "zai", "supportsThinkingTokenBudget": true, "thinkingTokenBudgetField": "thinking_budget",
        }));
        let params = capture_budget(
            &model,
            Some(ThinkingLevel::Medium),
            Some(json!({ "medium": 4096 })),
            None,
        )
        .await;
        assert_eq!(params["thinking_budget"], 4096);
        assert!(params.get("thinking_token_budget").is_none());
    }

    fn chat_template_budget_model() -> Model {
        vllm_model(json!({
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
                "enable_thinking": { "$var": "thinking.enabled" },
                "thinking_budget": { "$var": "thinking.budget" },
            },
        }))
    }

    #[tokio::test]
    async fn puts_the_clamped_budget_in_chat_template_kwargs() {
        let params = capture_budget(
            &chat_template_budget_model(),
            Some(ThinkingLevel::High),
            None,
            None,
        )
        .await;
        assert_eq!(
            params["chat_template_kwargs"],
            json!({ "enable_thinking": true, "thinking_budget": 16384 - 1024 })
        );
        assert!(params.get("thinking_token_budget").is_none());
    }

    #[tokio::test]
    async fn omits_thinking_budget_from_chat_template_kwargs_when_off() {
        let params = capture_budget(&chat_template_budget_model(), None, None, None).await;
        assert_eq!(
            params["chat_template_kwargs"],
            json!({ "enable_thinking": false })
        );
    }

    // openai-completions-vllm-priority.test.ts
    #[tokio::test]
    async fn sends_compat_vllm_priority_as_top_level_priority() {
        let mut model = openai_as_completions("gpt-4o-mini");
        model.compat = Some(serde_json::from_value(json!({ "vllmPriority": 10 })).unwrap());
        let payload = payload_for(&model, sys_hi(), with_key("test-key")).await;
        assert_eq!(payload["priority"], 10);
    }

    #[tokio::test]
    async fn omits_priority_when_vllm_priority_is_not_set() {
        let payload = payload_for(
            &openai_as_completions("gpt-4o-mini"),
            sys_hi(),
            with_key("test-key"),
        )
        .await;
        assert!(payload.get("priority").is_none());
    }

    // Pi catalog entries (packages/ai/src/providers/data/*.json, v1.0.2) for
    // providers this crate does not ship yet.
    const OPENROUTER_AUTO: &str = r#"{"id":"auto","name":"Auto","api":"openai-completions","provider":"openrouter","baseUrl":"https://openrouter.ai/api/v1","reasoning":true,"input":["text","image"],"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"contextWindow":2000000,"maxTokens":30000,"compat":{"supportsDeveloperRole":false,"thinkingFormat":"openrouter","supportsStrictMode":true,"sendSessionAffinityHeaders":true}}"#;
    const OPENROUTER_FABLE_BATCH: &str = r#"{"id":"anthropic/claude-fable-5.1:batch","name":"Anthropic: Claude Fable 5.1 (batch)","api":"openai-completions","baseUrl":"https://openrouter.ai/api/v1","provider":"openrouter","reasoning":true,"thinkingLevelMap":{"off":null,"minimal":null,"low":"low","medium":"medium","high":"high","xhigh":"xhigh","max":"max"},"input":["text","image"],"cost":{"input":5,"output":25,"cacheRead":0.125,"cacheWrite":6.25},"contextWindow":1000000,"maxTokens":128000,"compat":{"thinkingFormat":"openrouter","supportsStrictMode":true,"cacheControlFormat":"anthropic","sendSessionAffinityHeaders":true}}"#;
    const OPENROUTER_DEEPSEEK_V4_PRO: &str = r#"{"id":"deepseek/deepseek-v4-pro","name":"DeepSeek: DeepSeek V4 Pro 0423","api":"openai-completions","baseUrl":"https://openrouter.ai/api/v1","provider":"openrouter","reasoning":true,"thinkingLevelMap":{"off":"none","minimal":null,"low":null,"medium":null,"high":"high","xhigh":"xhigh","max":null},"input":["text"],"cost":{"input":0.2088,"output":0.4176,"cacheRead":0.0174,"cacheWrite":0},"contextWindow":1024000,"maxTokens":384000,"compat":{"supportsDeveloperRole":false,"thinkingFormat":"openrouter","supportsStrictMode":true,"sendSessionAffinityHeaders":true,"requiresReasoningContentOnAssistantMessages":true}}"#;
    const OPENROUTER_GPT_52_CODEX: &str = r#"{"id":"openai/gpt-5.2-codex","name":"OpenAI: GPT-5.2-Codex","api":"openai-completions","baseUrl":"https://openrouter.ai/api/v1","provider":"openrouter","reasoning":true,"thinkingLevelMap":{"off":null,"minimal":null,"low":"low","medium":"medium","high":"high","xhigh":"xhigh","max":null},"input":["text","image"],"cost":{"input":1.75,"output":14,"cacheRead":0.175,"cacheWrite":0},"contextWindow":400000,"maxTokens":128000,"compat":{"thinkingFormat":"openrouter","supportsStrictMode":true,"sendSessionAffinityHeaders":true}}"#;
    const OPENROUTER_DEEPSEEK_R1: &str = r#"{"id":"deepseek/deepseek-r1","name":"DeepSeek: R1","api":"openai-completions","baseUrl":"https://openrouter.ai/api/v1","provider":"openrouter","reasoning":true,"thinkingLevelMap":{"off":null},"input":["text"],"cost":{"input":0.7,"output":2.5,"cacheRead":0,"cacheWrite":0},"contextWindow":64000,"maxTokens":16000,"compat":{"supportsDeveloperRole":false,"thinkingFormat":"openrouter","supportsStrictMode":true,"sendSessionAffinityHeaders":true}}"#;
    const XIAOMI_MIMO: &str = r#"{"id":"mimo-v2.5-pro","name":"MiMo-V2.5-Pro","api":"openai-completions","provider":"xiaomi","baseUrl":"https://api.xiaomimimo.com/v1","compat":{"supportsStrictMode":true,"requiresReasoningContentOnAssistantMessages":true,"thinkingFormat":"deepseek"},"reasoning":true,"input":["text"],"cost":{"input":0.435,"output":0.87,"cacheRead":0.0036,"cacheWrite":0},"contextWindow":1048576,"maxTokens":131072}"#;
    const OPENCODE_GO_KIMI_K3: &str = r#"{"id":"kimi-k3","name":"Kimi K3","api":"openai-completions","provider":"opencode-go","baseUrl":"https://opencode.ai/zen/go/v1","reasoning":true,"input":["text","image"],"cost":{"input":3,"output":15,"cacheRead":0.3,"cacheWrite":0},"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsStrictMode":true,"maxTokensField":"max_tokens","supportsMidConvoSystemMessages":true,"supportsMidConvoToolAdditions":true},"contextWindow":1048576,"maxTokens":131072,"thinkingLevelMap":{"off":null,"minimal":null,"low":null,"medium":null,"high":null,"xhigh":null,"max":"max"}}"#;
    const OPENCODE_KIMI_K26: &str = r#"{"id":"kimi-k2.6","name":"Kimi K2.6","api":"openai-completions","provider":"opencode","baseUrl":"https://opencode.ai/zen/v1","reasoning":true,"input":["text","image"],"cost":{"input":0.95,"output":4,"cacheRead":0.16,"cacheWrite":0},"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsStrictMode":true,"thinkingFormat":"deepseek","supportsReasoningEffort":false,"maxTokensField":"max_tokens","supportsLongCacheRetention":false},"contextWindow":262144,"maxTokens":65536}"#;
    const MOONSHOT_KIMI_K27_CODE: &str = r#"{"id":"kimi-k2.7-code","name":"Kimi K2.7 Code","api":"openai-completions","provider":"moonshotai","baseUrl":"https://api.moonshot.ai/v1","reasoning":true,"input":["text","image"],"cost":{"input":0.95,"output":4,"cacheRead":0.19,"cacheWrite":0},"contextWindow":262144,"maxTokens":262144,"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":false,"maxTokensField":"max_tokens","supportsStrictMode":false,"thinkingFormat":"deepseek","supportsMidConvoSystemMessages":true},"thinkingLevelMap":{"off":null}}"#;
    const MOONSHOT_CN_KIMI_K26: &str = r#"{"id":"kimi-k2.6","name":"Kimi K2.6","api":"openai-completions","provider":"moonshotai-cn","baseUrl":"https://api.moonshot.cn/v1","reasoning":true,"input":["text","image"],"cost":{"input":0.95,"output":4,"cacheRead":0.16,"cacheWrite":0},"contextWindow":262144,"maxTokens":262144,"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":false,"maxTokensField":"max_tokens","supportsStrictMode":false,"thinkingFormat":"deepseek","supportsMidConvoSystemMessages":true}}"#;
    const DEEPSEEK_FLASH: &str = r#"{"id":"deepseek-flash","name":"DeepSeek V4.1 Flash","api":"openai-completions","baseUrl":"https://api.deepseek.com","provider":"deepseek","reasoning":true,"thinkingLevelMap":{"minimal":null,"low":"low","medium":null,"high":"high","max":"max"},"input":["text","image"],"cost":{"input":0.3,"output":1.2,"cacheRead":0.006,"cacheWrite":0},"contextWindow":1000000,"maxTokens":384000,"compat":{"supportsStore":false,"supportsDeveloperRole":false,"maxTokensField":"max_tokens","requiresReasoningContentOnAssistantMessages":true,"thinkingFormat":"deepseek","supportsStrictMode":true}}"#;
    const DEEPSEEK_V4_PRO: &str = r#"{"id":"deepseek-v4-pro","name":"DeepSeek V4 Pro","api":"openai-completions","baseUrl":"https://api.deepseek.com","provider":"deepseek","reasoning":true,"input":["text"],"cost":{"input":1.32,"output":3.96,"cacheRead":0.044,"cacheWrite":0},"contextWindow":1000000,"maxTokens":384000,"compat":{"supportsStore":false,"supportsDeveloperRole":false,"maxTokensField":"max_tokens","requiresReasoningContentOnAssistantMessages":true,"thinkingFormat":"deepseek","supportsStrictMode":true,"supportsMidConvoSystemMessages":true},"thinkingLevelMap":{"minimal":null,"low":null,"medium":null,"high":"high","max":"max"}}"#;
    const ZAI_GLM_52: &str = r#"{"id":"glm-5.2","name":"GLM-5.2","api":"openai-completions","provider":"zai","baseUrl":"https://api.z.ai/api/coding/paas/v4","reasoning":true,"thinkingLevelMap":{"off":"none","minimal":null,"low":null,"medium":null,"high":"high","xhigh":null,"max":"max"},"input":["text"],"cost":{"input":1.4,"output":4.4,"cacheRead":0.26,"cacheWrite":0},"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":true,"maxTokensField":"max_tokens","thinkingFormat":"zai","supportsStrictMode":true,"zaiToolStream":true},"contextWindow":1000000,"maxTokens":131072}"#;
    const ZAI_GLM_5_TURBO: &str = r#"{"id":"glm-5-turbo","name":"GLM-5-Turbo","api":"openai-completions","provider":"zai","baseUrl":"https://api.z.ai/api/coding/paas/v4","reasoning":true,"input":["text"],"cost":{"input":1.2,"output":4,"cacheRead":0.24,"cacheWrite":0},"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":false,"maxTokensField":"max_tokens","thinkingFormat":"zai","supportsStrictMode":true,"zaiToolStream":true},"contextWindow":200000,"maxTokens":131072}"#;
    const ANT_LING_RING: &str = r#"{"id":"Ring-2.6-1T","name":"Ring 2.6 1T","api":"openai-completions","baseUrl":"https://api.ant-ling.com/v1","provider":"ant-ling","reasoning":true,"input":["text"],"cost":{"input":0.06,"output":0.25,"cacheRead":0,"cacheWrite":0},"contextWindow":262144,"maxTokens":65536,"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":false,"maxTokensField":"max_tokens","thinkingFormat":"ant-ling","supportsStrictMode":true,"supportsLongCacheRetention":false},"thinkingLevelMap":{"off":null,"minimal":null,"low":null,"medium":null,"high":"high","xhigh":"xhigh"}}"#;
    const ANT_LING_FLASH: &str = r#"{"id":"Ling-2.6-flash","name":"Ling 2.6 Flash","api":"openai-completions","baseUrl":"https://api.ant-ling.com/v1","provider":"ant-ling","reasoning":false,"input":["text"],"cost":{"input":0.01,"output":0.02,"cacheRead":0,"cacheWrite":0},"contextWindow":262144,"maxTokens":65536,"compat":{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":false,"maxTokensField":"max_tokens","thinkingFormat":"ant-ling","supportsStrictMode":true,"supportsLongCacheRetention":false}}"#;
    const GROQ_GPT_OSS_20B: &str = r#"{"id":"openai/gpt-oss-20b","name":"GPT OSS 20B","api":"openai-completions","provider":"groq","baseUrl":"https://api.groq.com/openai/v1","reasoning":true,"input":["text"],"cost":{"input":0.075,"output":0.3,"cacheRead":0.0375,"cacheWrite":0},"contextWindow":131072,"maxTokens":65536,"compat":{"supportsStrictMode":true},"thinkingLevelMap":{"off":null,"minimal":null,"low":"low","medium":"medium","high":"high","xhigh":null,"max":null}}"#;
    const GROQ_QWEN_36: &str = r#"{"id":"qwen/qwen3.6-27b","name":"Qwen3.6 27B","api":"openai-completions","provider":"groq","baseUrl":"https://api.groq.com/openai/v1","reasoning":true,"input":["text","image"],"cost":{"input":0.6,"output":3,"cacheRead":0.3,"cacheWrite":0},"contextWindow":131072,"maxTokens":16384,"compat":{"supportsStrictMode":true},"thinkingLevelMap":{"off":"none","minimal":null,"low":null,"medium":null,"high":"default","xhigh":null,"max":null}}"#;
    const FIREWORKS_GLM_5P3: &str = r#"{"id":"accounts/fireworks/models/glm-5p3","name":"GLM 5.3","provider":"fireworks","reasoning":true,"input":["text"],"cost":{"input":1.4,"output":4.4,"cacheRead":0.26,"cacheWrite":0},"contextWindow":1048573,"maxTokens":262144,"api":"openai-completions","baseUrl":"https://api.fireworks.ai/inference/v1","compat":{"supportsStrictMode":true,"supportsStore":false,"supportsDeveloperRole":false,"sendSessionAffinityHeaders":true,"supportsLongCacheRetention":false},"thinkingLevelMap":{"off":null,"minimal":null,"low":"low","medium":null,"high":"high","xhigh":null,"max":"max"}}"#;
    const BASETEN_GLM_52: &str = r#"{"id":"zai-org/GLM-5.2","name":"GLM 5.2","api":"openai-completions","provider":"baseten","baseUrl":"https://inference.baseten.co/v1","reasoning":true,"thinkingLevelMap":{"off":"none","minimal":null,"low":null,"medium":null,"high":"high","xhigh":null,"max":"max"},"input":["text"],"cost":{"input":1.4,"output":4.4,"cacheRead":0.3,"cacheWrite":0},"compat":{"supportsStrictMode":true,"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":true,"supportsUsageInStreaming":true,"maxTokensField":"max_tokens","sendSessionAffinityHeaders":true,"supportsLongCacheRetention":false,"thinkingFormat":"baseten","chatTemplateArgs":{"enable_thinking":{"$var":"thinking.enabled"}}},"contextWindow":1048576,"maxTokens":262144}"#;

    fn pi_model(literal: &str) -> Model {
        model(serde_json::from_str(literal).unwrap())
    }

    /// `{ ...getModel(..), compat: undefined, api: "openai-completions" }`.
    fn without_compat(literal: &str) -> Model {
        let mut model = pi_model(literal);
        model.compat = None;
        model
    }

    fn user(text: &str) -> Value {
        json!({ "role": "user", "content": text, "timestamp": 1 })
    }

    fn zero_usage() -> Value {
        json!({ "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 } })
    }

    fn assistant(provider: &str, model_id: &str, content: Value, stop_reason: &str) -> Value {
        json!({
            "role": "assistant", "api": "openai-completions", "provider": provider, "model": model_id,
            "content": content, "usage": zero_usage(), "stopReason": stop_reason, "timestamp": 1,
        })
    }

    fn tool_result(id: &str, name: &str, content: Value) -> Value {
        json!({ "role": "toolResult", "toolCallId": id, "toolName": name, "content": content,
            "isError": false, "timestamp": 1 })
    }

    fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
        json!({ "name": name, "description": description,
            "parameters": { "type": "object", "properties": properties, "required": required } })
    }

    fn ping_tool() -> Value {
        tool(
            "ping",
            "Ping tool",
            json!({ "ok": { "type": "boolean" } }),
            &["ok"],
        )
    }

    fn prefer_strict_ping_tool() -> Value {
        let mut tool = tool(
            "ping",
            "Ping tool",
            json!({ "required": { "type": "string" }, "optional": { "type": "string" } }),
            &["required"],
        );
        tool["constrainedSampling"] = json!({ "type": "json_schema", "strict": "prefer" });
        tool
    }

    fn msgs(messages: Value) -> TranscriptContext {
        context(json!({ "messages": messages }))
    }

    fn msgs_with_tools(messages: Value, tools: Value) -> TranscriptContext {
        context(json!({ "messages": messages, "tools": tools }))
    }

    async fn simple_reasoning_payload(model: &Model, reasoning: Option<ThinkingLevel>) -> Value {
        simple_payload(model, msgs(json!([user("Hi")])), simple(reasoning)).await
    }

    async fn simple_max_tokens_payload(model: &Model, max_tokens: u32) -> Value {
        let mut options = simple(None);
        options.max_tokens = Some(max_tokens);
        simple_payload(model, msgs(json!([user("Hi")])), options).await
    }

    fn event_types(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
        events
            .iter()
            .map(AssistantMessageEvent::event_type)
            .collect()
    }

    fn tool_event_index(event: &AssistantMessageEvent) -> Option<usize> {
        match event {
            AssistantMessageEvent::ToolCallStart { content_index, .. }
            | AssistantMessageEvent::ToolCallDelta { content_index, .. }
            | AssistantMessageEvent::ToolCallEnd { content_index, .. } => Some(*content_index),
            _ => None,
        }
    }

    // openai-completions-tool-choice.test.ts
    #[tokio::test]
    async fn forwards_tool_choice_to_payload() {
        let ctx = msgs_with_tools(
            json!([user("Call ping with ok=true")]),
            json!([ping_tool()]),
        );
        let mut options = with_key("test");
        options.tool_choice = Some(json!("required"));
        let params = payload_for(&openai_as_completions("gpt-4o-mini"), ctx, options).await;
        assert_eq!(params["tool_choice"], "required");
        assert!(!params["tools"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn includes_tool_choice_when_no_tools_are_provided() {
        let mut options = simple(None);
        options.tool_choice = Some(ToolChoice::None);
        let params = simple_payload(
            &openai_as_completions("gpt-4o-mini"),
            msgs(json!([user("Summarize the conversation")])),
            options,
        )
        .await;
        assert_eq!(params["tool_choice"], "none");
        assert!(params.get("tools").is_none());
    }

    #[tokio::test]
    async fn omits_strict_when_compat_disables_strict_mode() {
        let mut model = openai_as_completions("gpt-4o-mini");
        model.compat =
            Some(serde_json::from_value(json!({ "supportsStrictMode": false })).unwrap());
        let ctx = msgs_with_tools(
            json!([user("Call ping with ok=true")]),
            json!([ping_tool()]),
        );
        let params = simple_payload(&model, ctx, simple(None)).await;
        let function = &params["tools"][0]["function"];
        assert!(function.is_object());
        assert!(function.get("strict").is_none());
    }

    #[tokio::test]
    async fn defaults_unknown_openai_compatible_endpoints_to_non_strict_tools() {
        let model = local_model(json!({ "provider": "local-vllm", "maxTokens": 8192 }));
        let ctx = msgs_with_tools(
            json!([user("Call ping")]),
            json!([prefer_strict_ping_tool()]),
        );
        let params = simple_payload(&model, ctx, simple(None)).await;
        let function = &params["tools"][0]["function"];
        assert!(function.get("strict").is_none());
        assert_eq!(function["parameters"]["required"], json!(["required"]));
    }

    #[tokio::test]
    async fn preserves_strict_tools_for_capable_built_in_models() {
        let ctx = msgs_with_tools(
            json!([user("Call ping")]),
            json!([prefer_strict_ping_tool()]),
        );
        let params = simple_payload(&pi_model(GROQ_GPT_OSS_20B), ctx, simple(None)).await;
        let function = &params["tools"][0]["function"];
        assert_eq!(function["strict"], true);
        assert_eq!(
            function["parameters"]["required"],
            json!(["required", "optional"])
        );
    }

    #[tokio::test]
    async fn maps_groq_reasoning_levels_through_the_thinking_level_map() {
        let qwen =
            simple_reasoning_payload(&pi_model(GROQ_QWEN_36), Some(ThinkingLevel::Medium)).await;
        assert_eq!(qwen["reasoning_effort"], "default");
        let gpt_oss =
            simple_reasoning_payload(&pi_model(GROQ_GPT_OSS_20B), Some(ThinkingLevel::Medium))
                .await;
        assert_eq!(gpt_oss["reasoning_effort"], "medium");
    }

    #[tokio::test]
    async fn zai_tool_stream_follows_tools() {
        let ctx = msgs_with_tools(
            json!([user("Call ping with ok=true")]),
            json!([ping_tool()]),
        );
        let params = simple_payload(&pi_model(ZAI_GLM_52), ctx, simple(None)).await;
        assert_eq!(params["tool_stream"], true);
        let params = simple_reasoning_payload(&pi_model(ZAI_GLM_52), None).await;
        assert!(params.get("tool_stream").is_none());
    }

    #[tokio::test]
    async fn maps_zai_glm_52_thinking_levels_to_reasoning_effort() {
        for (reasoning, effort) in [
            (ThinkingLevel::Low, "high"),
            (ThinkingLevel::Medium, "high"),
            (ThinkingLevel::High, "high"),
            (ThinkingLevel::Max, "max"),
        ] {
            let params = simple_reasoning_payload(&pi_model(ZAI_GLM_52), Some(reasoning)).await;
            assert_eq!(
                params["thinking"],
                json!({ "type": "enabled", "clear_thinking": false })
            );
            assert_eq!(params["reasoning_effort"], effort, "{reasoning:?}");
        }
    }

    #[tokio::test]
    async fn preserves_zai_thinking_when_replaying_reasoning_content() {
        let ctx = msgs(json!([
            user("Read README.md"),
            assistant(
                "zai",
                "glm-5.2",
                json!([
                    { "type": "thinking", "thinking": "prior reasoning", "thinkingSignature": "reasoning_content" },
                    { "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "README.md" } },
                ]),
                "toolUse"
            ),
            tool_result(
                "call_1",
                "read",
                json!([{ "type": "text", "text": "contents" }])
            ),
            user("Continue"),
        ]));
        let params = simple_payload(
            &pi_model(ZAI_GLM_52),
            ctx,
            simple(Some(ThinkingLevel::High)),
        )
        .await;
        assert_eq!(
            assistant_payload(&params)["reasoning_content"],
            "prior reasoning"
        );
        assert_eq!(
            params["thinking"],
            json!({ "type": "enabled", "clear_thinking": false })
        );
    }

    #[tokio::test]
    async fn omits_zai_reasoning_effort_when_thinking_is_off() {
        let params = simple_reasoning_payload(&pi_model(ZAI_GLM_52), None).await;
        assert_eq!(params["thinking"], json!({ "type": "disabled" }));
        assert!(params.get("reasoning_effort").is_none());
    }

    #[tokio::test]
    async fn maps_non_standard_finish_reasons_to_error() {
        let chunks = [
            json!({ "choices": [{ "delta": { "content": "partial" }, "finish_reason": null }] }),
            json!({ "choices": [{ "delta": {}, "finish_reason": "network_error" }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 }, "completion_tokens_details": { "reasoning_tokens": 0 } } }),
        ];
        let (_, response) = run_simple_chunks(
            pi_model(ZAI_GLM_52),
            msgs(json!([user("Hi")])),
            simple(None),
            &chunks,
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(
            response.error_message.as_deref(),
            Some("Provider finish_reason: network_error")
        );
    }

    #[tokio::test]
    async fn ignores_null_stream_chunks() {
        let chunks = [
            Value::Null,
            json!({ "id": "chatcmpl-test", "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-test", "choices": [{ "delta": {}, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 3, "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 }, "completion_tokens_details": { "reasoning_tokens": 0 } } }),
        ];
        let (_, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            msgs(json!([user("Reply with exactly OK")])),
            simple(None),
            &chunks,
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Stop);
        assert_eq!(response.error_message, None);
        assert_eq!(response.response_id.as_deref(), Some("chatcmpl-test"));
        assert_eq!(response.usage.total_tokens, 4);
        assert_eq!(
            content_json(&response),
            json!([{ "type": "text", "text": "OK" }])
        );
    }

    #[tokio::test]
    async fn errors_when_a_stream_ends_without_finish_reason() {
        let chunk = json!({ "id": "chatcmpl-truncated", "choices": [{ "delta": { "content": "partial answer" }, "finish_reason": null }] });
        let (_, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            msgs(json!([user("Reply with a longer sentence")])),
            simple(None),
            &[chunk.clone(), chunk],
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(
            response.error_message.as_deref(),
            Some("Stream ended without finish_reason")
        );
    }

    #[tokio::test]
    async fn accepts_streams_without_finish_reason_when_compat_disables_it() {
        let mut model = openai_as_completions("gpt-4o-mini");
        model.compat =
            Some(serde_json::from_value(json!({ "supportsFinishReason": false })).unwrap());
        let chunk = json!({ "id": "chatcmpl-no-finish-reason", "choices": [{ "delta": { "content": "complete answer" }, "finish_reason": null }] });
        let (_, response) = run_simple_chunks(
            model,
            msgs(json!([user("Reply with a complete answer")])),
            simple(None),
            &[chunk],
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Stop);
        assert_eq!(response.error_message, None);
        assert_eq!(
            content_json(&response),
            json!([{ "type": "text", "text": "complete answer" }])
        );
    }

    #[tokio::test]
    async fn ignores_empty_custom_objects_on_function_tool_call_deltas() {
        let chunk = json!({ "id": "chatcmpl-empty-custom", "choices": [{ "delta": { "tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": { "name": "read", "arguments": "{\"path\":\"README.md\"}" }, "custom": {},
        }] }, "finish_reason": "tool_calls" }] });
        let ctx = msgs_with_tools(json!([user("Read README.md")]), json!([read_tool()]));
        let (_, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            ctx,
            simple(None),
            &[chunk],
        )
        .await;
        assert_eq!(
            content_json(&response),
            json!([{ "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "README.md" } }])
        );
    }

    #[tokio::test]
    async fn coalesces_tool_call_deltas_by_stable_index_when_ids_mutate() {
        let call = |id: &str, name: Value, arguments: &str| json!({ "index": 0, "id": id, "type": "function", "function": { "name": name, "arguments": arguments } });
        let chunks = [
            json!({ "id": "chatcmpl-kimi-bad-stream", "choices": [{ "delta": { "tool_calls": [call("functions.read:0", json!("read"), "")] }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-kimi-bad-stream", "choices": [{ "delta": { "tool_calls": [call("chatcmpl-tool-a", Value::Null, "{\"path\":\"README")] }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-kimi-bad-stream", "choices": [{ "delta": { "tool_calls": [call("chatcmpl-tool-b", Value::Null, ".md\"}")] }, "finish_reason": "tool_calls" }],
                "usage": { "prompt_tokens": 10, "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 0 }, "completion_tokens_details": { "reasoning_tokens": 0 } } }),
        ];
        let ctx = msgs_with_tools(json!([user("Read README.md")]), json!([read_tool()]));
        let (events, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            ctx,
            simple(None),
            &chunks,
        )
        .await;
        let indexes: Vec<usize> = events.iter().filter_map(tool_event_index).collect();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(indexes, vec![0, 0, 0, 0, 0]);
        assert_eq!(
            content_json(&response),
            json!([{ "type": "toolCall", "id": "functions.read:0", "name": "read", "arguments": { "path": "README.md" } }])
        );
    }

    #[tokio::test]
    async fn accumulates_mixed_content_reasoning_and_parallel_tool_call_deltas() {
        let call = |index: Option<u64>, id: Option<&str>, name: Option<&str>, arguments: &str| {
            let mut call = json!({ "type": "function", "function": { "arguments": arguments } });
            if let Some(index) = index {
                call["index"] = json!(index);
            }
            if let Some(id) = id {
                call["id"] = json!(id);
            }
            if let Some(name) = name {
                call["function"]["name"] = json!(name);
            }
            call
        };
        let chunks = [
            json!({ "id": "chatcmpl-mixed-deltas", "choices": [{ "delta": {
                "content": "answer 1", "reasoning_content": "think 1",
                "tool_calls": [
                    call(Some(0), Some("tc_read_initial"), Some("read"), "{\"path\":\"README"),
                    call(Some(1), Some("tc_grep_initial"), Some("grep"), "{\"pattern\":\"TODO"),
                    call(None, Some("tc_list_no_index"), Some("list"), "{\"path\":\"packages"),
                    call(None, Some("tc_write_no_index"), Some("write"), "{\"path\":\"out"),
                ],
            }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-mixed-deltas", "choices": [{ "delta": {
                "content": " answer 2",
                "tool_calls": [
                    call(Some(1), Some("tc_grep_changed"), None, "\",\"path\":\"src"),
                    call(None, Some("tc_write_no_index"), None, ".txt\",\"content\":\"ok\"}"),
                    call(None, Some("tc_list_no_index"), None, "/ai\"}"),
                ],
            }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-mixed-deltas", "choices": [{ "delta": {
                "content": "\n", "reasoning_content": " think 2",
                "tool_calls": [
                    call(Some(0), Some("tc_read_changed"), None, ".md\"}"),
                    call(Some(1), None, None, "\"}"),
                ],
            }, "finish_reason": "tool_calls" }],
                "usage": { "prompt_tokens": 10, "completion_tokens": 8,
                    "prompt_tokens_details": { "cached_tokens": 0 }, "completion_tokens_details": { "reasoning_tokens": 2 } } }),
        ];
        let tools = json!([
            read_tool(),
            tool(
                "grep",
                "Search a file",
                json!({ "pattern": { "type": "string" }, "path": { "type": "string" } }),
                &["pattern", "path"]
            ),
            tool(
                "list",
                "List a directory",
                json!({ "path": { "type": "string" } }),
                &["path"]
            ),
            tool(
                "write",
                "Write a file",
                json!({ "path": { "type": "string" }, "content": { "type": "string" } }),
                &["path", "content"]
            ),
        ]);
        let ctx = msgs_with_tools(json!([user("Think, answer, and use tools.")]), tools);
        let (events, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            ctx,
            simple(None),
            &chunks,
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        let types = event_types(&events);
        let count = |kind: &str| types.iter().filter(|t| **t == kind).count();
        assert_eq!(
            [count("text_start"), count("text_delta"), count("text_end")],
            [1, 3, 1]
        );
        assert_eq!(
            [
                count("thinking_start"),
                count("thinking_delta"),
                count("thinking_end")
            ],
            [1, 2, 1]
        );
        assert_eq!(
            [
                count("toolcall_start"),
                count("toolcall_delta"),
                count("toolcall_end")
            ],
            [4, 9, 4]
        );
        let per_index = |index: usize| -> Vec<&'static str> {
            events
                .iter()
                .filter(|event| tool_event_index(event) == Some(index))
                .map(AssistantMessageEvent::event_type)
                .collect()
        };
        let two = vec![
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
        ];
        assert_eq!(per_index(2), two);
        assert_eq!(
            per_index(3),
            vec![
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end"
            ]
        );
        assert_eq!(per_index(4), two);
        assert_eq!(per_index(5), two);
        assert_eq!(
            content_json(&response),
            json!([
                { "type": "text", "text": "answer 1 answer 2\n" },
                { "type": "thinking", "thinking": "think 1 think 2", "thinkingSignature": "reasoning_content" },
                { "type": "toolCall", "id": "tc_read_initial", "name": "read", "arguments": { "path": "README.md" } },
                { "type": "toolCall", "id": "tc_grep_initial", "name": "grep", "arguments": { "pattern": "TODO", "path": "src" } },
                { "type": "toolCall", "id": "tc_list_no_index", "name": "list", "arguments": { "path": "packages/ai" } },
                { "type": "toolCall", "id": "tc_write_no_index", "name": "write", "arguments": { "path": "out.txt", "content": "ok" } },
            ])
        );
    }

    #[tokio::test]
    async fn chooses_system_or_developer_role_for_instructions() {
        let ctx = || context(json!({ "systemPrompt": "Follow instructions.", "messages": [] }));
        let role = |params: Value| params["messages"][0]["role"].as_str().unwrap().to_string();
        let params =
            simple_payload(&pi_model(OPENROUTER_DEEPSEEK_V4_PRO), ctx(), simple(None)).await;
        assert_eq!(role(params), "system");
        for literal in [OPENROUTER_GPT_52_CODEX, OPENROUTER_FABLE_BATCH] {
            let params = simple_payload(&pi_model(literal), ctx(), simple(None)).await;
            assert_eq!(role(params), "developer");
        }
        let params = simple_payload(&openai_as_completions("gpt-5.5"), ctx(), simple(None)).await;
        assert_eq!(role(params), "developer");
    }

    #[tokio::test]
    async fn replays_xiaomi_tool_calls_with_empty_reasoning_content() {
        let ctx = msgs(json!([
            assistant(
                "xiaomi",
                "mimo-v2.5-pro",
                json!([
                    { "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "README.md" } },
                ]),
                "toolUse"
            ),
            tool_result(
                "call_1",
                "read",
                json!([{ "type": "text", "text": "contents" }])
            ),
        ]));
        let params = simple_payload(
            &pi_model(XIAOMI_MIMO),
            ctx,
            simple(Some(ThinkingLevel::High)),
        )
        .await;
        let replayed = assistant_payload(&params);
        assert_eq!(replayed["role"], "assistant");
        assert_eq!(replayed["reasoning_content"], "");
        assert_eq!(params["thinking"], json!({ "type": "enabled" }));
        assert_eq!(params["reasoning_effort"], "high");
    }

    #[tokio::test]
    async fn normalizes_opencode_go_reasoning_deltas_to_reasoning_content() {
        let chunk = |id: &str| json!({ "id": id, "choices": [{ "delta": { "reasoning": "think" }, "finish_reason": "stop" }] });
        let (_, response) = run_simple_chunks(
            without_compat(OPENCODE_GO_KIMI_K3),
            msgs(json!([user("Hi")])),
            simple(None),
            &[chunk("chatcmpl-opencode-go-reasoning")],
        )
        .await;
        assert_eq!(
            content_json(&response),
            json!([{ "type": "thinking", "thinking": "think", "thinkingSignature": "reasoning_content" }])
        );
        let (_, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            msgs(json!([user("Hi")])),
            simple(None),
            &[chunk("chatcmpl-reasoning")],
        )
        .await;
        assert_eq!(
            content_json(&response),
            json!([{ "type": "thinking", "thinking": "think", "thinkingSignature": "reasoning" }])
        );
    }

    #[test]
    fn replays_opencode_go_reasoning_thinking_blocks_as_reasoning_content() {
        let mut model = without_compat(OPENCODE_GO_KIMI_K3);
        let ctx = msgs(json!([assistant(
            "opencode-go",
            "kimi-k3",
            json!([
                { "type": "thinking", "thinking": "think", "thinkingSignature": "reasoning" },
                { "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "README.md" } },
            ]),
            "stop"
        )]));
        model.compat = Some(
            serde_json::from_value(json!({
                "supportsStore": false, "supportsDeveloperRole": false, "supportsReasoningEffort": true,
                "supportsUsageInStreaming": true, "supportsFinishReason": true,
                "maxTokensField": "max_completion_tokens", "requiresToolResultName": false,
                "requiresAssistantAfterToolResult": false, "requiresThinkingAsText": false,
                "requiresReasoningContentOnAssistantMessages": false, "thinkingFormat": "openai",
                "openRouterRouting": {}, "vercelGatewayRouting": {}, "chatTemplateKwargs": {}, "chatTemplateArgs": {},
                "zaiToolStream": false, "supportsStrictMode": true, "supportsOpenAIGrammarTools": false,
                "sendSessionAffinityHeaders": false, "sessionAffinityFormat": "openai",
                "supportsLongCacheRetention": true,
            }))
            .unwrap(),
        );
        let messages = convert_messages(&model, &ctx, &get_compat(&model), None).unwrap();
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], "think");
        assert!(messages[0].get("reasoning").is_none());
    }

    #[tokio::test]
    async fn kimi_thinking_toggles() {
        let off = simple_reasoning_payload(&pi_model(OPENCODE_KIMI_K26), None).await;
        assert_eq!(off["thinking"], json!({ "type": "disabled" }));
        assert!(off.get("reasoning_effort").is_none());
        let on =
            simple_reasoning_payload(&pi_model(OPENCODE_KIMI_K26), Some(ThinkingLevel::High)).await;
        assert_eq!(on["thinking"], json!({ "type": "enabled" }));
        assert!(on.get("reasoning_effort").is_none());
        let mut code_cn = pi_model(MOONSHOT_KIMI_K27_CODE);
        code_cn.provider = "moonshotai-cn".into();
        code_cn.base_url = "https://api.moonshot.cn/v1".into();
        for model in [pi_model(MOONSHOT_KIMI_K27_CODE), code_cn] {
            let params = simple_reasoning_payload(&model, None).await;
            assert!(params.get("thinking").is_none(), "{}", model.provider);
            assert!(params.get("reasoning_effort").is_none());
        }
        let k26 = simple_reasoning_payload(&pi_model(MOONSHOT_CN_KIMI_K26), None).await;
        assert_eq!(k26["thinking"], json!({ "type": "disabled" }));
        assert!(k26.get("reasoning_effort").is_none());
    }

    #[tokio::test]
    async fn sends_max_tokens_for_max_tokens_field_models() {
        let custom = completions_model(json!({
            "id": "custom-deepseek-model", "name": "Custom DeepSeek Model", "provider": "custom-deepseek",
            "baseUrl": "https://api.deepseek.com", "reasoning": true, "maxTokens": 8192,
        }));
        let mut custom_upper = custom.clone();
        custom_upper.id = "custom-uppercase-deepseek-model".into();
        custom_upper.base_url = "https://API.DeepSeek.COM".into();
        let models = [
            pi_model(OPENCODE_GO_KIMI_K3),
            pi_model(OPENCODE_KIMI_K26),
            pi_model(DEEPSEEK_FLASH),
            pi_model(DEEPSEEK_V4_PRO),
            custom,
            custom_upper,
            pi_model(ZAI_GLM_5_TURBO),
            pi_model(ZAI_GLM_52),
        ];
        for model in models {
            let params = simple_max_tokens_payload(&model, 123).await;
            assert_eq!(params["max_tokens"], 123, "{}", model.id);
            assert!(
                params.get("max_completion_tokens").is_none(),
                "{}",
                model.id
            );
        }
    }

    #[tokio::test]
    async fn does_not_double_count_reasoning_tokens() {
        let chunk = json!({ "id": "chatcmpl-reasoning-usage", "choices": [{ "delta": {}, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 33,
                "prompt_tokens_details": { "cached_tokens": 0 }, "completion_tokens_details": { "reasoning_tokens": 21 } } });
        let (_, response) = run_simple_chunks(
            openai_as_completions("gpt-4o-mini"),
            msgs(json!([user("Use reasoning.")])),
            simple(None),
            &[chunk],
        )
        .await;
        assert_eq!(response.usage.input, 10);
        assert_eq!(response.usage.output, 33);
        assert_eq!(response.usage.total_tokens, 43);
    }

    #[tokio::test]
    async fn preserves_prompt_tokens_details_cache_fields() {
        let usage = json!({ "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 30 },
            "completion_tokens_details": { "reasoning_tokens": 0 } });
        let chunk_usage = [
            json!({ "id": "chatcmpl-cache-write", "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-cache-write", "choices": [{ "delta": {}, "finish_reason": "stop" }], "usage": usage }),
        ];
        let choice_usage = [
            json!({ "id": "chatcmpl-cache-write-choice", "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }] }),
            json!({ "id": "chatcmpl-cache-write-choice", "choices": [{ "delta": {}, "finish_reason": "stop", "usage": usage }] }),
        ];
        for chunks in [chunk_usage, choice_usage] {
            let (_, response) = run_simple_chunks(
                openai_as_completions("gpt-4o-mini"),
                msgs(json!([user("Reply with exactly OK")])),
                simple(None),
                &chunks,
            )
            .await;
            assert_eq!(response.usage.input, 20);
            assert_eq!(response.usage.cache_read, 50);
            assert_eq!(response.usage.cache_write, 30);
            assert_eq!(response.usage.total_tokens, 105);
        }
    }

    #[tokio::test]
    async fn uses_openrouter_reasoning_object_instead_of_reasoning_effort() {
        let params =
            simple_reasoning_payload(&pi_model(OPENROUTER_DEEPSEEK_R1), Some(ThinkingLevel::High))
                .await;
        assert_eq!(params["reasoning"], json!({ "effort": "high" }));
        assert!(params.get("reasoning_effort").is_none());
    }

    #[tokio::test]
    async fn uses_configurable_chat_template_boolean_thinking_kwargs() {
        let model = local_model(json!({
            "id": "deepseek-ai/DeepSeek-V3.1", "name": "DeepSeek V3.1 via vLLM", "provider": "local-vllm", "maxTokens": 8192,
            "compat": { "thinkingFormat": "chat-template", "supportsReasoningEffort": false,
                "chatTemplateKwargs": { "thinking": { "$var": "thinking.enabled" } } },
        }));
        for (reasoning, expected) in [(Some(ThinkingLevel::High), true), (None, false)] {
            let params = simple_reasoning_payload(&model, reasoning).await;
            assert_eq!(
                params["chat_template_kwargs"],
                json!({ "thinking": expected })
            );
            assert!(params.get("thinking").is_none());
            assert!(params.get("reasoning_effort").is_none());
        }
    }

    #[tokio::test]
    async fn uses_qwen_chat_template_thinking_kwargs() {
        let model = local_model(json!({
            "id": "Qwen/Qwen3-Coder", "name": "Qwen3 Coder via vLLM", "provider": "local-vllm", "maxTokens": 8192,
            "compat": { "thinkingFormat": "qwen-chat-template", "supportsReasoningEffort": false },
        }));
        for (reasoning, expected) in [(Some(ThinkingLevel::High), true), (None, false)] {
            let params = simple_reasoning_payload(&model, reasoning).await;
            assert_eq!(
                params["chat_template_kwargs"],
                json!({ "enable_thinking": expected, "preserve_thinking": true })
            );
            assert!(params.get("reasoning_effort").is_none());
        }
    }

    #[tokio::test]
    async fn uses_configurable_chat_template_effort_kwargs_with_static_kwargs() {
        let model = local_model(json!({
            "id": "unsloth/gpt-oss-120b-GGUF", "name": "GPT OSS via vLLM", "provider": "local-vllm", "maxTokens": 8192,
            "thinkingLevelMap": { "xhigh": "max" },
            "compat": { "thinkingFormat": "chat-template", "supportsReasoningEffort": false,
                "chatTemplateKwargs": { "preserve_thinking": true,
                    "reasoning_effort": { "$var": "thinking.effort", "omitWhenOff": true } } },
        }));
        let params = simple_reasoning_payload(&model, Some(ThinkingLevel::Xhigh)).await;
        assert_eq!(
            params["chat_template_kwargs"],
            json!({ "preserve_thinking": true, "reasoning_effort": "max" })
        );
        assert!(params.get("reasoning_effort").is_none());
    }

    #[tokio::test]
    async fn uses_ant_ling_compatibility_metadata() {
        let mut options = simple(Some(ThinkingLevel::High));
        options.max_tokens = Some(123);
        options.cache_retention = Some(CacheRetention::Long);
        options.session_id = Some("ant-ling-session".to_string());
        let ctx = context(json!({ "systemPrompt": "Follow instructions.", "messages": [] }));
        let params = simple_payload(&pi_model(ANT_LING_RING), ctx, options).await;
        assert_eq!(params["max_tokens"], 123);
        assert!(params.get("max_completion_tokens").is_none());
        assert_eq!(params["messages"][0]["role"], "system");
        assert_eq!(params["reasoning"], json!({ "effort": "high" }));
        for absent in [
            "reasoning_effort",
            "store",
            "prompt_cache_key",
            "prompt_cache_retention",
        ] {
            assert!(params.get(absent).is_none(), "{absent}");
        }
    }

    #[tokio::test]
    async fn omits_ant_ling_reasoning_for_unmapped_efforts_and_non_reasoning_models() {
        let mut options = with_key("test");
        options.reasoning_effort = Some(ThinkingLevel::Medium);
        let params =
            payload_for(&pi_model(ANT_LING_RING), msgs(json!([user("Hi")])), options).await;
        assert!(params.get("reasoning").is_none());
        let params =
            simple_reasoning_payload(&pi_model(ANT_LING_FLASH), Some(ThinkingLevel::High)).await;
        assert!(params.get("reasoning").is_none());
    }

    // openai-completions-empty-tools.test.ts
    #[tokio::test]
    async fn omits_tools_when_context_tools_are_empty_or_missing() {
        let model = openai_as_completions("gpt-4o-mini");
        let params = simple_payload(
            &model,
            msgs_with_tools(json!([user("hi")]), json!([])),
            simple(None),
        )
        .await;
        assert!(params.get("tools").is_none());
        let params = simple_payload(&model, msgs(json!([user("hi")])), simple(None)).await;
        assert!(params.get("tools").is_none());
    }

    #[tokio::test]
    async fn sends_default_and_explicit_max_tokens() {
        let model = openai_as_completions("gpt-4o-mini");
        let params = simple_payload(&model, msgs(json!([user("hi")])), simple(None)).await;
        assert!(params.get("max_tokens").is_none());
        assert_eq!(params["max_completion_tokens"], model.max_tokens);
        let params = simple_max_tokens_payload(&model, 1234).await;
        assert!(params.get("max_tokens").is_none());
        assert_eq!(params["max_completion_tokens"], 1234);
    }

    #[tokio::test]
    async fn clamps_max_tokens_to_remaining_context() {
        let mut model = openai_as_completions("gpt-4o-mini");
        model.context_window = 10000;
        model.max_tokens = 8000;
        let ctx = || msgs(json!([user(&"x".repeat(8000))]));
        let params = simple_payload(&model, ctx(), simple(None)).await;
        assert!(params.get("max_tokens").is_none());
        assert_eq!(params["max_completion_tokens"], 3904);
        let mut options = simple(None);
        options.max_tokens = Some(7000);
        let params = simple_payload(&model, ctx(), options).await;
        assert_eq!(params["max_completion_tokens"], 3904);
    }

    #[tokio::test]
    async fn still_emits_empty_tools_when_conversation_has_tool_history() {
        let ctx = msgs_with_tools(
            json!([
                user("use the tool"),
                assistant(
                    "openai",
                    "gpt-4o-mini",
                    json!([{ "type": "toolCall", "id": "t1", "name": "noop", "arguments": {} }]),
                    "toolUse"
                ),
                tool_result("t1", "noop", json!([{ "type": "text", "text": "done" }])),
            ]),
            json!([]),
        );
        let params = simple_payload(&openai_as_completions("gpt-4o-mini"), ctx, simple(None)).await;
        assert_eq!(params["tools"], json!([]));
    }

    // openai-completions-tool-result-images.test.ts
    fn image_model() -> Model {
        let mut model = openai_as_completions("gpt-4o-mini");
        model.input = vec![ModelInput::Text, ModelInput::Image];
        model
    }

    fn image_tool_result(id: &str) -> Value {
        tool_result(
            id,
            "read",
            json!([
                { "type": "text", "text": "Read image file [image/png]" },
                { "type": "image", "data": "ZmFrZQ==", "mimeType": "image/png" },
            ]),
        )
    }

    #[test]
    fn omits_empty_text_parts_from_user_messages_with_images() {
        let model = image_model();
        let ctx = msgs(json!([{ "role": "user", "timestamp": 1, "content": [
            { "type": "text", "text": "" },
            { "type": "image", "data": "ZmFrZQ==", "mimeType": "image/png" },
        ] }]));
        assert_eq!(
            Value::Array(convert_messages(&model, &ctx, &get_compat(&model), None).unwrap()),
            json!([{ "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "data:image/png;base64,ZmFrZQ==" } }] }])
        );
    }

    #[test]
    fn batches_tool_result_images_after_consecutive_tool_results() {
        let model = image_model();
        let ctx = msgs(json!([
            user("Read the images"),
            assistant(
                "openai",
                "gpt-4o-mini",
                json!([
                    { "type": "toolCall", "id": "tool-1", "name": "read", "arguments": { "path": "img-1.png" } },
                    { "type": "toolCall", "id": "tool-2", "name": "read", "arguments": { "path": "img-2.png" } },
                ]),
                "toolUse"
            ),
            image_tool_result("tool-1"),
            image_tool_result("tool-2"),
        ]));
        let messages = convert_messages(&model, &ctx, &get_compat(&model), None).unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "assistant", "tool", "tool", "user"]);
        let images = messages.last().unwrap()["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|part| part["type"] == "image_url")
            .count();
        assert_eq!(images, 2);
    }

    #[test]
    fn uses_no_tool_output_placeholder_for_empty_tool_results() {
        let model = image_model();
        let ctx = msgs(json!([
            user("Run the command"),
            assistant(
                "openai",
                "gpt-4o-mini",
                json!([
                    { "type": "toolCall", "id": "tool-1", "name": "bash", "arguments": { "command": "true" } },
                ]),
                "toolUse"
            ),
            tool_result("tool-1", "bash", json!([{ "type": "text", "text": "" }])),
        ]));
        let messages = convert_messages(&model, &ctx, &get_compat(&model), None).unwrap();
        let tool = messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool["content"], "(no tool output)");
    }

    // openai-completions-cache-control-format.test.ts
    fn custom_qwen() -> Model {
        completions_model(json!({
            "id": "custom-qwen", "name": "Custom Qwen", "provider": "openrouter",
            "baseUrl": "https://example.com/v1", "reasoning": true, "maxTokens": 32000,
            "compat": { "cacheControlFormat": "anthropic" },
        }))
    }

    async fn cache_control_payload(
        model: &Model,
        retention: Option<CacheRetention>,
        messages: Option<Value>,
    ) -> Value {
        let ctx = context(json!({
            "systemPrompt": "System prompt",
            "messages": messages.unwrap_or_else(|| json!([user("Hello")])),
            "tools": [read_tool()],
        }));
        let mut options = with_key("test-key");
        options.cache_retention = retention;
        payload_for(model, ctx, options).await
    }

    fn expect_anthropic_cache_markers(params: &Value) {
        let messages = params["messages"].as_array().unwrap();
        let instruction = messages
            .iter()
            .find(|m| m["role"] == "system" || m["role"] == "developer")
            .unwrap();
        assert_eq!(
            instruction["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(params["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            params["tools"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        assert_eq!(
            last["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[tokio::test]
    async fn applies_anthropic_style_cache_markers() {
        expect_anthropic_cache_markers(&cache_control_payload(&custom_qwen(), None, None).await);
        expect_anthropic_cache_markers(
            &cache_control_payload(&pi_model(OPENROUTER_FABLE_BATCH), None, None).await,
        );
    }

    #[tokio::test]
    async fn moves_the_conversation_cache_marker_to_a_tool_result() {
        let model = pi_model(OPENROUTER_FABLE_BATCH);
        let messages = json!([
            user("Read the file"),
            assistant(
                "openrouter",
                &model.id,
                json!([
                    { "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "README.md" } },
                ]),
                "toolUse"
            ),
            tool_result(
                "call_1",
                "read",
                json!([{ "type": "text", "text": "file contents" }])
            ),
        ]);
        let params = cache_control_payload(&model, None, Some(messages)).await;
        let messages = params["messages"].as_array().unwrap();
        assert_eq!(
            messages.iter().find(|m| m["role"] == "user").unwrap()["content"],
            "Read the file"
        );
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "tool");
        assert_eq!(
            last["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[tokio::test]
    async fn omits_anthropic_style_cache_markers_when_retention_is_none() {
        let params = cache_control_payload(&custom_qwen(), Some(CacheRetention::None), None).await;
        let messages = params["messages"].as_array().unwrap();
        let instruction = messages
            .iter()
            .find(|m| m["role"] == "system" || m["role"] == "developer")
            .unwrap();
        assert!(!instruction["content"].is_array());
        assert!(params["tools"][0].get("cache_control").is_none());
        assert!(messages.last().unwrap()["content"].is_string());
    }

    // openai-completions-prompt-cache.test.ts
    async fn prompt_cache_payload(
        model: &Model,
        retention: Option<CacheRetention>,
        session_id: &str,
    ) -> Value {
        let mut options = with_key("test-key");
        options.cache_retention = retention;
        options.session_id = Some(session_id.to_string());
        payload_for(model, sys_hi(), options).await
    }

    async fn prompt_cache_request(
        model: &Model,
        retention: Option<CacheRetention>,
        session_id: &str,
        headers: Option<ProviderHeaders>,
    ) -> CapturedRequest {
        let mut options = with_key("test-key");
        options.cache_retention = retention;
        options.session_id = Some(session_id.to_string());
        options.headers = headers;
        run_chunks(model.clone(), sys_hi(), options, &[stop_chunk()])
            .await
            .0
    }

    fn compat_model(compat: Value, base_url: Option<&str>) -> Model {
        let mut model = openai_as_completions("gpt-4o-mini");
        model.compat = Some(serde_json::from_value(compat).unwrap());
        if let Some(base_url) = base_url {
            model.base_url = base_url.to_string();
        }
        model
    }

    #[tokio::test]
    async fn sets_prompt_cache_fields_for_direct_openai_requests() {
        let model = openai_as_completions("gpt-4o-mini");
        let payload = prompt_cache_payload(&model, None, "session-123").await;
        assert_eq!(payload["prompt_cache_key"], "session-123");
        assert!(payload.get("prompt_cache_retention").is_none());
        let payload = prompt_cache_payload(&model, Some(CacheRetention::Long), "session-456").await;
        assert_eq!(payload["prompt_cache_key"], "session-456");
        assert_eq!(payload["prompt_cache_retention"], "24h");
        let payload = prompt_cache_payload(&model, None, &"x".repeat(67)).await;
        assert_eq!(payload["prompt_cache_key"], "x".repeat(64));
        let payload = prompt_cache_payload(&model, Some(CacheRetention::None), "session-789").await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[tokio::test]
    async fn omits_prompt_cache_fields_without_compatible_long_retention() {
        let model = compat_model(
            json!({ "supportsLongCacheRetention": false }),
            Some("https://proxy.example.com/v1"),
        );
        let payload =
            prompt_cache_payload(&model, Some(CacheRetention::Long), "session-proxy").await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[tokio::test]
    async fn uses_pi_cache_retention_for_direct_openai_requests() {
        let mut options = with_key("test-key");
        options.session_id = Some("session-env".to_string());
        options.env = Some([("PI_CACHE_RETENTION".to_string(), "long".to_string())].into());
        let payload = payload_for(&openai_as_completions("gpt-4o-mini"), sys_hi(), options).await;
        assert_eq!(payload["prompt_cache_key"], "session-env");
        assert_eq!(payload["prompt_cache_retention"], "24h");
    }

    #[tokio::test]
    async fn sends_known_session_affinity_headers() {
        let model = compat_model(
            json!({ "sendSessionAffinityHeaders": true }),
            Some("https://proxy.example.com/v1"),
        );
        let request = prompt_cache_request(&model, None, "session-affinity", None).await;
        assert_eq!(request.header("session_id"), Some("session-affinity"));
        assert_eq!(
            request.header("x-client-request-id"),
            Some("session-affinity")
        );
        assert_eq!(
            request.header("x-session-affinity"),
            Some("session-affinity")
        );
    }

    #[tokio::test]
    async fn sends_fireworks_and_baseten_session_affinity() {
        let mut router = pi_model(FIREWORKS_GLM_5P3);
        router.id = "accounts/fireworks/routers/glm-5p3-fast".into();
        for model in [pi_model(FIREWORKS_GLM_5P3), router] {
            let request = prompt_cache_request(&model, None, "fireworks-session", None).await;
            assert_eq!(
                request.header("x-session-affinity"),
                Some("fireworks-session"),
                "{}",
                model.id
            );
        }
        let request = prompt_cache_request(
            &pi_model(BASETEN_GLM_52),
            None,
            "baseten-catalog-session",
            None,
        )
        .await;
        assert_eq!(
            request.header("x-session-affinity"),
            Some("baseten-catalog-session")
        );
        assert_eq!(
            request.header("x-client-request-id"),
            Some("baseten-catalog-session")
        );
    }

    #[tokio::test]
    async fn uses_openai_nosession_format_when_configured() {
        let model = compat_model(
            json!({ "sendSessionAffinityHeaders": true, "sessionAffinityFormat": "openai-nosession" }),
            None,
        );
        // The payload keeps the api.openai.com base URL; the headers need a
        // real request, which the mock server's URL serves.
        let payload = prompt_cache_payload(&model, None, "session-nosession").await;
        assert!(payload.get("session_id").is_none());
        assert_eq!(payload["prompt_cache_key"], "session-nosession");
        let request = prompt_cache_request(&model, None, "session-nosession", None).await;
        assert_eq!(request.header("session_id"), None);
        assert_eq!(
            request.header("x-client-request-id"),
            Some("session-nosession")
        );
        assert_eq!(
            request.header("x-session-affinity"),
            Some("session-nosession")
        );
        assert_eq!(request.header("x-session-id"), None);
    }

    #[tokio::test]
    async fn uses_openrouter_session_affinity_header() {
        let proxy = compat_model(
            json!({ "sendSessionAffinityHeaders": true, "sessionAffinityFormat": "openrouter" }),
            Some("https://proxy.example.com/v1"),
        );
        for (model, session) in [
            (proxy, "session-proxy"),
            (pi_model(OPENROUTER_AUTO), "session-openrouter"),
        ] {
            let request = prompt_cache_request(&model, None, session, None).await;
            assert!(request.body.get("session_id").is_none());
            assert!(request.body.get("prompt_cache_key").is_none());
            assert_eq!(request.header("x-session-id"), Some(session));
            assert_eq!(request.header("session_id"), None);
            assert_eq!(request.header("x-client-request-id"), None);
            assert_eq!(request.header("x-session-affinity"), None);
        }
    }

    #[tokio::test]
    async fn omits_openrouter_session_affinity_data_when_disabled() {
        let mut model = compat_model(
            json!({ "sendSessionAffinityHeaders": false }),
            Some("https://openrouter.ai/api/v1"),
        );
        model.provider = "openrouter".into();
        let request = prompt_cache_request(&model, None, "session-openrouter", None).await;
        assert!(request.body.get("session_id").is_none());
        assert!(request.body.get("prompt_cache_key").is_none());
        assert_eq!(request.header("x-session-id"), None);
    }

    #[tokio::test]
    async fn omits_session_affinity_headers_when_retention_is_none() {
        let model = compat_model(
            json!({ "sendSessionAffinityHeaders": true }),
            Some("https://proxy.example.com/v1"),
        );
        let request =
            prompt_cache_request(&model, Some(CacheRetention::None), "session-affinity", None)
                .await;
        assert_eq!(request.header("session_id"), None);
        assert_eq!(request.header("x-client-request-id"), None);
        assert_eq!(request.header("x-session-affinity"), None);
    }

    #[tokio::test]
    async fn lets_explicit_headers_override_session_affinity_headers() {
        let model = compat_model(
            json!({ "sendSessionAffinityHeaders": true }),
            Some("https://proxy.example.com/v1"),
        );
        let headers: ProviderHeaders = [
            ("session_id", "override-session"),
            ("x-client-request-id", "override-request"),
            ("x-session-affinity", "override-affinity"),
        ]
        .into_iter()
        .map(|(name, value)| (name, Some(value.to_string())))
        .collect();
        let request = prompt_cache_request(&model, None, "session-affinity", Some(headers)).await;
        assert_eq!(request.header("session_id"), Some("override-session"));
        assert_eq!(
            request.header("x-client-request-id"),
            Some("override-request")
        );
        assert_eq!(
            request.header("x-session-affinity"),
            Some("override-affinity")
        );
    }

    // cache-retention.test.ts (OpenAI Completions Provider)
    #[tokio::test]
    async fn completions_cache_retention_for_proxies_and_opencode() {
        let proxy = || {
            completions_model(
                json!({ "provider": "test-openai-completions", "baseUrl": "https://my-proxy.example.com/v1" }),
            )
        };
        let payload =
            prompt_cache_payload(&proxy(), Some(CacheRetention::Long), "session-completions").await;
        assert_eq!(payload["prompt_cache_key"], "session-completions");
        assert_eq!(payload["prompt_cache_retention"], "24h");
        let mut no_long = proxy();
        no_long.compat =
            Some(serde_json::from_value(json!({ "supportsLongCacheRetention": false })).unwrap());
        let payload = prompt_cache_payload(
            &no_long,
            Some(CacheRetention::Long),
            "session-completions-false",
        )
        .await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
        let payload = prompt_cache_payload(
            &pi_model(OPENCODE_KIMI_K26),
            Some(CacheRetention::Long),
            "s",
        )
        .await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    // pre-generation-error.test.ts
    #[test]
    fn throws_synchronously_when_auth_is_missing() {
        let model = completions_model(
            json!({ "provider": "test-provider", "baseUrl": "https://example.invalid" }),
        );
        let error = stream_simple_openai_completions(model, msgs(json!([])), Default::default())
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "No API key for provider: test-provider");
    }

    // provider-error-body-regression.test.ts
    async fn error_body_result(body: Value) -> AssistantMessage {
        let server = MockServer::start(vec![MockResponse::status(
            403,
            &[("content-type", "application/json")],
            body.to_string(),
        )])
        .await;
        let mut model = completions_model(json!({
            "provider": "openrouter", "baseUrl": "https://openrouter.ai/api/v1", "contextWindow": 1000, "maxTokens": 100,
        }));
        model.base_url = server.url.clone();
        collect(stream_openai_completions(model, hi(), with_key("test")))
            .await
            .1
    }

    #[tokio::test]
    async fn surfaces_status_and_body_for_errors() {
        let output = error_body_result(json!({ "error": "blocked by gateway WAF" })).await;
        assert_eq!(output.stop_reason, StopReason::Error);
        let message = output.error_message.unwrap();
        assert!(message.contains("403"), "{message}");
        assert!(message.contains("blocked by gateway WAF"), "{message}");
        assert_ne!(message, "403 status code (no body)");
    }

    #[tokio::test]
    async fn does_not_double_print_openrouter_metadata_raw() {
        let output = error_body_result(json!({ "error": {
            "message": "Provider returned error", "code": 403,
            "metadata": { "raw": "upstream WAF blocked policy XYZ" },
        } }))
        .await;
        let message = output.error_message.unwrap();
        assert_eq!(
            message.matches("upstream WAF blocked policy XYZ").count(),
            1,
            "{message}"
        );
    }

    // sampling-options.test.ts (openai-completions cases)
    fn sampling_model(sampling_params: Option<Value>, overrides: Value) -> Model {
        let mut base = json!({
            "id": "custom-model", "name": "Custom Model", "provider": "custom-provider",
            "baseUrl": "http://127.0.0.1:9/v1", "maxTokens": 16384,
        });
        if let Some(sampling_params) = sampling_params {
            base["samplingParams"] = sampling_params;
        }
        for (key, value) in overrides.as_object().unwrap() {
            base[key] = value.clone();
        }
        completions_model(base)
    }

    fn sampling(params: Value) -> Option<crate::types::SamplingParams> {
        Some(serde_json::from_value(params).unwrap())
    }

    async fn sampling_payload(model: &Model, mut options: OpenAICompletionsOptions) -> Value {
        options.api_key = Some("fake-key".to_string());
        payload_for(model, msgs(json!([user("Hello")])), options).await
    }

    async fn simple_sampling_payload(model: &Model, mut options: SimpleStreamOptions) -> Value {
        options.api_key = Some("fake-key".to_string());
        simple_payload(model, msgs(json!([user("Hello")])), options).await
    }

    #[tokio::test]
    async fn merges_request_sampling_params_into_the_body() {
        let mut options = OpenAICompletionsOptions::default();
        options.sampling_params = sampling(json!({ "top_p": 0.95, "top_k": 0, "min_p": 0 }));
        let payload = sampling_payload(&sampling_model(None, json!({})), options).await;
        assert_eq!(payload["top_p"], 0.95);
        assert_eq!(payload["top_k"], 0);
        assert_eq!(payload["min_p"], 0);
        let payload = sampling_payload(&sampling_model(None, json!({})), Default::default()).await;
        assert!(payload.get("temperature").is_none());
        assert!(payload.get("top_p").is_none());
    }

    #[tokio::test]
    async fn applies_model_sampling_params_with_request_precedence() {
        let mut options = OpenAICompletionsOptions::default();
        options.sampling_params = sampling(json!({ "top_p": 0.5 }));
        let model = sampling_model(Some(json!({ "top_p": 0.95, "min_p": 0.05 })), json!({}));
        let payload = sampling_payload(&model, options).await;
        assert_eq!(payload["top_p"], 0.5);
        assert_eq!(payload["min_p"], 0.05);
    }

    #[tokio::test]
    async fn passes_sampling_params_through_stream_simple() {
        let mut options = SimpleStreamOptions::default();
        options.sampling_params = sampling(json!({ "top_p": 0.5 }));
        let payload = simple_sampling_payload(&sampling_model(None, json!({})), options).await;
        assert_eq!(payload["top_p"], 0.5);
    }

    #[tokio::test]
    async fn applies_sampling_params_for_the_effective_thinking_level() {
        let model = sampling_model(
            Some(json!({ "temperature": 1, "top_p": 0.95 })),
            json!({
                "reasoning": true,
                "thinkingLevelMap": { "low": null, "medium": null },
                "samplingParamsByThinkingLevel": { "high": { "temperature": 0.8, "top_k": 64 } },
            }),
        );
        let mut options = SimpleStreamOptions::default();
        options.reasoning = Some(ThinkingLevel::Low);
        let payload = simple_sampling_payload(&model, options).await;
        assert_eq!(payload["temperature"], 0.8);
        assert_eq!(payload["top_p"], 0.95);
        assert_eq!(payload["top_k"], 64);
    }

    #[tokio::test]
    async fn applies_off_sampling_params_when_reasoning_is_disabled() {
        let model = sampling_model(
            None,
            json!({ "samplingParamsByThinkingLevel": { "off": { "temperature": 0.7 } } }),
        );
        let payload = simple_sampling_payload(&model, SimpleStreamOptions::default()).await;
        assert_eq!(payload["temperature"], 0.7);
    }

    #[tokio::test]
    async fn merges_stream_option_keys_over_thinking_level_keys() {
        let model = sampling_model(
            None,
            json!({ "reasoning": true, "samplingParamsByThinkingLevel": { "low": { "temperature": 0.6, "top_p": 0.95 } } }),
        );
        let mut options = SimpleStreamOptions::default();
        options.reasoning = Some(ThinkingLevel::Low);
        options.sampling_params = sampling(json!({ "top_p": 0.5 }));
        let payload = simple_sampling_payload(&model, options).await;
        assert_eq!(payload["temperature"], 0.6);
        assert_eq!(payload["top_p"], 0.5);
    }

    #[tokio::test]
    async fn applies_thinking_level_params_between_model_and_request_params() {
        let model = sampling_model(
            Some(json!({ "temperature": 1, "top_p": 0.95 })),
            json!({ "reasoning": true, "samplingParamsByThinkingLevel": { "low": { "temperature": 0.6, "top_k": 64 } } }),
        );
        let mut options = OpenAICompletionsOptions::default();
        options.reasoning_effort = Some(ThinkingLevel::Low);
        options.sampling_params = sampling(json!({ "top_p": 0.5 }));
        let payload = sampling_payload(&model, options).await;
        assert_eq!(payload["temperature"], 0.6);
        assert_eq!(payload["top_p"], 0.5);
        assert_eq!(payload["top_k"], 64);
    }

    #[tokio::test]
    async fn sampling_params_override_named_request_fields() {
        let mut options = OpenAICompletionsOptions::default();
        options.temperature = Some(0.0);
        options.sampling_params = sampling(json!({ "temperature": 1 }));
        let payload = sampling_payload(&sampling_model(None, json!({})), options).await;
        assert_eq!(payload["temperature"], 1);
    }

    // openrouter-reasoning-options.test.ts (mandatory reasoning payloads)
    fn stealth_openrouter(thinking_level_map: Option<Value>) -> Model {
        let mut overrides = json!({
            "id": "stealth/ox-alpha", "name": "Ox Alpha", "provider": "openrouter",
            "baseUrl": "https://example.invalid/v1", "reasoning": true,
            "compat": { "thinkingFormat": "openrouter" },
        });
        if let Some(map) = thinking_level_map {
            overrides["thinkingLevelMap"] = map;
        }
        completions_model(overrides)
    }

    #[tokio::test]
    async fn openrouter_mandatory_reasoning_payloads() {
        let mandatory = json!({ "off": null, "minimal": null, "low": "low", "medium": null, "high": "high", "xhigh": null, "max": "max" });
        let ctx = || {
            context(json!({ "messages": [{ "role": "user", "content": "Hello", "timestamp": 0 }] }))
        };
        let payload = simple_payload(
            &stealth_openrouter(Some(mandatory.clone())),
            ctx(),
            simple(None),
        )
        .await;
        assert!(payload.get("reasoning").is_none());
        let payload = simple_payload(
            &stealth_openrouter(Some(mandatory)),
            ctx(),
            simple(Some(ThinkingLevel::Low)),
        )
        .await;
        assert_eq!(payload["reasoning"]["effort"], "low");
        let payload = simple_payload(&stealth_openrouter(None), ctx(), simple(None)).await;
        assert_eq!(payload["reasoning"]["effort"], "none");
    }
}
