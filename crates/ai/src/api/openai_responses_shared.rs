//! Port of `api/openai-responses-shared.ts`: message and tool conversion and
//! stream processing shared by the OpenAI Responses API modules.
//!
//! Requests and stream events are `serde_json::Value`s shaped like the
//! `openai` SDK types. Streaming scratch state (`partialJson`,
//! `customInput`) lives in side tables keyed by content index instead of on
//! the persisted tool call blocks.

use std::collections::HashMap;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use serde_json::{Map, Value, json};

use super::constrained_sampling::{
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta, get_grammar_tool_input,
    get_json_schema_tool_parameters, resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
};
use super::openai_client::{js_template_string, js_truthy};
use super::transform_messages::transform_messages;
use crate::models::calculate_cost;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Message, Model, ModelInput,
    ProviderStreamEventHook, StopReason, SystemMessage, TextContent, ThinkingContent, Tool,
    ToolCall, ToolResultContent, TranscriptContext, Usage, UsageCost, UserContent,
    UserMessageContent,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::transcript::{resolve_transcript, resolve_transcript_tools};
use crate::{Error, Result};

/// `ReadonlyMap<string, string>` from tool name to grammar input property,
/// as `createGrammarToolInputProperties()` builds it.
pub type GrammarToolInputProperties = HashMap<String, String>;

// =============================================================================
// Utilities
// =============================================================================

fn encode_text_signature_v1(id: Option<&Value>, phase: Option<&str>) -> String {
    let mut payload = Map::new();
    payload.insert("v".to_string(), json!(1));
    if let Some(id) = id {
        payload.insert("id".to_string(), id.clone());
    }
    if let Some(phase) = phase.filter(|phase| !phase.is_empty()) {
        payload.insert("phase".to_string(), json!(phase));
    }
    Value::Object(payload).to_string()
}

struct ParsedTextSignature {
    id: String,
    phase: Option<String>,
}

fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let signature = signature.filter(|signature| !signature.is_empty())?;
    if signature.starts_with('{')
        && let Ok(parsed) = serde_json::from_str::<Value>(signature)
        && parsed.get("v").and_then(Value::as_f64) == Some(1.0)
        && let Some(id) = parsed.get("id").and_then(Value::as_str)
    {
        let phase = parsed
            .get("phase")
            .and_then(Value::as_str)
            .filter(|phase| matches!(*phase, "commentary" | "final_answer"))
            .map(str::to_string);
        return Some(ParsedTextSignature {
            id: id.to_string(),
            phase,
        });
    }
    Some(ParsedTextSignature {
        id: signature.to_string(),
        phase: None,
    })
}

fn image_url(mime_type: &str, data: &str) -> String {
    format!("data:{mime_type};base64,{data}")
}

fn convert_tool_result_output(model: &Model, content: &[ToolResultContent]) -> Value {
    let text_result = content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(text.text.as_str()),
            UserContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<_> = content
        .iter()
        .filter_map(|content| match content {
            UserContent::Image(image) => Some(image),
            UserContent::Text(_) => None,
        })
        .collect();
    let has_text = !text_result.is_empty();

    if images.is_empty() || !model.input.contains(&ModelInput::Image) {
        let text = if has_text {
            text_result.as_str()
        } else if !images.is_empty() {
            "(see attached image)"
        } else {
            "(no tool output)"
        };
        return Value::String(sanitize_surrogates(text));
    }

    let mut output = Vec::new();
    if has_text {
        output.push(json!({ "type": "input_text", "text": sanitize_surrogates(&text_result) }));
    }
    for image in images {
        output.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": image_url(&image.mime_type, &image.data),
        }));
    }
    Value::Array(output)
}

/// `(responseServiceTier, requestServiceTier) => serviceTier`.
pub type ResolveServiceTier =
    Arc<dyn Fn(Option<&str>, Option<&str>) -> Option<String> + Send + Sync>;
/// `(usage, serviceTier) => void`.
pub type ApplyServiceTierPricing = Arc<dyn Fn(&mut Usage, Option<&str>) + Send + Sync>;

#[derive(Clone, Default)]
pub struct OpenAIResponsesStreamOptions {
    pub on_provider_stream_event: Option<ProviderStreamEventHook>,
    pub service_tier: Option<String>,
    pub grammar_tool_input_properties: Option<GrammarToolInputProperties>,
    pub resolve_service_tier: Option<ResolveServiceTier>,
    pub apply_service_tier_pricing: Option<ApplyServiceTierPricing>,
}

#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesMessagesOptions {
    pub include_system_prompt: Option<bool>,
    pub grammar_tool_input_properties: Option<GrammarToolInputProperties>,
    /// Whether later system messages are sent in place; otherwise they are folded into the leading prompt.
    pub supports_mid_convo_system_messages: Option<bool>,
    pub supports_additional_tools: Option<bool>,
    pub supports_tool_search: Option<bool>,
    pub tool_options: Option<ConvertResponsesToolsOptions>,
}

#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesToolsOptions {
    /// `strict?: boolean | null`: `None` is undefined, `Some(None)` is null.
    pub strict: Option<Option<bool>>,
    pub supports_strict_mode: Option<bool>,
    pub supports_openai_grammar_tools: Option<bool>,
    pub tool_search_result: Option<bool>,
}

// =============================================================================
// Message conversion
// =============================================================================

/// `part.replace(/[^a-zA-Z0-9_-]/g, "_")`, slice to 64, strip trailing `_`.
/// The regex has no `u` flag, so each UTF-16 code unit is replaced.
fn normalize_id_part(part: &str) -> String {
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
    sanitized.truncate(64);
    sanitized.trim_end_matches('_').to_string()
}

fn build_foreign_responses_item_id(item_id: &str) -> String {
    let mut normalized = format!("fc_{}", short_hash(item_id));
    normalized.truncate(64);
    normalized
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

pub fn convert_responses_messages(
    model: &Model,
    context: &TranscriptContext,
    allowed_tool_call_providers: &[&str],
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Result<Vec<Value>> {
    let default_options = ConvertResponsesMessagesOptions::default();
    let options = options.unwrap_or(&default_options);
    let normalized_context =
        resolve_transcript(context, options.supports_mid_convo_system_messages);
    let mut messages: Vec<Value> = Vec::new();

    let provider_allowed = allowed_tool_call_providers.contains(&model.provider.as_str());
    let normalize_tool_call_id = |id: &str, _target: &Model, source: &AssistantMessage| -> String {
        if !provider_allowed || !id.contains('|') {
            return normalize_id_part(id);
        }
        let mut parts = id.split('|');
        let call_id = parts.next().unwrap_or_default();
        let item_id = parts.next().unwrap_or_default();
        let normalized_call_id = normalize_id_part(call_id);
        let is_foreign_tool_call = source.provider != model.provider || source.api != model.api;
        let mut normalized_item_id = if is_foreign_tool_call {
            build_foreign_responses_item_id(item_id)
        } else {
            normalize_id_part(item_id)
        };
        // OpenAI Responses API requires item id to start with "fc"
        if !normalized_item_id.starts_with("fc_") {
            normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
        }
        format!("{normalized_call_id}|{normalized_item_id}")
    };

    let transformed_messages = transform_messages(
        &normalized_context.messages,
        model,
        Some(&normalize_tool_call_id),
    );
    let transcript_tools = resolve_transcript_tools(
        &normalized_context.messages,
        options.supports_additional_tools.unwrap_or(false)
            || options.supports_tool_search.unwrap_or(false),
    );
    let append_system_tool_additions =
        |messages: &mut Vec<Value>, message: &SystemMessage, seed: &str| -> Result<()> {
            let tools: &[Tool] = if transcript_tools.anchors_additions {
                message.tools_added.as_deref().unwrap_or_default()
            } else {
                &[]
            };
            if tools.is_empty() {
                return Ok(());
            }
            if options.supports_additional_tools == Some(true) {
                messages.push(json!({
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": convert_responses_tools(tools, options.tool_options.as_ref())?,
                }));
                return Ok(());
            }
            if options.supports_tool_search != Some(true) {
                return Ok(());
            }
            let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
            let call_id = format!(
                "pi_tool_load_{}",
                short_hash(&format!("{seed}:{}", names.join(",")))
            );
            messages.push(json!({
                "type": "tool_search_call",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "arguments": { "query": names.join(" "), "limit": names.len() },
            }));
            let tool_options = ConvertResponsesToolsOptions {
                tool_search_result: Some(true),
                ..options.tool_options.clone().unwrap_or_default()
            };
            messages.push(json!({
                "type": "tool_search_output",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "tools": convert_responses_tools(tools, Some(&tool_options))?,
            }));
            Ok(())
        };
    let include_initial_system_message = options.include_system_prompt.unwrap_or(true);
    let supports_developer_role = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_developer_role);
    let instruction_role = if model.reasoning && supports_developer_role != Some(false) {
        "developer"
    } else {
        "system"
    };
    let grammar_properties = options.grammar_tool_input_properties.as_ref();

    let mut msg_index = 0usize;
    for (source_index, msg) in transformed_messages.iter().enumerate() {
        let is_leading_system_message = source_index == 0 && matches!(msg, Message::System(_));
        match msg {
            Message::System(system) => {
                if !is_leading_system_message {
                    append_system_tool_additions(
                        &mut messages,
                        system,
                        &format!("system:{msg_index}"),
                    )?;
                }
                if !is_leading_system_message || include_initial_system_message {
                    let text = if is_leading_system_message {
                        get_system_message_text(system)
                    } else {
                        render_system_message_update(system)
                    };
                    if !text.is_empty() {
                        messages.push(json!({
                            "role": instruction_role,
                            "content": sanitize_surrogates(&text),
                        }));
                    }
                }
            }
            Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => {
                    messages.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": sanitize_surrogates(text) }],
                    }));
                }
                UserMessageContent::Parts(parts) => {
                    let content: Vec<Value> = parts
                        .iter()
                        .map(|item| match item {
                            UserContent::Text(text) => json!({
                                "type": "input_text",
                                "text": sanitize_surrogates(&text.text),
                            }),
                            UserContent::Image(image) => json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": image_url(&image.mime_type, &image.data),
                            }),
                        })
                        .collect();
                    if content.is_empty() {
                        continue;
                    }
                    messages.push(json!({ "role": "user", "content": content }));
                }
            },
            Message::Assistant(assistant) => {
                let mut output: Vec<Value> = Vec::new();
                let is_same_provider_and_api =
                    assistant.provider == model.provider && assistant.api == model.api;
                let is_same_model = is_same_provider_and_api && assistant.model == model.id;
                let is_different_model = is_same_provider_and_api && assistant.model != model.id;
                let mut text_block_index = 0usize;

                for block in &assistant.content {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            if let Some(signature) = thinking
                                .thinking_signature
                                .as_deref()
                                .filter(|signature| !signature.is_empty())
                            {
                                let reasoning_item: Value = serde_json::from_str(signature)?;
                                output.push(reasoning_item);
                            }
                        }
                        AssistantContent::Text(text_block) => {
                            let parsed_signature =
                                parse_text_signature(text_block.text_signature.as_deref());
                            let fallback_message_id = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            // OpenAI requires id to be max 64 characters
                            let msg_id = match parsed_signature.as_ref().map(|parsed| &parsed.id) {
                                Some(id) if !id.is_empty() => {
                                    if utf16_len(id) > 64 {
                                        format!("msg_{}", short_hash(id))
                                    } else {
                                        id.clone()
                                    }
                                }
                                _ => fallback_message_id,
                            };
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": sanitize_surrogates(&text_block.text),
                                    "annotations": [],
                                }],
                                "status": "completed",
                                "id": msg_id,
                            });
                            if let Some(phase) = parsed_signature.and_then(|parsed| parsed.phase) {
                                item["phase"] = json!(phase);
                            }
                            output.push(item);
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let mut parts = tool_call.id.split('|');
                            let call_id = parts.next().unwrap_or_default();
                            let item_id_raw = parts.next();
                            let custom_input_property = grammar_properties
                                .and_then(|properties| properties.get(&tool_call.name));

                            // For different-model messages, set id to undefined to avoid pairing validation.
                            // OpenAI tracks which item IDs were paired with rs_xxx reasoning items.
                            // By omitting the id, we avoid triggering that validation (like cross-provider does).
                            // Also drop ids that do not match the replayed item type: function_call ids must be fc_*
                            // and custom_tool_call ids must be ctc_*. Foreign tool call ids are normalized to fc_*, and
                            // a call can switch between the two types when grammar tool support differs.
                            let item_id_prefix = if custom_input_property.is_none() {
                                "fc_"
                            } else {
                                "ctc_"
                            };
                            let item_id = item_id_raw.filter(|item_id| {
                                !is_different_model && item_id.starts_with(item_id_prefix)
                            });

                            let mut item = Map::new();
                            if let Some(property) = custom_input_property {
                                item.insert("type".to_string(), json!("custom_tool_call"));
                                if let Some(item_id) = item_id {
                                    item.insert("id".to_string(), json!(item_id));
                                }
                                item.insert("call_id".to_string(), json!(call_id));
                                item.insert("name".to_string(), json!(tool_call.name));
                                item.insert(
                                    "input".to_string(),
                                    json!(sanitize_surrogates(get_grammar_tool_input(
                                        &tool_call.name,
                                        &tool_call.arguments,
                                        property,
                                    )?)),
                                );
                            } else {
                                item.insert("type".to_string(), json!("function_call"));
                                if let Some(item_id) = item_id {
                                    item.insert("id".to_string(), json!(item_id));
                                }
                                item.insert("call_id".to_string(), json!(call_id));
                                item.insert("name".to_string(), json!(tool_call.name));
                                item.insert(
                                    "arguments".to_string(),
                                    json!(tool_call.arguments.to_string()),
                                );
                            }
                            if is_same_model && let Some(namespace) = &tool_call.namespace {
                                item.insert("namespace".to_string(), json!(namespace));
                            }
                            output.push(Value::Object(item));
                        }
                    }
                }
                if output.is_empty() {
                    continue;
                }
                messages.extend(output);
            }
            Message::ToolResult(result) => {
                let call_id = result.tool_call_id.split('|').next().unwrap_or_default();
                let output = convert_tool_result_output(model, &result.content);
                let item_type = if grammar_properties
                    .is_some_and(|properties| properties.contains_key(&result.tool_name))
                {
                    "custom_tool_call_output"
                } else {
                    "function_call_output"
                };
                messages.push(json!({ "type": item_type, "call_id": call_id, "output": output }));
            }
        }
        if !is_leading_system_message {
            msg_index += 1;
        }
    }

    Ok(messages)
}

// =============================================================================
// Tool conversion
// =============================================================================

pub fn convert_responses_tools(
    tools: &[Tool],
    options: Option<&ConvertResponsesToolsOptions>,
) -> Result<Vec<Value>> {
    let default_options = ConvertResponsesToolsOptions::default();
    let options = options.unwrap_or(&default_options);
    let default_strict: Option<bool> = options.strict.unwrap_or(Some(false));
    let supports_strict_mode = options.supports_strict_mode.unwrap_or(true);
    let supports_openai_grammar_tools = options.supports_openai_grammar_tools.unwrap_or(false);
    let tool_search_result = options.tool_search_result == Some(true);

    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) =
                resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?
            {
                let mut converted = json!({
                    "type": "custom",
                    "name": tool.name,
                    "description": tool.description,
                    "format": {
                        "type": "grammar",
                        "syntax": grammar.format.as_str(),
                        "definition": grammar.definition,
                    },
                });
                if tool_search_result {
                    converted["defer_loading"] = json!(true);
                }
                return Ok(converted);
            }

            let constrained_strict =
                resolve_json_schema_strict_sampling(tool, supports_strict_mode, None)?;
            let strict = constrained_strict.or(default_strict);
            let mut function_tool = json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": get_json_schema_tool_parameters(tool, Some(strict == Some(true)))?,
            });
            if tool_search_result {
                function_tool["defer_loading"] = json!(true);
            }
            if supports_strict_mode {
                function_tool["strict"] = json!(strict);
            }
            Ok(function_tool)
        })
        .collect()
}

// =============================================================================
// Stream processing
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotType {
    Thinking,
    Text,
    ToolCall,
}

#[derive(Debug, Clone, Copy)]
struct ResponsesOutputSlot {
    slot_type: SlotType,
    content_index: usize,
}

struct CustomInput {
    property: String,
    json_buffer: GrammarToolInputJsonBuffer,
}

/// Mutable state of one `processResponsesStream()` run.
struct ResponsesStreamState<'a> {
    output: &'a mut AssistantMessage,
    stream: &'a AssistantMessageEventStream,
    options: Option<&'a OpenAIResponsesStreamOptions>,
    output_slots: HashMap<Option<u64>, ResponsesOutputSlot>,
    reasoning_blocks_by_id: HashMap<String, usize>,
    /// `StreamingToolCall.partialJson`, keyed by content index.
    partial_json: HashMap<usize, String>,
    /// `StreamingToolCall.customInput`, keyed by content index.
    custom_inputs: HashMap<usize, CustomInput>,
}

fn output_index(event: &Value) -> Option<u64> {
    event.get("output_index").and_then(Value::as_u64)
}

fn str_field<'v>(value: &'v Value, key: &str) -> Option<&'v str> {
    value.get(key).and_then(Value::as_str)
}

/// `value || 0` for a token count.
fn token_count(value: Option<&Value>) -> u32 {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .map(|number| number as u32)
        .unwrap_or(0)
}

impl ResponsesStreamState<'_> {
    fn push(&self, event: AssistantMessageEvent) {
        self.stream.push(event);
    }

    fn apply_message_phase_stop_reason(&mut self, item: &Value) {
        if str_field(item, "type") == Some("message")
            && str_field(item, "phase") == Some("final_answer")
        {
            self.output.stop_reason = StopReason::Stop;
        }
    }

    fn get_slot(&self, index: Option<u64>, slot_type: SlotType) -> Option<ResponsesOutputSlot> {
        self.output_slots
            .get(&index)
            .copied()
            .filter(|slot| slot.slot_type == slot_type)
    }

    fn thinking_mut(&mut self, content_index: usize) -> &mut ThinkingContent {
        match &mut self.output.content[content_index] {
            AssistantContent::Thinking(block) => block,
            _ => unreachable!("thinking slot points at a thinking block"),
        }
    }

    fn text_mut(&mut self, content_index: usize) -> &mut TextContent {
        match &mut self.output.content[content_index] {
            AssistantContent::Text(block) => block,
            _ => unreachable!("text slot points at a text block"),
        }
    }

    fn tool_call_mut(&mut self, content_index: usize) -> &mut ToolCall {
        match &mut self.output.content[content_index] {
            AssistantContent::ToolCall(block) => block,
            _ => unreachable!("tool call slot points at a tool call block"),
        }
    }

    fn push_tool_call_delta(&self, slot: ResponsesOutputSlot, delta: Option<String>) {
        let Some(delta) = delta else {
            return;
        };
        self.push(AssistantMessageEvent::ToolCallDelta {
            content_index: slot.content_index,
            delta,
            partial: self.output.clone(),
        });
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
        let property = custom_input.property.clone();
        let mut arguments = Map::new();
        arguments.insert(property, json!(next_input));
        self.tool_call_mut(content_index).arguments = Value::Object(arguments);
        Ok(delta)
    }

    fn add_block(
        &mut self,
        index: Option<u64>,
        slot_type: SlotType,
        block: AssistantContent,
    ) -> ResponsesOutputSlot {
        self.output.content.push(block);
        let slot = ResponsesOutputSlot {
            slot_type,
            content_index: self.output.content.len() - 1,
        };
        self.output_slots.insert(index, slot);
        slot
    }

    fn create_slot(&mut self, index: Option<u64>, item: &Value) -> Option<ResponsesOutputSlot> {
        match str_field(item, "type") {
            Some("reasoning") => {
                let slot = self.add_block(
                    index,
                    SlotType::Thinking,
                    AssistantContent::Thinking(ThinkingContent::default()),
                );
                self.push(AssistantMessageEvent::ThinkingStart {
                    content_index: slot.content_index,
                    partial: self.output.clone(),
                });
                Some(slot)
            }
            Some("message") => {
                self.apply_message_phase_stop_reason(item);
                let slot = self.add_block(index, SlotType::Text, AssistantContent::text(""));
                self.push(AssistantMessageEvent::TextStart {
                    content_index: slot.content_index,
                    partial: self.output.clone(),
                });
                Some(slot)
            }
            Some("function_call") => {
                let block = ToolCall {
                    id: format!(
                        "{}|{}",
                        js_template_string(item.get("call_id")),
                        js_template_string(item.get("id"))
                    ),
                    name: str_field(item, "name").unwrap_or_default().to_string(),
                    arguments: json!({}),
                    thought_signature: None,
                    namespace: str_field(item, "namespace").map(str::to_string),
                };
                let partial_json = str_field(item, "arguments").unwrap_or_default().to_string();
                let slot =
                    self.add_block(index, SlotType::ToolCall, AssistantContent::ToolCall(block));
                self.partial_json.insert(slot.content_index, partial_json);
                self.push(AssistantMessageEvent::ToolCallStart {
                    content_index: slot.content_index,
                    partial: self.output.clone(),
                });
                Some(slot)
            }
            Some("custom_tool_call") => {
                let name = str_field(item, "name").unwrap_or_default().to_string();
                let input_property = self
                    .options
                    .and_then(|options| options.grammar_tool_input_properties.as_ref())
                    .and_then(|properties| properties.get(&name))
                    .cloned()
                    .unwrap_or_else(|| "input".to_string());
                let input = str_field(item, "input").unwrap_or_default();
                let mut arguments = Map::new();
                arguments.insert(input_property.clone(), json!(input));
                let block = ToolCall {
                    id: format!(
                        "{}|{}",
                        js_template_string(item.get("call_id")),
                        js_template_string(item.get("id"))
                    ),
                    name,
                    arguments: Value::Object(arguments),
                    thought_signature: None,
                    namespace: str_field(item, "namespace").map(str::to_string),
                };
                let slot =
                    self.add_block(index, SlotType::ToolCall, AssistantContent::ToolCall(block));
                self.custom_inputs.insert(
                    slot.content_index,
                    CustomInput {
                        property: input_property,
                        json_buffer: GrammarToolInputJsonBuffer::default(),
                    },
                );
                self.push(AssistantMessageEvent::ToolCallStart {
                    content_index: slot.content_index,
                    partial: self.output.clone(),
                });
                Some(slot)
            }
            _ => None,
        }
    }

    fn get_or_create_slot(
        &mut self,
        index: Option<u64>,
        item: &Value,
    ) -> Option<ResponsesOutputSlot> {
        match self.output_slots.get(&index) {
            Some(slot) => Some(*slot),
            None => self.create_slot(index, item),
        }
    }

    // Azure OpenAI can omit reasoning.encrypted_content from response.output_item.done
    // and provide it only in response.completed.response.output. Backfill the
    // persisted reasoning signature from the terminal response to keep store:false
    // multi-turn replay stateless. See https://github.com/earendil-works/pi/issues/6409.
    fn backfill_reasoning_signatures(&mut self, response_output: &[Value]) -> Result<()> {
        for item in response_output {
            if str_field(item, "type") != Some("reasoning")
                || !js_truthy(item.get("encrypted_content"))
            {
                continue;
            }
            let Some(content_index) = str_field(item, "id")
                .and_then(|id| self.reasoning_blocks_by_id.get(id))
                .copied()
            else {
                continue;
            };
            let block = self.thinking_mut(content_index);
            let Some(signature) = block
                .thinking_signature
                .as_deref()
                .filter(|signature| !signature.is_empty())
            else {
                continue;
            };
            let mut stored_item: Value = serde_json::from_str(signature)?;
            if js_truthy(stored_item.get("encrypted_content")) {
                continue;
            }
            if let Some(stored) = stored_item.as_object_mut() {
                stored.insert(
                    "encrypted_content".to_string(),
                    item["encrypted_content"].clone(),
                );
            }
            block.thinking_signature = Some(stored_item.to_string());
        }
        Ok(())
    }

    fn finalize_response(&mut self, model: &Model, response: &Value) -> Result<()> {
        let response_output = response
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        self.backfill_reasoning_signatures(&response_output)?;
        if let Some(id) = response.get("id").filter(|id| js_truthy(Some(id))) {
            self.output.response_id = Some(js_template_string(Some(id)));
        }
        if let Some(usage) = response.get("usage").filter(|usage| js_truthy(Some(usage))) {
            let input_details = usage.get("input_tokens_details");
            let cached_tokens =
                token_count(input_details.and_then(|details| details.get("cached_tokens")));
            let cache_write_tokens =
                token_count(input_details.and_then(|details| details.get("cache_write_tokens")));
            let input_tokens = token_count(usage.get("input_tokens"));
            self.output.usage = Usage {
                // OpenAI includes cached and cache-write tokens in input_tokens, so subtract both.
                input: input_tokens
                    .saturating_sub(cached_tokens)
                    .saturating_sub(cache_write_tokens),
                output: token_count(usage.get("output_tokens")),
                cache_read: cached_tokens,
                cache_write: cache_write_tokens,
                cache_write_1h: None,
                reasoning: Some(token_count(
                    usage
                        .get("output_tokens_details")
                        .and_then(|details| details.get("reasoning_tokens")),
                )),
                total_tokens: token_count(usage.get("total_tokens")),
                cost: UsageCost::default(),
            };
        }
        calculate_cost(model, &mut self.output.usage);
        if let Some(options) = self.options
            && let Some(apply_service_tier_pricing) = &options.apply_service_tier_pricing
        {
            let response_service_tier = str_field(response, "service_tier");
            let service_tier = match &options.resolve_service_tier {
                Some(resolve) => resolve(response_service_tier, options.service_tier.as_deref()),
                None => response_service_tier
                    .map(str::to_string)
                    .or_else(|| options.service_tier.clone()),
            };
            apply_service_tier_pricing(&mut self.output.usage, service_tier.as_deref());
        }
        // Map status to stop reason. For incomplete responses, retain the provider's
        // specific reason so max-output truncation and content filtering stay distinct.
        let status = response.get("status").filter(|status| !status.is_null());
        let incomplete_reason = response
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty());
        self.output.raw_stop_reason = match incomplete_reason {
            Some(reason) => Some(format!("{}.{reason}", js_template_string(status))),
            None => status.map(|status| js_template_string(Some(status))),
        };
        let mapped_stop = map_stop_reason(status, incomplete_reason)?;
        self.output.stop_reason = mapped_stop.stop_reason;
        self.output.error_message = mapped_stop.error_message;
        if self.output.stop_reason == StopReason::Stop
            && self
                .output
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_)))
        {
            self.output.stop_reason = StopReason::ToolUse;
        }
        Ok(())
    }

    fn push_thinking_delta(&mut self, slot: ResponsesOutputSlot, delta: String) {
        self.thinking_mut(slot.content_index)
            .thinking
            .push_str(&delta);
        self.push(AssistantMessageEvent::ThinkingDelta {
            content_index: slot.content_index,
            delta,
            partial: self.output.clone(),
        });
    }

    fn push_text_delta(&mut self, slot: ResponsesOutputSlot, delta: String) {
        self.text_mut(slot.content_index).text.push_str(&delta);
        self.push(AssistantMessageEvent::TextDelta {
            content_index: slot.content_index,
            delta,
            partial: self.output.clone(),
        });
    }

    fn handle_output_item_done(&mut self, index: Option<u64>, item: &Value) -> Result<()> {
        self.apply_message_phase_stop_reason(item);
        let slot = self.get_or_create_slot(index, item);
        let item_type = str_field(item, "type");
        let Some(slot) = slot else {
            return Ok(());
        };

        if item_type == Some("reasoning") && slot.slot_type == SlotType::Thinking {
            let join_texts = |key: &str| -> String {
                item.get(key)
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .map(|part| str_field(part, "text").unwrap_or_default())
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .unwrap_or_default()
            };
            let summary_text = join_texts("summary");
            let content_text = join_texts("content");
            let block = self.thinking_mut(slot.content_index);
            if !summary_text.is_empty() {
                block.thinking = summary_text;
            } else if !content_text.is_empty() {
                block.thinking = content_text;
            }
            block.thinking_signature = Some(item.to_string());
            let thinking = block.thinking.clone();
            if let Some(id) = str_field(item, "id") {
                self.reasoning_blocks_by_id
                    .insert(id.to_string(), slot.content_index);
            }
            self.push(AssistantMessageEvent::ThinkingEnd {
                content_index: slot.content_index,
                content: thinking,
                partial: self.output.clone(),
            });
            self.output_slots.remove(&index);
        } else if item_type == Some("message") && slot.slot_type == SlotType::Text {
            let text = item
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .map(|part| {
                            let key = if str_field(part, "type") == Some("output_text") {
                                "text"
                            } else {
                                "refusal"
                            };
                            str_field(part, key).unwrap_or_default()
                        })
                        .collect::<String>()
                })
                .unwrap_or_default();
            let signature = encode_text_signature_v1(
                item.get("id").filter(|id| !id.is_null()),
                str_field(item, "phase"),
            );
            let block = self.text_mut(slot.content_index);
            block.text = text.clone();
            block.text_signature = Some(signature);
            self.push(AssistantMessageEvent::TextEnd {
                content_index: slot.content_index,
                content: text,
                partial: self.output.clone(),
            });
            self.output_slots.remove(&index);
        } else if item_type == Some("function_call")
            && slot.slot_type == SlotType::ToolCall
            && self.partial_json.contains_key(&slot.content_index)
        {
            // Finalize in-place and strip the scratch buffer so replay only
            // carries parsed arguments.
            let partial_json = self
                .partial_json
                .remove(&slot.content_index)
                .unwrap_or_default();
            let source = str_field(item, "arguments")
                .filter(|arguments| !arguments.is_empty())
                .map(str::to_string)
                .or(Some(partial_json).filter(|partial| !partial.is_empty()))
                .unwrap_or_else(|| "{}".to_string());
            let block = self.tool_call_mut(slot.content_index);
            block.arguments = parse_streaming_json(Some(&source));
            if let Some(namespace) = str_field(item, "namespace") {
                block.namespace = Some(namespace.to_string());
            }
            let tool_call = block.clone();
            self.push(AssistantMessageEvent::ToolCallEnd {
                content_index: slot.content_index,
                tool_call,
                partial: self.output.clone(),
            });
            self.output_slots.remove(&index);
        } else if item_type == Some("custom_tool_call")
            && slot.slot_type == SlotType::ToolCall
            && self.custom_inputs.contains_key(&slot.content_index)
        {
            let input = match item.get("input").filter(|input| !input.is_null()) {
                Some(input) => js_template_string(Some(input)),
                None => self.get_custom_tool_call_input(slot.content_index),
            };
            let delta = self.append_custom_tool_call_input(slot.content_index, &input, true)?;
            self.push_tool_call_delta(slot, delta);
            if let Some(namespace) = str_field(item, "namespace") {
                self.tool_call_mut(slot.content_index).namespace = Some(namespace.to_string());
            }
            self.custom_inputs.remove(&slot.content_index);
            let tool_call = self.tool_call_mut(slot.content_index).clone();
            self.push(AssistantMessageEvent::ToolCallEnd {
                content_index: slot.content_index,
                tool_call,
                partial: self.output.clone(),
            });
            self.output_slots.remove(&index);
        }
        Ok(())
    }
}

pub async fn process_responses_stream<S>(
    openai_stream: S,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    model: &Model,
    options: Option<&OpenAIResponsesStreamOptions>,
) -> Result<()>
where
    S: Stream<Item = Result<Value>>,
{
    let mut saw_terminal_response_event = false;
    let mut state = ResponsesStreamState {
        output,
        stream,
        options,
        output_slots: HashMap::new(),
        reasoning_blocks_by_id: HashMap::new(),
        partial_json: HashMap::new(),
        custom_inputs: HashMap::new(),
    };

    futures::pin_mut!(openai_stream);
    while let Some(event) = openai_stream.next().await {
        let event = event?;
        if let Some(hook) = options.and_then(|options| options.on_provider_stream_event.as_ref()) {
            hook(&event, model).await;
        }
        let index = output_index(&event);
        match str_field(&event, "type").unwrap_or_default() {
            "response.created" => {
                state.output.response_id = event
                    .get("response")
                    .and_then(|response| response.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            "response.output_item.added" => {
                let item = event.get("item").cloned().unwrap_or(Value::Null);
                state.create_slot(index, &item);
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let Some(slot) = state.get_slot(index, SlotType::Thinking) else {
                    continue;
                };
                state.push_thinking_delta(slot, js_template_string(event.get("delta")));
            }
            "response.reasoning_summary_part.done" => {
                let Some(slot) = state.get_slot(index, SlotType::Thinking) else {
                    continue;
                };
                state.push_thinking_delta(slot, "\n\n".to_string());
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let Some(slot) = state.get_slot(index, SlotType::Text) else {
                    continue;
                };
                state.push_text_delta(slot, js_template_string(event.get("delta")));
            }
            "response.function_call_arguments.delta" => {
                let Some(slot) = state.get_slot(index, SlotType::ToolCall) else {
                    continue;
                };
                let Some(partial_json) = state.partial_json.get_mut(&slot.content_index) else {
                    continue;
                };
                let delta = js_template_string(event.get("delta"));
                partial_json.push_str(&delta);
                let arguments = parse_streaming_json(Some(partial_json));
                state.tool_call_mut(slot.content_index).arguments = arguments;
                state.push_tool_call_delta(slot, Some(delta));
            }
            "response.function_call_arguments.done" => {
                let Some(slot) = state.get_slot(index, SlotType::ToolCall) else {
                    continue;
                };
                let Some(partial_json) = state.partial_json.get_mut(&slot.content_index) else {
                    continue;
                };
                let arguments = str_field(&event, "arguments")
                    .unwrap_or_default()
                    .to_string();
                let previous_partial_json = std::mem::replace(partial_json, arguments.clone());
                state.tool_call_mut(slot.content_index).arguments =
                    parse_streaming_json(Some(&arguments));

                if let Some(delta) = arguments.strip_prefix(previous_partial_json.as_str())
                    && !delta.is_empty()
                {
                    state.push_tool_call_delta(slot, Some(delta.to_string()));
                }
            }
            "response.custom_tool_call_input.delta" => {
                let Some(slot) = state.get_slot(index, SlotType::ToolCall) else {
                    continue;
                };
                if !state.custom_inputs.contains_key(&slot.content_index) {
                    continue;
                }
                let next = state.get_custom_tool_call_input(slot.content_index)
                    + &js_template_string(event.get("delta"));
                let delta =
                    state.append_custom_tool_call_input(slot.content_index, &next, false)?;
                state.push_tool_call_delta(slot, delta);
            }
            "response.custom_tool_call_input.done" => {
                let Some(slot) = state.get_slot(index, SlotType::ToolCall) else {
                    continue;
                };
                if !state.custom_inputs.contains_key(&slot.content_index) {
                    continue;
                }
                let input = js_template_string(event.get("input"));
                let delta =
                    state.append_custom_tool_call_input(slot.content_index, &input, true)?;
                state.push_tool_call_delta(slot, delta);
            }
            "response.output_item.done" => {
                let item = event.get("item").cloned().unwrap_or(Value::Null);
                state.handle_output_item_done(index, &item)?;
            }
            "response.completed" | "response.incomplete" => {
                saw_terminal_response_event = true;
                let response = event.get("response").cloned().unwrap_or(Value::Null);
                state.finalize_response(model, &response)?;
            }
            "error" => {
                return Err(Error::message(format!(
                    "Error Code {}: {}",
                    js_template_string(event.get("code")),
                    js_template_string(event.get("message"))
                )));
            }
            "response.failed" => {
                let response = event.get("response");
                state.output.raw_stop_reason = response
                    .and_then(|response| response.get("status"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let error = response.and_then(|response| response.get("error"));
                let details_reason = response
                    .and_then(|response| response.get("incomplete_details"))
                    .and_then(|details| details.get("reason"));
                let message = if let Some(error) = error.filter(|error| js_truthy(Some(error))) {
                    let code = error
                        .get("code")
                        .filter(|code| js_truthy(Some(code)))
                        .map(|code| js_template_string(Some(code)))
                        .unwrap_or_else(|| "unknown".to_string());
                    let message = error
                        .get("message")
                        .filter(|message| js_truthy(Some(message)))
                        .map(|message| js_template_string(Some(message)))
                        .unwrap_or_else(|| "no message".to_string());
                    format!("{code}: {message}")
                } else if js_truthy(details_reason) {
                    format!("incomplete: {}", js_template_string(details_reason))
                } else {
                    "Unknown error (no error details in response)".to_string()
                };
                return Err(Error::message(message));
            }
            _ => {}
        }
    }
    if !saw_terminal_response_event {
        return Err(Error::message(
            "OpenAI Responses stream ended before a terminal response event",
        ));
    }
    // The agent runs every tool call in the final message. Refuse to hand over calls whose
    // output_item.done never arrived: their arguments may be cut off or mixed up, e.g. when a
    // non-compliant server omits output_index. Finished calls have their scratch buffers removed.
    if state.output.stop_reason == StopReason::ToolUse {
        for (content_index, block) in state.output.content.iter().enumerate() {
            let AssistantContent::ToolCall(tool_call) = block else {
                continue;
            };
            if state.partial_json.contains_key(&content_index)
                || state.custom_inputs.contains_key(&content_index)
            {
                return Err(Error::message(format!(
                    "OpenAI Responses stream completed with an unfinished tool call: {} ({})",
                    tool_call.name, tool_call.id
                )));
            }
        }
    }
    Ok(())
}

struct MappedStop {
    stop_reason: StopReason,
    error_message: Option<String>,
}

fn map_stop_reason(status: Option<&Value>, incomplete_reason: Option<&str>) -> Result<MappedStop> {
    let stop = |stop_reason| {
        Ok(MappedStop {
            stop_reason,
            error_message: None,
        })
    };
    let Some(status) = status.filter(|status| js_truthy(Some(status))) else {
        return stop(StopReason::Stop);
    };
    match status.as_str() {
        Some("completed") => stop(StopReason::Stop),
        Some("incomplete") => {
            if incomplete_reason == Some("max_output_tokens") {
                return stop(StopReason::Length);
            }
            Ok(MappedStop {
                stop_reason: StopReason::Error,
                error_message: Some(match incomplete_reason {
                    Some(reason) => format!("Response incomplete: {reason}"),
                    None => "Response incomplete without a provider reason".to_string(),
                }),
            })
        }
        Some("failed" | "cancelled") => stop(StopReason::Error),
        // These two are wonky ...
        Some("in_progress" | "queued") => stop(StopReason::Stop),
        _ => Err(Error::message(format!(
            "Unhandled stop reason: {}",
            js_template_string(Some(status))
        ))),
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::{Value, json};

    use super::*;
    use crate::api::constrained_sampling::make_strict_json_schema;
    use crate::api::openai_client::test_support::{context, gpt5_mini, model, pending_output};
    use crate::types::{AssistantMessageEvent, Tool};

    const OPENAI_PROVIDERS: &[&str] = &["openai", "openai-codex", "opencode"];

    fn usage() -> Value {
        json!({
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        })
    }

    fn events(values: Vec<Value>) -> impl Stream<Item = Result<Value>> {
        futures::stream::iter(values.into_iter().map(Ok))
    }

    async fn drain(stream: &AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
        stream.end(None);
        stream.clone().collect().await
    }

    async fn process(
        values: Vec<Value>,
        model: &Model,
        options: Option<&OpenAIResponsesStreamOptions>,
    ) -> (Result<()>, AssistantMessage, Vec<AssistantMessageEvent>) {
        let mut output = pending_output(model);
        let stream = AssistantMessageEventStream::new();
        let result =
            process_responses_stream(events(values), &mut output, &stream, model, options).await;
        let events = drain(&stream).await;
        (result, output, events)
    }

    fn grammar_options(name: &str, property: &str) -> OpenAIResponsesStreamOptions {
        OpenAIResponsesStreamOptions {
            grammar_tool_input_properties: Some([(name.to_string(), property.to_string())].into()),
            ..Default::default()
        }
    }

    fn convert_options(name: &str, property: &str) -> ConvertResponsesMessagesOptions {
        ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some([(name.to_string(), property.to_string())].into()),
            ..Default::default()
        }
    }

    fn find<'a>(items: &'a [Value], item_type: &str) -> Option<&'a Value> {
        items.iter().find(|item| item["type"] == item_type)
    }

    fn codex_model() -> Model {
        let mut model = crate::api::openai_client::test_support::openai_model("gpt-5.5");
        model.provider = "openai-codex".to_string();
        model.api = "openai-codex-responses".to_string();
        model
    }

    // openai-responses-empty-tool-result.test.ts
    #[test]
    fn empty_tool_results_use_the_no_output_placeholder() {
        let model = crate::api::openai_client::test_support::openai_model("gpt-4o-mini");
        let ctx = context(json!({
            "messages": [
                { "role": "user", "content": "Run the command", "timestamp": 1 },
                {
                    "role": "assistant",
                    "content": [{ "type": "toolCall", "id": "tool-1", "name": "bash", "arguments": { "command": "true" } }],
                    "api": model.api, "provider": model.provider, "model": model.id,
                    "usage": usage(), "stopReason": "toolUse", "timestamp": 2,
                },
                {
                    "role": "toolResult", "toolCallId": "tool-1", "toolName": "bash",
                    "content": [{ "type": "text", "text": "" }], "isError": false, "timestamp": 3,
                },
            ],
        }));
        let input = convert_responses_messages(&model, &ctx, OPENAI_PROVIDERS, None).unwrap();
        let output = find(&input, "function_call_output").unwrap();
        assert_eq!(output["output"], "(no tool output)");
    }

    // openai-responses-foreign-toolcall-id.test.ts
    #[test]
    fn foreign_copilot_item_ids_are_hashed_into_fc_ids() {
        const RAW: &str = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";
        let model = codex_model();
        let ctx = context(json!({
            "systemPrompt": "You are concise.",
            "messages": [
                { "role": "user", "content": "Use the tool.", "timestamp": 1 },
                {
                    "role": "assistant",
                    "content": [{ "type": "toolCall", "id": RAW, "name": "edit", "arguments": { "path": "src/styles/app.css" } }],
                    "api": "openai-responses", "provider": "github-copilot", "model": "gpt-5.5",
                    "usage": usage(), "stopReason": "toolUse", "timestamp": 2,
                },
                {
                    "role": "toolResult", "toolCallId": RAW, "toolName": "edit",
                    "content": [{ "type": "text", "text": "ok" }], "isError": false, "timestamp": 3,
                },
            ],
        }));
        let input = convert_responses_messages(&model, &ctx, OPENAI_PROVIDERS, None).unwrap();
        let call = find(&input, "function_call").unwrap();
        let expected = format!("fc_{}", short_hash(RAW.split('|').nth(1).unwrap()));
        assert_eq!(call["id"], json!(expected));
        let id = call["id"].as_str().unwrap();
        assert!(id.len() <= 64);
        assert!(
            id[3..]
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        );
    }

    // openai-responses-message-id.test.ts
    #[test]
    fn fallback_message_ids_are_unique_per_text_block() {
        let model = codex_model();
        let ctx = context(json!({
            "systemPrompt": "You are concise.",
            "messages": [
                { "role": "user", "content": "hello", "timestamp": 1 },
                {
                    "role": "assistant",
                    "content": [
                        { "type": "thinking", "thinking": "private reasoning" },
                        { "type": "text", "text": "visible answer" },
                    ],
                    "api": "anthropic-messages", "provider": "anthropic", "model": "claude-opus-4-8",
                    "usage": usage(), "stopReason": "stop", "timestamp": 2,
                },
            ],
        }));
        let input = convert_responses_messages(&model, &ctx, OPENAI_PROVIDERS, None).unwrap();
        let ids: Vec<&str> = input
            .iter()
            .filter(|item| item["type"] == "message")
            .filter_map(|item| item["id"].as_str())
            .collect();
        assert_eq!(ids, ["msg_pi_1", "msg_pi_1_1"]);
    }

    fn namespace_model() -> Model {
        model(json!({
            "id": "gpt-5.4", "name": "GPT-5.4", "api": "openai-responses", "provider": "openai",
            "baseUrl": "https://api.openai.com/v1", "reasoning": true, "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 400000, "maxTokens": 128000,
        }))
    }

    fn function_call_events() -> Vec<Value> {
        vec![
            json!({ "type": "response.output_item.added", "sequence_number": 0, "output_index": 0,
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup", "arguments": "" } }),
            json!({ "type": "response.output_item.done", "sequence_number": 1, "output_index": 0,
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup",
                    "arguments": "{\"value\":\"hello\"}", "namespace": "dynamic_tools" } }),
            json!({ "type": "response.completed", "sequence_number": 2, "response": { "id": "resp_test", "status": "completed" } }),
        ]
    }

    fn assistant_context(output: &AssistantMessage) -> TranscriptContext {
        context(
            json!({ "messages": [serde_json::to_value(Message::Assistant(output.clone())).unwrap()] }),
        )
    }

    fn tool_call(output: &AssistantMessage) -> &ToolCall {
        match &output.content[0] {
            AssistantContent::ToolCall(tool_call) => tool_call,
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    // openai-responses-namespace.test.ts
    #[tokio::test]
    async fn omits_an_absent_error_message() {
        let model = namespace_model();
        let (result, output, _) = process(function_call_events(), &model, None).await;
        result.unwrap();
        assert_eq!(output.error_message, None);
    }

    #[tokio::test]
    async fn round_trips_a_function_namespace_received_on_done() {
        let model = namespace_model();
        let (result, output, _) = process(function_call_events(), &model, None).await;
        result.unwrap();
        let call = tool_call(&output);
        assert_eq!(call.id, "call_test|fc_test");
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments, json!({ "value": "hello" }));
        assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));

        let replayed =
            convert_responses_messages(&model, &assistant_context(&output), &["openai"], None)
                .unwrap();
        assert_eq!(
            find(&replayed, "function_call").unwrap(),
            &json!({
                "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup",
                "arguments": "{\"value\":\"hello\"}", "namespace": "dynamic_tools",
            })
        );
    }

    #[tokio::test]
    async fn round_trips_a_custom_tool_namespace_received_on_done() {
        let model = namespace_model();
        let values = vec![
            json!({ "type": "response.output_item.added", "sequence_number": 0, "output_index": 0,
                "item": { "type": "custom_tool_call", "id": "ctc_test", "call_id": "call_test", "name": "query", "input": "" } }),
            json!({ "type": "response.output_item.done", "sequence_number": 1, "output_index": 0,
                "item": { "type": "custom_tool_call", "id": "ctc_test", "call_id": "call_test", "name": "query",
                    "input": "hello", "namespace": "dynamic_tools" } }),
            json!({ "type": "response.completed", "sequence_number": 2, "response": { "id": "resp_test", "status": "completed" } }),
        ];
        let options = grammar_options("query", "input");
        let (result, output, _) = process(values, &model, Some(&options)).await;
        result.unwrap();
        let call = tool_call(&output);
        assert_eq!(call.id, "call_test|ctc_test");
        assert_eq!(call.arguments, json!({ "input": "hello" }));
        assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));

        let replayed = convert_responses_messages(
            &model,
            &assistant_context(&output),
            &["openai"],
            Some(&convert_options("query", "input")),
        )
        .unwrap();
        assert_eq!(
            find(&replayed, "custom_tool_call").unwrap(),
            &json!({
                "type": "custom_tool_call", "id": "ctc_test", "call_id": "call_test", "name": "query",
                "input": "hello", "namespace": "dynamic_tools",
            })
        );
    }

    #[test]
    fn drops_namespaces_when_the_target_cannot_replay_them() {
        let base = namespace_model();
        let mut output = pending_output(&base);
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: "call_function|fc_test".to_string(),
            name: "lookup".to_string(),
            arguments: json!({ "value": "hello" }),
            thought_signature: None,
            namespace: Some("dynamic_tools".to_string()),
        }));
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: "call_custom|ctc_test".to_string(),
            name: "query".to_string(),
            arguments: json!({ "input": "hello" }),
            thought_signature: None,
            namespace: Some("dynamic_tools".to_string()),
        }));
        let mut other_id = base.clone();
        other_id.id = "gpt-5.2".to_string();
        let mut azure = base.clone();
        azure.provider = "azure-openai-responses".to_string();
        let mut codex = base.clone();
        codex.api = "openai-codex-responses".to_string();
        codex.provider = "openai-codex".to_string();
        codex.id = "gpt-5.3-codex-spark".to_string();
        for target in [other_id, azure, codex] {
            let replayed = convert_responses_messages(
                &target,
                &assistant_context(&output),
                &["openai"],
                Some(&convert_options("query", "input")),
            )
            .unwrap();
            let function_call = find(&replayed, "function_call").unwrap();
            let custom_call = find(&replayed, "custom_tool_call").unwrap();
            assert!(function_call.get("namespace").is_none());
            assert!(custom_call.get("namespace").is_none());
        }
    }

    #[test]
    fn ordinary_function_calls_get_no_namespace() {
        let model = namespace_model();
        let mut output = pending_output(&model);
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: "call_test|fc_test".to_string(),
            name: "lookup".to_string(),
            arguments: json!({ "value": "hello" }),
            thought_signature: None,
            namespace: None,
        }));
        let replayed =
            convert_responses_messages(&model, &assistant_context(&output), &["openai"], None)
                .unwrap();
        assert!(
            find(&replayed, "function_call")
                .unwrap()
                .get("namespace")
                .is_none()
        );
    }

    // openai-responses-partial-json-cleanup.test.ts
    #[tokio::test]
    async fn persisted_tool_calls_carry_only_parsed_arguments() {
        let model = gpt5_mini("openai-responses");
        let arguments_json = "{\"path\":\"README.md\",\"content\":\"updated\"}";
        let values = vec![
            json!({ "type": "response.output_item.added",
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": "" } }),
            json!({ "type": "response.function_call_arguments.delta", "delta": "{\"path\":\"README.md\"" }),
            json!({ "type": "response.function_call_arguments.delta", "delta": ",\"content\":\"updated\"}" }),
            json!({ "type": "response.function_call_arguments.done", "arguments": arguments_json }),
            json!({ "type": "response.output_item.done",
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": arguments_json } }),
            json!({ "type": "response.completed", "sequence_number": 5, "response": { "id": "resp_test", "status": "completed" } }),
        ];
        let (result, output, events) = process(values, &model, None).await;
        result.unwrap();
        assert_eq!(output.content.len(), 1);
        let persisted = tool_call(&output);
        assert_eq!(
            persisted.arguments,
            json!({ "path": "README.md", "content": "updated" })
        );
        let persisted_json = serde_json::to_value(persisted).unwrap();
        assert!(persisted_json.get("partialJson").is_none());
        let end = events
            .iter()
            .find_map(|event| match event {
                AssistantMessageEvent::ToolCallEnd { tool_call, .. } => Some(tool_call),
                _ => None,
            })
            .unwrap();
        assert_eq!(end, persisted);
    }

    // openai-responses-terminal-event.test.ts
    fn early_eof_events() -> Vec<Value> {
        vec![
            json!({ "type": "response.created", "sequence_number": 0, "response": { "id": "resp_early_eof" } }),
            json!({ "type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_early_eof", "summary": [] } }),
            json!({ "type": "response.reasoning_text.delta", "sequence_number": 2, "output_index": 0, "content_index": 0,
                "item_id": "rs_early_eof", "delta": "partial reasoning before the stream ends" }),
        ]
    }

    fn incomplete_events(reason: &str) -> Vec<Value> {
        vec![json!({
            "type": "response.incomplete", "sequence_number": 0,
            "response": {
                "id": "resp_incomplete", "status": "incomplete", "incomplete_details": { "reason": reason },
                "usage": { "input_tokens": 30, "output_tokens": 12, "total_tokens": 42,
                    "input_tokens_details": { "cached_tokens": 5 } },
            },
        })]
    }

    fn phased_message_events(phases: [&str; 2], incomplete: bool) -> Vec<Value> {
        let mut values = vec![
            json!({ "type": "response.output_item.added", "sequence_number": 0, "output_index": 0,
                "item": { "type": "message", "id": "msg_phase", "role": "assistant", "status": "in_progress",
                    "content": [], "phase": phases[0] } }),
            json!({ "type": "response.output_item.done", "sequence_number": 1, "output_index": 0,
                "item": { "type": "message", "id": "msg_phase", "role": "assistant", "status": "completed",
                    "content": [{ "type": "output_text", "text": "answer", "annotations": [] }], "phase": phases[1] } }),
        ];
        values.push(if incomplete {
            json!({ "type": "response.incomplete", "sequence_number": 2, "response": { "id": "resp_phase",
                "status": "incomplete", "incomplete_details": { "reason": "max_output_tokens" } } })
        } else {
            json!({ "type": "response.completed", "sequence_number": 2, "response": { "id": "resp_phase", "status": "completed" } })
        });
        values
    }

    fn observed_stop_reasons(events: &[AssistantMessageEvent]) -> Vec<StopReason> {
        events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::TextStart { partial, .. }
                | AssistantMessageEvent::TextDelta { partial, .. }
                | AssistantMessageEvent::TextEnd { partial, .. }
                | AssistantMessageEvent::ThinkingStart { partial, .. }
                | AssistantMessageEvent::ThinkingDelta { partial, .. }
                | AssistantMessageEvent::ThinkingEnd { partial, .. }
                | AssistantMessageEvent::ToolCallStart { partial, .. }
                | AssistantMessageEvent::ToolCallDelta { partial, .. }
                | AssistantMessageEvent::ToolCallEnd { partial, .. }
                | AssistantMessageEvent::Start { partial } => Some(partial.stop_reason),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn rejects_streams_that_end_before_a_terminal_event() {
        let model = gpt5_mini("openai-responses");
        let (result, _, _) = process(early_eof_events(), &model, None).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream ended before a terminal response event"
        );
    }

    #[tokio::test]
    async fn rejects_completed_streams_with_unfinished_tool_calls() {
        let model = gpt5_mini("openai-responses");
        let values = vec![
            json!({ "type": "response.output_item.added", "sequence_number": 0, "output_index": 0,
                "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "bash", "arguments": "" } }),
            json!({ "type": "response.function_call_arguments.delta", "sequence_number": 1, "output_index": 0,
                "item_id": "fc_1", "delta": "{\"command\":\"rm -rf /tmp/build" }),
            json!({ "type": "response.completed", "sequence_number": 2, "response": { "id": "resp_unfinished", "status": "completed" } }),
        ];
        let (result, _, _) = process(values, &model, None).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream completed with an unfinished tool call: bash (call_1|fc_1)"
        );
    }

    // https://github.com/earendil-works/pi/issues/9974
    #[tokio::test]
    async fn rejects_parallel_tool_calls_without_output_index() {
        let model = gpt5_mini("openai-responses");
        let call = |n: &str, arguments: &str| json!({ "type": "function_call", "id": format!("fc_{n}"), "call_id": format!("call_{n}"), "name": "bash", "arguments": arguments });
        let values = vec![
            json!({ "type": "response.output_item.added", "item": call("a", "") }),
            json!({ "type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": "{\"command\":\"echo a\"}" }),
            json!({ "type": "response.output_item.added", "item": call("b", "") }),
            json!({ "type": "response.function_call_arguments.delta", "item_id": "fc_b", "delta": "{\"command\":\"echo b\"}" }),
            json!({ "type": "response.output_item.done", "item": call("a", "{\"command\":\"echo a\"}") }),
            json!({ "type": "response.output_item.done", "item": call("b", "{\"command\":\"echo b\"}") }),
            json!({ "type": "response.completed", "response": { "id": "resp_no_output_index", "status": "completed" } }),
        ];
        let (result, _, _) = process(values, &model, None).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream completed with an unfinished tool call: bash (call_a|fc_a)"
        );
    }

    #[tokio::test]
    async fn tracks_message_phases() {
        let model = gpt5_mini("openai-responses");
        for (phases, expected) in [
            (
                ["commentary", "commentary"],
                [StopReason::Pending, StopReason::Pending],
            ),
            (
                ["final_answer", "final_answer"],
                [StopReason::Stop, StopReason::Stop],
            ),
            (
                ["commentary", "final_answer"],
                [StopReason::Pending, StopReason::Stop],
            ),
        ] {
            let (result, output, events) =
                process(phased_message_events(phases, false), &model, None).await;
            result.unwrap();
            assert_eq!(observed_stop_reasons(&events), expected);
            assert_eq!(output.stop_reason, StopReason::Stop);
        }
    }

    #[tokio::test]
    async fn incomplete_terminal_reasons_replace_a_provisional_final_answer_stop() {
        let model = gpt5_mini("openai-responses");
        let (result, output, events) = process(
            phased_message_events(["final_answer", "final_answer"], true),
            &model,
            None,
        )
        .await;
        result.unwrap();
        assert_eq!(
            observed_stop_reasons(&events),
            [StopReason::Stop, StopReason::Stop]
        );
        assert_eq!(output.stop_reason, StopReason::Length);
    }

    #[tokio::test]
    async fn finalizes_completed_terminal_events_as_stop() {
        let model = gpt5_mini("openai-responses");
        let values = vec![json!({
            "type": "response.completed", "sequence_number": 0,
            "response": { "id": "resp_completed", "status": "completed",
                "usage": { "input_tokens": 20, "output_tokens": 7, "total_tokens": 27,
                    "input_tokens_details": { "cached_tokens": 2, "cache_write_tokens": 3 } } },
        })];
        let (result, output, _) = process(values, &model, None).await;
        result.unwrap();
        assert_eq!(output.response_id.as_deref(), Some("resp_completed"));
        assert_eq!(output.stop_reason, StopReason::Stop);
        assert_eq!(output.raw_stop_reason.as_deref(), Some("completed"));
        assert_eq!(
            (
                output.usage.input,
                output.usage.output,
                output.usage.cache_read,
                output.usage.cache_write,
                output.usage.total_tokens
            ),
            (15, 7, 2, 3, 27)
        );
    }

    #[tokio::test]
    async fn finalizes_incomplete_terminal_events() {
        let model = gpt5_mini("openai-responses");
        let (result, output, _) =
            process(incomplete_events("max_output_tokens"), &model, None).await;
        result.unwrap();
        assert_eq!(output.response_id.as_deref(), Some("resp_incomplete"));
        assert_eq!(output.stop_reason, StopReason::Length);
        assert_eq!(
            output.raw_stop_reason.as_deref(),
            Some("incomplete.max_output_tokens")
        );
        assert_eq!(
            (
                output.usage.input,
                output.usage.output,
                output.usage.cache_read,
                output.usage.cache_write,
                output.usage.total_tokens
            ),
            (25, 12, 5, 0, 42)
        );

        for reason in ["content_filter", "max_time_limit"] {
            let (result, output, _) = process(incomplete_events(reason), &model, None).await;
            result.unwrap();
            assert_eq!(output.stop_reason, StopReason::Error);
            assert_eq!(output.raw_stop_reason, Some(format!("incomplete.{reason}")));
            assert_eq!(
                output.error_message,
                Some(format!("Response incomplete: {reason}"))
            );
        }
    }

    #[tokio::test]
    async fn rejects_failed_terminal_events_with_the_provider_error() {
        let model = gpt5_mini("openai-responses");
        let values = vec![json!({
            "type": "response.failed", "sequence_number": 0,
            "response": { "id": "resp_failed", "status": "failed", "error": { "code": "server_error", "message": "boom" } },
        })];
        let (result, output, _) = process(values, &model, None).await;
        assert_eq!(result.unwrap_err().to_string(), "server_error: boom");
        assert_eq!(output.raw_stop_reason.as_deref(), Some("failed"));
    }

    #[tokio::test]
    async fn error_events_become_coded_errors() {
        let model = gpt5_mini("openai-responses");
        let values = vec![json!({ "type": "error", "code": "rate_limit", "message": "slow down" })];
        let (result, _, _) = process(values, &model, None).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "Error Code rate_limit: slow down"
        );
    }

    #[tokio::test]
    async fn backfills_reasoning_signatures_from_the_terminal_response() {
        let model = gpt5_mini("openai-responses");
        let values = vec![
            json!({ "type": "response.output_item.added", "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_1", "summary": [] } }),
            json!({ "type": "response.output_item.done", "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_1", "summary": [{ "type": "summary_text", "text": "thought" }] } }),
            json!({ "type": "response.completed", "response": { "status": "completed",
                "output": [{ "type": "reasoning", "id": "rs_1", "encrypted_content": "secret" }] } }),
        ];
        let (result, output, _) = process(values, &model, None).await;
        result.unwrap();
        let AssistantContent::Thinking(thinking) = &output.content[0] else {
            panic!("expected thinking");
        };
        assert_eq!(thinking.thinking, "thought");
        let signature: Value =
            serde_json::from_str(thinking.thinking_signature.as_deref().unwrap()).unwrap();
        assert_eq!(signature["encrypted_content"], "secret");
        assert_eq!(signature["id"], "rs_1");
    }

    // openai-responses-tool-result-images.test.ts (offline payload check)
    #[test]
    fn tool_result_images_stay_in_function_call_output() {
        let mut model = gpt5_mini("openai-responses");
        model.input = vec![ModelInput::Text, ModelInput::Image];
        let ctx = context(json!({
            "messages": [
                { "role": "user", "content": "Call the tool", "timestamp": 1 },
                {
                    "role": "assistant",
                    "content": [{ "type": "toolCall", "id": "call_1|fc_1", "name": "get_circle_with_description", "arguments": {} }],
                    "api": "openai-responses", "provider": "openai", "model": "gpt-5-mini",
                    "usage": usage(), "stopReason": "toolUse", "timestamp": 2,
                },
                {
                    "role": "toolResult", "toolCallId": "call_1|fc_1", "toolName": "get_circle_with_description",
                    "content": [
                        { "type": "text", "text": "A red circle with a diameter of 100 pixels." },
                        { "type": "image", "data": "aW1n", "mimeType": "image/png" },
                    ],
                    "isError": false, "timestamp": 3,
                },
            ],
        }));
        let input = convert_responses_messages(&model, &ctx, OPENAI_PROVIDERS, None).unwrap();
        let index = input
            .iter()
            .position(|item| item["type"] == "function_call_output")
            .unwrap();
        assert_eq!(
            input[index]["output"],
            json!([
                { "type": "input_text", "text": "A red circle with a diameter of 100 pixels." },
                { "type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,aW1n" },
            ])
        );
        assert!(input[index + 1..].iter().all(|item| item["role"] != "user"));

        model.input = vec![ModelInput::Text];
        let input = convert_responses_messages(&model, &ctx, OPENAI_PROVIDERS, None).unwrap();
        // transformMessages() replaces images the model cannot read with a placeholder.
        assert_eq!(
            find(&input, "function_call_output").unwrap()["output"],
            "A red circle with a diameter of 100 pixels.\n(tool image omitted: model does not support images)"
        );
    }

    // constrained-sampling.test.ts
    fn make_model() -> Model {
        model(json!({
            "id": "gpt-test", "name": "GPT Test", "api": "openai-responses", "provider": "openai",
            "baseUrl": "https://api.openai.com/v1", "reasoning": false, "input": ["text", "image"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 128000, "maxTokens": 4096,
        }))
    }

    fn make_tool(extra: Value) -> Tool {
        let mut tool = json!({
            "name": "sample_tool",
            "description": "Sample tool",
            "parameters": {
                "type": "object",
                "properties": { "payload": { "type": "string" } },
                "required": ["payload"],
                "additionalProperties": false,
            },
        });
        for (key, value) in extra.as_object().unwrap() {
            tool[key] = value.clone();
        }
        serde_json::from_value(tool).unwrap()
    }

    fn tools_options(
        strict_mode: Option<bool>,
        grammar: Option<bool>,
    ) -> ConvertResponsesToolsOptions {
        ConvertResponsesToolsOptions {
            supports_strict_mode: strict_mode,
            supports_openai_grammar_tools: grammar,
            ..Default::default()
        }
    }

    #[test]
    fn converts_supported_constraints_and_falls_back_when_unsupported() {
        let strict = convert_responses_tools(
            &[make_tool(
                json!({ "constrainedSampling": { "type": "json_schema", "strict": "prefer" } }),
            )],
            None,
        )
        .unwrap();
        assert_eq!(strict[0]["type"], "function");
        assert_eq!(strict[0]["strict"], true);

        let error = convert_responses_tools(
            &[make_tool(
                json!({ "constrainedSampling": { "type": "json_schema", "strict": "require" } }),
            )],
            Some(&tools_options(Some(false), None)),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Tool \"sample_tool\" requires JSON-schema constrained sampling")
        );

        let grammar_tool = make_tool(json!({
            "constrainedSampling": { "type": "grammar", "variants": { "openai_lark": "start: /[a-z]+/" } },
        }));
        let converted = convert_responses_tools(
            std::slice::from_ref(&grammar_tool),
            Some(&tools_options(None, Some(true))),
        )
        .unwrap();
        assert_eq!(converted[0]["type"], "custom");
        assert_eq!(converted[0]["name"], "sample_tool");
        assert_eq!(
            converted[0]["format"],
            json!({ "type": "grammar", "syntax": "lark", "definition": "start: /[a-z]+/" })
        );
        let error = convert_responses_tools(
            &[make_tool(
                json!({ "constrainedSampling": { "type": "grammar", "variants": {} } }),
            )],
            Some(&tools_options(None, Some(true))),
        )
        .unwrap_err();
        assert!(error.to_string().contains(
            "Tool \"sample_tool\" cannot use grammar constrained sampling: no supported grammar variant was provided"
        ));

        let fallback = convert_responses_tools(
            std::slice::from_ref(&grammar_tool),
            Some(&tools_options(Some(false), Some(false))),
        )
        .unwrap();
        assert_eq!(fallback[0]["type"], "function");
        assert!(fallback[0].get("strict").is_none());

        assert_eq!(
            convert_responses_tools(&[make_tool(json!({ "constrainedSampling": false }))], None)
                .unwrap(),
            convert_responses_tools(&[make_tool(json!({}))], None).unwrap()
        );
    }

    #[test]
    fn unconvertible_strict_schemas_fall_back_to_non_strict() {
        let parameters = json!({
            "type": "object",
            "properties": { "child": { "$ref": "https://example.com/child.json" } },
            "required": ["child"],
        });
        assert!(make_strict_json_schema(&parameters, None).is_err());
        let tool = make_tool(json!({
            "parameters": parameters,
            "constrainedSampling": { "type": "json_schema", "strict": "prefer" },
        }));
        let converted = convert_responses_tools(
            std::slice::from_ref(&tool),
            Some(&tools_options(Some(true), None)),
        )
        .unwrap();
        assert_eq!(converted[0]["strict"], false);
        assert_eq!(converted[0]["parameters"], tool.parameters);
    }

    fn grammar_replay_context(
        api: &str,
        provider: &str,
        model_id: &str,
        arguments: Value,
    ) -> TranscriptContext {
        context(json!({
            "messages": [
                {
                    "role": "assistant", "api": api, "provider": provider, "model": model_id,
                    "content": [{ "type": "toolCall", "id": "call_1|ctc_1", "name": "sample_tool", "arguments": arguments }],
                    "usage": usage(), "stopReason": "toolUse", "timestamp": 1,
                },
                {
                    "role": "toolResult", "toolCallId": "call_1|ctc_1", "toolName": "sample_tool",
                    "content": [{ "type": "text", "text": "done" }], "isError": false, "timestamp": 2,
                },
            ],
        }))
    }

    #[test]
    fn replays_grammar_calls_as_custom_items() {
        let options = convert_options("sample_tool", "payload");
        for invalid in [json!({}), json!({ "payload": 42 })] {
            let ctx = grammar_replay_context("openai-responses", "openai", "gpt-test", invalid);
            let error =
                convert_responses_messages(&make_model(), &ctx, &["openai"], Some(&options))
                    .unwrap_err();
            assert!(error.to_string().contains(
                "Grammar tool call \"sample_tool\" requires argument \"payload\" to be a string"
            ));
        }
        let ctx = grammar_replay_context(
            "openai-responses",
            "openai",
            "gpt-test",
            json!({ "payload": "abc" }),
        );
        let messages =
            convert_responses_messages(&make_model(), &ctx, &["openai"], Some(&options)).unwrap();
        assert!(messages.contains(&json!({
            "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": "abc",
        })));
        assert!(messages.contains(&json!({
            "type": "custom_tool_call_output", "call_id": "call_1", "output": "done",
        })));
    }

    // earendil-works/radius#115: a gateway forwards another model's history as a foreign provider.
    #[test]
    fn drops_foreign_item_ids_when_replaying_grammar_calls() {
        let options = convert_options("sample_tool", "payload");
        let ctx = grammar_replay_context(
            "pi-messages",
            "radius",
            "gpt-other",
            json!({ "payload": "abc" }),
        );
        let messages =
            convert_responses_messages(&make_model(), &ctx, &["openai"], Some(&options)).unwrap();
        let call = find(&messages, "custom_tool_call").unwrap();
        assert_eq!(call["call_id"], "call_1");
        assert_eq!(call["input"], "abc");
        assert!(call.get("id").is_none());
    }

    #[tokio::test]
    async fn starts_custom_tool_calls_with_their_initial_input() {
        let values = vec![
            json!({ "type": "response.output_item.added", "output_index": 0,
                "item": { "type": "custom_tool_call", "call_id": "call_1", "id": "ctc_1", "name": "sample_tool", "input": "a" } }),
            json!({ "type": "response.custom_tool_call_input.delta", "output_index": 0, "item_id": "ctc_1", "delta": "b" }),
            json!({ "type": "response.custom_tool_call_input.done", "output_index": 0, "item_id": "ctc_1", "input": "abc" }),
            json!({ "type": "response.output_item.done", "output_index": 0,
                "item": { "type": "custom_tool_call", "call_id": "call_1", "id": "ctc_1", "name": "sample_tool", "input": "abc" } }),
            json!({ "type": "response.completed",
                "response": { "status": "completed", "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 } } }),
        ];
        let options = grammar_options("sample_tool", "payload");
        let (result, output, events) = process(values, &make_model(), Some(&options)).await;
        result.unwrap();
        assert_eq!(output.stop_reason, StopReason::ToolUse);
        let starts: Vec<Value> = events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallStart {
                    content_index,
                    partial,
                } => match &partial.content[*content_index] {
                    AssistantContent::ToolCall(call) => Some(call.arguments.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(starts, [json!({ "payload": "a" })]);
        assert_eq!(
            serde_json::to_value(&output.content).unwrap(),
            json!([{ "type": "toolCall", "id": "call_1|ctc_1", "name": "sample_tool", "arguments": { "payload": "abc" } }])
        );
        let deltas: String = events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&deltas).unwrap(),
            json!({ "payload": "abc" })
        );
    }

    #[test]
    fn text_signatures_round_trip_ids_and_phases() {
        let model = make_model();
        let mut output = pending_output(&model);
        output.content.push(AssistantContent::Text(TextContent {
            text: "hi".to_string(),
            text_signature: Some(encode_text_signature_v1(
                Some(&json!("msg_1")),
                Some("final_answer"),
            )),
        }));
        output.content.push(AssistantContent::Text(TextContent {
            text: "legacy".to_string(),
            text_signature: Some("x".repeat(70)),
        }));
        let replayed =
            convert_responses_messages(&model, &assistant_context(&output), &["openai"], None)
                .unwrap();
        assert_eq!(replayed[0]["id"], "msg_1");
        assert_eq!(replayed[0]["phase"], "final_answer");
        assert_eq!(
            replayed[1]["id"],
            json!(format!("msg_{}", short_hash(&"x".repeat(70))))
        );
        assert!(replayed[1].get("phase").is_none());
    }
}
