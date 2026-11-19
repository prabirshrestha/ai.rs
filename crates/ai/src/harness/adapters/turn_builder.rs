//! [rust] Builds ai.rs [`AgentEvent`]s and messages from the output of an
//! agent that runs its own loop (Claude Code, Codex, ACP agents, OpenCode).
//!
//! The events follow [`run_agent_loop`](crate::run_agent_loop): `AgentStart`,
//! then per model step `TurnStart`, the assistant message's `MessageStart`,
//! `MessageUpdate`s and `MessageEnd`, each tool's `ToolExecutionStart` and
//! `ToolExecutionEnd` with its tool result message, and `TurnEnd`; finally
//! `AgentEnd` with every new message.

#![allow(dead_code)]

use std::collections::HashMap;

use serde_json::Value;

use crate::harness::session::{HarnessEvent, HarnessTurnSender};
use crate::{
    AgentEvent, AgentMessage, AgentToolResult, AssistantContent, AssistantMessage,
    AssistantMessageEvent, Message, StopReason, TextContent, ThinkingContent, ToolCall,
    ToolResultMessage, Usage,
};

fn now_millis() -> u64 {
    crate::utils::time::now_millis()
}

struct OpenMessage {
    message: AssistantMessage,
    /// Block key → content index.
    blocks: HashMap<String, usize>,
    /// Content index → streamed tool arguments.
    tool_json: HashMap<usize, String>,
    /// Content indexes whose end was emitted.
    ended: Vec<usize>,
}

/// Accumulates one harness turn.
pub struct TurnBuilder {
    sender: HarnessTurnSender,
    api: String,
    provider: String,
    model: String,
    new_messages: Vec<AgentMessage>,
    open: Option<OpenMessage>,
    /// The last assistant message of the current step, once ended.
    step_message: Option<AssistantMessage>,
    step_tool_results: Vec<ToolResultMessage>,
    in_step: bool,
    started: bool,
    /// Tool call id → (tool name, arguments) of started tools.
    running_tools: HashMap<String, (String, Value)>,
    ended_tools: Vec<String>,
}

impl TurnBuilder {
    pub fn new(
        sender: HarnessTurnSender,
        api: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            sender,
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
            new_messages: Vec::new(),
            open: None,
            step_message: None,
            step_tool_results: Vec::new(),
            in_step: false,
            started: false,
            running_tools: HashMap::new(),
            ended_tools: Vec::new(),
        }
    }

    pub fn sender(&self) -> &HarnessTurnSender {
        &self.sender
    }

    fn emit(&self, event: AgentEvent) {
        self.sender.emit(HarnessEvent::Agent(event));
    }

    pub fn emit_harness(&self, event: HarnessEvent) {
        self.sender.emit(event);
    }

    /// The model reported by the agent, for the next messages.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
        if let Some(open) = &mut self.open {
            open.message.model = self.model.clone();
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Start the turn with the prompt.
    pub fn start(&mut self, prompt: AgentMessage) {
        if self.started {
            return;
        }
        self.started = true;
        self.emit(AgentEvent::AgentStart);
        self.begin_step();
        self.emit(AgentEvent::MessageStart {
            message: prompt.clone(),
        });
        self.emit(AgentEvent::MessageEnd {
            message: prompt.clone(),
        });
        self.new_messages.push(prompt);
    }

    /// A user message added while the turn runs (steering).
    pub fn user_message(&mut self, message: AgentMessage) {
        self.close_message(None, None, None);
        self.emit(AgentEvent::MessageStart {
            message: message.clone(),
        });
        self.emit(AgentEvent::MessageEnd {
            message: message.clone(),
        });
        self.new_messages.push(message);
    }

    fn begin_step(&mut self) {
        if !self.in_step {
            self.in_step = true;
            self.emit(AgentEvent::TurnStart);
        }
    }

    fn end_step(&mut self) {
        if !self.in_step {
            return;
        }
        if let Some(message) = self.step_message.take() {
            let tool_results = std::mem::take(&mut self.step_tool_results);
            self.emit(AgentEvent::TurnEnd {
                message: Message::Assistant(message),
                tool_results,
            });
            self.in_step = false;
        }
    }

    fn partial(&self) -> AssistantMessage {
        self.open
            .as_ref()
            .map(|open| open.message.clone())
            .expect("an open message")
    }

    fn update(&self, event: AssistantMessageEvent) {
        let message = Message::Assistant(self.partial());
        self.emit(AgentEvent::MessageUpdate {
            message,
            assistant_message_event: event,
        });
    }

    /// The open assistant message, else the last closed one.
    pub fn current_assistant(&self) -> Option<AssistantMessage> {
        if let Some(open) = &self.open {
            return Some(open.message.clone());
        }
        self.new_messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant(message) => Some(message.clone()),
                _ => None,
            })
    }

    /// Whether an assistant message is open.
    pub fn has_open_message(&self) -> bool {
        self.open.is_some()
    }

    /// Open an assistant message, ending the previous step when it has one.
    pub fn begin_message(&mut self) {
        if self.open.is_some() {
            return;
        }
        if self.step_message.is_some() {
            self.end_step();
        }
        self.begin_step();
        let message = AssistantMessage {
            content: Vec::new(),
            api: self.api.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: now_millis(),
        };
        self.open = Some(OpenMessage {
            message,
            blocks: HashMap::new(),
            tool_json: HashMap::new(),
            ended: Vec::new(),
        });
        self.emit(AgentEvent::MessageStart {
            message: Message::Assistant(self.partial()),
        });
        self.update(AssistantMessageEvent::Start {
            partial: self.partial(),
        });
    }

    pub fn set_response_id(&mut self, id: impl Into<String>) {
        self.begin_message();
        self.open.as_mut().unwrap().message.response_id = Some(id.into());
    }

    fn block_index(&mut self, key: &str) -> Option<usize> {
        self.open.as_ref()?.blocks.get(key).copied()
    }

    fn push_block(&mut self, key: &str, content: AssistantContent) -> usize {
        self.begin_message();
        let open = self.open.as_mut().unwrap();
        open.message.content.push(content);
        let index = open.message.content.len() - 1;
        open.blocks.insert(key.to_string(), index);
        index
    }

    /// Append streamed text to block `key`.
    pub fn text_delta(&mut self, key: &str, delta: &str) {
        let index = match self.block_index(key) {
            Some(index) => index,
            None => {
                let index = self.push_block(
                    key,
                    AssistantContent::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    }),
                );
                self.update(AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: self.partial(),
                });
                index
            }
        };
        if delta.is_empty() {
            return;
        }
        if let Some(AssistantContent::Text(text)) =
            self.open.as_mut().unwrap().message.content.get_mut(index)
        {
            text.text.push_str(delta);
        }
        self.update(AssistantMessageEvent::TextDelta {
            content_index: index,
            delta: delta.to_string(),
            partial: self.partial(),
        });
    }

    /// Append streamed reasoning to block `key`.
    pub fn thinking_delta(&mut self, key: &str, delta: &str) {
        let index = match self.block_index(key) {
            Some(index) => index,
            None => {
                let index = self.push_block(
                    key,
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: None,
                        redacted: None,
                    }),
                );
                self.update(AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: self.partial(),
                });
                index
            }
        };
        if delta.is_empty() {
            return;
        }
        if let Some(AssistantContent::Thinking(thinking)) =
            self.open.as_mut().unwrap().message.content.get_mut(index)
        {
            thinking.thinking.push_str(delta);
        }
        self.update(AssistantMessageEvent::ThinkingDelta {
            content_index: index,
            delta: delta.to_string(),
            partial: self.partial(),
        });
    }

    /// Set the signature of reasoning block `key`.
    pub fn thinking_signature(&mut self, key: &str, signature: &str) {
        if self.block_index(key).is_none() {
            self.thinking_delta(key, "");
        }
        let index = self.block_index(key).unwrap();
        if let Some(AssistantContent::Thinking(thinking)) =
            self.open.as_mut().unwrap().message.content.get_mut(index)
        {
            thinking
                .thinking_signature
                .get_or_insert_with(String::new)
                .push_str(signature);
        }
    }

    /// Start a tool call block whose arguments stream as JSON text.
    pub fn tool_call_start(&mut self, key: &str, id: &str, name: &str) {
        if self.block_index(key).is_some() {
            return;
        }
        let index = self.push_block(
            key,
            AssistantContent::ToolCall(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: Value::Object(Default::default()),
                thought_signature: None,
            }),
        );
        self.update(AssistantMessageEvent::ToolCallStart {
            content_index: index,
            partial: self.partial(),
        });
    }

    pub fn tool_call_delta(&mut self, key: &str, delta: &str) {
        let Some(index) = self.block_index(key) else {
            return;
        };
        let open = self.open.as_mut().unwrap();
        let json = open.tool_json.entry(index).or_default();
        json.push_str(delta);
        if let Ok(value) = serde_json::from_str::<Value>(json)
            && let Some(AssistantContent::ToolCall(call)) = open.message.content.get_mut(index)
        {
            call.arguments = value;
        }
        self.update(AssistantMessageEvent::ToolCallDelta {
            content_index: index,
            delta: delta.to_string(),
            partial: self.partial(),
        });
    }

    /// Set the final arguments of tool call block `key`.
    pub fn tool_call_arguments(&mut self, key: &str, arguments: Value) {
        if let Some(index) = self.block_index(key)
            && let Some(AssistantContent::ToolCall(call)) =
                self.open.as_mut().unwrap().message.content.get_mut(index)
        {
            call.arguments = arguments;
        }
    }

    /// Rename the tool of tool call block `key`.
    pub fn tool_call_name(&mut self, key: &str, name: &str) {
        if let Some(index) = self.block_index(key)
            && let Some(AssistantContent::ToolCall(call)) =
                self.open.as_mut().unwrap().message.content.get_mut(index)
        {
            call.name = name.to_string();
        }
    }

    /// A complete tool call.
    pub fn tool_call(&mut self, key: &str, id: &str, name: &str, arguments: Value) {
        self.tool_call_start(key, id, name);
        self.tool_call_arguments(key, arguments);
        self.end_block(key);
    }

    /// A complete text block.
    pub fn text(&mut self, key: &str, text: &str) {
        self.text_delta(key, text);
        self.end_block(key);
    }

    /// A complete reasoning block.
    pub fn thinking(&mut self, key: &str, text: &str, signature: Option<&str>) {
        self.thinking_delta(key, text);
        if let Some(signature) = signature {
            self.thinking_signature(key, signature);
        }
        self.end_block(key);
    }

    /// Whether block `key` exists in the open message.
    pub fn has_block(&self, key: &str) -> bool {
        self.open
            .as_ref()
            .is_some_and(|open| open.blocks.contains_key(key))
    }

    /// End block `key`.
    pub fn end_block(&mut self, key: &str) {
        let Some(index) = self.block_index(key) else {
            return;
        };
        let open = self.open.as_mut().unwrap();
        if open.ended.contains(&index) {
            return;
        }
        open.ended.push(index);
        let content = open.message.content[index].clone();
        let event = match content {
            AssistantContent::Text(text) => AssistantMessageEvent::TextEnd {
                content_index: index,
                content: text.text,
                partial: self.partial(),
            },
            AssistantContent::Thinking(thinking) => AssistantMessageEvent::ThinkingEnd {
                content_index: index,
                content: thinking.thinking,
                partial: self.partial(),
            },
            AssistantContent::ToolCall(tool_call) => AssistantMessageEvent::ToolCallEnd {
                content_index: index,
                tool_call,
                partial: self.partial(),
            },
        };
        self.update(event);
    }

    /// Add usage to the open message.
    pub fn add_usage(&mut self, usage: &Usage) {
        self.begin_message();
        let total = &mut self.open.as_mut().unwrap().message.usage;
        total.input += usage.input;
        total.output += usage.output;
        total.cache_read += usage.cache_read;
        total.cache_write += usage.cache_write;
        total.total_tokens += usage.total_tokens;
        if let Some(reasoning) = usage.reasoning {
            *total.reasoning.get_or_insert(0) += reasoning;
        }
    }

    /// Replace the usage of the open message.
    pub fn set_usage(&mut self, usage: Usage) {
        if let Some(open) = &mut self.open {
            open.message.usage = usage;
        }
    }

    /// End the open assistant message.
    pub fn close_message(
        &mut self,
        stop_reason: Option<StopReason>,
        usage: Option<Usage>,
        error_message: Option<String>,
    ) {
        let Some(mut open) = self.open.take() else {
            return;
        };
        let indexes: Vec<String> = open.blocks.keys().cloned().collect();
        self.open = Some(open);
        for key in indexes {
            self.end_block(&key);
        }
        open = self.open.take().unwrap();
        let has_tool_calls = open
            .message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)));
        open.message.stop_reason = stop_reason.unwrap_or(if has_tool_calls {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        });
        if let Some(usage) = usage {
            open.message.usage = usage;
        }
        open.message.error_message = error_message;
        let message = open.message;
        let event = if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
            AssistantMessageEvent::Error {
                reason: message.stop_reason,
                error: message.clone(),
            }
        } else {
            AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message: message.clone(),
            }
        };
        self.emit(AgentEvent::MessageUpdate {
            message: Message::Assistant(message.clone()),
            assistant_message_event: event,
        });
        self.emit(AgentEvent::MessageEnd {
            message: Message::Assistant(message.clone()),
        });
        self.new_messages.push(Message::Assistant(message.clone()));
        self.step_message = Some(message);
    }

    /// A tool started running.
    pub fn tool_start(&mut self, id: &str, name: &str, args: Value) {
        if self.running_tools.contains_key(id) || self.ended_tools.iter().any(|t| t == id) {
            return;
        }
        self.running_tools
            .insert(id.to_string(), (name.to_string(), args.clone()));
        self.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: id.to_string(),
            tool_name: name.to_string(),
            args,
        });
    }

    pub fn is_tool_running(&self, id: &str) -> bool {
        self.running_tools.contains_key(id)
    }

    pub fn is_tool_ended(&self, id: &str) -> bool {
        self.ended_tools.iter().any(|t| t == id)
    }

    /// Partial output of a running tool.
    pub fn tool_update(&mut self, id: &str, partial_result: AgentToolResult) {
        let Some((name, args)) = self.running_tools.get(id).cloned() else {
            return;
        };
        self.emit(AgentEvent::ToolExecutionUpdate {
            tool_call_id: id.to_string(),
            tool_name: name,
            args,
            partial_result,
        });
    }

    /// A tool finished. Starts it first when it was not reported as started.
    pub fn tool_end(&mut self, id: &str, name: &str, result: AgentToolResult, is_error: bool) {
        if self.is_tool_ended(id) {
            return;
        }
        if !self.running_tools.contains_key(id) {
            self.tool_start(id, name, Value::Object(Default::default()));
        }
        let (name, _) = self.running_tools.remove(id).unwrap();
        self.ended_tools.push(id.to_string());
        // The tool's results follow its call.
        if self.open.is_some() {
            self.close_message(None, None, None);
        }
        self.emit(AgentEvent::ToolExecutionEnd {
            tool_call_id: id.to_string(),
            tool_name: name.clone(),
            result: result.clone(),
            is_error,
        });
        let message = ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: name,
            content: result.content,
            details: result.details,
            usage: result.usage,
            added_tool_names: result.added_tool_names,
            is_error,
            timestamp: now_millis(),
        };
        self.emit(AgentEvent::MessageStart {
            message: Message::ToolResult(message.clone()),
        });
        self.emit(AgentEvent::MessageEnd {
            message: Message::ToolResult(message.clone()),
        });
        self.new_messages.push(Message::ToolResult(message.clone()));
        self.step_tool_results.push(message);
    }

    /// Messages so far.
    pub fn messages(&self) -> &[AgentMessage] {
        &self.new_messages
    }

    /// End the turn: close what is open and return the new messages.
    /// `error` ends an open (or new) assistant message as failed.
    pub fn finish(mut self, error: Option<(StopReason, String)>) -> Vec<AgentMessage> {
        if !self.started {
            self.started = true;
            self.emit(AgentEvent::AgentStart);
        }
        let pending: Vec<(String, String)> = self
            .running_tools
            .iter()
            .map(|(id, (name, _))| (id.clone(), name.clone()))
            .collect();
        for (id, name) in pending {
            let text = match &error {
                Some((_, message)) => message.clone(),
                None => "The tool did not report a result.".to_string(),
            };
            self.tool_end(&id, &name, AgentToolResult::text(text), true);
        }
        if let Some((reason, message)) = error {
            self.begin_message();
            self.close_message(Some(reason), None, Some(message));
        } else {
            self.close_message(None, None, None);
        }
        self.end_step();
        self.emit(AgentEvent::AgentEnd {
            messages: self.new_messages.clone(),
        });
        self.new_messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::session::HarnessTurn;
    use serde_json::json;

    fn kinds(events: &[HarnessEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                HarnessEvent::Agent(event) => Some(
                    match event {
                        AgentEvent::AgentStart => "agent_start",
                        AgentEvent::AgentEnd { .. } => "agent_end",
                        AgentEvent::TurnStart => "turn_start",
                        AgentEvent::TurnEnd { .. } => "turn_end",
                        AgentEvent::MessageStart { .. } => "message_start",
                        AgentEvent::MessageUpdate { .. } => "message_update",
                        AgentEvent::MessageEnd { .. } => "message_end",
                        AgentEvent::ToolExecutionStart { .. } => "tool_start",
                        AgentEvent::ToolExecutionUpdate { .. } => "tool_update",
                        AgentEvent::ToolExecutionEnd { .. } => "tool_end",
                    }
                    .to_string(),
                ),
                _ => None,
            })
            .filter(|kind| kind != "message_update")
            .collect()
    }

    #[tokio::test]
    async fn follows_the_agent_loop_event_order() {
        let turn = HarnessTurn::spawn(|sender| async move {
            let mut builder = TurnBuilder::new(sender, "claude-code", "anthropic", "m");
            builder.start(Message::user_text("hi"));
            builder.text_delta("t", "Let me ");
            builder.text_delta("t", "look.");
            builder.tool_call("c", "call-1", "read", json!({ "file_path": "a" }));
            builder.close_message(None, None, None);
            builder.tool_start("call-1", "read", json!({ "file_path": "a" }));
            builder.tool_end("call-1", "read", AgentToolResult::text("x"), false);
            builder.text("t2", "Done.");
            Ok(builder.finish(None))
        });
        let (events, messages) = turn.collect().await.unwrap();
        assert_eq!(
            kinds(&events),
            [
                "agent_start",
                "turn_start",
                "message_start",
                "message_end",
                "message_start",
                "message_end",
                "tool_start",
                "tool_end",
                "message_start",
                "message_end",
                "turn_end",
                "turn_start",
                "message_start",
                "message_end",
                "turn_end",
                "agent_end",
            ]
        );
        assert_eq!(messages.len(), 4);
        match &messages[1] {
            Message::Assistant(message) => {
                assert_eq!(message.stop_reason, StopReason::ToolUse);
                assert!(
                    matches!(&message.content[0], AssistantContent::Text(t) if t.text == "Let me look.")
                );
            }
            other => panic!("{other:?}"),
        }
        match &messages[3] {
            Message::Assistant(message) => assert_eq!(message.stop_reason, StopReason::Stop),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn fails_running_tools_and_reports_errors() {
        let turn = HarnessTurn::spawn(|sender| async move {
            let mut builder = TurnBuilder::new(sender, "a", "p", "m");
            builder.start(Message::user_text("hi"));
            builder.tool_start("call-1", "bash", json!({}));
            Ok(builder.finish(Some((StopReason::Error, "boom".into()))))
        });
        let (_, messages) = turn.collect().await.unwrap();
        match (&messages[1], &messages[2]) {
            (Message::ToolResult(result), Message::Assistant(message)) => {
                assert!(result.is_error);
                assert_eq!(message.stop_reason, StopReason::Error);
                assert_eq!(message.error_message.as_deref(), Some("boom"));
            }
            other => panic!("{other:?}"),
        }
    }
}
