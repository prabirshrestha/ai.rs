//! [rust] Runs the caller's tools and approvals for agents that run their own
//! loop, the way [`run_agent_loop`](crate::run_agent_loop) runs them:
//! argument preparation and validation, `before_tool_call`, `execute` and
//! `after_tool_call`.

#![allow(dead_code)]

use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    AfterToolCallContext, AfterToolCallFn, AgentContext, AgentToolResult, AgentToolUpdateCallback,
    AssistantMessage, BeforeToolCallContext, BeforeToolCallFn, DynAgentTool, ToolCall,
};

fn error_result(message: impl Into<String>) -> AgentToolResult {
    let mut result = AgentToolResult::text(message);
    result.details = Some(json!({}));
    result
}

/// The caller's tools and tool hooks for one turn.
#[derive(Clone, Default)]
pub struct HostTools {
    pub tools: Vec<DynAgentTool>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub cancellation_token: Option<CancellationToken>,
}

/// The decision on a tool call the agent asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    Allow { args: Value },
    Deny { reason: String },
}

impl HostTools {
    pub fn find(&self, name: &str) -> Option<DynAgentTool> {
        self.tools
            .iter()
            .find(|tool| tool.definition().name == name)
            .cloned()
    }

    fn context(&self) -> AgentContext {
        AgentContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: self.tools.clone(),
        }
    }

    /// Ask `before_tool_call` about a call. Without a hook, `default` decides.
    pub async fn approve(
        &self,
        tool_call: &ToolCall,
        assistant_message: &AssistantMessage,
        default: ApprovalDecision,
    ) -> ApprovalDecision {
        let Some(before) = &self.before_tool_call else {
            return default;
        };
        let args = Arc::new(StdMutex::new(tool_call.arguments.clone()));
        let result = before(
            BeforeToolCallContext {
                assistant_message: assistant_message.clone(),
                tool_call: tool_call.clone(),
                args: args.clone(),
                context: self.context(),
            },
            self.cancellation_token.clone(),
        )
        .await;
        match result {
            Ok(Some(result)) if result.block => ApprovalDecision::Deny {
                reason: result
                    .reason
                    .unwrap_or_else(|| "Tool execution was blocked".to_string()),
            },
            Ok(_) => ApprovalDecision::Allow {
                args: args.lock().unwrap().clone(),
            },
            Err(error) => ApprovalDecision::Deny {
                reason: error.to_string(),
            },
        }
    }

    /// Run a host tool. Returns the result and whether it is an error.
    pub async fn run(
        &self,
        tool_call: &ToolCall,
        assistant_message: &AssistantMessage,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> (AgentToolResult, bool) {
        let Some(tool) = self.find(&tool_call.name) else {
            return (
                error_result(format!("Tool {} not found", tool_call.name)),
                true,
            );
        };
        let prepared = match tool.prepare_arguments(tool_call.arguments.clone()) {
            Ok(args) => args,
            Err(error) => return (error_result(error.to_string()), true),
        };
        let mut prepared_call = tool_call.clone();
        prepared_call.arguments = prepared;
        let args = match crate::utils::validation::validate_tool_arguments(
            &tool.definition(),
            &prepared_call,
        ) {
            Ok(args) => args,
            Err(error) => return (error_result(error.to_string()), true),
        };
        prepared_call.arguments = args;
        let args = match self
            .approve(
                &prepared_call,
                assistant_message,
                ApprovalDecision::Allow {
                    args: prepared_call.arguments.clone(),
                },
            )
            .await
        {
            ApprovalDecision::Allow { args } => args,
            ApprovalDecision::Deny { reason } => return (error_result(reason), true),
        };
        if self
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return (error_result("Operation aborted"), true);
        }
        let (mut result, mut is_error) = match tool
            .execute(
                &tool_call.id,
                args.clone(),
                self.cancellation_token.clone(),
                on_update,
            )
            .await
        {
            Ok(result) => (result, false),
            Err(error) => (error_result(error.to_string()), true),
        };
        if let Some(after) = &self.after_tool_call {
            match after(
                AfterToolCallContext {
                    assistant_message: assistant_message.clone(),
                    tool_call: tool_call.clone(),
                    args,
                    result: result.clone(),
                    is_error,
                    context: self.context(),
                },
                self.cancellation_token.clone(),
            )
            .await
            {
                Ok(Some(after)) => {
                    if let Some(content) = after.content {
                        result.content = content;
                    }
                    if let Some(details) = after.details.filter(|details| !details.is_null()) {
                        result.details = Some(details);
                    }
                    if let Some(usage) = after.usage {
                        result.usage = Some(usage);
                    }
                    if let Some(terminate) = after.terminate {
                        result.terminate = terminate;
                    }
                    if let Some(next) = after.is_error {
                        is_error = next;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    result = error_result(error.to_string());
                    is_error = true;
                }
            }
        }
        (result, is_error)
    }
}

/// The text of a tool result, for agents that take text only.
pub fn tool_result_text(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|part| match part {
            crate::ToolResultContent::Text(text) => Some(text.text.as_str()),
            crate::ToolResultContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A tool result as MCP `CallToolResult` content.
pub fn tool_result_mcp_content(result: &AgentToolResult) -> Vec<Value> {
    result
        .content
        .iter()
        .map(|part| match part {
            crate::ToolResultContent::Text(text) => json!({ "type": "text", "text": text.text }),
            crate::ToolResultContent::Image(image) => {
                json!({ "type": "image", "data": image.data, "mimeType": image.mime_type })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentToolBuilder, BeforeToolCallResult};

    fn assistant() -> AssistantMessage {
        let model = crate::providers::faux::register_faux_provider(None).get_model();
        AssistantMessage::empty_for(&model)
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments: args,
            thought_signature: None,
        }
    }

    fn echo() -> DynAgentTool {
        AgentToolBuilder::new("echo")
            .description("Echo")
            .parameters(json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            }))
            .execute(|args| async move {
                Ok(AgentToolResult::text(args["text"].as_str().unwrap_or("")))
            })
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn runs_validates_and_approves_tools() {
        let tools = HostTools {
            tools: vec![echo()],
            ..Default::default()
        };
        let (result, is_error) = tools
            .run(&call("echo", json!({ "text": "hi" })), &assistant(), None)
            .await;
        assert!(!is_error);
        assert_eq!(tool_result_text(&result), "hi");

        let (_, is_error) = tools
            .run(&call("echo", json!({})), &assistant(), None)
            .await;
        assert!(is_error);
        let (result, _) = tools
            .run(&call("nope", json!({})), &assistant(), None)
            .await;
        assert_eq!(tool_result_text(&result), "Tool nope not found");

        let denying = HostTools {
            tools: vec![echo()],
            before_tool_call: Some(Arc::new(|_, _| {
                Box::pin(async {
                    Ok(Some(BeforeToolCallResult {
                        block: true,
                        reason: Some("no".into()),
                    }))
                })
            })),
            ..Default::default()
        };
        let (result, is_error) = denying
            .run(&call("echo", json!({ "text": "hi" })), &assistant(), None)
            .await;
        assert!(is_error);
        assert_eq!(tool_result_text(&result), "no");
        assert_eq!(
            denying
                .approve(
                    &call("bash", json!({})),
                    &assistant(),
                    ApprovalDecision::Allow { args: json!({}) }
                )
                .await,
            ApprovalDecision::Deny {
                reason: "no".into()
            }
        );
    }
}
