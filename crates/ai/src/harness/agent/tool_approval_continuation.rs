//! Port of `agent/harness-agent-tool-approval-continuation.ts`.

use std::collections::{HashMap, HashSet};

use crate::harness::errors::{HarnessError, HarnessResult};
use crate::harness::sdk::messages::{
    AssistantContent, AssistantContentPart, ModelMessage, ToolApprovalRequest,
    ToolApprovalResponse, ToolContentPart, ToolResultOutput,
};

/// `collectHarnessAgentToolApprovalContinuations({ messages })`: approval
/// decisions in the trailing tool message that continue a suspended turn.
///
/// Responses whose tool call already has a result are skipped. A synthetic
/// `execution-denied` result does not count as a result.
pub fn collect_harness_agent_tool_approval_continuations(
    messages: &[ModelMessage],
) -> HarnessResult<Vec<ToolApprovalResponse>> {
    let Some(ModelMessage::Tool(last)) = messages.last() else {
        return Ok(vec![]);
    };

    let mut tool_call_ids: HashSet<&str> = HashSet::new();
    let mut approval_requests: HashMap<&str, &ToolApprovalRequest> = HashMap::new();
    for message in messages {
        let ModelMessage::Assistant(m) = message else {
            continue;
        };
        let AssistantContent::Parts(parts) = &m.content else {
            continue;
        };
        for part in parts {
            match part {
                AssistantContentPart::ToolCall(c) => {
                    tool_call_ids.insert(&c.tool_call_id);
                }
                AssistantContentPart::ToolApprovalRequest(r) => {
                    approval_requests.insert(&r.approval_id, r);
                }
                _ => {}
            }
        }
    }

    let tool_result_ids: HashSet<&str> = last
        .content
        .iter()
        .filter_map(|part| match part {
            ToolContentPart::ToolResult(r)
                if !matches!(r.output, ToolResultOutput::ExecutionDenied { .. }) =>
            {
                Some(r.tool_call_id.as_str())
            }
            _ => None,
        })
        .collect();

    let mut continuations = Vec::new();
    for part in &last.content {
        let ToolContentPart::ToolApprovalResponse(response) = part else {
            continue;
        };
        let Some(request) = approval_requests.get(response.approval_id.as_str()) else {
            return Err(HarnessError::new(format!(
                "Tool approval response '{}' does not match a prior tool approval request.",
                response.approval_id
            ))
            .into());
        };
        if tool_result_ids.contains(request.tool_call_id.as_str()) {
            continue;
        }
        if !tool_call_ids.contains(request.tool_call_id.as_str()) {
            return Err(HarnessError::new(format!(
                "Tool approval request '{}' references unknown tool call '{}'.",
                request.approval_id, request.tool_call_id
            ))
            .into());
        }
        continuations.push(response.clone());
    }
    Ok(continuations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// What `convertToModelMessages` produces for an assistant UI message
    /// with tool parts.
    fn messages(value: Value) -> Vec<ModelMessage> {
        serde_json::from_value(value).unwrap()
    }

    fn approval_responded(provider_executed: bool) -> Vec<ModelMessage> {
        let pe = if provider_executed {
            json!(true)
        } else {
            Value::Null
        };
        let mut call1 = json!({"type": "tool-call", "toolCallId": "call-1", "toolName": "weather", "input": {"city": "Paris"}});
        let mut call2 = json!({"type": "tool-call", "toolCallId": "call-2", "toolName": "weather", "input": {"city": "Tokyo"}});
        let mut r1 =
            json!({"type": "tool-approval-response", "approvalId": "approval-1", "approved": true});
        let mut r2 = json!({"type": "tool-approval-response", "approvalId": "approval-2", "approved": false, "reason": "User denied the weather lookup."});
        if provider_executed {
            for v in [&mut call1, &mut call2, &mut r1, &mut r2] {
                v["providerExecuted"] = pe.clone();
            }
        }
        messages(json!([
            {"role": "assistant", "content": [
                call1,
                {"type": "tool-approval-request", "approvalId": "approval-1", "toolCallId": "call-1"},
                call2,
                {"type": "tool-approval-request", "approvalId": "approval-2", "toolCallId": "call-2"}
            ]},
            {"role": "tool", "content": [
                r1,
                r2,
                {"type": "tool-result", "toolCallId": "call-2", "toolName": "weather",
                 "output": {"type": "execution-denied", "reason": "User denied the weather lookup."}}
            ]}
        ]))
    }

    #[test]
    fn preserves_mixed_decisions_from_ui_messages() {
        for provider_executed in [false, true] {
            let decisions: Vec<(String, bool, Option<String>)> =
                collect_harness_agent_tool_approval_continuations(&approval_responded(
                    provider_executed,
                ))
                .unwrap()
                .into_iter()
                .map(|r| (r.approval_id, r.approved, r.reason))
                .collect();
            assert_eq!(
                decisions,
                vec![
                    ("approval-1".into(), true, None),
                    (
                        "approval-2".into(),
                        false,
                        Some("User denied the weather lookup.".into())
                    ),
                ],
                "providerExecuted: {provider_executed}"
            );
        }
    }

    #[test]
    fn preserves_a_denial_when_every_approval_is_denied() {
        let messages = messages(json!([
            {"role": "assistant", "content": [
                {"type": "tool-call", "toolCallId": "call-1", "toolName": "weather", "input": {"city": "Paris"}},
                {"type": "tool-approval-request", "approvalId": "approval-1", "toolCallId": "call-1"}
            ]},
            {"role": "tool", "content": [
                {"type": "tool-approval-response", "approvalId": "approval-1", "approved": false},
                {"type": "tool-result", "toolCallId": "call-1", "toolName": "weather",
                 "output": {"type": "execution-denied"}}
            ]}
        ]));
        assert_eq!(
            serde_json::to_value(
                collect_harness_agent_tool_approval_continuations(&messages)
                    .unwrap()
                    .into_iter()
                    .map(ToolContentPart::ToolApprovalResponse)
                    .collect::<Vec<_>>()
            )
            .unwrap(),
            json!([{"type": "tool-approval-response", "approvalId": "approval-1", "approved": false}])
        );
    }

    #[test]
    fn ignores_approvals_with_completed_tool_results() {
        let messages = messages(json!([
            {"role": "assistant", "content": [
                {"type": "tool-call", "toolCallId": "call-1", "toolName": "weather", "input": {"city": "Paris"}},
                {"type": "tool-approval-request", "approvalId": "approval-1", "toolCallId": "call-1"},
                {"type": "tool-call", "toolCallId": "call-2", "toolName": "weather", "input": {"city": "Tokyo"}},
                {"type": "tool-approval-request", "approvalId": "approval-2", "toolCallId": "call-2"}
            ]},
            {"role": "tool", "content": [
                {"type": "tool-approval-response", "approvalId": "approval-1", "approved": true},
                {"type": "tool-result", "toolCallId": "call-1", "toolName": "weather",
                 "output": {"type": "json", "value": {"celsius": 12}}},
                {"type": "tool-approval-response", "approvalId": "approval-2", "approved": false},
                {"type": "tool-result", "toolCallId": "call-2", "toolName": "weather",
                 "output": {"type": "error-text", "value": "Tool call execution denied."}}
            ]}
        ]));
        assert!(
            collect_harness_agent_tool_approval_continuations(&messages)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_unknown_approvals_and_tool_calls() {
        let unknown_approval = messages(json!([
            {"role": "tool", "content": [
                {"type": "tool-approval-response", "approvalId": "missing", "approved": true}
            ]}
        ]));
        assert_eq!(
            collect_harness_agent_tool_approval_continuations(&unknown_approval)
                .unwrap_err()
                .to_string(),
            "Tool approval response 'missing' does not match a prior tool approval request."
        );
        let unknown_call = messages(json!([
            {"role": "assistant", "content": [
                {"type": "tool-approval-request", "approvalId": "a", "toolCallId": "c"}
            ]},
            {"role": "tool", "content": [
                {"type": "tool-approval-response", "approvalId": "a", "approved": true}
            ]}
        ]));
        assert_eq!(
            collect_harness_agent_tool_approval_continuations(&unknown_call)
                .unwrap_err()
                .to_string(),
            "Tool approval request 'a' references unknown tool call 'c'."
        );
    }
}
