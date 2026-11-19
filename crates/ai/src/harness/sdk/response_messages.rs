//! Port of `ai` `toResponseMessages`.

use std::collections::HashMap;

use crate::harness::errors::HarnessResult;
use crate::harness::provider::JsonValue;
use crate::harness::sdk::content::ContentPart;
use crate::harness::sdk::messages::{
    AssistantContent, AssistantContentPart, AssistantModelMessage, ModelMessage, ReasoningPart,
    TextPart, ToolApprovalRequest, ToolApprovalResponse, ToolCallPart, ToolContentPart,
    ToolModelMessage, ToolResultOutput, ToolResultPart,
};
use crate::harness::sdk::tool::{
    ToolModelOutputErrorMode, ToolSet, create_tool_error_model_output, create_tool_model_output,
};

/// `toResponseMessages({ content, tools })`.
///
/// [rust] `model_outputs` stands in for the harness's per-call
/// `toModelOutput` override: an entry wins over the tool's own
/// `toModelOutput` for successful results.
pub async fn to_response_messages(
    content: &[ContentPart],
    tools: Option<&ToolSet>,
    model_outputs: Option<&HashMap<String, ToolResultOutput>>,
) -> HarnessResult<Vec<ModelMessage>> {
    let model_output =
        |tool_call_id: &str, tool_name: &str, input: &JsonValue, output: &JsonValue| {
            let override_output = model_outputs.and_then(|m| m.get(tool_call_id)).cloned();
            let tool = tools.and_then(|t| t.get(tool_name));
            let tool_call_id = tool_call_id.to_string();
            let input = input.clone();
            let output = output.clone();
            async move {
                match override_output {
                    Some(o) => Ok(o),
                    None => {
                        create_tool_model_output(
                            &tool_call_id,
                            &input,
                            &output,
                            tool,
                            ToolModelOutputErrorMode::None,
                        )
                        .await
                    }
                }
            }
        };

    let mut messages = Vec::new();
    let mut tool_call_order: HashMap<String, usize> = HashMap::new();
    let mut assistant: Vec<AssistantContentPart> = Vec::new();

    for part in content {
        match part {
            ContentPart::ToolResult(r) if r.provider_executed != Some(true) => continue,
            ContentPart::ToolError(e) if e.provider_executed != Some(true) => continue,
            ContentPart::Text(t) if t.text.is_empty() && t.provider_metadata.is_none() => {
                continue;
            }
            _ => {}
        }
        match part {
            ContentPart::Text(t) => assistant.push(AssistantContentPart::Text(TextPart {
                text: t.text.clone(),
                provider_options: t.provider_metadata.clone(),
            })),
            ContentPart::Reasoning(r) => {
                assistant.push(AssistantContentPart::Reasoning(ReasoningPart {
                    text: r.text.clone(),
                    provider_options: r.provider_metadata.clone(),
                }))
            }
            ContentPart::ToolCall(c) => {
                let next = tool_call_order.len();
                tool_call_order
                    .entry(c.tool_call_id.clone())
                    .or_insert(next);
                let input = if c.is_invalid()
                    && !c.input.is_object()
                    && !c.input.is_array()
                    && !c.input.is_null()
                {
                    JsonValue::Object(Default::default())
                } else {
                    c.input.clone()
                };
                assistant.push(AssistantContentPart::ToolCall(ToolCallPart {
                    tool_call_id: c.tool_call_id.clone(),
                    tool_name: c.tool_name.clone(),
                    input,
                    provider_options: c.provider_metadata.clone(),
                    provider_executed: c.provider_executed,
                }));
            }
            ContentPart::ToolResult(r) => {
                let output = model_output(
                    &r.tool_call_id,
                    &r.tool_name,
                    r.input.as_ref().unwrap_or(&JsonValue::Null),
                    &r.output,
                )
                .await?;
                assistant.push(AssistantContentPart::ToolResult(ToolResultPart {
                    tool_call_id: r.tool_call_id.clone(),
                    tool_name: r.tool_name.clone(),
                    output,
                    provider_options: r.provider_metadata.clone(),
                }));
            }
            ContentPart::ToolError(e) => {
                assistant.push(AssistantContentPart::ToolResult(ToolResultPart {
                    tool_call_id: e.tool_call_id.clone(),
                    tool_name: e.tool_name.clone(),
                    output: create_tool_error_model_output(
                        &e.error,
                        ToolModelOutputErrorMode::Json,
                    ),
                    provider_options: e.provider_metadata.clone(),
                }));
            }
            ContentPart::ToolApprovalRequest(a) => assistant.push(
                AssistantContentPart::ToolApprovalRequest(ToolApprovalRequest {
                    approval_id: a.approval_id.clone(),
                    tool_call_id: a.tool_call.tool_call_id.clone(),
                    reason: a.reason.clone(),
                    is_automatic: a.is_automatic,
                    signature: a.signature.clone(),
                    input_schema_input: None,
                }),
            ),
            ContentPart::ToolApprovalResponse(_) => {}
        }
    }

    if !assistant.is_empty() {
        messages.push(ModelMessage::Assistant(AssistantModelMessage {
            content: AssistantContent::Parts(assistant),
            provider_options: None,
        }));
    }

    let mut tool_content: Vec<ToolContentPart> = Vec::new();
    for part in content {
        match part {
            ContentPart::ToolApprovalResponse(a) => {
                tool_content.push(ToolContentPart::ToolApprovalResponse(
                    ToolApprovalResponse {
                        approval_id: a.approval_id.clone(),
                        approved: a.approved,
                        reason: a.reason.clone(),
                        provider_executed: a.provider_executed,
                    },
                ));
                if !a.approved {
                    tool_content.push(ToolContentPart::ToolResult(ToolResultPart {
                        tool_call_id: a.tool_call.tool_call_id.clone(),
                        tool_name: a.tool_call.tool_name.clone(),
                        output: ToolResultOutput::ExecutionDenied {
                            reason: a.reason.clone(),
                            provider_options: None,
                        },
                        provider_options: None,
                    }));
                }
            }
            ContentPart::ToolResult(r) if r.provider_executed != Some(true) => {
                let output = model_output(
                    &r.tool_call_id,
                    &r.tool_name,
                    r.input.as_ref().unwrap_or(&JsonValue::Null),
                    &r.output,
                )
                .await?;
                tool_content.push(ToolContentPart::ToolResult(ToolResultPart {
                    tool_call_id: r.tool_call_id.clone(),
                    tool_name: r.tool_name.clone(),
                    output,
                    provider_options: r.provider_metadata.clone(),
                }));
            }
            ContentPart::ToolError(e) if e.provider_executed != Some(true) => {
                tool_content.push(ToolContentPart::ToolResult(ToolResultPart {
                    tool_call_id: e.tool_call_id.clone(),
                    tool_name: e.tool_name.clone(),
                    output: create_tool_error_model_output(
                        &e.error,
                        ToolModelOutputErrorMode::Text,
                    ),
                    provider_options: e.provider_metadata.clone(),
                }));
            }
            _ => {}
        }
    }

    if !tool_content.is_empty() {
        messages.push(ModelMessage::Tool(ToolModelMessage {
            content: sort_tool_results_by_call_order(tool_content, &tool_call_order),
            provider_options: None,
        }));
    }
    Ok(messages)
}

fn sort_tool_results_by_call_order(
    content: Vec<ToolContentPart>,
    order: &HashMap<String, usize>,
) -> Vec<ToolContentPart> {
    let mut results: Vec<(usize, ToolResultPart)> = content
        .iter()
        .filter_map(|p| match p {
            ToolContentPart::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .enumerate()
        .collect();
    results.sort_by(|(ai, a), (bi, b)| {
        match (order.get(&a.tool_call_id), order.get(&b.tool_call_id)) {
            (None, None) => ai.cmp(bi),
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (Some(x), Some(y)) => x.cmp(y).then(ai.cmp(bi)),
        }
    });
    let mut sorted = results.into_iter().map(|(_, r)| r);
    content
        .into_iter()
        .map(|p| match p {
            ToolContentPart::ToolResult(_) => {
                ToolContentPart::ToolResult(sorted.next().expect("same number of results"))
            }
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sdk::content::{TextContent, TypedToolCall};
    use crate::harness::sdk::error_value::ErrorValue;
    use serde_json::json;

    #[tokio::test]
    async fn builds_assistant_and_tool_messages_in_call_order() {
        let c1 = TypedToolCall::new("c1", "a", json!({}));
        let c2 = TypedToolCall::new("c2", "b", json!({}));
        let content = vec![
            ContentPart::Text(TextContent {
                text: "hi".into(),
                ..Default::default()
            }),
            ContentPart::Text(TextContent::default()),
            ContentPart::ToolCall(c1.clone()),
            ContentPart::ToolCall(c2.clone()),
            ContentPart::ToolError(c2.to_error(ErrorValue::message("bad"))),
            ContentPart::ToolResult(c1.to_result(json!("ok"))),
        ];
        let messages = to_response_messages(&content, None, None).await.unwrap();
        assert_eq!(
            serde_json::to_value(&messages).unwrap(),
            json!([
                {"role": "assistant", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool-call", "toolCallId": "c1", "toolName": "a", "input": {}},
                    {"type": "tool-call", "toolCallId": "c2", "toolName": "b", "input": {}}
                ]},
                {"role": "tool", "content": [
                    {"type": "tool-result", "toolCallId": "c1", "toolName": "a", "output": {"type": "text", "value": "ok"}},
                    {"type": "tool-result", "toolCallId": "c2", "toolName": "b", "output": {"type": "error-text", "value": "Error: bad"}}
                ]}
            ])
        );
    }

    #[tokio::test]
    async fn model_output_overrides_win() {
        let c1 = TypedToolCall::new("c1", "a", json!({}));
        let mut overrides = HashMap::new();
        overrides.insert(
            "c1".to_string(),
            ToolResultOutput::Text {
                value: "override".into(),
                provider_options: None,
            },
        );
        let messages = to_response_messages(
            &[ContentPart::ToolResult(c1.to_result(json!({"x": 1})))],
            None,
            Some(&overrides),
        )
        .await
        .unwrap();
        assert_eq!(
            serde_json::to_value(&messages).unwrap()[0]["content"][0]["output"],
            json!({"type": "text", "value": "override"})
        );
    }
}
