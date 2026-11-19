//! Port of `agent/harness-agent-tool-result-continuation.ts`.

use crate::harness::sdk::messages::{ModelMessage, ToolContentPart, ToolResultPart};

/// `collectHarnessAgentToolResultContinuations({ messages })`: the
/// client-provided tool results of the trailing tool message.
pub fn collect_harness_agent_tool_result_continuations(
    messages: &[ModelMessage],
) -> Vec<ToolResultPart> {
    let Some(ModelMessage::Tool(last)) = messages.last() else {
        return vec![];
    };
    last.content
        .iter()
        .filter_map(|part| match part {
            ToolContentPart::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn messages(value: serde_json::Value) -> Vec<ModelMessage> {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn collects_results_only_from_the_trailing_tool_message() {
        let messages = messages(json!([
            {"role": "tool", "content": [{
                "type": "tool-result", "toolCallId": "old", "toolName": "weather",
                "output": {"type": "json", "value": {"celsius": 18}}
            }]},
            {"role": "user", "content": "continue"}
        ]));
        assert!(collect_harness_agent_tool_result_continuations(&messages).is_empty());
    }

    #[test]
    fn converts_model_output_wrappers_into_harness_results() {
        let content = json!([
            {"type": "tool-result", "toolCallId": "text", "toolName": "tool",
             "output": {"type": "text", "value": "done"}},
            {"type": "tool-result", "toolCallId": "json", "toolName": "tool",
             "output": {"type": "json", "value": {"done": true}},
             "providerOptions": {"fake": {"nativeRequest": {"requestId": "native-1"}}}},
            {"type": "tool-result", "toolCallId": "error-text", "toolName": "tool",
             "output": {"type": "error-text", "value": "failed"}},
            {"type": "tool-result", "toolCallId": "error-json", "toolName": "tool",
             "output": {"type": "error-json", "value": {"message": "failed"}}},
            {"type": "tool-result", "toolCallId": "denied", "toolName": "tool",
             "output": {"type": "execution-denied", "reason": "not allowed"}},
            {"type": "tool-result", "toolCallId": "content", "toolName": "tool",
             "output": {"type": "content", "value": [{"type": "text", "text": "structured"}]}}
        ]);
        let messages = messages(json!([{"role": "tool", "content": content}]));
        assert_eq!(
            serde_json::to_value(
                collect_harness_agent_tool_result_continuations(&messages)
                    .into_iter()
                    .map(ToolContentPart::ToolResult)
                    .collect::<Vec<_>>()
            )
            .unwrap(),
            content
        );
    }
}
