//! Port of `harness-v1-lifecycle-state.ts`.

use serde::{Deserialize, Serialize};

use crate::harness::provider::JsonValue;
use crate::harness::sdk::messages::{ProviderOptions, ToolResultPart};
use crate::harness::spec::tools::{HarnessV1Skill, HarnessV1ToolSpec};

use crate::harness::spec::literal::{HarnessV1SpecificationVersion, string_literal};

string_literal!(
    /// `type: 'resume-session'`.
    ResumeSessionType = "resume-session"
);
string_literal!(
    /// `type: 'continue-turn'`.
    ContinueTurnType = "continue-turn"
);

/// `kind` of a pending approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessV1PendingToolApprovalKind {
    Builtin,
    Custom,
}

/// `HarnessV1PendingToolApproval`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1PendingToolApproval {
    pub approval_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: String,
    pub kind: HarnessV1PendingToolApprovalKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_name: Option<String>,
}

/// `HarnessV1PendingToolResult['completedResult']`: executed while
/// suspending; submitted on resume without running the tool again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1CompletedToolResult {
    #[serde(default)]
    pub output: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolResultPart>,
}

/// `HarnessV1PendingToolResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1PendingToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_result: Option<HarnessV1CompletedToolResult>,
}

/// `HarnessV1TurnSettings`: framework-owned settings captured when a turn
/// begins.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1TurnSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub skills: Vec<HarnessV1Skill>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub tools: Vec<HarnessV1ToolSpec>,
}

/// `HarnessV1ResumeSessionState` (`type: 'resume-session'`).
///
/// [rust] Deserializing rejects top-level `pendingToolApprovals` /
/// `pendingToolResults`, as TS's `validateLifecycleStateData` does: the
/// struct has no such fields to carry them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", try_from = "RawResumeSessionState")]
pub struct HarnessV1ResumeSessionState {
    pub r#type: ResumeSessionType,
    pub harness_id: String,
    pub specification_version: HarnessV1SpecificationVersion,
    /// Adapter-defined payload.
    pub data: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continue_from: Option<HarnessV1ContinueTurnState>,
}

/// [`HarnessV1ResumeSessionState`] as deserialized, before the check for
/// top-level pending state.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawResumeSessionState {
    r#type: ResumeSessionType,
    harness_id: String,
    specification_version: HarnessV1SpecificationVersion,
    data: JsonValue,
    #[serde(default)]
    continue_from: Option<HarnessV1ContinueTurnState>,
    #[serde(default)]
    pending_tool_approvals: Option<JsonValue>,
    #[serde(default)]
    pending_tool_results: Option<JsonValue>,
}

impl TryFrom<RawResumeSessionState> for HarnessV1ResumeSessionState {
    type Error = String;

    fn try_from(raw: RawResumeSessionState) -> Result<Self, Self::Error> {
        if raw.pending_tool_approvals.is_some() {
            return Err("Resume session state cannot contain pending tool approvals; unfinished turns must be stored as `continueFrom`.".into());
        }
        if raw.pending_tool_results.is_some() {
            return Err("Resume session state cannot contain pending tool results; unfinished turns must be stored as `continueFrom`.".into());
        }
        Ok(Self {
            r#type: raw.r#type,
            harness_id: raw.harness_id,
            specification_version: raw.specification_version,
            data: raw.data,
            continue_from: raw.continue_from,
        })
    }
}

/// `HarnessV1ContinueTurnState` (`type: 'continue-turn'`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ContinueTurnState {
    pub r#type: ContinueTurnType,
    pub harness_id: String,
    pub specification_version: HarnessV1SpecificationVersion,
    /// Adapter-defined payload.
    pub data: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_tool_approvals: Option<Vec<HarnessV1PendingToolApproval>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_tool_results: Option<Vec<HarnessV1PendingToolResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_settings: Option<HarnessV1TurnSettings>,
}

impl HarnessV1ResumeSessionState {
    pub fn new(harness_id: impl Into<String>, data: JsonValue) -> Self {
        Self {
            r#type: ResumeSessionType,
            harness_id: harness_id.into(),
            specification_version: HarnessV1SpecificationVersion,
            data,
            continue_from: None,
        }
    }
}

impl HarnessV1ContinueTurnState {
    pub fn new(harness_id: impl Into<String>, data: JsonValue) -> Self {
        Self {
            r#type: ContinueTurnType,
            harness_id: harness_id.into(),
            specification_version: HarnessV1SpecificationVersion,
            data,
            pending_tool_approvals: None,
            pending_tool_results: None,
            turn_settings: None,
        }
    }
}

/// `HarnessV1LifecycleState`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HarnessV1LifecycleState {
    ResumeSession(HarnessV1ResumeSessionState),
    ContinueTurn(HarnessV1ContinueTurnState),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lifecycle_states_round_trip_with_type_tags() {
        let state = json!({
            "type": "resume-session",
            "harnessId": "codex",
            "specificationVersion": "harness-v1",
            "data": { "threadId": "t" },
            "continueFrom": {
                "type": "continue-turn",
                "harnessId": "codex",
                "specificationVersion": "harness-v1",
                "data": null,
                "pendingToolApprovals": [{
                    "approvalId": "a", "toolCallId": "c", "toolName": "bash",
                    "input": "{}", "kind": "builtin", "providerExecuted": true
                }],
                "pendingToolResults": [{
                    "toolCallId": "c2", "toolName": "lookup", "input": "{}",
                    "completedResult": { "output": 1, "isError": false }
                }],
                "turnSettings": { "skills": [], "tools": [{ "name": "lookup" }] }
            }
        });
        let parsed: HarnessV1LifecycleState = serde_json::from_value(state.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), state);
        let resume: HarnessV1ResumeSessionState = serde_json::from_value(state.clone()).unwrap();
        assert_eq!(serde_json::to_value(&resume).unwrap(), state);
        assert!(serde_json::from_value::<HarnessV1ContinueTurnState>(state).is_err());
    }
}
