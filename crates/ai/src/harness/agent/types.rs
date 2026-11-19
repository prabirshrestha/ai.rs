//! Port of `agent/harness-agent-types.ts` and
//! `agent/harness-agent-tool-types.ts`.
//!
//! [rust] `HarnessAllTools` / `HarnessBuiltinToolsOf` are type-level merges
//! of typed tool sets; Rust tool sets are untyped maps, so they are not
//! mirrored.

use crate::harness::spec::*;

pub type HarnessAgentAdapter = dyn HarnessV1;
pub type HarnessSandboxTemplate = HarnessV1SandboxTemplate;
pub type HarnessAgentBuiltinTool = HarnessV1BuiltinTool;
pub type HarnessAgentBuiltinToolName = HarnessV1BuiltinToolName;
pub type HarnessAgentBuiltinToolUseKind = HarnessV1BuiltinToolUseKind;
pub type HarnessAgentStartOptions = HarnessV1StartOptions;
pub type HarnessAgentAdapterSession = dyn HarnessV1Session;
pub type HarnessAgentPrompt = HarnessV1Prompt;
pub type HarnessAgentPromptControl = dyn HarnessV1PromptControl;
pub type HarnessAgentPromptTurnOptions = HarnessV1PromptTurnOptions;
pub type HarnessAgentContinueTurnOptions = HarnessV1ContinueTurnOptions;
pub type HarnessAgentStreamPart = HarnessV1StreamPart;
pub type HarnessAgentToolSpec = HarnessV1ToolSpec;
pub type HarnessAgentLifecycleState = HarnessV1LifecycleState;
pub type HarnessAgentResumeSessionState = HarnessV1ResumeSessionState;
pub type HarnessAgentContinueTurnState = HarnessV1ContinueTurnState;
pub type HarnessAgentPendingToolApproval = HarnessV1PendingToolApproval;
pub type HarnessAgentPendingToolResult = HarnessV1PendingToolResult;
pub type HarnessAgentSkill = HarnessV1Skill;
pub type HarnessAgentPermissionMode = HarnessV1PermissionMode;
