//! Port of `agent/internal/permission-mode.ts`.

use crate::harness::agent::settings::HarnessAgentToolApprovalConfiguration;
use crate::harness::sdk::tool_approval::ToolApprovalStatusObject;
use crate::harness::spec::tools::HarnessV1PermissionMode;

/// `DEFAULT_PERMISSION_MODE`.
pub const DEFAULT_PERMISSION_MODE: HarnessV1PermissionMode = HarnessV1PermissionMode::AllowAll;

/// `resolvePermissionMode`.
pub fn resolve_permission_mode(
    permission_mode: Option<HarnessV1PermissionMode>,
) -> HarnessV1PermissionMode {
    permission_mode.unwrap_or(DEFAULT_PERMISSION_MODE)
}

/// `permissionModeNeedsBuiltinSupport`.
pub fn permission_mode_needs_builtin_support(permission_mode: HarnessV1PermissionMode) -> bool {
    permission_mode != HarnessV1PermissionMode::AllowAll
}

/// `CustomToolApprovalDecision`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CustomToolApprovalDecision {
    Allow { reason: Option<String> },
    Deny { reason: Option<String> },
    Request,
}

/// `resolveCustomToolApproval`.
pub fn resolve_custom_tool_approval(
    tool_name: &str,
    tool_approval: Option<&HarnessAgentToolApprovalConfiguration>,
) -> CustomToolApprovalDecision {
    let status = tool_approval
        .and_then(|a| a.get(tool_name))
        .map(|s| s.normalize())
        .unwrap_or(ToolApprovalStatusObject::NotApplicable);
    match status {
        ToolApprovalStatusObject::NotApplicable => {
            CustomToolApprovalDecision::Allow { reason: None }
        }
        ToolApprovalStatusObject::Approved { reason } => {
            CustomToolApprovalDecision::Allow { reason }
        }
        ToolApprovalStatusObject::Denied { reason } => CustomToolApprovalDecision::Deny { reason },
        ToolApprovalStatusObject::UserApproval { .. } => CustomToolApprovalDecision::Request,
    }
}
