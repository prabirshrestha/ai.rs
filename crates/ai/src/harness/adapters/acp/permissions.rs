//! Port of `bridge/permission-controller.ts` and
//! `bridge/permission-locations.ts`: which ACP permission requests are
//! approved without asking the caller.

use std::sync::Arc;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::sandbox_paths::are_paths_inside_roots;
use crate::harness::paths::is_absolute_posix;
use crate::harness::sandbox::SandboxSession;
use crate::harness::tools::PermissionMode;
use crate::harness::utils::is_sandbox_write_protected_path;

/// Skills in `.agents/skills` load into later sessions, so writes there
/// always ask outside `allow-all`.
pub const ACP_BUILTIN_WRITE_PROTECTED_PATHS: [&str; 1] = [".agents/skills"];

/// The options a permission request must offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionOptions {
    pub allow_once: String,
    pub reject_once: String,
}

/// The `allow_once` and `reject_once` options, or which are missing.
pub fn permission_options(request: &Value) -> Result<PermissionOptions, Vec<&'static str>> {
    let find = |kind: &str| {
        request
            .get("options")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|option| option.get("kind").and_then(Value::as_str) == Some(kind))
            .and_then(|option| option.get("optionId").and_then(Value::as_str))
            .map(str::to_string)
    };
    match (find("allow_once"), find("reject_once")) {
        (Some(allow_once), Some(reject_once)) => Ok(PermissionOptions {
            allow_once,
            reject_once,
        }),
        (allow, reject) => {
            let mut missing = Vec::new();
            if allow.is_none() {
                missing.push("allow_once");
            }
            if reject.is_none() {
                missing.push("reject_once");
            }
            Err(missing)
        }
    }
}

pub fn selected(option_id: &str) -> Value {
    json!({ "outcome": { "outcome": "selected", "optionId": option_id } })
}

pub fn cancelled() -> Value {
    json!({ "outcome": { "outcome": "cancelled" } })
}

/// `shouldAutoApprove`.
pub fn should_auto_approve(permission_mode: PermissionMode, kind: Option<&str>) -> bool {
    if permission_mode == PermissionMode::AllowAll {
        return true;
    }
    if matches!(kind, Some("read" | "search" | "think" | "fetch")) {
        return true;
    }
    permission_mode == PermissionMode::AllowEdits && is_write_kind(kind)
}

pub fn is_write_kind(kind: Option<&str>) -> bool {
    matches!(kind, Some("edit" | "delete" | "move"))
}

/// What the auto-approval checks need.
#[derive(Clone)]
pub struct AutoApproval {
    pub sandbox: Arc<dyn SandboxSession>,
    pub work_dir: String,
    pub permission_mode: PermissionMode,
    pub has_permission_mode_mapping: bool,
    pub write_protected_paths: Vec<String>,
    pub abort: Option<CancellationToken>,
}

impl AutoApproval {
    /// Whether the request is approved without the caller.
    pub async fn approves(&self, tool_call: &Value) -> bool {
        let kind = tool_call.get("kind").and_then(Value::as_str);
        if self.has_permission_mode_mapping || !should_auto_approve(self.permission_mode, kind) {
            return false;
        }
        if self.permission_mode == PermissionMode::AllowAll {
            return true;
        }
        if !self.are_locations_trusted(tool_call).await {
            return false;
        }
        !(is_write_kind(kind) && self.are_write_locations_protected(tool_call).await)
    }

    fn location_paths(tool_call: &Value) -> Option<Vec<Option<String>>> {
        let locations = tool_call.get("locations")?.as_array()?;
        Some(
            locations
                .iter()
                .map(|location| {
                    location
                        .get("path")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect(),
        )
    }

    /// `areACPToolLocationsTrusted`.
    pub async fn are_locations_trusted(&self, tool_call: &Value) -> bool {
        let paths = Self::location_paths(tool_call).unwrap_or_default();
        if paths.is_empty() {
            // File operations need explicit locations.
            let kind = tool_call.get("kind").and_then(Value::as_str);
            return matches!(kind, Some("think" | "fetch"));
        }
        let Some(paths) = paths.into_iter().collect::<Option<Vec<String>>>() else {
            return false;
        };
        are_paths_inside_roots(
            self.sandbox.as_ref(),
            &self.work_dir,
            &self.work_dir,
            &[],
            &paths,
            self.abort.clone(),
        )
        .await
    }

    /// `areACPToolWriteLocationsProtected`.
    pub async fn are_write_locations_protected(&self, tool_call: &Value) -> bool {
        let paths = Self::location_paths(tool_call).unwrap_or_default();
        let Some(paths) = paths
            .into_iter()
            .map(|path| path.filter(|path| !path.trim().is_empty()))
            .collect::<Option<Vec<String>>>()
        else {
            return true;
        };
        if paths.is_empty() {
            return true;
        }
        let absolute: Vec<String> = paths
            .iter()
            .map(|path| {
                if is_absolute_posix(path) {
                    path.clone()
                } else {
                    format!("{}/{path}", self.work_dir)
                }
            })
            .collect();
        let mut protected: Vec<String> = ACP_BUILTIN_WRITE_PROTECTED_PATHS
            .iter()
            .map(|path| path.to_string())
            .collect();
        protected.extend(self.write_protected_paths.iter().cloned());
        is_sandbox_write_protected_path(
            self.sandbox.as_ref(),
            &self.work_dir,
            &absolute,
            &protected,
            self.abort.clone(),
        )
        .await
        .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_approves_by_mode_and_kind() {
        use PermissionMode::*;
        assert!(should_auto_approve(AllowAll, Some("execute")));
        assert!(should_auto_approve(AllowReads, Some("read")));
        assert!(should_auto_approve(AllowReads, Some("fetch")));
        assert!(!should_auto_approve(AllowReads, Some("edit")));
        assert!(should_auto_approve(AllowEdits, Some("move")));
        assert!(!should_auto_approve(AllowEdits, Some("execute")));
        assert!(!should_auto_approve(AllowEdits, None));
    }

    #[test]
    fn requires_allow_and_reject_options() {
        let request = json!({ "options": [
            { "optionId": "a", "kind": "allow_once", "name": "Allow" },
            { "optionId": "r", "kind": "reject_once", "name": "Reject" },
        ] });
        assert_eq!(
            permission_options(&request),
            Ok(PermissionOptions {
                allow_once: "a".into(),
                reject_once: "r".into()
            })
        );
        assert_eq!(
            permission_options(&json!({ "options": [] })),
            Err(vec!["allow_once", "reject_once"])
        );
    }
}
