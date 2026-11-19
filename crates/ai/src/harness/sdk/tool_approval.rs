//! Port of `ai` `ToolApprovalStatus` (`tool-approval-configuration.ts`).

use serde::{Deserialize, Serialize};

/// The object form of `ToolApprovalStatus`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolApprovalStatusObject {
    NotApplicable,
    Approved {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Denied {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    UserApproval {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

/// The string form of `ToolApprovalStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolApprovalStatusName {
    NotApplicable,
    Approved,
    Denied,
    UserApproval,
}

/// `ToolApprovalStatus` (without `undefined`, which is `None` at use sites).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolApprovalStatus {
    Name(ToolApprovalStatusName),
    Object(ToolApprovalStatusObject),
}

impl ToolApprovalStatus {
    /// The object form: a bare string becomes `{ type }`.
    pub fn normalize(&self) -> ToolApprovalStatusObject {
        match self {
            Self::Object(o) => o.clone(),
            Self::Name(ToolApprovalStatusName::NotApplicable) => {
                ToolApprovalStatusObject::NotApplicable
            }
            Self::Name(ToolApprovalStatusName::Approved) => {
                ToolApprovalStatusObject::Approved { reason: None }
            }
            Self::Name(ToolApprovalStatusName::Denied) => {
                ToolApprovalStatusObject::Denied { reason: None }
            }
            Self::Name(ToolApprovalStatusName::UserApproval) => {
                ToolApprovalStatusObject::UserApproval { reason: None }
            }
        }
    }
}

impl From<ToolApprovalStatusName> for ToolApprovalStatus {
    fn from(name: ToolApprovalStatusName) -> Self {
        Self::Name(name)
    }
}

impl From<ToolApprovalStatusObject> for ToolApprovalStatus {
    fn from(object: ToolApprovalStatusObject) -> Self {
        Self::Object(object)
    }
}
