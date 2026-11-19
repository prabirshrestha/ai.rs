//! Builtin tools, skills, permission modes and tool filtering.

use std::sync::OnceLock;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Tool;
use crate::harness::error::HarnessError;
use crate::harness::questions_tool::questions_tool;

/// A skill the harness installs for the agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessSkill {
    /// Stable identifier (kebab-case slug).
    pub name: String,
    /// Short, model-facing description.
    pub description: String,
    /// Full skill content.
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<HarnessSkillFile>>,
}

/// An extra file shipped with a skill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessSkillFile {
    /// Skill-relative POSIX path.
    pub path: String,
    /// UTF-8 text content.
    pub content: String,
}

/// Which builtin tools run without asking `before_tool_call` first.
///
/// Host tools are not affected: approve those in `before_tool_call`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    AllowReads,
    AllowEdits,
    #[default]
    AllowAll,
}

/// Which builtin tools are active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "mode",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum BuiltinToolFilter {
    Allow { tool_names: Vec<String> },
    Deny { tool_names: Vec<String> },
}

impl BuiltinToolFilter {
    /// Whether `tool_name` stays active under `filter`.
    pub fn includes(filter: Option<&Self>, tool_name: &str) -> bool {
        match filter {
            None => true,
            Some(Self::Allow { tool_names }) => tool_names.iter().any(|name| name == tool_name),
            Some(Self::Deny { tool_names }) => !tool_names.iter().any(|name| name == tool_name),
        }
    }

    /// The tool result text for a call to a filtered-out builtin.
    pub fn denial_reason(tool_name: &str) -> String {
        format!("Tool '{tool_name}' is inactive due to the HarnessAgent tool filtering policy.")
    }
}

/// The cross-harness builtin tool vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BuiltinToolName {
    Read,
    Write,
    Edit,
    Bash,
    Grep,
    Glob,
    WebSearch,
    AskUserQuestions,
}

/// Every [`BuiltinToolName`], in registry order.
pub const BUILTIN_TOOL_NAMES: [BuiltinToolName; 8] = [
    BuiltinToolName::Read,
    BuiltinToolName::Write,
    BuiltinToolName::Edit,
    BuiltinToolName::Bash,
    BuiltinToolName::Grep,
    BuiltinToolName::Glob,
    BuiltinToolName::WebSearch,
    BuiltinToolName::AskUserQuestions,
];

impl BuiltinToolName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Edit => "edit",
            Self::Bash => "bash",
            Self::Grep => "grep",
            Self::Glob => "glob",
            Self::WebSearch => "webSearch",
            Self::AskUserQuestions => "askUserQuestions",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        BUILTIN_TOOL_NAMES
            .iter()
            .copied()
            .find(|candidate| candidate.as_str() == name)
    }
}

/// What a builtin tool does, for [`PermissionMode`] decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuiltinToolKind {
    Readonly,
    Edit,
    Bash,
}

/// A tool the harness's agent runs itself.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltinTool {
    /// The tool as the model sees it, keyed by its harness name.
    pub tool: Tool,
    /// The agent's own name for the tool, when it differs.
    pub native_name: Option<String>,
    /// The cross-harness name, when the tool has one.
    pub common_name: Option<BuiltinToolName>,
    pub kind: Option<BuiltinToolKind>,
}

impl From<Tool> for BuiltinTool {
    fn from(tool: Tool) -> Self {
        Self {
            tool,
            native_name: None,
            common_name: None,
            kind: None,
        }
    }
}

impl BuiltinTool {
    /// Whether `mode` lets this tool run without asking.
    pub fn is_allowed_by(&self, mode: PermissionMode) -> bool {
        match mode {
            PermissionMode::AllowAll => true,
            PermissionMode::AllowEdits => {
                matches!(
                    self.kind,
                    Some(BuiltinToolKind::Readonly | BuiltinToolKind::Edit)
                )
            }
            PermissionMode::AllowReads => self.kind == Some(BuiltinToolKind::Readonly),
        }
    }
}

fn object_schema(fields: &[&str]) -> Value {
    let properties: serde_json::Map<String, Value> = fields
        .iter()
        .map(|field| ((*field).to_string(), json!({ "type": "string" })))
        .collect();
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "required": fields,
        "additionalProperties": false
    })
}

fn baseline(name: BuiltinToolName, description: &str, fields: &[&str]) -> Tool {
    Tool {
        name: name.as_str().into(),
        description: description.into(),
        parameters: object_schema(fields),
        constrained_sampling: None,
    }
}

/// The builtin vocabulary with its baseline input schemas.
pub fn builtin_tools() -> &'static IndexMap<BuiltinToolName, Tool> {
    static TOOLS: OnceLock<IndexMap<BuiltinToolName, Tool>> = OnceLock::new();
    TOOLS.get_or_init(|| {
        use BuiltinToolName as N;
        IndexMap::from([
            (
                N::Read,
                baseline(N::Read, "Read file contents", &["file_path"]),
            ),
            (
                N::Write,
                baseline(
                    N::Write,
                    "Write content to a file",
                    &["file_path", "content"],
                ),
            ),
            (
                N::Edit,
                baseline(
                    N::Edit,
                    "Edit a file by replacing text",
                    &["file_path", "old_string", "new_string"],
                ),
            ),
            (
                N::Bash,
                baseline(N::Bash, "Execute a shell command", &["command"]),
            ),
            (
                N::Grep,
                baseline(N::Grep, "Search file contents with regex", &["pattern"]),
            ),
            (
                N::Glob,
                baseline(N::Glob, "Find files matching a glob pattern", &["pattern"]),
            ),
            (
                N::WebSearch,
                baseline(N::WebSearch, "Search the web", &["query"]),
            ),
            (N::AskUserQuestions, questions_tool()),
        ])
    })
}

/// Options of [`common_tool`].
#[derive(Debug, Clone)]
pub struct CommonToolOptions {
    pub native_name: String,
    pub kind: Option<BuiltinToolKind>,
    pub description: Option<String>,
    pub parameters: Value,
}

/// A builtin that maps to a cross-harness name. Its schema must accept
/// everything the baseline schema accepts.
pub fn common_tool(
    common_name: BuiltinToolName,
    options: CommonToolOptions,
) -> Result<BuiltinTool, HarnessError> {
    let standard = &builtin_tools()[&common_name];
    check_superset(&standard.parameters, &options.parameters).map_err(|reason| {
        HarnessError::new(format!(
            "adapter input schema must be a superset of the standard schema ('{}' / '{}'): {reason}",
            common_name.as_str(),
            options.native_name
        ))
    })?;
    Ok(BuiltinTool {
        tool: Tool {
            name: common_name.as_str().into(),
            description: options.description.unwrap_or_default(),
            parameters: options.parameters,
            constrained_sampling: None,
        },
        native_name: Some(options.native_name),
        common_name: Some(common_name),
        kind: options.kind,
    })
}

fn required_of(schema: &Value) -> Vec<&str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn types_compatible(standard: &Value, adapter: &Value) -> bool {
    let set = |value: &Value| -> Vec<String> {
        match value {
            Value::String(value) => vec![value.clone()],
            Value::Array(values) => values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            _ => vec![],
        }
    };
    match (standard.get("type"), adapter.get("type")) {
        (_, None) => true,
        (None, Some(_)) => false,
        (Some(standard), Some(adapter)) => {
            let adapter = set(adapter);
            set(standard).iter().all(|kind| adapter.contains(kind))
        }
    }
}

fn check_superset(standard: &Value, adapter: &Value) -> Result<(), String> {
    if !types_compatible(standard, adapter) {
        return Err("incompatible type".into());
    }
    let standard_required = required_of(standard);
    for required in required_of(adapter) {
        if !standard_required.contains(&required) {
            return Err(format!(
                "requires '{required}', which the standard does not"
            ));
        }
    }
    let adapter_properties = adapter.get("properties").and_then(Value::as_object);
    if let Some(standard_properties) = standard.get("properties").and_then(Value::as_object) {
        for (name, standard_property) in standard_properties {
            match adapter_properties.and_then(|properties| properties.get(name)) {
                Some(adapter_property) => check_superset(standard_property, adapter_property)
                    .map_err(|error| format!("property '{name}': {error}"))?,
                None if adapter.get("additionalProperties") == Some(&json!(false)) => {
                    return Err(format!("missing property '{name}'"));
                }
                None => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(properties: Value, required: &[&str]) -> Value {
        json!({ "type": "object", "properties": properties, "required": required })
    }

    fn options(native_name: &str, parameters: Value) -> CommonToolOptions {
        CommonToolOptions {
            native_name: native_name.into(),
            kind: None,
            description: None,
            parameters,
        }
    }

    #[test]
    fn names_follow_the_registry() {
        let keys: Vec<_> = builtin_tools().keys().copied().collect();
        assert_eq!(keys, BUILTIN_TOOL_NAMES.to_vec());
        for name in BUILTIN_TOOL_NAMES {
            assert_eq!(BuiltinToolName::from_name(name.as_str()), Some(name));
            assert_eq!(serde_json::to_value(name).unwrap(), json!(name.as_str()));
            assert_eq!(builtin_tools()[&name].name, name.as_str());
        }
        assert_eq!(BuiltinToolName::from_name("notARealName"), None);
    }

    #[test]
    fn accepts_exact_and_superset_schemas() {
        let tool = common_tool(
            BuiltinToolName::Read,
            options(
                "Read",
                schema(json!({ "file_path": { "type": "string" } }), &["file_path"]),
            ),
        )
        .unwrap();
        assert_eq!(tool.native_name.as_deref(), Some("Read"));
        assert_eq!(tool.common_name, Some(BuiltinToolName::Read));
        assert_eq!(tool.tool.name, "read");
        assert!(
            common_tool(
                BuiltinToolName::Read,
                options(
                    "Read",
                    schema(
                        json!({ "file_path": { "type": "string" }, "offset": { "type": "number" } }),
                        &["file_path"],
                    ),
                ),
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_missing_fields_and_incompatible_types() {
        assert!(
            common_tool(
                BuiltinToolName::Read,
                options(
                    "Read",
                    schema(json!({ "path": { "type": "string" } }), &["path"])
                ),
            )
            .is_err()
        );
        assert!(
            common_tool(
                BuiltinToolName::Bash,
                options(
                    "Bash",
                    schema(json!({ "command": { "type": "number" } }), &["command"]),
                ),
            )
            .is_err()
        );
    }

    #[test]
    fn tool_filtering() {
        let allow = BuiltinToolFilter::Allow {
            tool_names: vec!["bash".into()],
        };
        let deny = BuiltinToolFilter::Deny {
            tool_names: vec!["bash".into()],
        };
        assert!(BuiltinToolFilter::includes(None, "x"));
        assert!(BuiltinToolFilter::includes(Some(&allow), "bash"));
        assert!(!BuiltinToolFilter::includes(Some(&allow), "read"));
        assert!(!BuiltinToolFilter::includes(Some(&deny), "bash"));
        assert!(BuiltinToolFilter::includes(Some(&deny), "read"));
        assert_eq!(
            BuiltinToolFilter::denial_reason("bash"),
            "Tool 'bash' is inactive due to the HarnessAgent tool filtering policy."
        );
        assert_eq!(
            serde_json::to_value(&allow).unwrap(),
            json!({ "mode": "allow", "toolNames": ["bash"] })
        );
    }

    #[test]
    fn permission_modes_follow_tool_kinds() {
        let tool = |kind| BuiltinTool {
            kind,
            ..BuiltinTool::from(builtin_tools()[&BuiltinToolName::Read].clone())
        };
        let read = tool(Some(BuiltinToolKind::Readonly));
        let edit = tool(Some(BuiltinToolKind::Edit));
        let bash = tool(Some(BuiltinToolKind::Bash));
        assert_eq!(PermissionMode::default(), PermissionMode::AllowAll);
        assert!(read.is_allowed_by(PermissionMode::AllowReads));
        assert!(!edit.is_allowed_by(PermissionMode::AllowReads));
        assert!(edit.is_allowed_by(PermissionMode::AllowEdits));
        assert!(!bash.is_allowed_by(PermissionMode::AllowEdits));
        assert!(bash.is_allowed_by(PermissionMode::AllowAll));
    }
}
