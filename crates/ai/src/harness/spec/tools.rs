//! Port of `harness-v1-tool-spec.ts`, `harness-v1-builtin-tool.ts`,
//! `harness-v1-tool-filtering.ts`, `harness-v1-permission-mode.ts` and
//! `harness-v1-skill.ts`.

use std::sync::OnceLock;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::harness::errors::HarnessError;
use crate::harness::provider::JsonValue;
use crate::harness::sdk::tool::{Schema, Tool, json_schema};
use crate::harness::spec::questions_tool::harness_v1_questions_tool;

/// `HarnessV1ToolSpec`: a host-defined tool made available to the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ToolSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema (`JSONSchema7`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<JsonValue>,
}

/// `HarnessV1Skill`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1Skill {
    /// Stable identifier (kebab-case slug).
    pub name: String,
    /// Short, model-facing description.
    pub description: String,
    /// Full skill content.
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<HarnessV1SkillFile>>,
}

/// `HarnessV1SkillFile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1SkillFile {
    /// Skill-relative POSIX path.
    pub path: String,
    /// UTF-8 text content.
    pub content: String,
}

/// `HarnessV1PermissionMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HarnessV1PermissionMode {
    AllowReads,
    AllowEdits,
    AllowAll,
}

/// `HarnessV1BuiltinToolFiltering`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "mode",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum HarnessV1BuiltinToolFiltering {
    Allow { tool_names: Vec<String> },
    Deny { tool_names: Vec<String> },
}

/// `isHarnessV1BuiltinToolIncluded`.
pub fn is_harness_v1_builtin_tool_included(
    tool_name: &str,
    tool_filtering: Option<&HarnessV1BuiltinToolFiltering>,
) -> bool {
    match tool_filtering {
        None => true,
        Some(HarnessV1BuiltinToolFiltering::Allow { tool_names }) => {
            tool_names.iter().any(|n| n == tool_name)
        }
        Some(HarnessV1BuiltinToolFiltering::Deny { tool_names }) => {
            !tool_names.iter().any(|n| n == tool_name)
        }
    }
}

/// `getHarnessV1BuiltinToolFilteringDenialReason`.
pub fn get_harness_v1_builtin_tool_filtering_denial_reason(tool_name: &str) -> String {
    format!("Tool '{tool_name}' is inactive due to the HarnessAgent tool filtering policy.")
}

/// `HarnessV1BuiltinToolName`: keys of `HARNESS_V1_BUILTIN_TOOLS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HarnessV1BuiltinToolName {
    Read,
    Write,
    Edit,
    Bash,
    Grep,
    Glob,
    WebSearch,
    AskUserQuestions,
}

impl HarnessV1BuiltinToolName {
    /// The TS key.
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

    /// Parse a TS key.
    pub fn from_name(name: &str) -> Option<Self> {
        HARNESS_V1_BUILTIN_TOOL_NAMES
            .iter()
            .copied()
            .find(|n| n.as_str() == name)
    }
}

/// `HARNESS_V1_BUILTIN_TOOL_NAMES`, in registry order.
pub const HARNESS_V1_BUILTIN_TOOL_NAMES: [HarnessV1BuiltinToolName; 8] = [
    HarnessV1BuiltinToolName::Read,
    HarnessV1BuiltinToolName::Write,
    HarnessV1BuiltinToolName::Edit,
    HarnessV1BuiltinToolName::Bash,
    HarnessV1BuiltinToolName::Grep,
    HarnessV1BuiltinToolName::Glob,
    HarnessV1BuiltinToolName::WebSearch,
    HarnessV1BuiltinToolName::AskUserQuestions,
];

/// `HarnessV1BuiltinToolUseKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HarnessV1BuiltinToolUseKind {
    Readonly,
    Edit,
    Bash,
}

/// `HarnessV1BuiltinTool`: an AI SDK `Tool` plus the harness fields.
#[derive(Debug, Clone)]
pub struct HarnessV1BuiltinTool {
    pub tool: Tool,
    pub native_name: Option<String>,
    pub common_name: Option<HarnessV1BuiltinToolName>,
    pub tool_use_kind: Option<HarnessV1BuiltinToolUseKind>,
}

impl From<Tool> for HarnessV1BuiltinTool {
    /// A built-in declared with `tool()` directly (no common name).
    fn from(tool: Tool) -> Self {
        Self {
            tool,
            native_name: None,
            common_name: None,
            tool_use_kind: None,
        }
    }
}

fn object_schema(fields: &[&str]) -> JsonValue {
    let properties: serde_json::Map<String, JsonValue> = fields
        .iter()
        .map(|f| ((*f).to_string(), json!({ "type": "string" })))
        .collect();
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "required": fields,
        "additionalProperties": false
    })
}

fn baseline(description: &str, fields: &[&str]) -> Tool {
    // `outputSchema: z.unknown()`.
    Tool::new(json_schema(object_schema(fields)))
        .with_description(description)
        .with_output_schema(json_schema(json!({})))
}

/// `HARNESS_V1_BUILTIN_TOOLS`: the cross-harness vocabulary with baseline
/// input schemas.
pub fn harness_v1_builtin_tools() -> &'static IndexMap<HarnessV1BuiltinToolName, Tool> {
    static TOOLS: OnceLock<IndexMap<HarnessV1BuiltinToolName, Tool>> = OnceLock::new();
    TOOLS.get_or_init(|| {
        use HarnessV1BuiltinToolName as N;
        IndexMap::from([
            (N::Read, baseline("Read file contents", &["file_path"])),
            (
                N::Write,
                baseline("Write content to a file", &["file_path", "content"]),
            ),
            (
                N::Edit,
                baseline(
                    "Edit a file by replacing text",
                    &["file_path", "old_string", "new_string"],
                ),
            ),
            (N::Bash, baseline("Execute a shell command", &["command"])),
            (
                N::Grep,
                baseline("Search file contents with regex", &["pattern"]),
            ),
            (
                N::Glob,
                baseline("Find files matching a glob pattern", &["pattern"]),
            ),
            (N::WebSearch, baseline("Search the web", &["query"])),
            (N::AskUserQuestions, harness_v1_questions_tool()),
        ])
    })
}

/// Options of [`common_tool`].
#[derive(Debug, Clone)]
pub struct CommonToolOptions {
    pub native_name: String,
    pub tool_use_kind: Option<HarnessV1BuiltinToolUseKind>,
    pub description: Option<String>,
    pub input_schema: Schema,
}

/// `commonTool(commonName, opts)`.
///
/// [rust] TS rejects an adapter schema that is not a superset of the
/// baseline at compile time. Rust checks it at runtime and returns an error:
/// every property the baseline requires must be declared by the adapter
/// schema with a compatible `type` (or with no type), and the adapter may
/// only require properties the baseline requires.
pub fn common_tool(
    common_name: HarnessV1BuiltinToolName,
    opts: CommonToolOptions,
) -> Result<HarnessV1BuiltinTool, HarnessError> {
    let standard = &harness_v1_builtin_tools()[&common_name]
        .input_schema
        .json_schema;
    check_superset(standard, &opts.input_schema.json_schema).map_err(|reason| {
        HarnessError::new(format!(
            "ERROR: adapter input schema must be a superset of the standard schema \
             ('{}' / '{}'): {reason}",
            common_name.as_str(),
            opts.native_name
        ))
    })?;
    let mut tool = Tool::new(opts.input_schema);
    tool.description = opts.description;
    Ok(HarnessV1BuiltinTool {
        tool,
        native_name: Some(opts.native_name),
        common_name: Some(common_name),
        tool_use_kind: opts.tool_use_kind,
    })
}

fn required_of(schema: &JsonValue) -> Vec<&str> {
    schema
        .get("required")
        .and_then(JsonValue::as_array)
        .map(|r| r.iter().filter_map(JsonValue::as_str).collect())
        .unwrap_or_default()
}

fn types_compatible(standard: &JsonValue, adapter: &JsonValue) -> bool {
    match (standard.get("type"), adapter.get("type")) {
        (_, None) => true,
        (None, Some(_)) => false,
        (Some(s), Some(a)) => {
            let set = |v: &JsonValue| -> Vec<String> {
                match v {
                    JsonValue::String(s) => vec![s.clone()],
                    JsonValue::Array(a) => a
                        .iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect(),
                    _ => vec![],
                }
            };
            let a = set(a);
            set(s).iter().all(|t| a.contains(t))
        }
    }
}

fn check_superset(standard: &JsonValue, adapter: &JsonValue) -> Result<(), String> {
    if !types_compatible(standard, adapter) {
        return Err("incompatible type".into());
    }
    let std_required = required_of(standard);
    for r in required_of(adapter) {
        if !std_required.contains(&r) {
            return Err(format!("requires '{r}', which the standard does not"));
        }
    }
    let std_props = standard.get("properties").and_then(JsonValue::as_object);
    let adapter_props = adapter.get("properties").and_then(JsonValue::as_object);
    if let Some(std_props) = std_props {
        for (name, std_prop) in std_props {
            match adapter_props.and_then(|p| p.get(name)) {
                Some(adapter_prop) => check_superset(std_prop, adapter_prop)
                    .map_err(|e| format!("property '{name}': {e}"))?,
                None => {
                    // An open schema still accepts the property.
                    if adapter.get("additionalProperties") == Some(&json!(false)) {
                        return Err(format!("missing property '{name}'"));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // harness-v1-builtin-tool.test-d.ts (runtime form of the type tests)
    #[test]
    fn names_are_derived_from_the_registry_keys() {
        let keys: Vec<_> = harness_v1_builtin_tools().keys().copied().collect();
        assert_eq!(keys, HARNESS_V1_BUILTIN_TOOL_NAMES.to_vec());
        for n in HARNESS_V1_BUILTIN_TOOL_NAMES {
            assert_eq!(HarnessV1BuiltinToolName::from_name(n.as_str()), Some(n));
            assert_eq!(serde_json::to_value(n).unwrap(), json!(n.as_str()));
        }
    }

    fn read_schema(props: JsonValue, required: &[&str]) -> Schema {
        json_schema(json!({ "type": "object", "properties": props, "required": required }))
    }

    #[test]
    fn accepts_a_schema_that_exactly_matches_the_standard() {
        let t = common_tool(
            HarnessV1BuiltinToolName::Read,
            CommonToolOptions {
                native_name: "Read".into(),
                tool_use_kind: None,
                description: None,
                input_schema: read_schema(
                    json!({ "file_path": { "type": "string" } }),
                    &["file_path"],
                ),
            },
        )
        .unwrap();
        assert_eq!(t.native_name.as_deref(), Some("Read"));
        assert_eq!(t.common_name, Some(HarnessV1BuiltinToolName::Read));
    }

    #[test]
    fn accepts_a_schema_that_is_a_strict_superset_of_the_standard() {
        assert!(common_tool(
            HarnessV1BuiltinToolName::Read,
            CommonToolOptions {
                native_name: "Read".into(),
                tool_use_kind: Some(HarnessV1BuiltinToolUseKind::Readonly),
                description: None,
                input_schema: read_schema(
                    json!({ "file_path": { "type": "string" }, "offset": { "type": "number" } }),
                    &["file_path"]
                ),
            },
        )
        .is_ok());
    }

    #[test]
    fn rejects_a_schema_missing_a_required_standard_field() {
        assert!(
            common_tool(
                HarnessV1BuiltinToolName::Read,
                CommonToolOptions {
                    native_name: "Read".into(),
                    tool_use_kind: None,
                    description: None,
                    input_schema: read_schema(json!({ "path": { "type": "string" } }), &["path"]),
                },
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_an_incompatible_field_type() {
        assert!(
            common_tool(
                HarnessV1BuiltinToolName::Bash,
                CommonToolOptions {
                    native_name: "Bash".into(),
                    tool_use_kind: None,
                    description: None,
                    input_schema: read_schema(
                        json!({ "command": { "type": "number" } }),
                        &["command"]
                    ),
                },
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_an_unknown_common_name() {
        assert_eq!(HarnessV1BuiltinToolName::from_name("notARealName"), None);
    }

    #[test]
    fn tool_filtering() {
        let allow = HarnessV1BuiltinToolFiltering::Allow {
            tool_names: vec!["bash".into()],
        };
        let deny = HarnessV1BuiltinToolFiltering::Deny {
            tool_names: vec!["bash".into()],
        };
        assert!(is_harness_v1_builtin_tool_included("x", None));
        assert!(is_harness_v1_builtin_tool_included("bash", Some(&allow)));
        assert!(!is_harness_v1_builtin_tool_included("read", Some(&allow)));
        assert!(!is_harness_v1_builtin_tool_included("bash", Some(&deny)));
        assert!(is_harness_v1_builtin_tool_included("read", Some(&deny)));
        assert_eq!(
            get_harness_v1_builtin_tool_filtering_denial_reason("bash"),
            "Tool 'bash' is inactive due to the HarnessAgent tool filtering policy."
        );
        assert_eq!(
            serde_json::to_value(&allow).unwrap(),
            json!({ "mode": "allow", "toolNames": ["bash"] })
        );
    }
}
