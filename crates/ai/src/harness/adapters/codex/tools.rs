//! Codex's builtin tools, tool filtering (`codex-tool-filtering.ts`), the
//! trusted `apply_patch` filtering hook (`codex-tool-filtering-hook.ts`) and
//! dynamic tool names.

use std::collections::HashSet;
use std::sync::LazyLock;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::Tool;
use crate::harness::adapters::builtin_catalog::load_builtin_tool_catalog;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::tools::{BuiltinTool, BuiltinToolFilter};

/// `CODEX_BUILTIN_TOOLS`, keyed by harness name.
pub fn codex_builtin_tools() -> &'static IndexMap<String, BuiltinTool> {
    static TOOLS: LazyLock<IndexMap<String, BuiltinTool>> =
        LazyLock::new(|| load_builtin_tool_catalog(include_str!("builtin_tools.json")));
    &TOOLS
}

/// `CodexBuiltinToolPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodexBuiltinToolPolicy {
    pub bash_disabled: bool,
    pub web_search_disabled: bool,
    pub apply_patch_disabled: bool,
    pub view_image_disabled: bool,
    pub disable_environments: bool,
    pub deny_apply_patch_with_hook: bool,
    pub web_search_live: bool,
}

/// `resolveCodexBuiltinToolPolicy`.
pub fn resolve_codex_builtin_tool_policy(
    filter: Option<&BuiltinToolFilter>,
    web_search: bool,
) -> CodexBuiltinToolPolicy {
    let included = |name: &str| BuiltinToolFilter::includes(filter, name);
    let bash_disabled = !included("bash");
    let web_search_disabled = !included("webSearch");
    let apply_patch_disabled = !included("apply_patch");
    let view_image_disabled = !included("view_image");
    let disable_environments = bash_disabled && apply_patch_disabled && view_image_disabled;
    CodexBuiltinToolPolicy {
        bash_disabled,
        web_search_disabled,
        apply_patch_disabled,
        view_image_disabled,
        disable_environments,
        deny_apply_patch_with_hook: apply_patch_disabled && !disable_environments,
        web_search_live: web_search && !web_search_disabled,
    }
}

/// `toTomlInline`: a JSON value as an inline TOML value for `--config`.
pub fn to_toml_inline(value: &Value) -> HarnessResult<String> {
    Ok(match value {
        Value::String(text) => serde_json::to_string(text)?,
        Value::Bool(value) => value.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(to_toml_inline)
                .collect::<HarnessResult<Vec<_>>>()?
                .join(",")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, item)| Ok(format!(
                    "{}={}",
                    serde_json::to_string(key)?,
                    to_toml_inline(item)?
                )))
                .collect::<HarnessResult<Vec<_>>>()?
                .join(",")
        ),
        Value::Null => {
            return Err(
                HarnessError::new("Codex configuration contains an unsupported value.").into(),
            );
        }
    })
}

const HOOK_MATCHER: &str = "^apply_patch$";
const HOOK_TIMEOUT_SECONDS: u64 = 10;
pub const HOOK_COMMAND: &str = "printf '%s\\n' \"Tool 'apply_patch' is inactive due to the HarnessAgent tool filtering policy.\" >&2; exit 2";

/// `TrustedApplyPatchHook`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrustedApplyPatchHook {
    pub cli_overrides: Vec<String>,
    pub thread_config: Map<String, Value>,
    pub key: String,
    pub hash: String,
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, item)| (key.clone(), canonicalize(item)))
                    .collect(),
            )
        }
        other => other.clone(),
    }
}

pub(crate) fn sha256_hex(text: &str) -> String {
    ring::digest::digest(&ring::digest::SHA256, text.as_bytes())
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `createTrustedApplyPatchHook`: a `PreToolUse` hook that blocks
/// `apply_patch`, trusted up front so Codex runs it.
pub fn create_trusted_apply_patch_hook(
    codex_config: &Map<String, Value>,
) -> HarnessResult<TrustedApplyPatchHook> {
    let mut thread_config = codex_config.clone();
    let mut hooks = thread_config
        .remove("hooks")
        .and_then(|hooks| hooks.as_object().cloned())
        .unwrap_or_default();
    let dotted: Vec<String> = thread_config
        .keys()
        .filter(|key| key.starts_with("hooks."))
        .cloned()
        .collect();
    for key in dotted {
        let value = thread_config.remove(&key).unwrap();
        let name = &key["hooks.".len()..];
        if let Some(state_key) = name.strip_prefix("state.") {
            let state = hooks.entry("state").or_insert_with(|| json!({}));
            if !state.is_object() {
                *state = json!({});
            }
            state[state_key] = value;
        } else {
            hooks.insert(name.to_string(), value);
        }
    }
    let user_groups = match hooks.get("PreToolUse") {
        None => Vec::new(),
        Some(Value::Array(groups)) => groups.clone(),
        Some(_) => {
            return Err(
                HarnessError::new("Codex config hooks.PreToolUse must be an array.").into(),
            );
        }
    };
    let handler = json!({
        "type": "command",
        "command": HOOK_COMMAND,
        "timeout": HOOK_TIMEOUT_SECONDS,
        "async": false,
    });
    let hash = format!(
        "sha256:{}",
        sha256_hex(
            &canonicalize(&json!({
                "event_name": "pre_tool_use",
                "matcher": HOOK_MATCHER,
                "hooks": [handler],
            }))
            .to_string()
        )
    );
    let key = format!(
        "/<session-flags>/config.toml:pre_tool_use:{}:0",
        user_groups.len()
    );
    let mut groups = user_groups;
    groups.push(json!({ "matcher": HOOK_MATCHER, "hooks": [handler] }));
    hooks.insert("PreToolUse".into(), Value::Array(groups));
    let state = hooks.entry("state").or_insert_with(|| json!({}));
    if !state.is_object() {
        *state = json!({});
    }
    state[&key] = json!({ "enabled": true, "trusted_hash": hash });
    Ok(TrustedApplyPatchHook {
        cli_overrides: vec![
            format!("hooks={}", to_toml_inline(&Value::Object(hooks))?),
            "features.hooks=true".to_string(),
        ],
        thread_config,
        key,
        hash,
    })
}

/// `assertTrustedApplyPatchHook`: the `hooks/list` response shows the hook
/// loaded and trusted.
pub fn assert_trusted_apply_patch_hook(
    response: &Value,
    hook: &TrustedApplyPatchHook,
) -> HarnessResult<()> {
    let entry = response
        .get("data")
        .and_then(Value::as_array)
        .filter(|data| data.len() == 1)
        .map(|data| &data[0]);
    let matches: Vec<&Value> = entry
        .and_then(|entry| entry.get("hooks"))
        .and_then(Value::as_array)
        .map(|hooks| {
            hooks
                .iter()
                .filter(|item| item.get("key").and_then(Value::as_str) == Some(&hook.key))
                .collect()
        })
        .unwrap_or_default();
    let ok = matches.len() == 1 && {
        let found = matches[0];
        let field = |name: &str| found.get(name);
        field("eventName") == Some(&json!("preToolUse"))
            && field("handlerType") == Some(&json!("command"))
            && field("command") == Some(&json!(HOOK_COMMAND))
            && field("matcher") == Some(&json!(HOOK_MATCHER))
            && field("currentHash") == Some(&json!(hook.hash))
            && field("enabled") == Some(&json!(true))
            && field("trustStatus") == Some(&json!("trusted"))
            && !entry
                .and_then(|entry| entry.get("errors"))
                .and_then(Value::as_array)
                .is_some_and(|errors| !errors.is_empty())
    };
    if ok {
        Ok(())
    } else {
        Err(HarnessError::new(
            "Codex app-server did not load the trusted apply_patch filtering hook.",
        )
        .into())
    }
}

/// `createDynamicTools`: the caller's tools as Codex dynamic tools. Names
/// Codex does not accept get a stable alias.
pub struct DynamicTools {
    pub specs: Vec<Value>,
    pub original_name_by_alias: IndexMap<String, String>,
}

fn is_valid_dynamic_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && name != "mcp"
        && !name.starts_with("mcp__")
}

pub fn create_dynamic_tools(tools: &[Tool]) -> DynamicTools {
    let valid: HashSet<&str> = tools
        .iter()
        .map(|tool| tool.name.as_str())
        .filter(|name| is_valid_dynamic_tool_name(name))
        .collect();
    let mut used = HashSet::new();
    let mut original_name_by_alias = IndexMap::new();
    let specs = tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let mut alias = tool.name.clone();
            if !is_valid_dynamic_tool_name(&alias) || used.contains(&alias) {
                let base = format!("ai_sdk_tool_{}", &sha256_hex(&tool.name)[..16]);
                alias = base.clone();
                let mut suffix = index;
                while used.contains(&alias) || valid.contains(alias.as_str()) {
                    alias = format!("{base}_{suffix}");
                    suffix += 1;
                }
            }
            used.insert(alias.clone());
            original_name_by_alias.insert(alias.clone(), tool.name.clone());
            json!({
                "type": "function",
                "name": alias,
                "description": tool.description,
                "inputSchema": tool.parameters,
            })
        })
        .collect();
    DynamicTools {
        specs,
        original_name_by_alias,
    }
}

/// `fingerprintCodexTurnConfiguration`: dynamic tools and instructions are
/// fixed when a thread starts, so a change starts a new thread.
pub fn fingerprint_turn_configuration(instructions: Option<&str>, tools: &[Tool]) -> String {
    let tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            })
        })
        .collect();
    let canonical = canonicalize(&json!({ "instructions": instructions, "tools": tools }));
    format!("v2:{}", sha256_hex(&canonical.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Tool {
        Tool {
            name: name.into(),
            description: "d".into(),
            parameters: json!({ "type": "object" }),
            constrained_sampling: None,
        }
    }

    #[test]
    fn resolves_the_tool_policy() {
        let policy = resolve_codex_builtin_tool_policy(None, true);
        assert!(policy.web_search_live);
        assert!(!policy.disable_environments);
        let deny = BuiltinToolFilter::Deny {
            tool_names: vec!["apply_patch".into()],
        };
        let policy = resolve_codex_builtin_tool_policy(Some(&deny), false);
        assert!(policy.deny_apply_patch_with_hook);
        assert!(!policy.web_search_live);
        let allow = BuiltinToolFilter::Allow {
            tool_names: vec!["webSearch".into()],
        };
        let policy = resolve_codex_builtin_tool_policy(Some(&allow), true);
        assert!(policy.disable_environments);
        assert!(!policy.deny_apply_patch_with_hook);
        assert!(policy.web_search_live);
    }

    #[test]
    fn writes_inline_toml() {
        assert_eq!(
            to_toml_inline(&json!({ "a": [1, "x", true], "b": { "c": "d" } })).unwrap(),
            r#"{"a"=[1,"x",true],"b"={"c"="d"}}"#
        );
        assert!(to_toml_inline(&Value::Null).is_err());
    }

    #[test]
    fn creates_the_trusted_hook() {
        let config: Map<String, Value> = serde_json::from_value(json!({
            "model": "x",
            "hooks.state.other": { "enabled": false },
            "hooks": { "PreToolUse": [{ "matcher": "x", "hooks": [] }] },
        }))
        .unwrap();
        let hook = create_trusted_apply_patch_hook(&config).unwrap();
        assert_eq!(
            hook.thread_config,
            serde_json::from_value::<Map<String, Value>>(json!({ "model": "x" })).unwrap()
        );
        assert_eq!(hook.key, "/<session-flags>/config.toml:pre_tool_use:1:0");
        assert!(hook.hash.starts_with("sha256:"));
        assert!(hook.cli_overrides[0].starts_with("hooks={"));
        assert!(hook.cli_overrides[0].contains("\"other\"={\"enabled\"=false}"));
        assert_eq!(hook.cli_overrides[1], "features.hooks=true");
        let response = json!({ "data": [{ "hooks": [{
            "key": hook.key, "eventName": "preToolUse", "handlerType": "command",
            "command": HOOK_COMMAND, "matcher": "^apply_patch$", "currentHash": hook.hash,
            "enabled": true, "trustStatus": "trusted",
        }], "errors": [] }] });
        assert_trusted_apply_patch_hook(&response, &hook).unwrap();
        assert!(assert_trusted_apply_patch_hook(&json!({ "data": [] }), &hook).is_err());
    }

    #[test]
    fn aliases_invalid_dynamic_tool_names() {
        let tools = create_dynamic_tools(&[
            tool("lookup"),
            tool("mcp__x"),
            tool("has space"),
            tool("lookup"),
        ]);
        let names: Vec<&str> = tools
            .specs
            .iter()
            .map(|spec| spec["name"].as_str().unwrap())
            .collect();
        assert_eq!(names[0], "lookup");
        assert!(names[1].starts_with("ai_sdk_tool_"));
        assert!(names[2].starts_with("ai_sdk_tool_"));
        assert!(names[3].starts_with("ai_sdk_tool_") && names[3] != names[0]);
        assert_eq!(tools.original_name_by_alias[names[2]], "has space");
        assert_eq!(codex_builtin_tools().len(), 4);
        assert_eq!(
            fingerprint_turn_configuration(Some("a"), &[tool("x")]),
            fingerprint_turn_configuration(Some("a"), &[tool("x")])
        );
        assert_ne!(
            fingerprint_turn_configuration(Some("a"), &[tool("x")]),
            fingerprint_turn_configuration(None, &[tool("x")])
        );
    }
}
