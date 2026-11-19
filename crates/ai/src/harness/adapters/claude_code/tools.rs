//! Claude Code's builtin tools: names, kinds, tool filtering and the
//! approval rules (`NATIVE_TO_COMMON`, `NATIVE_TOOL_KINDS`,
//! `tool-filtering.ts`, `needsHostApproval` and `permission-paths.ts` of the
//! TS bridge).

use std::sync::LazyLock;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::builtin_catalog::load_builtin_tool_catalog;
use crate::harness::paths::{is_absolute_posix, normalize_posix};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};
use crate::harness::tools::{BuiltinTool, BuiltinToolFilter, BuiltinToolKind, PermissionMode};
use crate::harness::utils::{is_sandbox_write_protected_path, shell_quote};

/// The MCP server that serves the caller's tools.
pub const HOST_TOOL_SERVER: &str = "harness-tools";
/// The prefix Claude Code gives the caller's tools.
pub const HOST_TOOL_PREFIX: &str = "mcp__harness-tools__";

/// Every native tool the Claude Code CLI can run, keyed by harness name.
pub fn claude_code_builtin_tools() -> &'static IndexMap<String, BuiltinTool> {
    static TOOLS: LazyLock<IndexMap<String, BuiltinTool>> =
        LazyLock::new(|| load_builtin_tool_catalog(include_str!("builtin_tools.json")));
    &TOOLS
}

/// `NATIVE_TO_COMMON`.
const NATIVE_TO_COMMON: [(&str, &str); 8] = [
    ("Read", "read"),
    ("Write", "write"),
    ("Edit", "edit"),
    ("Bash", "bash"),
    ("Glob", "glob"),
    ("Grep", "grep"),
    ("WebSearch", "webSearch"),
    ("AskUserQuestion", "askUserQuestions"),
];

/// `CLAUDE_CODE_WRITE_PROTECTED_PATHS`: workspace paths that hold Claude
/// Code configuration or skills, which can run code or change the agent's
/// behavior in later sessions.
pub const CLAUDE_CODE_WRITE_PROTECTED_PATHS: [&str; 3] = [".agents/skills", ".claude", ".mcp.json"];

/// `NATIVE_TOOL_KINDS`.
pub const NATIVE_TOOL_KINDS: [(&str, BuiltinToolKind); 26] = [
    ("Read", BuiltinToolKind::Readonly),
    ("Glob", BuiltinToolKind::Readonly),
    ("Grep", BuiltinToolKind::Readonly),
    ("WebSearch", BuiltinToolKind::Readonly),
    ("WebFetch", BuiltinToolKind::Readonly),
    ("TaskGet", BuiltinToolKind::Readonly),
    ("TaskList", BuiltinToolKind::Readonly),
    ("TaskOutput", BuiltinToolKind::Readonly),
    ("ListAgents", BuiltinToolKind::Readonly),
    ("ListMcpResources", BuiltinToolKind::Readonly),
    ("ReadMcpResource", BuiltinToolKind::Readonly),
    ("Write", BuiltinToolKind::Edit),
    ("Edit", BuiltinToolKind::Edit),
    ("NotebookEdit", BuiltinToolKind::Edit),
    ("TodoWrite", BuiltinToolKind::Edit),
    ("TaskCreate", BuiltinToolKind::Edit),
    ("TaskUpdate", BuiltinToolKind::Edit),
    ("TaskStop", BuiltinToolKind::Edit),
    ("EnterWorktree", BuiltinToolKind::Edit),
    ("ExitWorktree", BuiltinToolKind::Edit),
    ("ExitPlanMode", BuiltinToolKind::Edit),
    ("Skill", BuiltinToolKind::Readonly),
    ("AskUserQuestion", BuiltinToolKind::Readonly),
    ("ToolSearch", BuiltinToolKind::Readonly),
    ("Bash", BuiltinToolKind::Bash),
    ("Monitor", BuiltinToolKind::Bash),
];

fn native_tool_kind(native_name: &str) -> Option<BuiltinToolKind> {
    NATIVE_TOOL_KINDS
        .iter()
        .find(|(name, _)| *name == native_name)
        .map(|(_, kind)| *kind)
}

/// `PUBLIC_TO_NATIVE` of `tool-filtering.ts`.
const PUBLIC_TO_NATIVE: [(&str, &str); 26] = [
    ("read", "Read"),
    ("write", "Write"),
    ("edit", "Edit"),
    ("bash", "Bash"),
    ("glob", "Glob"),
    ("grep", "Grep"),
    ("webSearch", "WebSearch"),
    ("WebFetch", "WebFetch"),
    ("NotebookEdit", "NotebookEdit"),
    ("TodoWrite", "TodoWrite"),
    ("Agent", "Agent"),
    ("TaskCreate", "TaskCreate"),
    ("TaskGet", "TaskGet"),
    ("TaskUpdate", "TaskUpdate"),
    ("TaskList", "TaskList"),
    ("TaskStop", "TaskStop"),
    ("TaskOutput", "TaskOutput"),
    ("Monitor", "Monitor"),
    ("ListMcpResources", "ListMcpResources"),
    ("ReadMcpResource", "ReadMcpResource"),
    ("ExitPlanMode", "ExitPlanMode"),
    ("EnterWorktree", "EnterWorktree"),
    ("ExitWorktree", "ExitWorktree"),
    ("askUserQuestions", "AskUserQuestion"),
    ("Skill", "Skill"),
    ("ToolSearch", "ToolSearch"),
];

/// `toCommonName`: the harness name of a native tool. The caller's tools
/// lose their MCP prefix.
pub fn to_harness_name(native_name: &str) -> String {
    if let Some(name) = native_name.strip_prefix(HOST_TOOL_PREFIX) {
        return name.to_string();
    }
    NATIVE_TO_COMMON
        .iter()
        .find(|(native, _)| *native == native_name)
        .map_or_else(|| native_name.to_string(), |(_, common)| common.to_string())
}

fn to_native_name(tool_name: &str) -> String {
    PUBLIC_TO_NATIVE
        .iter()
        .find(|(public, _)| *public == tool_name)
        .map_or_else(|| tool_name.to_string(), |(_, native)| native.to_string())
}

/// `resolveNativeTools`: the `--tools` list of an allow filter.
pub fn resolve_native_tools(filter: Option<&BuiltinToolFilter>) -> Option<Vec<String>> {
    match filter {
        Some(BuiltinToolFilter::Allow { tool_names }) => {
            Some(tool_names.iter().map(|name| to_native_name(name)).collect())
        }
        _ => None,
    }
}

/// `resolveInactiveNativeTools`: the native tools a filter turns off.
pub fn resolve_inactive_native_tools(filter: Option<&BuiltinToolFilter>) -> Vec<String> {
    match filter {
        None => Vec::new(),
        Some(BuiltinToolFilter::Allow { tool_names }) => PUBLIC_TO_NATIVE
            .iter()
            .filter(|(public, _)| !tool_names.iter().any(|name| name == public))
            .map(|(_, native)| native.to_string())
            .collect(),
        Some(BuiltinToolFilter::Deny { tool_names }) => {
            tool_names.iter().map(|name| to_native_name(name)).collect()
        }
    }
}

/// `createPermissionSettings`: `ask` rules for the tools the mode does not
/// allow, so Claude Code asks before they run.
pub fn create_permission_settings(
    permission_mode: PermissionMode,
    auto_mode: bool,
    inactive_native_tools: &[String],
) -> Option<Map<String, Value>> {
    let mut ask_rules: Vec<String> = Vec::new();
    for (native_name, kind) in NATIVE_TOOL_KINDS {
        let asks = inactive_native_tools.iter().any(|name| name == native_name)
            || (!auto_mode
                && match permission_mode {
                    PermissionMode::AllowReads => {
                        matches!(kind, BuiltinToolKind::Edit | BuiltinToolKind::Bash)
                    }
                    PermissionMode::AllowEdits => kind == BuiltinToolKind::Bash,
                    PermissionMode::AllowAll => false,
                });
        let rule = format!("{native_name}(*)");
        if asks && !ask_rules.contains(&rule) {
            ask_rules.push(rule);
        }
    }
    if ask_rules.is_empty() {
        return None;
    }
    let mut settings = Map::new();
    settings.insert("permissions".into(), json!({ "ask": ask_rules }));
    settings.insert(
        "sandbox".into(),
        json!({ "autoAllowBashIfSandboxed": false }),
    );
    Some(settings)
}

/// `toGatewayModelOverride`: Claude Code misreads AI Gateway ids with a
/// dotted version such as `anthropic/claude-opus-5.5`; map the id it expects
/// to the gateway id.
fn to_gateway_model_override(model: Option<&str>) -> Option<(String, String)> {
    let model = model?;
    let id = model.strip_prefix("anthropic/")?;
    let id = match id.rfind('[') {
        Some(start) if id.ends_with(']') && !id[start + 1..id.len() - 1].contains(']') => {
            &id[..start]
        }
        _ => id,
    };
    id.contains('.')
        .then(|| (id.replace('.', "-"), model.to_string()))
}

/// `createGatewayModelOverrides`.
pub fn create_gateway_model_overrides(
    model: Option<&str>,
    env: &IndexMap<String, String>,
) -> Option<Map<String, Value>> {
    let mut overrides = Map::new();
    for source in [
        env.get("ANTHROPIC_DEFAULT_OPUS_MODEL").map(String::as_str),
        env.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
            .map(String::as_str),
        env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL").map(String::as_str),
        env.get("CLAUDE_CODE_SUBAGENT_MODEL").map(String::as_str),
        model,
    ] {
        if let Some((key, value)) = to_gateway_model_override(source) {
            overrides.insert(key, Value::String(value));
        }
    }
    (!overrides.is_empty()).then_some(overrides)
}

/// `nativeToolRequiresApproval`.
pub fn native_tool_requires_approval(native_name: &str, permission_mode: PermissionMode) -> bool {
    if permission_mode == PermissionMode::AllowAll {
        return false;
    }
    if native_name == "EnterWorktree" || native_name == "ExitWorktree" {
        return true;
    }
    let Some(kind) = native_tool_kind(native_name) else {
        return true;
    };
    match permission_mode {
        PermissionMode::AllowEdits => kind == BuiltinToolKind::Bash,
        _ => matches!(kind, BuiltinToolKind::Edit | BuiltinToolKind::Bash),
    }
}

/// What a tool call touches, for the path checks.
pub struct ApprovalContext<'a> {
    pub sandbox: &'a dyn SandboxSession,
    pub work_dir: &'a str,
    pub cwd: &'a str,
    pub readable_skill_roots: &'a [String],
    pub additional_directories: &'a [String],
    pub permission_mode: PermissionMode,
    pub inactive_native_tools: &'a [String],
    pub abort: Option<CancellationToken>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Access {
    Read,
    Write,
}

fn glob_needs_approval(pattern: Option<&Value>) -> bool {
    let Some(pattern) = pattern.and_then(Value::as_str) else {
        return true;
    };
    pattern.contains("..")
        || pattern.contains(['{', '}', '(', ')', '[', ']'])
        || pattern.contains('\\')
        || pattern.starts_with('/')
}

/// `needsHostApproval`: whether a native tool call must be approved by the
/// caller under the permission mode.
pub async fn needs_host_approval(
    context: &ApprovalContext<'_>,
    tool_name: &str,
    tool_input: &Value,
    blocked_path: Option<&str>,
) -> bool {
    if tool_name.starts_with(HOST_TOOL_PREFIX) {
        return false;
    }
    if context
        .inactive_native_tools
        .iter()
        .any(|name| name == tool_name)
        || native_tool_requires_approval(tool_name, context.permission_mode)
    {
        return true;
    }
    let empty = Map::new();
    let args = tool_input.as_object().unwrap_or(&empty);
    if tool_name == "Glob" && glob_needs_approval(args.get("pattern")) {
        return true;
    }
    let (mut paths, access): (Vec<Option<&Value>>, Access) = match tool_name {
        "Read" => (vec![args.get("file_path")], Access::Read),
        "Write" | "Edit" => (vec![args.get("file_path")], Access::Write),
        "NotebookEdit" => (vec![args.get("notebook_path")], Access::Write),
        "Glob" | "Grep" => (vec![args.get("path")], Access::Read),
        _ => {
            if blocked_path.is_none() {
                return false;
            }
            (Vec::new(), Access::Read)
        }
    };
    let cwd_value = Value::String(context.cwd.to_string());
    if matches!(tool_name, "Glob" | "Grep") && paths[0].is_none() {
        paths[0] = Some(&cwd_value);
    }
    let blocked = blocked_path.map(|path| Value::String(path.to_string()));
    if let Some(blocked) = &blocked {
        paths.push(Some(blocked));
    }
    let mut strings = Vec::new();
    for path in paths {
        match path.and_then(Value::as_str) {
            Some(path) => strings.push(path.to_string()),
            None => return true,
        }
    }
    if !are_file_tool_paths_trusted(context, &strings, access).await {
        return true;
    }
    access == Access::Write && is_write_protected(context, &strings).await
}

/// `isProtectedClaudeFileWrite`: in auto mode, only writes to protected paths
/// ask the caller.
pub async fn is_protected_claude_file_write(
    context: &ApprovalContext<'_>,
    tool_name: &str,
    tool_input: &Value,
) -> bool {
    let target = match tool_name {
        "Write" | "Edit" => tool_input.get("file_path"),
        "NotebookEdit" => tool_input.get("notebook_path"),
        _ => return false,
    };
    let Some(target) = target.and_then(Value::as_str) else {
        return true;
    };
    is_write_protected(context, &[target.to_string()]).await
}

async fn is_write_protected(context: &ApprovalContext<'_>, paths: &[String]) -> bool {
    let absolute: Vec<String> = paths
        .iter()
        .map(|path| {
            if is_absolute_posix(path) {
                path.clone()
            } else {
                format!("{}/{path}", context.cwd)
            }
        })
        .collect();
    let protected: Vec<String> = CLAUDE_CODE_WRITE_PROTECTED_PATHS
        .iter()
        .map(|path| path.to_string())
        .collect();
    is_sandbox_write_protected_path(
        context.sandbox,
        context.work_dir,
        &absolute,
        &protected,
        context.abort.clone(),
    )
    .await
    .unwrap_or(true)
}

fn relative_inside(root: &str, candidate: &str) -> bool {
    let root = root.trim_end_matches('/');
    let root = if root.is_empty() { "/" } else { root };
    candidate == root
        || root == "/"
        || candidate
            .strip_prefix(root)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Resolves paths in the sandbox the way `resolvePermissionPath` does: the
/// real path of the deepest existing ancestor plus the missing segments.
/// Prints one line per path, or `!` when it cannot be resolved.
const RESOLVE_PATHS_SCRIPT: &str = r#"
resolve() {
  p=$1; missing=""
  while [ ! -e "$p" ] && [ ! -L "$p" ]; do
    seg=${p##*/}; parent=${p%/*}
    [ -z "$parent" ] && parent=/
    if [ "$parent" = "$p" ] || [ "$seg" = "." ] || [ "$seg" = ".." ]; then echo '!'; return; fi
    missing="/$seg$missing"; p=$parent
  done
  real=$(realpath "$p" 2>/dev/null || readlink -f "$p" 2>/dev/null) || { echo '!'; return; }
  [ -z "$real" ] && { echo '!'; return; }
  [ "$real" = "/" ] && [ -n "$missing" ] && real=""
  printf '%s%s\n' "$real" "$missing"
}
for p in "$@"; do resolve "$p"; done
"#;

async fn resolve_sandbox_paths(
    context: &ApprovalContext<'_>,
    paths: &[String],
) -> Option<Vec<String>> {
    let args: Vec<String> = paths.iter().map(|path| shell_quote(path)).collect();
    let command = format!(
        "sh -c {} resolve {}",
        shell_quote(RESOLVE_PATHS_SCRIPT),
        args.join(" ")
    );
    let result = context
        .sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command,
                abort_signal: context.abort.clone(),
                ..Default::default()
            },
            max_output_length: None,
        })
        .await
        .ok()?;
    if result.exit_code != 0 {
        return None;
    }
    let lines: Vec<String> = result.stdout.lines().map(str::to_string).collect();
    (lines.len() == paths.len()).then_some(lines)
}

/// `areFileToolPathsTrusted`: every path stays inside the work dir, an
/// additional directory or (for reads) a skill root, after symlinks.
pub(crate) async fn are_file_tool_paths_trusted(
    context: &ApprovalContext<'_>,
    paths: &[String],
    access: Access,
) -> bool {
    if paths.is_empty() {
        return false;
    }
    let mut root_inputs = vec![context.work_dir.to_string()];
    root_inputs.extend(
        context
            .additional_directories
            .iter()
            .filter(|root| is_absolute_posix(root))
            .cloned(),
    );
    if access == Access::Read {
        root_inputs.extend(context.readable_skill_roots.iter().cloned());
    }
    let mut candidates = Vec::new();
    for path in paths {
        if path.is_empty() || path != path.trim() || path == "~" || path.starts_with("~/") {
            return false;
        }
        if !is_absolute_posix(path) && !is_absolute_posix(context.cwd) {
            return false;
        }
        let absolute = if is_absolute_posix(path) {
            path.clone()
        } else {
            format!("{}/{path}", context.cwd)
        };
        let normalized = normalize_posix(&absolute);
        candidates.push(absolute);
        candidates.push(normalized);
    }
    let mut all = root_inputs.clone();
    all.extend(candidates.iter().cloned());
    let Some(resolved) = resolve_sandbox_paths(context, &all).await else {
        return false;
    };
    let (roots, resolved_candidates) = resolved.split_at(root_inputs.len());
    // The work dir must resolve; other roots are skipped when missing.
    if roots[0] == "!" {
        return false;
    }
    let roots: Vec<&String> = roots.iter().filter(|root| *root != "!").collect();
    resolved_candidates.iter().all(|candidate| {
        candidate != "!" && roots.iter().any(|root| relative_inside(root, candidate))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_names() {
        assert_eq!(to_harness_name("Read"), "read");
        assert_eq!(to_harness_name("AskUserQuestion"), "askUserQuestions");
        assert_eq!(to_harness_name("mcp__harness-tools__lookup"), "lookup");
        assert_eq!(to_harness_name("WebFetch"), "WebFetch");
        assert_eq!(claude_code_builtin_tools().len(), 48);
        assert_eq!(
            claude_code_builtin_tools()["askUserQuestions"]
                .native_name
                .as_deref(),
            Some("AskUserQuestion")
        );
    }

    #[test]
    fn resolves_tool_filtering() {
        let allow = BuiltinToolFilter::Allow {
            tool_names: vec!["read".into(), "bash".into()],
        };
        assert_eq!(
            resolve_native_tools(Some(&allow)),
            Some(vec!["Read".to_string(), "Bash".to_string()])
        );
        let inactive = resolve_inactive_native_tools(Some(&allow));
        assert!(inactive.contains(&"Write".to_string()));
        assert!(!inactive.contains(&"Read".to_string()));
        let deny = BuiltinToolFilter::Deny {
            tool_names: vec!["webSearch".into()],
        };
        assert_eq!(resolve_native_tools(Some(&deny)), None);
        assert_eq!(resolve_inactive_native_tools(Some(&deny)), ["WebSearch"]);
    }

    #[test]
    fn creates_permission_settings() {
        assert_eq!(
            create_permission_settings(PermissionMode::AllowAll, false, &[]),
            None
        );
        let settings = create_permission_settings(PermissionMode::AllowEdits, false, &[]).unwrap();
        assert_eq!(
            settings["permissions"]["ask"],
            json!(["Bash(*)", "Monitor(*)"])
        );
        let settings =
            create_permission_settings(PermissionMode::AllowReads, true, &["Read".into()]).unwrap();
        assert_eq!(settings["permissions"]["ask"], json!(["Read(*)"]));
        assert_eq!(settings["sandbox"]["autoAllowBashIfSandboxed"], false);
    }

    #[test]
    fn maps_gateway_model_overrides() {
        let env = IndexMap::from([(
            "ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(),
            "anthropic/claude-haiku-5.5".to_string(),
        )]);
        let overrides =
            create_gateway_model_overrides(Some("anthropic/claude-opus-5.5[1m]"), &env).unwrap();
        assert_eq!(
            overrides["claude-opus-5-5"],
            "anthropic/claude-opus-5.5[1m]"
        );
        assert_eq!(overrides["claude-haiku-5-5"], "anthropic/claude-haiku-5.5");
        assert_eq!(
            create_gateway_model_overrides(Some("claude-opus-5-5"), &IndexMap::new()),
            None
        );
    }

    #[test]
    fn decides_native_approvals() {
        assert!(!native_tool_requires_approval(
            "Bash",
            PermissionMode::AllowAll
        ));
        assert!(native_tool_requires_approval(
            "Bash",
            PermissionMode::AllowEdits
        ));
        assert!(!native_tool_requires_approval(
            "Write",
            PermissionMode::AllowEdits
        ));
        assert!(native_tool_requires_approval(
            "Write",
            PermissionMode::AllowReads
        ));
        assert!(!native_tool_requires_approval(
            "Read",
            PermissionMode::AllowReads
        ));
        assert!(native_tool_requires_approval(
            "EnterWorktree",
            PermissionMode::AllowEdits
        ));
        assert!(native_tool_requires_approval(
            "Unknown",
            PermissionMode::AllowEdits
        ));
        assert!(glob_needs_approval(Some(&json!("../x"))));
        assert!(glob_needs_approval(Some(&json!("{a,b}"))));
        assert!(glob_needs_approval(Some(&json!("/etc/*"))));
        assert!(!glob_needs_approval(Some(&json!("src/**/*.rs"))));
    }

    #[tokio::test]
    async fn checks_paths_in_the_sandbox() {
        use crate::harness::adapters::local_sandbox::LocalSandbox;
        let dir = crate::harness::adapters::test_temp_dir::mkdtemp("cc-paths-");
        let work = dir.path().join("work");
        std::fs::create_dir_all(work.join(".claude")).unwrap();
        std::fs::write(work.join("a.txt"), "a").unwrap();
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, work.join("link")).unwrap();
        let work = work.to_string_lossy().to_string();
        let sandbox = LocalSandbox::new(&work);
        let context = ApprovalContext {
            sandbox: &sandbox,
            work_dir: &work,
            cwd: &work,
            readable_skill_roots: &[],
            additional_directories: &[],
            permission_mode: PermissionMode::AllowEdits,
            inactive_native_tools: &[],
            abort: None,
        };
        let read = |path: &str| json!({ "file_path": path });
        assert!(!needs_host_approval(&context, "Read", &read("a.txt"), None).await);
        assert!(!needs_host_approval(&context, "Read", &read("new/b.txt"), None).await);
        assert!(needs_host_approval(&context, "Read", &read("link/x"), None).await);
        assert!(needs_host_approval(&context, "Read", &read("../x"), None).await);
        assert!(needs_host_approval(&context, "Read", &read("~/x"), None).await);
        assert!(!needs_host_approval(&context, "Write", &read("a.txt"), None).await);
        assert!(needs_host_approval(&context, "Write", &read(".claude/settings.json"), None).await);
        assert!(needs_host_approval(&context, "Bash", &json!({ "command": "ls" }), None).await);
        assert!(!needs_host_approval(&context, "TodoWrite", &json!({}), None).await);
        assert!(needs_host_approval(&context, "TodoWrite", &json!({}), Some("/etc/passwd")).await);
        assert!(!needs_host_approval(&context, "Grep", &json!({ "pattern": "x" }), None).await);
        assert!(is_protected_claude_file_write(&context, "Edit", &read(".claude/x")).await);
        assert!(!is_protected_claude_file_write(&context, "Edit", &read("a.txt")).await);
    }
}
