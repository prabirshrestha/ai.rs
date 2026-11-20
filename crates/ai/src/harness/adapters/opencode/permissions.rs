//! Port of `bridge/permission-paths.ts`, `bridge/question-tool.ts`,
//! `bridge/tool-relay-auth.ts` and the permission rules of `bridge/index.ts`
//! (`selectPermissionReply`), run on the host against the sandbox.

use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::sandbox_paths::are_paths_inside_roots;
use crate::harness::paths::{is_absolute_posix, normalize_posix};
use crate::harness::questions_tool::QuestionsToolOutput;
use crate::harness::sandbox::SandboxSession;
use crate::harness::tools::PermissionMode;
use crate::harness::utils::is_sandbox_write_protected_path;

use super::events::to_wire_tool_name;

/// Workspace paths that hold OpenCode configuration or skills it discovers,
/// which can run code or change the agent's behavior in later sessions.
/// Writes to them always require host approval unless the permission mode is
/// `allow-all`.
pub const OPENCODE_WRITE_PROTECTED_PATHS: [&str; 5] = [
    ".agents/skills",
    ".opencode",
    "opencode.json",
    "opencode.jsonc",
    ".claude/skills",
];

/// `PermissionToolCall`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PermissionToolCall {
    pub native_name: String,
    pub input: Map<String, Value>,
}

/// `OpenCodePermissionPathOptions`.
pub(super) struct PermissionPathOptions<'a> {
    pub action: &'a str,
    pub resources: &'a [String],
    pub metadata: Option<&'a Map<String, Value>>,
    pub tool_call: Option<&'a PermissionToolCall>,
    pub work_dir: &'a str,
    pub readable_skill_roots: &'a [String],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    Read,
    Write,
}

/// `resolveOpenCodePermissionTargets`.
pub(super) fn resolve_permission_targets(
    options: &PermissionPathOptions<'_>,
) -> Option<(Access, Vec<String>)> {
    let args = options.tool_call.map(|call| &call.input);
    let native_name = options
        .tool_call
        .map(|call| call.native_name.as_str())
        .unwrap_or_default();
    let access = if matches!(options.action, "edit" | "write")
        || matches!(native_name, "write" | "edit" | "apply_patch")
    {
        Access::Write
    } else {
        Access::Read
    };
    let metadata_filepath = options
        .metadata
        .and_then(|metadata| metadata.get("filepath"))
        .and_then(Value::as_str);
    let targets: Vec<Option<&Value>> = if let Some(Value::Array(files)) =
        options.metadata.and_then(|metadata| metadata.get("files"))
    {
        if files.is_empty() {
            return None;
        }
        files
            .iter()
            .flat_map(|file| match file.as_object() {
                None => vec![None],
                Some(file) => {
                    let is_move = file.get("type").and_then(Value::as_str) == Some("move")
                        || file.get("movePath").is_some_and(|path| !path.is_null());
                    if is_move {
                        vec![file.get("filePath"), file.get("movePath")]
                    } else {
                        vec![file.get("filePath")]
                    }
                }
            })
            .collect()
    } else if let Some(args) =
        args.filter(|_| matches!(native_name, "read" | "view" | "write" | "edit"))
    {
        vec![args.get("filePath")]
    } else if let Some(args) =
        args.filter(|_| matches!(native_name, "glob" | "grep" | "list" | "ls"))
    {
        match args.get("path").filter(|path| !path.is_null()) {
            Some(path) => vec![Some(path)],
            None => return Some((access, vec![options.work_dir.to_string()])),
        }
    } else if let Some(filepath) = metadata_filepath.filter(|_| options.action != "edit") {
        return Some((access, vec![filepath.to_string()]));
    } else if let Some(filepath) =
        metadata_filepath.filter(|path| options.action == "edit" && is_absolute_posix(path))
    {
        return Some((access, vec![filepath.to_string()]));
    } else if options.action == "read"
        && !options.resources.is_empty()
        && options.resources.iter().all(|path| is_absolute_posix(path))
    {
        return Some((access, options.resources.to_vec()));
    } else {
        return None;
    };
    let targets: Option<Vec<String>> = targets
        .into_iter()
        .map(|target| target.and_then(Value::as_str).map(str::to_string))
        .collect();
    Some((access, targets?))
}

/// `areOpenCodePermissionPathsTrusted`.
pub(super) async fn are_permission_paths_trusted(
    sandbox: &dyn SandboxSession,
    options: &PermissionPathOptions<'_>,
    abort: Option<CancellationToken>,
) -> bool {
    let Some((access, targets)) = resolve_permission_targets(options) else {
        return false;
    };
    let optional_roots: &[String] = if access == Access::Read {
        options.readable_skill_roots
    } else {
        &[]
    };
    are_paths_inside_roots(
        sandbox,
        options.work_dir,
        options.work_dir,
        optional_roots,
        &targets,
        abort,
    )
    .await
}

/// `areOpenCodePermissionPathsWriteProtected`.
pub(super) async fn are_permission_paths_write_protected(
    sandbox: &dyn SandboxSession,
    options: &PermissionPathOptions<'_>,
    abort: Option<CancellationToken>,
) -> bool {
    let Some((access, targets)) = resolve_permission_targets(options) else {
        return true;
    };
    if access != Access::Write {
        return false;
    }
    let absolute: Vec<String> = targets
        .iter()
        .map(|target| {
            if is_absolute_posix(target) {
                target.clone()
            } else {
                normalize_posix(&format!("{}/{target}", options.work_dir))
            }
        })
        .collect();
    let protected: Vec<String> = OPENCODE_WRITE_PROTECTED_PATHS
        .iter()
        .map(|path| path.to_string())
        .collect();
    is_sandbox_write_protected_path(sandbox, options.work_dir, &absolute, &protected, abort)
        .await
        .unwrap_or(true)
}

/// `BUILTIN_TOOL_PERMISSION`: the OpenCode permission of a builtin tool.
pub(super) fn builtin_tool_permission(tool_name: &str) -> Option<&'static str> {
    Some(match tool_name {
        "askUserQuestions" => "question",
        "read" => "read",
        "write" | "edit" => "edit",
        "bash" => "bash",
        "glob" => "glob",
        "grep" => "grep",
        "ls" => "list",
        "webfetch" => "webfetch",
        "skill" => "skill",
        "todowrite" => "todowrite",
        "agent" => "task",
        _ => return None,
    })
}

/// `TOOL_KIND`.
fn tool_kind(tool_name: &str) -> Option<&'static str> {
    Some(match tool_name {
        "read" | "glob" | "grep" | "ls" | "webfetch" => "readonly",
        "write" | "edit" | "skill" | "todowrite" => "edit",
        "bash" | "agent" => "bash",
        _ => return None,
    })
}

/// `toPermissionToolName`.
pub(super) fn to_permission_tool_name(action: &str) -> String {
    let normalized = action.to_lowercase();
    let contains = |needle: &str| normalized.contains(needle);
    if contains("bash") || contains("shell") {
        "bash".into()
    } else if contains("edit") {
        "edit".into()
    } else if contains("write") {
        "write".into()
    } else if contains("webfetch") {
        "webfetch".into()
    } else if contains("task") || contains("agent") {
        "agent".into()
    } else if contains("list") {
        "ls".into()
    } else if contains("grep") {
        "grep".into()
    } else if contains("glob") {
        "glob".into()
    } else if contains("read") {
        "read".into()
    } else {
        to_wire_tool_name(&normalized)
    }
}

/// The reply to a permission request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PermissionReply {
    Once,
    Always,
    Reject(Option<String>),
}

impl PermissionReply {
    pub fn body(&self) -> Value {
        match self {
            Self::Once => json!({ "reply": "once" }),
            Self::Always => json!({ "reply": "always" }),
            Self::Reject(None) => json!({ "reply": "reject" }),
            Self::Reject(Some(message)) => json!({ "reply": "reject", "message": message }),
        }
    }
}

/// What `selectPermissionReply` decides before asking the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PermissionDecision {
    Reply(PermissionReply),
    /// Ask the host's `before_tool_call`.
    Ask,
}

/// One permission request.
pub(super) struct PermissionRequest<'a> {
    pub action: &'a str,
    pub resources: &'a [String],
    pub metadata: Option<&'a Map<String, Value>>,
    pub tool_call: Option<&'a PermissionToolCall>,
}

/// `selectPermissionReply`, up to the host approval.
pub(super) async fn select_permission_reply(
    sandbox: &dyn SandboxSession,
    request: &PermissionRequest<'_>,
    permission_mode: PermissionMode,
    inactive_builtin_tool_names: &[String],
    work_dir: &str,
    readable_skill_roots: &[String],
    abort: Option<CancellationToken>,
) -> PermissionDecision {
    let tool_name = to_permission_tool_name(request.action);
    if inactive_builtin_tool_names.contains(&tool_name) {
        return PermissionDecision::Ask;
    }
    if permission_mode == PermissionMode::AllowAll {
        return PermissionDecision::Reply(PermissionReply::Always);
    }
    let native_name = request.tool_call.map(|call| call.native_name.as_str());
    let file_permission = request.action == "external_directory"
        || matches!(
            tool_name.as_str(),
            "read" | "write" | "edit" | "glob" | "grep" | "ls"
        );
    let kind = match native_name {
        Some(native) if request.action == "external_directory" => {
            if native == "apply_patch" {
                "edit"
            } else {
                tool_kind(&to_wire_tool_name(native)).unwrap_or("bash")
            }
        }
        _ => tool_kind(&tool_name).unwrap_or("bash"),
    };
    let allowed = match permission_mode {
        PermissionMode::AllowEdits => kind == "readonly" || kind == "edit",
        _ => kind == "readonly",
    };
    if allowed {
        if !file_permission {
            return PermissionDecision::Reply(PermissionReply::Once);
        }
        let options = PermissionPathOptions {
            action: request.action,
            resources: request.resources,
            metadata: request.metadata,
            tool_call: request.tool_call,
            work_dir,
            readable_skill_roots,
        };
        if are_permission_paths_trusted(sandbox, &options, abort.clone()).await
            && !are_permission_paths_write_protected(sandbox, &options, abort).await
        {
            return PermissionDecision::Reply(PermissionReply::Once);
        }
    }
    PermissionDecision::Ask
}

/// `toHarnessQuestionsInput`.
pub(super) fn to_harness_questions_input(questions: &[Value]) -> Value {
    let questions: Vec<Value> = questions
        .iter()
        .enumerate()
        .map(|(question_index, question)| {
            let options: Vec<Value> = question
                .get("options")
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .enumerate()
                .map(|(option_index, option)| {
                    let mut entry = Map::new();
                    entry.insert("id".into(), json!(format!("option-{}", option_index + 1)));
                    entry.insert(
                        "label".into(),
                        option.get("label").cloned().unwrap_or(json!("")),
                    );
                    if let Some(description) = option.get("description").filter(|d| !d.is_null()) {
                        entry.insert("description".into(), description.clone());
                    }
                    Value::Object(entry)
                })
                .collect();
            let mut entry = Map::new();
            entry.insert(
                "id".into(),
                json!(format!("question-{}", question_index + 1)),
            );
            entry.insert(
                "question".into(),
                question.get("question").cloned().unwrap_or(json!("")),
            );
            if let Some(header) = question.get("header").filter(|header| !header.is_null()) {
                entry.insert("header".into(), header.clone());
            }
            entry.insert("options".into(), Value::Array(options));
            if let Some(multiple) = question.get("multiple").filter(|value| !value.is_null()) {
                entry.insert("allowMultiple".into(), multiple.clone());
            }
            if let Some(custom) = question.get("custom").filter(|value| !value.is_null()) {
                entry.insert("allowFreeForm".into(), custom.clone());
            }
            Value::Object(entry)
        })
        .collect();
    json!({ "allowPartialAnswers": true, "questions": questions })
}

/// `toOpenCodeQuestionResponse`: `None` rejects the question.
pub(super) fn to_opencode_question_response(
    questions: &[Value],
    output: &QuestionsToolOutput,
) -> Option<Vec<Vec<String>>> {
    let answers = match output {
        QuestionsToolOutput::Declined | QuestionsToolOutput::Cancelled => return None,
        QuestionsToolOutput::Answered { answers }
        | QuestionsToolOutput::PartiallyAnswered { answers } => answers,
    };
    Some(
        questions
            .iter()
            .enumerate()
            .map(|(question_index, question)| {
                let Some(answer) = answers.get(&format!("question-{}", question_index + 1)) else {
                    return Vec::new();
                };
                let options = question
                    .get("options")
                    .and_then(Value::as_array)
                    .map_or(&[][..], Vec::as_slice);
                let mut labels: Vec<String> = answer
                    .option_ids
                    .iter()
                    .filter_map(|option_id| {
                        let index: usize = option_id.strip_prefix("option-")?.parse().ok()?;
                        let option = options.get(index.checked_sub(1)?)?;
                        option.get("label")?.as_str().map(str::to_string)
                    })
                    .collect();
                if let Some(freeform) = &answer.freeform {
                    labels.push(freeform.clone());
                }
                labels
            })
            .collect(),
    )
}

/// `normalizeJsonValue`: objects with sorted keys and no nulls from
/// `undefined`.
pub(super) fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonical_json).collect()),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|key| (key.clone(), canonical_json(&map[key])))
                    .collect(),
            )
        }
        other => other.clone(),
    }
}

/// `toolRelayCallKey`.
pub(super) fn tool_relay_call_key(tool_name: &str, input: &Value) -> String {
    let input = if input.is_null() {
        json!({})
    } else {
        input.clone()
    };
    format!("{tool_name}\0{}", canonical_json(&input))
}

/// `ToolRelayAuthorizer`: a host tool request from OpenCode's MCP client
/// runs only after the event stream reported the matching tool call. The
/// value is the OpenCode tool call id.
pub(super) struct ToolRelayAuthorizer<T> {
    ttl: Duration,
    authorizations: Vec<(String, String, Instant)>,
    pending: Vec<(String, T, Instant)>,
}

impl<T> ToolRelayAuthorizer<T> {
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            authorizations: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// `authorizeToolCall`: returns the waiting request it releases.
    pub fn authorize(&mut self, key: String, call_id: String, now: Instant) -> Option<(T, String)> {
        self.authorizations.retain(|(_, _, expires)| *expires > now);
        if let Some(index) = self
            .pending
            .iter()
            .position(|(pending, _, _)| *pending == key)
        {
            let (_, waiter, _) = self.pending.remove(index);
            return Some((waiter, call_id));
        }
        self.authorizations.push((key, call_id, now + self.ttl));
        None
    }

    /// `waitForToolCallAuthorization`: `Ok(call id)` when already
    /// authorized, else the request waits (`Err(())`).
    pub fn wait(&mut self, key: String, waiter: T, now: Instant) -> Option<String> {
        self.authorizations.retain(|(_, _, expires)| *expires > now);
        if let Some(index) = self
            .authorizations
            .iter()
            .position(|(authorized, _, _)| *authorized == key)
        {
            return Some(self.authorizations.remove(index).1);
        }
        self.pending.push((key, waiter, now + self.ttl));
        None
    }

    /// Requests whose time ran out.
    pub fn expire(&mut self, now: Instant) -> Vec<T> {
        let (expired, pending): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|(_, _, expires)| *expires <= now);
        self.pending = pending;
        expired.into_iter().map(|(_, waiter, _)| waiter).collect()
    }

    /// `close`.
    pub fn close(&mut self) -> Vec<T> {
        self.authorizations.clear();
        std::mem::take(&mut self.pending)
            .into_iter()
            .map(|(_, waiter, _)| waiter)
            .collect()
    }
}

#[cfg(test)]
#[path = "permissions_tests.rs"]
mod tests;
