//! [rust] What a harness offers on this machine: whether it is signed in,
//! its models with the efforts each takes, and its slash commands. Read with
//! [`HarnessSession::probe`](crate::harness::HarnessSession::probe) from a
//! short-lived process, so a model picker can be filled before any turn runs.
//!
//! The parsers here take the harness's own answers and do no I/O.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ThinkingLevel;

/// One model a harness can run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessModelInfo {
    /// The id the harness takes as its model.
    pub id: String,
    pub name: String,
    /// The thinking levels the model takes. Empty when it takes none.
    pub thinking_levels: Vec<ThinkingLevel>,
    /// Whether the harness runs it when no model is named.
    pub is_default: bool,
}

/// One slash command a harness offers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessCommandInfo {
    pub name: String,
    pub description: String,
    /// What its argument is, when it takes one.
    pub hint: String,
}

/// What [`HarnessSession::probe`](crate::harness::HarnessSession::probe) found.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessProbe {
    /// `None` when the harness does not say.
    pub signed_in: Option<bool>,
    /// The account or the way it is signed in, such as an email.
    pub account: Option<String>,
    /// Why the harness could not be read, such as a sign-in it asks for.
    pub error: Option<String>,
    pub models: Vec<HarnessModelInfo>,
    pub commands: Vec<HarnessCommandInfo>,
}

/// A thinking level by the name harnesses give efforts.
pub fn thinking_level_named(name: &str) -> Option<ThinkingLevel> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" | "x-high" | "extra-high" | "extra_high" => ThinkingLevel::Xhigh,
        "max" => ThinkingLevel::Max,
        _ => return None,
    })
}

fn str_of<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn items(value: &Value) -> impl Iterator<Item = &Value> {
    value.as_array().into_iter().flatten()
}

fn dedupe(mut models: Vec<HarnessModelInfo>) -> Vec<HarnessModelInfo> {
    let mut seen = std::collections::HashSet::new();
    models.retain(|model| seen.insert(model.id.clone()));
    models
}

/// Codex's `account/read`: signed in with an account, or through a provider
/// of its own configuration that needs none.
pub fn codex_account(result: &Value) -> (bool, Option<String>) {
    let account = &result["account"];
    let signed_in = account.is_object() || result["requiresOpenaiAuth"] != Value::Bool(true);
    let label = if account.is_object() {
        str_of(account, "email")
            .or_else(|| str_of(account, "type"))
            .map(str::to_string)
    } else if signed_in {
        Some("configured provider".into())
    } else {
        None
    };
    (signed_in, label)
}

/// Codex's `model/list`: its visible models by model id, with the efforts
/// each supports.
pub fn codex_models(result: &Value) -> Vec<HarnessModelInfo> {
    dedupe(
        items(&result["data"])
            .filter(|model| model["hidden"] != Value::Bool(true))
            .filter_map(|model| {
                let id = str_of(model, "model")?.to_string();
                Some(HarnessModelInfo {
                    name: str_of(model, "displayName")
                        .filter(|name| !name.is_empty())
                        .unwrap_or(&id)
                        .to_string(),
                    thinking_levels: items(&model["supportedReasoningEfforts"])
                        .filter_map(|effort| {
                            str_of(effort, "reasoningEffort").and_then(thinking_level_named)
                        })
                        .collect(),
                    is_default: model["isDefault"] == Value::Bool(true),
                    id,
                })
            })
            .collect(),
    )
}

/// The efforts Claude Code takes.
pub const CLAUDE_CODE_THINKING_LEVELS: [ThinkingLevel; 5] = [
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::Xhigh,
    ThinkingLevel::Max,
];

/// `claude auth status`: its JSON says, else its exit code does.
pub fn claude_code_auth(succeeded: bool, stdout: &str) -> (bool, Option<String>) {
    match serde_json::from_str::<Value>(stdout.trim()) {
        Ok(value) => (
            value["loggedIn"].as_bool().unwrap_or(succeeded),
            str_of(&value, "email")
                .or_else(|| str_of(&value, "authMethod"))
                .map(str::to_string),
        ),
        Err(_) => (succeeded, None),
    }
}

/// The models in Claude Code's answer to the `initialize` control request.
/// Haiku takes no effort.
pub fn claude_code_models(response: &Value) -> Vec<HarnessModelInfo> {
    dedupe(
        items(&response["response"]["models"])
            .filter_map(|model| {
                let id = str_of(model, "value")?.to_string();
                let name = str_of(model, "displayName").unwrap_or(&id).to_string();
                let haiku = format!("{id} {name}").to_lowercase().contains("haiku");
                Some(HarnessModelInfo {
                    is_default: id == "default",
                    thinking_levels: if haiku {
                        Vec::new()
                    } else {
                        CLAUDE_CODE_THINKING_LEVELS.to_vec()
                    },
                    id,
                    name,
                })
            })
            .collect(),
    )
}

/// The commands in Claude Code's answer to `initialize`, without its skills
/// (a description ending in where the skill lives, such as `(user)`) and
/// internal commands.
pub fn claude_code_commands(response: &Value) -> Vec<HarnessCommandInfo> {
    items(&response["response"]["commands"])
        .filter_map(|command| {
            let name = str_of(command, "name")?;
            let description = str_of(command, "description").unwrap_or("");
            let skill = ["(user)", "(project)", "(plugin)"]
                .iter()
                .any(|place| description.ends_with(place));
            (!skill && !name.starts_with('_')).then(|| HarnessCommandInfo {
                name: name.to_string(),
                description: description.to_string(),
                hint: str_of(command, "argumentHint").unwrap_or("").to_string(),
            })
        })
        .collect()
}

/// The commands of an ACP `available_commands_update`, or `None` for another
/// update.
pub fn acp_commands(update: &Value) -> Option<Vec<HarnessCommandInfo>> {
    (update["sessionUpdate"] == "available_commands_update").then(|| {
        items(&update["availableCommands"])
            .filter_map(|command| {
                Some(HarnessCommandInfo {
                    name: str_of(command, "name")?.to_string(),
                    description: str_of(command, "description").unwrap_or("").to_string(),
                    hint: command["input"]["hint"].as_str().unwrap_or("").to_string(),
                })
            })
            .collect()
    })
}

/// An ACP agent's model config option: the one with id `model`, else the
/// first of category `model`. Some agents list their provider under
/// category `model` too, before their models.
pub fn acp_model_option(config: &[Value]) -> Option<&Value> {
    let is_model = |option: &&Value| option["category"] == "model";
    config
        .iter()
        .filter(is_model)
        .find(|option| option["id"] == "model")
        .or_else(|| config.iter().find(is_model))
}

/// The models an ACP agent offers in a new session (the `session/new`
/// result): its model list, else its model config option. Embedding models
/// cannot run a chat and are left out.
pub fn acp_models(session: &Value) -> Vec<HarnessModelInfo> {
    let config = session["configOptions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let thinking_levels: Vec<ThinkingLevel> = config
        .iter()
        .find(|option| {
            option["category"] == "thought_level"
                || str_of(option, "id").is_some_and(|id| id.to_lowercase().contains("effort"))
        })
        .map(|option| {
            items(&option["options"])
                .filter_map(|choice| str_of(choice, "value").and_then(thinking_level_named))
                .collect()
        })
        .unwrap_or_default();
    let model = |id: &str, name: Option<&str>, current: &Value| HarnessModelInfo {
        id: id.to_string(),
        name: name.unwrap_or(id).to_string(),
        thinking_levels: thinking_levels.clone(),
        is_default: current.as_str() == Some(id),
    };
    let models = &session["models"];
    let mut list: Vec<HarnessModelInfo> = items(&models["availableModels"])
        .filter_map(|entry| {
            Some(model(
                str_of(entry, "modelId")?,
                str_of(entry, "name"),
                &models["currentModelId"],
            ))
        })
        .collect();
    if list.is_empty()
        && let Some(option) = acp_model_option(&config)
    {
        list = items(&option["options"])
            .filter(|choice| {
                !str_of(choice, "value")
                    .is_some_and(|value| value.to_lowercase().contains("embedding"))
            })
            .filter_map(|choice| {
                Some(model(
                    str_of(choice, "value")?,
                    str_of(choice, "name"),
                    &option["currentValue"],
                ))
            })
            .collect();
    }
    dedupe(list)
}

/// Whether an error from an agent asks for a sign-in.
pub fn is_sign_in_error(error: &str) -> bool {
    let error = error.to_lowercase();
    error.contains("auth")
        || error.contains("sign in")
        || error.contains("log in")
        || error.contains("login")
        || error.contains("needs access to vercel ai gateway")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn codex_account_reads_email_or_configured_provider() {
        let signed = json!({ "account": { "type": "chatgpt", "email": "a@b.c" } });
        assert_eq!(codex_account(&signed), (true, Some("a@b.c".into())));
        let provider = json!({ "account": null, "requiresOpenaiAuth": false });
        assert_eq!(
            codex_account(&provider),
            (true, Some("configured provider".into()))
        );
        let none = json!({ "account": null, "requiresOpenaiAuth": true });
        assert_eq!(codex_account(&none), (false, None));
    }

    #[test]
    fn codex_models_skip_hidden_and_read_efforts() {
        let result = json!({ "data": [
            { "id": "e1", "model": "gpt-5.5", "displayName": "GPT-5.5", "isDefault": true,
              "supportedReasoningEfforts": [{ "reasoningEffort": "low" }, { "reasoningEffort": "xhigh" }] },
            { "id": "e2", "model": "secret", "hidden": true },
            { "id": "e3", "model": "gpt-5.5", "displayName": "dup" },
            { "id": "e4", "model": "mini", "displayName": "" },
        ]});
        let models = codex_models(&result);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-5.5");
        assert_eq!(models[0].name, "GPT-5.5");
        assert!(models[0].is_default);
        assert_eq!(
            models[0].thinking_levels,
            [ThinkingLevel::Low, ThinkingLevel::Xhigh]
        );
        assert_eq!(models[1].name, "mini");
        assert!(models[1].thinking_levels.is_empty());
    }

    #[test]
    fn claude_code_auth_reads_json_or_exit_code() {
        assert_eq!(
            claude_code_auth(true, r#"{"loggedIn":true,"email":"x@y.z"}"#),
            (true, Some("x@y.z".into()))
        );
        assert_eq!(
            claude_code_auth(true, r#"{"loggedIn":false}"#),
            (false, None)
        );
        assert_eq!(claude_code_auth(false, "not json"), (false, None));
    }

    #[test]
    fn claude_code_models_give_haiku_no_effort() {
        let response = json!({ "response": { "models": [
            { "value": "default", "displayName": "Default" },
            { "value": "haiku", "displayName": "Haiku 4.5" },
        ]}});
        let models = claude_code_models(&response);
        assert!(models[0].is_default);
        assert_eq!(models[0].thinking_levels.len(), 5);
        assert!(models[1].thinking_levels.is_empty());
    }

    #[test]
    fn claude_code_commands_leave_out_skills_and_internal_ones() {
        let response = json!({ "response": { "commands": [
            { "name": "review", "description": "Review a PR", "argumentHint": "<pr>" },
            { "name": "my-skill", "description": "Does things (user)" },
            { "name": "_hidden", "description": "" },
        ]}});
        let commands = claude_code_commands(&response);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].hint, "<pr>");
    }

    #[test]
    fn acp_models_read_the_model_list_first() {
        let session = json!({
            "models": { "currentModelId": "b", "availableModels": [
                { "modelId": "a", "name": "A" }, { "modelId": "b" }] },
            "configOptions": [{ "id": "effort", "options": [{ "value": "low" }, { "value": "high" }] }],
        });
        let models = acp_models(&session);
        assert_eq!(models.len(), 2);
        assert!(models[1].is_default);
        assert_eq!(models[1].name, "b");
        assert_eq!(
            models[0].thinking_levels,
            [ThinkingLevel::Low, ThinkingLevel::High]
        );
    }

    #[test]
    fn acp_models_fall_back_to_the_model_option_without_embeddings() {
        let session = json!({ "configOptions": [
            { "id": "provider", "category": "model", "currentValue": "gw",
              "options": [{ "value": "gw" }] },
            { "id": "model", "category": "model", "currentValue": "p/chat",
              "options": [{ "value": "p/chat" }, { "value": "p/text-embedding-3-small" }] },
            { "id": "thinking", "category": "thought_level", "options": [{ "value": "medium" }] },
        ]});
        let models = acp_models(&session);
        let ids: Vec<_> = models.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids, ["p/chat"]);
        assert!(models[0].is_default);
        assert_eq!(models[0].thinking_levels, [ThinkingLevel::Medium]);
    }

    #[test]
    fn acp_commands_read_only_command_updates() {
        let update = json!({ "sessionUpdate": "available_commands_update", "availableCommands": [
            { "name": "plan", "description": "Plan", "input": { "hint": "goal" } }] });
        let commands = acp_commands(&update).unwrap();
        assert_eq!(commands[0].hint, "goal");
        assert!(acp_commands(&json!({ "sessionUpdate": "agent_message_chunk" })).is_none());
    }

    #[test]
    fn thinking_levels_by_name() {
        assert_eq!(thinking_level_named("XHigh"), Some(ThinkingLevel::Xhigh));
        assert_eq!(thinking_level_named("none"), None);
    }
}
