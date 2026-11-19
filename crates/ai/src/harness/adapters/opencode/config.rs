//! Port of `buildOpenCodeConfig`, `buildProviderConfig` and
//! `bridge/opencode-project-isolation.ts`: the configuration the OpenCode
//! server starts with, built on the host from the sandbox's environment.

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::error::HarnessResult;
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};
use crate::harness::tools::PermissionMode;
use crate::harness::utils::shell_quote;

use super::HOST_TOOL_MCP_SERVER_NAME;
use super::auth::to_opencode_gateway_base_url;
use super::permissions::builtin_tool_permission;

/// What [`build_opencode_config`] needs.
pub(super) struct ConfigInput<'a> {
    pub opencode_config: Option<&'a Map<String, Value>>,
    pub mcp_servers: Option<&'a Map<String, Value>>,
    pub permission_mode: PermissionMode,
    pub inactive_builtin_tool_names: &'a [String],
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub headers: Option<&'a IndexMap<String, String>>,
    /// The OpenCode server's environment (`procEnv` of the bridge).
    pub env: &'a IndexMap<String, String>,
    pub client_app: &'a str,
    pub skills_dir: &'a str,
    /// [`resolve_opencode_instructions`].
    pub instructions: Option<Vec<String>>,
    /// The command OpenCode starts to reach the host tool relay.
    pub relay_command: Option<Vec<String>>,
    /// Whether the relay serves the restrictive-mode `read` tool.
    pub read_guard: bool,
}

/// `withoutAgentPolicyOverrides`: agents cannot override the harness
/// permissions or tool filtering.
pub(super) fn without_agent_policy_overrides(
    input: Option<&Map<String, Value>>,
) -> Map<String, Value> {
    let mut config = input.cloned().unwrap_or_default();
    for key in ["agent", "mode"] {
        let Some(Value::Object(agents)) = config.get_mut(key) else {
            continue;
        };
        for agent in agents.values_mut() {
            if let Value::Object(agent) = agent {
                agent.remove("permission");
                agent.remove("tools");
            }
        }
    }
    config
}

fn present<'a>(env: &'a IndexMap<String, String>, name: &str) -> Option<&'a str> {
    env.get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

/// `splitModel`.
pub(super) fn split_model<'a>(
    model: Option<&'a str>,
    provider: Option<&'a str>,
) -> (Option<&'a str>, Option<&'a str>) {
    match model {
        None | Some("") => (None, None),
        Some(model) => match model.split_once('/') {
            Some((provider_id, model_id)) => (Some(provider_id), Some(model_id)),
            None => (provider, Some(model)),
        },
    }
}

fn models(model_id: Option<&str>) -> Option<Value> {
    model_id.map(|id| json!({ id: { "id": id, "name": id } }))
}

fn query_params(env: &IndexMap<String, String>) -> Option<Value> {
    let parsed: Value = serde_json::from_str(present(env, "OPENAI_QUERY_PARAMS_JSON")?).ok()?;
    parsed.is_object().then_some(parsed)
}

/// `buildProviderConfig`.
pub(super) fn build_provider_config(input: &ConfigInput<'_>) -> Option<Value> {
    let env = input.env;
    let (model_provider, model_id) = split_model(input.model, input.provider);
    let provider_id = model_provider
        .or(input.provider)
        .or(present(env, "OPENAI_NAME"))
        .unwrap_or("anthropic")
        .to_string();
    let headers = input.headers.map(|headers| json!(headers));

    if let (Some(api_key), Some(base_url)) = (
        present(env, "AI_GATEWAY_API_KEY"),
        present(env, "AI_GATEWAY_BASE_URL"),
    ) {
        let mut gateway_headers = input.headers.cloned().unwrap_or_default();
        gateway_headers.insert("x-client-app".into(), input.client_app.to_string());
        let mut provider = json!({
            "options": {
                "apiKey": api_key,
                "baseURL": to_opencode_gateway_base_url(base_url),
                "headers": gateway_headers,
            },
        });
        if provider_id == "google" || provider_id == "mistral" {
            provider["npm"] = json!("@ai-sdk/openai-compatible");
        }
        if let Some(models) = models(model_id) {
            provider["models"] = models;
        }
        return Some(json!({ provider_id: provider }));
    }

    let api_key = present(env, "OPENAI_API_KEY");
    let base_url = present(env, "OPENAI_BASE_URL");
    if (present(env, "OPENAI_NAME").is_some()
        || (provider_id != "anthropic" && provider_id != "openai"))
        && (api_key.is_some() || base_url.is_some())
    {
        let id = present(env, "OPENAI_NAME")
            .unwrap_or(&provider_id)
            .to_string();
        let mut options = Map::new();
        if let Some(key) = api_key {
            options.insert("apiKey".into(), json!(key));
        }
        if let Some(url) = base_url {
            options.insert("baseURL".into(), json!(url));
        }
        if let Some(headers) = &headers {
            options.insert("headers".into(), headers.clone());
        }
        if let Some(params) = query_params(env) {
            options.insert("queryParams".into(), params);
        }
        let mut provider = json!({ "options": options });
        if let Some(models) = models(model_id) {
            provider["models"] = models;
        }
        return Some(json!({ id: provider }));
    }

    if provider_id == "anthropic"
        && (present(env, "ANTHROPIC_API_KEY").is_some()
            || present(env, "ANTHROPIC_AUTH_TOKEN").is_some()
            || present(env, "ANTHROPIC_BASE_URL").is_some())
    {
        let mut options = Map::new();
        for (name, key) in [
            ("ANTHROPIC_API_KEY", "apiKey"),
            ("ANTHROPIC_AUTH_TOKEN", "authToken"),
            ("ANTHROPIC_BASE_URL", "baseURL"),
        ] {
            if let Some(value) = present(env, name) {
                options.insert(key.into(), json!(value));
            }
        }
        if let Some(headers) = &headers {
            options.insert("headers".into(), headers.clone());
        }
        return Some(json!({ "anthropic": { "options": options } }));
    }

    if provider_id == "openai" && (api_key.is_some() || base_url.is_some()) {
        let mut options = Map::new();
        for (name, key) in [
            ("OPENAI_API_KEY", "apiKey"),
            ("OPENAI_BASE_URL", "baseURL"),
            ("OPENAI_ORGANIZATION", "organization"),
            ("OPENAI_PROJECT", "project"),
        ] {
            if let Some(value) = present(env, name) {
                options.insert(key.into(), json!(value));
            }
        }
        if let Some(headers) = &headers {
            options.insert("headers".into(), headers.clone());
        }
        if let Some(params) = query_params(env) {
            options.insert("queryParams".into(), params);
        }
        return Some(json!({ "openai": { "options": options } }));
    }
    None
}

/// `buildOpenCodeConfig`.
///
/// [rust] In restrictive permission modes the native `read` tool is denied
/// and the host tool relay serves `read` instead, in place of the TS read
/// plugin: OpenCode's native read loads nearby instruction files without a
/// permission request.
pub(super) fn build_opencode_config(input: &ConfigInput<'_>) -> Map<String, Value> {
    let mut config = without_agent_policy_overrides(input.opencode_config);
    config.insert("share".into(), json!("disabled"));
    config.insert("autoupdate".into(), json!(false));
    let restrictive = input.permission_mode != PermissionMode::AllowAll;
    let reads = if restrictive { "ask" } else { "allow" };
    let mut permission = Map::new();
    for name in ["read", "glob", "grep", "list"] {
        permission.insert(name.into(), json!(reads));
    }
    for name in [
        "edit",
        "bash",
        "external_directory",
        "webfetch",
        "doom_loop",
        "task",
    ] {
        permission.insert(name.into(), json!("ask"));
    }
    permission.insert("question".into(), json!("allow"));
    if input.read_guard {
        permission.insert("read".into(), json!("deny"));
    }
    if let Some(instructions) = &input.instructions {
        config.insert("instructions".into(), json!(instructions));
    }
    if let Some(model) = input.model.filter(|model| !model.is_empty()) {
        config.insert("model".into(), json!(model));
    }
    config.insert("skills".into(), json!({ "paths": [input.skills_dir] }));
    for tool_name in input.inactive_builtin_tool_names {
        if let Some(name) = builtin_tool_permission(tool_name) {
            permission.insert(name.into(), json!("deny"));
        }
    }
    config.insert("permission".into(), Value::Object(permission));
    if let Some(provider) = build_provider_config(input) {
        config.insert("provider".into(), provider);
    }
    let mut mcp = input.mcp_servers.cloned().unwrap_or_default();
    if let Some(command) = &input.relay_command {
        mcp.insert(
            HOST_TOOL_MCP_SERVER_NAME.into(),
            json!({ "type": "local", "enabled": true, "command": command }),
        );
    }
    if !mcp.is_empty() {
        config.insert("mcp".into(), Value::Object(mcp));
    }
    config
}

fn has_glob_characters(value: &str) -> bool {
    value.contains(['*', '?', '[', ']', '{', '}'])
}

fn posix_dirname(path: &str) -> &str {
    match path.trim_end_matches('/').rfind('/') {
        Some(0) => "/",
        Some(index) => &path[..index],
        None => ".",
    }
}

async fn run(
    sandbox: &dyn SandboxSession,
    command: String,
    abort: Option<CancellationToken>,
) -> HarnessResult<String> {
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command,
                abort_signal: abort,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    Ok(if result.exit_code == 0 {
        result.stdout
    } else {
        String::new()
    })
}

/// The sandbox's real path of `directory`, or `directory` when it does not
/// exist.
pub(super) async fn sandbox_realpath(
    sandbox: &dyn SandboxSession,
    directory: &str,
    abort: Option<CancellationToken>,
) -> HarnessResult<String> {
    let output = run(
        sandbox,
        format!("cd -- {} 2>/dev/null && pwd -P", shell_quote(directory)),
        abort,
    )
    .await?;
    let resolved = output.trim_end_matches('\n');
    Ok(if resolved.is_empty() {
        directory.to_string()
    } else {
        resolved.to_string()
    })
}

/// `resolveOpenCodeInstructions`: the workspace `AGENTS.md`, then the
/// configured instruction files, globs expanded in the sandbox.
///
/// [rust] Globs expand with bash `globstar`; brace patterns are not
/// expanded.
pub(super) async fn resolve_opencode_instructions(
    sandbox: &dyn SandboxSession,
    work_dir: &str,
    host_instructions: Option<&Value>,
    abort: Option<CancellationToken>,
) -> HarnessResult<Option<Vec<String>>> {
    let entries: Vec<String> = match host_instructions {
        None => Vec::new(),
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
            .iter()
            .map(|item| item.as_str().unwrap().to_string())
            .collect(),
        Some(_) => return Ok(None),
    };
    let directory = sandbox_realpath(sandbox, work_dir, abort.clone()).await?;
    let join = |entry: &str| {
        crate::harness::paths::normalize_posix(&format!(
            "{}/{entry}",
            directory.trim_end_matches('/')
        ))
    };
    let mut instructions = vec![join("AGENTS.md")];
    for entry in entries {
        if entry.starts_with("http://")
            || entry.starts_with("https://")
            || entry.starts_with("~/")
            || entry.starts_with('/')
        {
            instructions.push(entry);
        } else if !has_glob_characters(posix_dirname(&entry)) {
            instructions.push(join(&entry));
        } else {
            let script = format!(
                "cd -- {} && shopt -s globstar nullglob && IFS= && p={} && for f in $p; do [ -f \"$f\" ] && printf '%s\\0' \"$f\"; done",
                shell_quote(&directory),
                shell_quote(&entry),
            );
            let output = run(
                sandbox,
                format!("bash -c {}", shell_quote(&script)),
                abort.clone(),
            )
            .await?;
            let mut matches: Vec<String> = output
                .split('\0')
                .filter(|path| !path.is_empty())
                .map(join)
                .collect();
            matches.sort();
            instructions.extend(matches);
        }
    }
    Ok(Some(instructions))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(env: &'a IndexMap<String, String>) -> ConfigInput<'a> {
        ConfigInput {
            opencode_config: None,
            mcp_servers: None,
            permission_mode: PermissionMode::AllowAll,
            inactive_builtin_tool_names: &[],
            model: None,
            provider: None,
            headers: None,
            env,
            client_app: "ai-sdk-harness-opencode/test",
            skills_dir: "/home/u/.agents/skills",
            instructions: None,
            relay_command: None,
            read_guard: false,
        }
    }

    fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn strips_agent_policy_overrides() {
        let config = json!({
            "agent": { "build": { "permission": {}, "tools": {}, "model": "m" }, "x": 1 },
            "mode": { "plan": { "tools": {} } },
        });
        assert_eq!(
            Value::Object(without_agent_policy_overrides(config.as_object())),
            json!({ "agent": { "build": { "model": "m" }, "x": 1 }, "mode": { "plan": {} } })
        );
    }

    #[test]
    fn builds_the_permission_map_for_each_mode() {
        let env = env(&[]);
        let config = build_opencode_config(&input(&env));
        assert_eq!(config["permission"]["read"], "allow");
        assert_eq!(config["permission"]["bash"], "ask");
        assert_eq!(config["permission"]["question"], "allow");
        assert_eq!(config["share"], "disabled");
        assert_eq!(config["autoupdate"], false);
        assert_eq!(
            config["skills"],
            json!({ "paths": ["/home/u/.agents/skills"] })
        );
        assert!(config.get("mcp").is_none());

        let inactive = vec!["bash".to_string(), "agent".to_string()];
        let mut restrictive = input(&env);
        restrictive.permission_mode = PermissionMode::AllowReads;
        restrictive.inactive_builtin_tool_names = &inactive;
        restrictive.read_guard = true;
        restrictive.relay_command = Some(vec!["sh".into(), "-c".into(), "x".into()]);
        let config = build_opencode_config(&restrictive);
        assert_eq!(config["permission"]["read"], "deny");
        assert_eq!(config["permission"]["glob"], "ask");
        assert_eq!(config["permission"]["bash"], "deny");
        assert_eq!(config["permission"]["task"], "deny");
        assert_eq!(
            config["mcp"]["harness-tools"],
            json!({ "type": "local", "enabled": true, "command": ["sh", "-c", "x"] })
        );
    }

    #[test]
    fn builds_provider_configs() {
        let headers = IndexMap::from([("x-a".to_string(), "1".to_string())]);
        let gateway = env(&[
            ("AI_GATEWAY_API_KEY", "k"),
            ("AI_GATEWAY_BASE_URL", "https://g/"),
        ]);
        let mut options = input(&gateway);
        options.model = Some("google/gemini");
        options.headers = Some(&headers);
        assert_eq!(
            build_provider_config(&options),
            Some(json!({ "google": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "apiKey": "k",
                    "baseURL": "https://g/v1",
                    "headers": { "x-a": "1", "x-client-app": "ai-sdk-harness-opencode/test" },
                },
                "models": { "gemini": { "id": "gemini", "name": "gemini" } },
            } }))
        );

        let compatible = env(&[
            ("OPENAI_NAME", "local"),
            ("OPENAI_BASE_URL", "http://llm"),
            ("OPENAI_QUERY_PARAMS_JSON", "{\"v\":\"1\"}"),
        ]);
        let mut options = input(&compatible);
        options.model = Some("qwen");
        assert_eq!(
            build_provider_config(&options),
            Some(json!({ "local": {
                "options": { "baseURL": "http://llm", "queryParams": { "v": "1" } },
                "models": { "qwen": { "id": "qwen", "name": "qwen" } },
            } }))
        );

        let anthropic = env(&[("ANTHROPIC_API_KEY", "a")]);
        assert_eq!(
            build_provider_config(&input(&anthropic)),
            Some(json!({ "anthropic": { "options": { "apiKey": "a" } } }))
        );

        let openai = env(&[("OPENAI_API_KEY", "o"), ("OPENAI_PROJECT", "p")]);
        let mut options = input(&openai);
        options.model = Some("openai/gpt-5");
        assert_eq!(
            build_provider_config(&options),
            Some(json!({ "openai": { "options": { "apiKey": "o", "project": "p" } } }))
        );
        assert_eq!(build_provider_config(&input(&env(&[]))), None);
    }

    #[tokio::test]
    async fn resolves_instruction_paths_in_the_sandbox() {
        use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
        use crate::harness::adapters::test_temp_dir::mkdtemp;
        let temp = mkdtemp("opencode-instructions-");
        let root = canonical(temp.path());
        std::fs::create_dir_all(format!("{root}/docs/a")).unwrap();
        std::fs::write(format!("{root}/docs/a/x.md"), "").unwrap();
        std::fs::write(format!("{root}/docs/y.md"), "").unwrap();
        let sandbox = LocalSandbox::new(root.clone());
        let resolved = resolve_opencode_instructions(
            &sandbox,
            &root,
            Some(&json!([
                "rules.md",
                "/abs.md",
                "https://x/y.md",
                "docs/**/*.md"
            ])),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            resolved,
            vec![
                format!("{root}/AGENTS.md"),
                format!("{root}/rules.md"),
                "/abs.md".to_string(),
                "https://x/y.md".to_string(),
                format!("{root}/docs/a/x.md"),
                format!("{root}/docs/y.md"),
            ]
        );
        assert_eq!(
            resolve_opencode_instructions(&sandbox, &root, Some(&json!("x")), None)
                .await
                .unwrap(),
            None
        );
    }
}
