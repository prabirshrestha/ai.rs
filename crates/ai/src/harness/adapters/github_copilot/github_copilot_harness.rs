//! Port of `harness-github-copilot/src/github-copilot-harness.ts`.

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use crate::harness::adapters::acp::v1::harness::ACPV1Harness;
use crate::harness::adapters::acp::v1::settings::{
    ACPCredentialBrokeringOptions, ACPGatewayProviderAuthentication, ACPHostToolMCPTransport,
    ACPInstructionMapping, ACPModelMapping, ACPNonTextContentType, ACPNpmLockedSource,
    ACPProviderAuthentication, ACPResolveAuthenticationEnvironmentOptions, ACPSource,
    ACPV1Settings,
};
use crate::harness::adapters::acp::{
    ACPAuthenticationMode, ACPClientApp, ACPHarnessSettings, ACPResolveHostToolCallOptions,
    ACPResolvedHostToolCall, ACPToolCall, create_acp,
};
use crate::harness::adapters::zod_schema::{
    Field, array, boolean, enumeration, loose_object, nonnegative_integer, number, optional,
    required, string,
};
use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::sdk::tool::Tool;
use crate::harness::spec::network_sandbox::{
    HarnessV1PortEndpoint, HarnessV1RequestTransformation,
    HarnessV1RequestTransformationKeyValueMatcher, HarnessV1RequestTransformationMatch,
    HarnessV1RequestTransformationMatcher, HarnessV1RequestTransformationTransform,
};
use crate::harness::spec::tools::{
    CommonToolOptions, HarnessV1BuiltinTool, HarnessV1BuiltinToolName, HarnessV1BuiltinToolUseKind,
    common_tool,
};
use crate::harness::spec::{HarnessV1CredentialForwarding, HarnessV1CredentialForwardingOptions};
use crate::harness::utils::{
    SandboxChannelReconnectOptions, create_credential_request_transformation,
    is_sandbox_credential_placeholder,
};

use super::VERSION;
use super::github_copilot_gateway_model::{
    GitHubCopilotGatewayModelHarness, with_github_copilot_gateway_model,
};
use super::github_copilot_subscription::{
    normalize_github_host, resolve_github_copilot_subscription_environment,
};

const GITHUB_COPILOT_BRIDGE_PACKAGE_JSON: &str =
    include_str!("../../../../assets/harness/github-copilot/package.json");
const GITHUB_COPILOT_BRIDGE_PNPM_LOCK_YAML: &str =
    include_str!("../../../../assets/harness/github-copilot/pnpm-lockfile.yaml");
const GITHUB_COPILOT_BRIDGE_PNPM_WORKSPACE_YAML: &str =
    include_str!("../../../../assets/harness/github-copilot/pnpm-workspace.yaml");

const GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES: [&str; 3] =
    ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"];
const GITHUB_OAUTH_TOKEN_BODY_LENGTH: usize = 36;
const GITHUB_OAUTH_SANDBOX_CREDENTIAL_MARKER: &str = "aisdkhc";
const GITHUB_COPILOT_GATEWAY_MODEL: &str = "openai/gpt-5.5";

/// `GitHubCopilotAuthenticationMode`.
pub type GitHubCopilotAuthenticationMode = ACPAuthenticationMode;

/// `reasoningEffort` of [`GitHubCopilotHarnessSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GitHubCopilotReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl GitHubCopilotReasoningEffort {
    /// The TS value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// `GitHubCopilotHarnessSettings`.
#[derive(Clone, Default)]
pub struct GitHubCopilotHarnessSettings {
    pub auth: Option<GitHubCopilotAuthenticationMode>,
    pub credential_forwarding: Option<HarnessV1CredentialForwarding>,
    pub reasoning_effort: Option<GitHubCopilotReasoningEffort>,
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
    pub port: Option<u16>,
    pub port_endpoint: Option<HarnessV1PortEndpoint>,
    pub startup_timeout_ms: Option<u64>,
    /// Abort a running turn when its host disconnects and does not reconnect
    /// within this many milliseconds, so the turn stops spending. A
    /// suspended turn is exempt. Unset keeps the turn running until it
    /// finishes.
    pub host_disconnect_timeout_ms: Option<u64>,
    /// Configures reconnection attempts after an established bridge
    /// connection drops. The reconnect window includes connection
    /// establishment and backoff delays. Defaults to 30 seconds with
    /// exponential backoff from 50 milliseconds up to 2 seconds.
    pub reconnect: Option<SandboxChannelReconnectOptions>,
    /// WebSocket ping interval in milliseconds. Defaults to 20_000. Set to 0
    /// to disable. Two unanswered pings terminate the socket and trigger
    /// reconnect.
    pub keep_alive_ms: Option<u64>,
}

/// `GITHUB_COPILOT_CLIENT_APP`.
fn github_copilot_client_app() -> ACPClientApp {
    ACPClientApp {
        name: "ai-sdk-harness-github-copilot".into(),
        version: VERSION.into(),
    }
}

fn tool(
    title: Option<&str>,
    fields: &[Field],
    tool_use_kind: Option<HarnessV1BuiltinToolUseKind>,
) -> HarnessV1BuiltinTool {
    let mut tool = Tool::new(loose_object(fields));
    tool.title = title.map(str::to_string);
    HarnessV1BuiltinTool {
        tool,
        native_name: None,
        common_name: None,
        tool_use_kind,
    }
}

fn common(
    common_name: HarnessV1BuiltinToolName,
    tool_use_kind: HarnessV1BuiltinToolUseKind,
    fields: &[Field],
    title: Option<&str>,
) -> HarnessV1BuiltinTool {
    let mut tool = common_tool(
        common_name,
        CommonToolOptions {
            native_name: common_name.as_str().into(),
            tool_use_kind: Some(tool_use_kind),
            description: None,
            input_schema: loose_object(fields),
        },
    )
    .expect("GitHub Copilot built-in schemas extend the standard schemas");
    if let Some(title) = title {
        tool.tool.title = Some(title.into());
    }
    tool
}

/// `GITHUB_COPILOT_BUILTIN_TOOLS`.
///
/// This catalog reflects the stable, non-MCP tool surface emitted by GitHub
/// Copilot CLI >=1.0.82. Loose object schemas preserve compatibility when
/// the CLI adds fields without changing the established inputs.
pub fn github_copilot_builtin_tools() -> IndexMap<String, HarnessV1BuiltinTool> {
    use HarnessV1BuiltinToolName as N;
    use HarnessV1BuiltinToolUseKind::{Bash, Edit, Readonly};
    IndexMap::from([
        (
            "bash".into(),
            common(
                N::Bash,
                Bash,
                &[
                    required("command", string()),
                    optional("description", string()),
                    optional("shellId", string()),
                    optional("mode", enumeration(&["sync", "async"])),
                    optional("detach", boolean()),
                    optional("initial_wait", number()),
                ],
                None,
            ),
        ),
        (
            "read_bash".into(),
            tool(
                Some("Reading shell output"),
                &[required("shellId", string()), required("delay", number())],
                Some(Readonly),
            ),
        ),
        (
            "stop_bash".into(),
            tool(
                Some("Stopping shell session"),
                &[required("shellId", string())],
                Some(Bash),
            ),
        ),
        (
            "list_bash".into(),
            tool(Some("list_bash"), &[], Some(Readonly)),
        ),
        (
            "view".into(),
            tool(
                Some("Viewing "),
                &[
                    required("path", string()),
                    optional("view_range", array(number())),
                    optional("forceReadLargeFiles", boolean()),
                ],
                Some(Readonly),
            ),
        ),
        (
            "create".into(),
            tool(
                Some("Creating "),
                &[required("path", string()), required("file_text", string())],
                Some(Edit),
            ),
        ),
        (
            "edit".into(),
            tool(
                Some("Editing "),
                &[
                    required("path", string()),
                    optional("old_str", string()),
                    required("new_str", string()),
                ],
                Some(Edit),
            ),
        ),
        (
            "web_fetch".into(),
            tool(
                Some("Fetching "),
                &[
                    required("url", string()),
                    optional("max_length", number()),
                    optional("start_index", number()),
                    optional("raw", boolean()),
                ],
                Some(Readonly),
            ),
        ),
        (
            "skill".into(),
            tool(Some("Using skill: "), &[required("skill", string())], None),
        ),
        (
            "sql".into(),
            tool(
                None,
                &[
                    required("description", string()),
                    required("query", string()),
                    optional("database", enumeration(&["session", "session_store"])),
                ],
                None,
            ),
        ),
        (
            "read_agent".into(),
            tool(
                Some("read_agent"),
                &[
                    required("agent_id", string()),
                    optional("wait", boolean()),
                    optional("timeout", number()),
                    optional("since_turn", nonnegative_integer()),
                ],
                Some(Readonly),
            ),
        ),
        (
            "list_agents".into(),
            tool(
                Some("list_agents"),
                &[
                    optional("include_completed", boolean()),
                    optional("scope", enumeration(&["siblings", "children", "all"])),
                ],
                Some(Readonly),
            ),
        ),
        (
            "write_agent".into(),
            tool(
                Some("write_agent"),
                &[
                    required("message", string()),
                    optional("agent_id", string()),
                    optional("agent_ids", array(string())),
                    optional("scope", enumeration(&["siblings", "children"])),
                ],
                None,
            ),
        ),
        (
            "grep".into(),
            common(
                N::Grep,
                Readonly,
                &[
                    required("pattern", string()),
                    optional("path", string()),
                    optional("paths", array(string())),
                    optional("glob", string()),
                    optional(
                        "output_mode",
                        enumeration(&["content", "files_with_matches", "count"]),
                    ),
                    optional("case_insensitive", boolean()),
                    optional("multiline", boolean()),
                    optional("head_limit", number()),
                    optional("context", number()),
                    optional("before_context", number()),
                    optional("after_context", number()),
                ],
                Some("Searching for "),
            ),
        ),
        (
            "glob".into(),
            common(
                N::Glob,
                Readonly,
                &[
                    required("pattern", string()),
                    optional("paths", array(string())),
                ],
                Some("Finding files matching "),
            ),
        ),
        (
            "task".into(),
            tool(
                None,
                &[
                    required("name", string()),
                    required("prompt", string()),
                    required(
                        "agent_type",
                        enumeration(&[
                            "explore",
                            "task",
                            "general-purpose",
                            "code-review",
                            "research",
                            "security-review",
                        ]),
                    ),
                    required("description", string()),
                    optional("model", string()),
                    optional("context_tier", enumeration(&["default", "long_context"])),
                    optional("mode", enumeration(&["sync", "background"])),
                ],
                None,
            ),
        ),
    ])
}

/// The harness `createGitHubCopilot` returns.
pub type GitHubCopilotHarness = GitHubCopilotGatewayModelHarness<ACPV1Harness>;

/// `createGitHubCopilot(settings)`.
pub fn create_github_copilot(
    settings: GitHubCopilotHarnessSettings,
) -> HarnessResult<GitHubCopilotHarness> {
    let harness = create_acp(github_copilot_acp_settings(settings))?;
    Ok(with_github_copilot_gateway_model(
        harness,
        GITHUB_COPILOT_GATEWAY_MODEL,
    ))
}

/// `githubCopilot`: the default GitHub Copilot harness instance. Equivalent
/// to `create_github_copilot(GitHubCopilotHarnessSettings::default())`.
pub fn github_copilot() -> HarnessResult<GitHubCopilotHarness> {
    create_github_copilot(GitHubCopilotHarnessSettings::default())
}

/// [rust] The settings `createGitHubCopilot` passes to `createACP`, split out
/// so the tests can inspect them as the TS tests inspect the mocked call.
pub(crate) fn github_copilot_acp_settings(
    settings: GitHubCopilotHarnessSettings,
) -> ACPHarnessSettings {
    let mcp_tool_title_prefixes: Vec<String> = std::iter::once("github-mcp-server-".to_string())
        .chain(
            settings
                .mcp_servers
                .iter()
                .flat_map(|servers| servers.keys())
                .map(|name| format!("{name}-")),
        )
        .collect();

    let mut acp = ACPV1Settings::new(
        "github-copilot",
        ACPSource::NpmLocked(ACPNpmLockedSource {
            package_json: GITHUB_COPILOT_BRIDGE_PACKAGE_JSON.into(),
            pnpm_lock_yaml: GITHUB_COPILOT_BRIDGE_PNPM_LOCK_YAML.into(),
            pnpm_workspace_yaml: Some(GITHUB_COPILOT_BRIDGE_PNPM_WORKSPACE_YAML.into()),
        }),
        "copilot",
        ACPModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        Arc::new(move |tool_call: &ACPToolCall| {
            Ok(mcp_tool_title_prefixes
                .iter()
                .any(|prefix| tool_call.title.starts_with(prefix.as_str())))
        }),
        Arc::new(resolve_github_copilot_host_tool_call),
    );
    acp.version = Some("v1".into());
    // GitHub Copilot CLI does not expose an option to disable project-level
    // configuration. This ACP integration cannot isolate repository-owned
    // configuration, so working directory contents must be trusted.
    let mut args: Vec<String> = ["--acp", "--stdio", "--no-auto-update"]
        .map(String::from)
        .to_vec();
    if let Some(reasoning_effort) = settings.reasoning_effort {
        args.push(format!("--reasoning-effort={}", reasoning_effort.as_str()));
    }
    acp.args = Some(args);
    acp.auth = settings.auth;
    acp.resolve_authentication_environment = Some(Arc::new(
        |options: ACPResolveAuthenticationEnvironmentOptions| {
            async move {
                resolve_github_copilot_subscription_environment(
                    options.auth.as_ref(),
                    options.env,
                    None,
                )
                .await
            }
            .boxed()
        },
    ));
    // The Copilot CLI drops every MCP server that a client passes in ACP
    // `session/new` unless it is declared with the `http` or `sse`
    // transport, so host tools must be exposed over the bridge's HTTP MCP
    // endpoint.
    acp.host_tool_mcp_transport = Some(ACPHostToolMCPTransport::Http);
    // Images from tool outputs are only processed with direct GitHub Copilot
    // authentication. For third parties like AI Gateway, GitHub Copilot does
    // not fetch the model catalog and therefore its check for image support
    // from a tool output leads to it passing only the text.
    acp.non_text_content_types = Some(vec![ACPNonTextContentType::Image]);
    acp.forward_env = Some(vec!["COPILOT_GH_HOST".into(), "GH_HOST".into()]);
    acp.credential_env = Some(
        GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES
            .map(String::from)
            .to_vec(),
    );
    acp.credential_brokering = Some(Arc::new(github_copilot_credential_brokering));
    acp.credential_forwarding = Some(create_github_copilot_credential_forwarding(
        settings.credential_forwarding,
    ));
    acp.provider_authentication = Some(ACPProviderAuthentication {
        gateway: ACPGatewayProviderAuthentication {
            env: IndexMap::from([
                (
                    "COPILOT_PROVIDER_BASE_URL".to_string(),
                    json!({ "$source": "gateway-base-url", "ensureSuffix": "/v1" }),
                ),
                ("COPILOT_PROVIDER_TYPE".to_string(), json!("openai")),
                (
                    "COPILOT_PROVIDER_API_KEY".to_string(),
                    json!({ "$source": "gateway-api-key" }),
                ),
                ("COPILOT_PROVIDER_WIRE_API".to_string(), json!("responses")),
                (
                    "COPILOT_MODEL".to_string(),
                    json!(GITHUB_COPILOT_GATEWAY_MODEL),
                ),
                (
                    "COPILOT_PROVIDER_HEADERS".to_string(),
                    json!({ "$source": "client-app", "prefix": "x-client-app: " }),
                ),
            ]),
        },
    });
    acp.skills_directory = Some(".copilot/skills".into());
    acp.instruction_mapping = Some(ACPInstructionMapping::Filesystem {
        path: ".copilot/copilot-instructions.md".into(),
    });
    // GitHub Copilot CLI loads hooks, MCP servers, skills, custom agents, and
    // extensions from these repository locations when a session starts, some
    // of which run shell commands outside of tool approval. The rest of
    // `.github` is left writable because it also holds non-agent files such
    // as CI workflows.
    acp.write_protected_paths = Some(
        [
            ".github/hooks",
            ".github/mcp.json",
            ".github/copilot",
            ".github/skills",
            ".github/agents",
            ".github/extensions",
            ".claude",
            ".mcp.json",
            ".vscode/mcp.json",
        ]
        .map(String::from)
        .to_vec(),
    );
    acp.mcp_servers = settings.mcp_servers;

    let mut harness = ACPHarnessSettings::new(acp);
    harness.client_app = Some(github_copilot_client_app());
    harness.builtin_tools = Some(github_copilot_builtin_tools());
    harness.port = settings.port;
    harness.port_endpoint = settings.port_endpoint;
    harness.startup_timeout_ms = settings.startup_timeout_ms;
    harness.host_disconnect_timeout_ms = settings.host_disconnect_timeout_ms;
    harness.reconnect = settings.reconnect;
    harness.keep_alive_ms = settings.keep_alive_ms;
    harness
}

fn non_empty<'a>(env: Option<&'a IndexMap<String, String>>, name: &str) -> Option<&'a str> {
    env.and_then(|env| env.get(name))
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

fn authorization_transformation(
    host: &str,
    placeholder: &str,
    credential: &str,
) -> HarnessV1RequestTransformation {
    HarnessV1RequestTransformation {
        r#match: HarnessV1RequestTransformationMatch {
            host: host.into(),
            path: None,
            method: None,
            query_string: None,
            headers: Some(vec![HarnessV1RequestTransformationKeyValueMatcher {
                key: Some(HarnessV1RequestTransformationMatcher::Exact(
                    "Authorization".into(),
                )),
                value: Some(HarnessV1RequestTransformationMatcher::Exact(
                    placeholder.into(),
                )),
            }]),
        },
        transform: HarnessV1RequestTransformationTransform {
            headers: IndexMap::from([("Authorization".to_string(), credential.to_string())]),
        },
    }
}

fn host_transformation(
    host: &str,
    headers: &IndexMap<String, String>,
) -> HarnessV1RequestTransformation {
    HarnessV1RequestTransformation {
        r#match: HarnessV1RequestTransformationMatch {
            host: host.into(),
            path: None,
            method: None,
            query_string: None,
            headers: None,
        },
        transform: HarnessV1RequestTransformationTransform {
            headers: headers.clone(),
        },
    }
}

fn github_copilot_credential_brokering(
    options: ACPCredentialBrokeringOptions<'_>,
) -> HarnessResult<Vec<HarnessV1RequestTransformation>> {
    let env = options.env;
    let sandbox_env = options.sandbox_env;
    let mut transformations = Vec::new();
    let github_host = normalize_github_host(
        env.get("COPILOT_GH_HOST")
            .or_else(|| env.get("GH_HOST"))
            .map(String::as_str)
            .unwrap_or("github.com"),
    )?;
    let is_github_com = github_host == "github.com";
    let mut github_hosts = vec![github_host.clone(), format!("*.{github_host}")];
    if is_github_com {
        github_hosts.extend(["githubcopilot.com".into(), "*.githubcopilot.com".into()]);
    }

    for host in &github_hosts {
        for name in GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES {
            if let (Some(credential), Some(placeholder)) =
                (non_empty(Some(env), name), non_empty(sandbox_env, name))
            {
                for scheme in ["Bearer", "token"] {
                    transformations.push(authorization_transformation(
                        host,
                        &format!("{scheme} {placeholder}"),
                        &format!("{scheme} {credential}"),
                    ));
                }
            }
        }
    }

    let provider_credential = non_empty(Some(env), "COPILOT_PROVIDER_API_KEY");
    let sandbox_provider_credential = non_empty(sandbox_env, "COPILOT_PROVIDER_API_KEY");
    let provider_base_url = non_empty(Some(env), "COPILOT_PROVIDER_BASE_URL");
    if let (Some(provider_credential), Some(sandbox_provider_credential), Some(provider_base_url)) = (
        provider_credential,
        sandbox_provider_credential,
        provider_base_url,
    ) {
        let mut transform_headers = options.headers.cloned().unwrap_or_default();
        transform_headers.insert(
            "Authorization".into(),
            format!("Bearer {provider_credential}"),
        );
        transformations.push(create_credential_request_transformation(
            provider_base_url,
            &IndexMap::from([(
                "Authorization".to_string(),
                format!("Bearer {sandbox_provider_credential}"),
            )]),
            &transform_headers,
        )?);
    } else if let Some(headers) = options.headers {
        let copilot_hosts = if is_github_com {
            vec![
                "githubcopilot.com".to_string(),
                "*.githubcopilot.com".into(),
            ]
        } else {
            vec![github_host.clone(), format!("*.{github_host}")]
        };
        for host in &copilot_hosts {
            transformations.push(host_transformation(host, headers));
        }
    }

    Ok(transformations)
}

fn create_github_copilot_credential_forwarding(
    credential_forwarding: Option<HarnessV1CredentialForwarding>,
) -> HarnessV1CredentialForwarding {
    Arc::new(move |options: HarnessV1CredentialForwardingOptions| {
        let sandbox_credential = if GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES
            .contains(&options.environment_variable_name.as_str())
            && is_sandbox_credential_placeholder(&options.credential)
        {
            create_github_oauth_sandbox_credential_placeholder(&options.credential)
        } else {
            options.credential
        };
        match &credential_forwarding {
            None => async move { Ok(sandbox_credential) }.boxed(),
            Some(forward) => forward(HarnessV1CredentialForwardingOptions {
                credential: sandbox_credential,
                environment_variable_name: options.environment_variable_name,
            }),
        }
    })
}

/// Copilot CLI validates environment credentials as GitHub token formats
/// before making the request that credential brokering can transform.
/// GitHub OAuth tokens use `gho_` followed by a 36-character alphanumeric
/// body.
fn create_github_oauth_sandbox_credential_placeholder(credential: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, credential.as_bytes());
    let hex: String = digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let token_body =
        &hex[..GITHUB_OAUTH_TOKEN_BODY_LENGTH - GITHUB_OAUTH_SANDBOX_CREDENTIAL_MARKER.len()];
    format!("gho_{GITHUB_OAUTH_SANDBOX_CREDENTIAL_MARKER}{token_body}")
}

/// GitHub Copilot reports host tool calls without a programmatic name, using
/// `${serverName}-${toolName}` as the title and the tool input as raw input.
fn resolve_github_copilot_host_tool_call(
    options: ACPResolveHostToolCallOptions<'_>,
) -> Result<Option<ACPResolvedHostToolCall>, BoxError> {
    let tool_call = options.tool_call;
    if matches!(tool_call.name, Some(Some(_))) {
        return Ok(None);
    }
    let Some(raw_input) = tool_call.raw_input.as_ref().and_then(JsonValue::as_object) else {
        return Ok(None);
    };
    let tool_name = options
        .tool_names
        .iter()
        .find(|name| tool_call.title == format!("{}-{name}", options.server_name));
    Ok(tool_name.map(|tool_name| ACPResolvedHostToolCall {
        tool_name: tool_name.clone(),
        input: raw_input.clone().into_iter().collect(),
    }))
}
