//! Port of `@ai-sdk/harness-github-copilot`: the GitHub Copilot CLI
//! (`copilot --acp --stdio`) on the ACP adapter.

mod subscription;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::harness::adapters::acp::{
    AcpAuthenticationMode, AcpClientApp, AcpCredentialBrokering, AcpGatewayProviderAuthentication,
    AcpHarness, AcpHarnessSettings, AcpHostToolMcpTransport, AcpInstructionMapping,
    AcpModelMapping, AcpProviderAuthentication, AcpResolveHostToolCall,
    AcpResolveHostToolCallOptions, AcpResolvedHostToolCall, AcpSource,
};
use crate::harness::adapters::builtin_catalog::load_builtin_tool_catalog;
use crate::harness::auth::{HarnessCredentialForwarding, HarnessCredentialForwardingOptions};
use crate::harness::error::BoxError;
use crate::harness::network::{
    HarnessRequestTransformation, HarnessRequestTransformationKeyValueMatcher,
    HarnessRequestTransformationMatch, HarnessRequestTransformationMatcher,
    HarnessRequestTransformationTransform,
};
use crate::harness::tools::BuiltinTool;
use crate::harness::utils::sandbox_credential_brokering::{
    create_credential_request_transformation, is_sandbox_credential_placeholder,
};

pub use subscription::{
    FindGitHubCliExecutable, GitHubCopilotSubscription, ReadGitHubCliToken,
    ReadGitHubCopilotSubscription, ReadGitHubCopilotSubscriptionOptions, ReadSecureCredential,
    find_host_github_cli_executable, normalize_github_host, read_github_copilot_subscription,
    resolve_github_copilot_subscription_environment,
};

pub const GITHUB_COPILOT_HARNESS_ID: &str = "github-copilot";
/// `VERSION` of the ported `@ai-sdk/harness-github-copilot` package.
pub const VERSION: &str = "1.0.85";
/// The `@github/copilot` version the bootstrap installs.
pub const GITHUB_COPILOT_CLI_VERSION: &str = "1.0.106";
/// The model of AI Gateway sessions, which the CLI does not report.
pub const GITHUB_COPILOT_GATEWAY_MODEL: &str = "openai/gpt-5.5";

const BUILTIN_TOOLS_JSON: &str = include_str!("builtin_tools.json");
const GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES: [&str; 3] =
    ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"];
const GITHUB_OAUTH_TOKEN_BODY_LENGTH: usize = 36;
const GITHUB_OAUTH_SANDBOX_CREDENTIAL_MARKER: &str = "aisdkhc";

/// `GitHubCopilotAuthenticationMode`.
pub type GitHubCopilotAuthenticationMode = AcpAuthenticationMode;

/// `reasoningEffort`: the CLI's `--reasoning-effort`.
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

/// `GitHubCopilotHarnessSettings`. The bridge settings of the TS adapter
/// (port, reconnection, keep-alive) have no counterpart: the Rust ACP
/// client talks to the CLI's stdio directly.
#[derive(Clone, Default)]
pub struct GitHubCopilotHarnessSettings {
    /// GitHub authentication (a native Copilot login, `gh`, or a token
    /// variable), AI Gateway BYOK routing, or an isolated environment.
    pub auth: Option<GitHubCopilotAuthenticationMode>,
    /// Customizes each credential value before it is forwarded into a
    /// sandbox process.
    pub credential_forwarding: Option<HarnessCredentialForwarding>,
    pub reasoning_effort: Option<GitHubCopilotReasoningEffort>,
    /// MCP server definitions keyed by server name, in ACP format.
    pub mcp_servers: Option<IndexMap<String, Value>>,
    /// [rust] Run this agent command (a sandbox path or command, already
    /// shell-quoted) instead of installing the CLI with the bootstrap.
    pub executable_override: Option<String>,
    /// [rust] Extra arguments, after the adapter's.
    pub extra_args: Vec<String>,
    /// [rust] Extra environment variables for the agent.
    pub env: IndexMap<String, String>,
}

/// `GITHUB_COPILOT_BUILTIN_TOOLS`: the stable, non-MCP tool surface of
/// GitHub Copilot CLI >=1.0.82, with loose object schemas.
pub fn github_copilot_builtin_tools() -> IndexMap<String, BuiltinTool> {
    load_builtin_tool_catalog(BUILTIN_TOOLS_JSON)
}

fn github_copilot_builtin_tool_titles() -> IndexMap<String, String> {
    let catalog: IndexMap<String, Value> =
        serde_json::from_str(BUILTIN_TOOLS_JSON).expect("the builtin tool catalog is valid");
    catalog
        .into_iter()
        .filter_map(|(name, entry)| Some((name, entry.get("title")?.as_str()?.to_string())))
        .collect()
}

/// `createGitHubCopilot(settings)`.
pub fn create_github_copilot(
    settings: GitHubCopilotHarnessSettings,
) -> Result<AcpHarness, BoxError> {
    AcpHarness::new(github_copilot_acp_settings(settings))
}

/// The ACP settings [`create_github_copilot`] uses.
pub fn github_copilot_acp_settings(settings: GitHubCopilotHarnessSettings) -> AcpHarnessSettings {
    let mut acp = AcpHarnessSettings::new(
        GITHUB_COPILOT_HARNESS_ID,
        AcpClientApp {
            name: "ai-sdk-harness-github-copilot".into(),
            version: VERSION.into(),
        },
        AcpSource::Npm {
            package: "@github/copilot".into(),
            version: GITHUB_COPILOT_CLI_VERSION.into(),
        },
        "copilot",
        AcpModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        resolve_github_copilot_host_tool_call(),
    );
    acp.args = ["--acp", "--stdio", "--no-auto-update"]
        .map(String::from)
        .to_vec();
    if let Some(effort) = settings.reasoning_effort {
        acp.args
            .push(format!("--reasoning-effort={}", effort.as_str()));
    }
    acp.args.extend(settings.extra_args.iter().cloned());
    acp.env.extend(settings.env.clone());
    if settings.executable_override.is_some() {
        acp.executable_override = settings.executable_override.clone();
    }
    acp.auth = settings.auth;
    acp.resolve_authentication_environment = Some(Arc::new(|auth, env| {
        async move { resolve_github_copilot_subscription_environment(auth.as_ref(), env, None).await }
            .boxed()
    }));
    // The CLI's stdio MCP servers cannot reach host tools in a sandbox, so
    // host tools are served over HTTP.
    acp.host_tool_mcp_transport = AcpHostToolMcpTransport::Http;
    acp.image_content = true;
    acp.forward_env = vec!["COPILOT_GH_HOST".into(), "GH_HOST".into()];
    acp.credential_env = Some(
        GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES
            .map(String::from)
            .to_vec(),
    );
    acp.credential_brokering = Some(github_copilot_credential_brokering());
    acp.credential_forwarding = Some(github_copilot_credential_forwarding(
        settings.credential_forwarding,
    ));
    acp.provider_authentication = Some(AcpProviderAuthentication {
        gateway: AcpGatewayProviderAuthentication {
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
    acp.instruction_mapping = Some(AcpInstructionMapping::Filesystem {
        path: ".copilot/copilot-instructions.md".into(),
    });
    // Project files the CLI loads as hooks, MCP servers, agents, skills or
    // extensions.
    acp.write_protected_paths = [
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
    .to_vec();
    acp.builtin_tools = github_copilot_builtin_tools();
    acp.builtin_tool_titles = github_copilot_builtin_tool_titles();
    acp.mcp_servers = settings.mcp_servers;
    // `withGitHubCopilotGatewayModel`: with a BYOK provider the CLI reports
    // no model, and runs `COPILOT_MODEL`.
    acp.fallback_model = Some(GITHUB_COPILOT_GATEWAY_MODEL.into());
    acp
}

fn exact(value: impl Into<String>) -> Option<HarnessRequestTransformationMatcher> {
    Some(HarnessRequestTransformationMatcher::Exact(value.into()))
}

fn authorization_swap(host: &str, sandbox: &str, real: &str) -> HarnessRequestTransformation {
    HarnessRequestTransformation {
        r#match: HarnessRequestTransformationMatch {
            host: host.to_string(),
            path: None,
            method: None,
            query_string: None,
            headers: Some(vec![HarnessRequestTransformationKeyValueMatcher {
                key: exact("Authorization"),
                value: exact(sandbox),
            }]),
        },
        transform: HarnessRequestTransformationTransform {
            headers: IndexMap::from([("Authorization".to_string(), real.to_string())]),
        },
    }
}

/// Swaps sandbox placeholders for GitHub tokens on the GitHub host (and
/// Copilot's API hosts on github.com), and for the AI Gateway key on the
/// provider base URL.
fn github_copilot_credential_brokering() -> AcpCredentialBrokering {
    Arc::new(|options| {
        let env = options.env;
        let set = |map: Option<&IndexMap<String, String>>, name: &str| {
            map.and_then(|map| map.get(name))
                .filter(|value| !value.is_empty())
                .cloned()
        };
        let github_host = normalize_github_host(
            env.get("COPILOT_GH_HOST")
                .or_else(|| env.get("GH_HOST"))
                .map(String::as_str)
                .unwrap_or("github.com"),
        )?;
        let mut github_hosts = vec![github_host.clone(), format!("*.{github_host}")];
        if github_host == "github.com" {
            github_hosts.extend(["githubcopilot.com".into(), "*.githubcopilot.com".into()]);
        }
        let mut transformations = Vec::new();
        for host in &github_hosts {
            for name in GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES {
                if let (Some(real), Some(sandbox)) =
                    (set(Some(env), name), set(options.sandbox_env, name))
                {
                    for scheme in ["Bearer", "token"] {
                        transformations.push(authorization_swap(
                            host,
                            &format!("{scheme} {sandbox}"),
                            &format!("{scheme} {real}"),
                        ));
                    }
                }
            }
        }
        let provider_credential = set(Some(env), "COPILOT_PROVIDER_API_KEY");
        let sandbox_provider_credential = set(options.sandbox_env, "COPILOT_PROVIDER_API_KEY");
        let provider_base_url = set(Some(env), "COPILOT_PROVIDER_BASE_URL");
        if let (Some(credential), Some(sandbox_credential), Some(base_url)) = (
            provider_credential,
            sandbox_provider_credential,
            provider_base_url,
        ) {
            let mut headers = options.headers.cloned().unwrap_or_default();
            headers.insert("Authorization".into(), format!("Bearer {credential}"));
            transformations.push(create_credential_request_transformation(
                &base_url,
                &IndexMap::from([(
                    "Authorization".to_string(),
                    format!("Bearer {sandbox_credential}"),
                )]),
                &headers,
            )?);
        } else if let Some(headers) = options.headers {
            let copilot_hosts = if github_host == "github.com" {
                vec![
                    "githubcopilot.com".to_string(),
                    "*.githubcopilot.com".into(),
                ]
            } else {
                vec![github_host.clone(), format!("*.{github_host}")]
            };
            for host in copilot_hosts {
                transformations.push(HarnessRequestTransformation {
                    r#match: HarnessRequestTransformationMatch {
                        host,
                        path: None,
                        method: None,
                        query_string: None,
                        headers: None,
                    },
                    transform: HarnessRequestTransformationTransform {
                        headers: headers.clone(),
                    },
                });
            }
        }
        Ok(transformations)
    })
}

/// Copilot accepts only GitHub-shaped tokens, so a GitHub token's sandbox
/// placeholder is reshaped into a `gho_` token.
fn github_copilot_credential_forwarding(
    credential_forwarding: Option<HarnessCredentialForwarding>,
) -> HarnessCredentialForwarding {
    Arc::new(move |options: HarnessCredentialForwardingOptions| {
        let credential = if GITHUB_COPILOT_TOKEN_ENVIRONMENT_VARIABLES
            .contains(&options.environment_variable_name.as_str())
            && is_sandbox_credential_placeholder(&options.credential)
        {
            github_oauth_sandbox_credential_placeholder(&options.credential)
        } else {
            options.credential
        };
        match &credential_forwarding {
            None => async move { Ok(credential) }.boxed(),
            Some(forward) => forward(HarnessCredentialForwardingOptions {
                credential,
                environment_variable_name: options.environment_variable_name,
            }),
        }
    })
}

/// A `gho_` token whose body is the placeholder's marked SHA-256 digest:
/// still unique per placeholder, and shaped like a GitHub OAuth token.
fn github_oauth_sandbox_credential_placeholder(credential: &str) -> String {
    let digest = crate::harness::adapters::acp::auth::sha256_hex(credential);
    let body =
        &digest[..GITHUB_OAUTH_TOKEN_BODY_LENGTH - GITHUB_OAUTH_SANDBOX_CREDENTIAL_MARKER.len()];
    format!("gho_{GITHUB_OAUTH_SANDBOX_CREDENTIAL_MARKER}{body}")
}

/// Copilot reports host tool calls with no name, the MCP arguments as the
/// raw input, and a `<server>-<tool>` title.
pub fn resolve_github_copilot_host_tool_call() -> AcpResolveHostToolCall {
    Arc::new(|options: AcpResolveHostToolCallOptions<'_>| {
        let tool_call = options.tool_call;
        if tool_call.get("name").is_some_and(|name| !name.is_null()) {
            return None;
        }
        let raw_input = tool_call
            .get("rawInput")
            .filter(|input| input.is_object())?;
        let title = tool_call.get("title")?.as_str()?;
        let tool_name = options
            .tool_names
            .iter()
            .find(|name| title == format!("{}-{name}", options.server_name))?;
        Some(AcpResolvedHostToolCall {
            tool_name: tool_name.clone(),
            input: raw_input.clone(),
        })
    })
}
