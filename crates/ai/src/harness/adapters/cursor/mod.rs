//! Port of `@ai-sdk/harness-cursor`: the Cursor CLI agent (`agent acp`)
//! on the ACP adapter.

mod subscription;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::harness::adapters::acp::{
    AcpAuthenticationMode, AcpClientApp, AcpCredentialBrokering, AcpHarness, AcpHarnessSettings,
    AcpModelMapping, AcpResolveHostToolCall, AcpResolveHostToolCallOptions,
    AcpResolvedHostToolCall, AcpSource,
};
use crate::harness::adapters::builtin_catalog::load_builtin_tool_catalog;
use crate::harness::auth::{HarnessAuthentication, HarnessCredentialForwarding};
use crate::harness::error::BoxError;
use crate::harness::network::{
    HarnessRequestTransformation, HarnessRequestTransformationKeyValueMatcher,
    HarnessRequestTransformationMatch, HarnessRequestTransformationMatcher,
    HarnessRequestTransformationTransform,
};
use crate::harness::tools::BuiltinTool;

pub use subscription::{
    ReadCursorSubscriptionOptions, read_cursor_subscription, resolve_cursor_auth_path,
    resolve_cursor_subscription_environment,
};

pub const CURSOR_HARNESS_ID: &str = "cursor";
/// `VERSION` of the ported `@ai-sdk/harness-cursor` package.
pub const VERSION: &str = "1.0.103";

const BUILTIN_TOOLS_JSON: &str = include_str!("builtin_tools.json");

/// `CursorAuthenticationMode`.
pub type CursorAuthenticationMode = AcpAuthenticationMode;

/// `CursorHarnessSettings`. The bridge settings of the TS adapter (port,
/// reconnection, keep-alive) have no counterpart: the Rust ACP client talks
/// to the agent's stdio directly.
#[derive(Clone, Default)]
pub struct CursorHarnessSettings {
    /// Declares the provider authentication configured in Cursor, or
    /// supplies an isolated environment for Cursor CLI authentication. The
    /// adapter cannot change provider routing and warns for explicit
    /// routing modes.
    pub auth: Option<CursorAuthenticationMode>,
    /// Customizes each credential value before it is forwarded into a
    /// sandbox process.
    pub credential_forwarding: Option<HarnessCredentialForwarding>,
    /// MCP server definitions keyed by server name, in Cursor's native
    /// format.
    pub mcp_servers: Option<IndexMap<String, Value>>,
}

/// `CURSOR_BUILTIN_TOOLS`. `mcpToolCall` is dynamic and `truncatedToolCall`
/// is a transport sentinel, so neither is a builtin.
pub fn cursor_builtin_tools() -> IndexMap<String, BuiltinTool> {
    load_builtin_tool_catalog(BUILTIN_TOOLS_JSON)
}

/// Each builtin tool's ACP tool call title.
fn cursor_builtin_tool_titles() -> IndexMap<String, String> {
    let catalog: IndexMap<String, Value> =
        serde_json::from_str(BUILTIN_TOOLS_JSON).expect("the builtin tool catalog is valid");
    catalog
        .into_iter()
        .filter_map(|(name, entry)| Some((name, entry.get("title")?.as_str()?.to_string())))
        .collect()
}

/// `createCursor(settings)`.
pub fn create_cursor(settings: CursorHarnessSettings) -> Result<AcpHarness, BoxError> {
    AcpHarness::new(cursor_acp_settings(settings))
}

/// The ACP settings [`create_cursor`] uses.
pub fn cursor_acp_settings(settings: CursorHarnessSettings) -> AcpHarnessSettings {
    if let Some(warning) = cursor_authentication_warning(settings.auth.as_ref()) {
        eprintln!("{warning}");
    }
    let mut acp = AcpHarnessSettings::new(
        CURSOR_HARNESS_ID,
        AcpClientApp {
            name: "ai-sdk-harness-cursor".into(),
            version: VERSION.into(),
        },
        AcpSource::InstallCommand {
            command: "curl https://cursor.com/install -fsS | bash".into(),
        },
        "agent",
        AcpModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        resolve_cursor_host_tool_call(),
    );
    acp.image_content = true;
    acp.resolve_authentication_environment = Some(Arc::new(|auth, env| {
        async move { resolve_cursor_subscription_environment(auth.as_ref(), env).await }.boxed()
    }));
    acp.credential_brokering = Some(cursor_credential_brokering(settings.auth.as_ref()));
    acp.auth = settings.auth;
    acp.credential_forwarding = settings.credential_forwarding;
    acp.mcp_servers = settings.mcp_servers;
    acp.builtin_tools = cursor_builtin_tools();
    acp.builtin_tool_titles = cursor_builtin_tool_titles();
    acp.args = vec![
        "--disable-auto-update".into(),
        "--disable-project-configs".into(),
        "acp".into(),
    ];
    acp.write_protected_paths = vec![".cursor".into()];
    acp.client_capabilities = Some(
        json!({ "_meta": { "parameterizedModelPicker": true } })
            .as_object()
            .cloned()
            .unwrap_or_default(),
    );
    acp.credential_env = Some(vec!["CURSOR_API_KEY".into()]);
    acp
}

/// `warnCursorAuthenticationConfiguration`: Cursor's own configuration
/// controls provider routing.
fn cursor_authentication_warning(auth: Option<&CursorAuthenticationMode>) -> Option<String> {
    let (mode, detail) = match auth? {
        HarnessAuthentication::AiGateway => (
            "ai-gateway",
            "Configure an AI Gateway API key as Cursor's OpenAI API key and set Override OpenAI Base URL to https://ai-gateway.vercel.sh/cursor/v1.",
        ),
        HarnessAuthentication::Choice(_) => (
            "direct",
            "Configure Cursor to use its direct model-provider routing.",
        ),
        _ => return None,
    };
    Some(format!(
        "[cursor] auth: \"{mode}\" cannot configure Cursor provider authentication. {detail} CURSOR_API_KEY is still required for Cursor CLI authentication."
    ))
}

fn exact(value: impl Into<String>) -> Option<HarnessRequestTransformationMatcher> {
    Some(HarnessRequestTransformationMatcher::Exact(value.into()))
}

/// Swaps the sandbox's placeholder `CURSOR_API_KEY` for the real key on
/// Cursor's key exchange, and adds configured headers to model requests.
fn cursor_credential_brokering(auth: Option<&CursorAuthenticationMode>) -> AcpCredentialBrokering {
    let gateway = matches!(auth, Some(HarnessAuthentication::AiGateway));
    Arc::new(move |options| {
        let mut transformations = Vec::new();
        if let (Some(key), Some(sandbox_key)) = (
            options
                .env
                .get("CURSOR_API_KEY")
                .filter(|key| !key.is_empty()),
            options
                .sandbox_env
                .and_then(|sandbox_env| sandbox_env.get("CURSOR_API_KEY"))
                .filter(|key| !key.is_empty()),
        ) {
            transformations.push(HarnessRequestTransformation {
                r#match: HarnessRequestTransformationMatch {
                    host: "api2.cursor.sh".into(),
                    path: exact("/auth/exchange_user_api_key"),
                    method: Some(vec!["POST".into()]),
                    query_string: None,
                    headers: Some(vec![HarnessRequestTransformationKeyValueMatcher {
                        key: exact("Authorization"),
                        value: exact(format!("Bearer {sandbox_key}")),
                    }]),
                },
                transform: HarnessRequestTransformationTransform {
                    headers: IndexMap::from([(
                        "Authorization".to_string(),
                        format!("Bearer {key}"),
                    )]),
                },
            });
        }
        if let Some(headers) = options.headers {
            transformations.push(HarnessRequestTransformation {
                r#match: HarnessRequestTransformationMatch {
                    host: if gateway {
                        "ai-gateway.vercel.sh".into()
                    } else {
                        "api2.cursor.sh".into()
                    },
                    path: gateway.then(|| {
                        HarnessRequestTransformationMatcher::StartsWith("/cursor/v1".into())
                    }),
                    method: None,
                    query_string: None,
                    headers: None,
                },
                transform: HarnessRequestTransformationTransform {
                    headers: headers.clone(),
                },
            });
        }
        Ok(transformations)
    })
}

/// Cursor reports host tool calls with a `{ providerIdentifier, toolName,
/// args }` raw input, where `providerIdentifier` is the MCP server name.
pub fn resolve_cursor_host_tool_call() -> AcpResolveHostToolCall {
    Arc::new(|options: AcpResolveHostToolCallOptions<'_>| {
        let raw_input = options.tool_call.get("rawInput")?.as_object()?;
        if raw_input.get("providerIdentifier")?.as_str()? != options.server_name {
            return None;
        }
        let tool_name = raw_input.get("toolName")?.as_str()?;
        if !options.tool_names.iter().any(|name| name == tool_name) {
            return None;
        }
        let args = raw_input.get("args").filter(|args| args.is_object())?;
        Some(AcpResolvedHostToolCall {
            tool_name: tool_name.to_string(),
            input: args.clone(),
        })
    })
}
