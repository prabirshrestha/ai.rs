//! Port of `harness-cursor/src/cursor-harness.ts`.

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::{Value as JsonValue, json};

use crate::harness::adapters::acp::v1::harness::ACPV1Harness;
use crate::harness::adapters::acp::v1::settings::{
    ACPCredentialBrokeringOptions, ACPInstallCommandSource, ACPModelMapping, ACPNonTextContentType,
    ACPResolveAuthenticationEnvironmentOptions, ACPSource, ACPV1Settings,
};
use crate::harness::adapters::acp::{
    ACPAuthenticationMode, ACPClientApp, ACPHarnessSettings, ACPResolveHostToolCallOptions,
    ACPResolvedHostToolCall, ACPToolCall, create_acp,
};
use crate::harness::adapters::builtin_tool_catalog::load_builtin_tool_catalog;
use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::spec::HarnessV1CredentialForwarding;
use crate::harness::spec::network_sandbox::{
    HarnessV1PortEndpoint, HarnessV1RequestTransformation,
    HarnessV1RequestTransformationKeyValueMatcher, HarnessV1RequestTransformationMatch,
    HarnessV1RequestTransformationMatcher, HarnessV1RequestTransformationTransform,
};
use crate::harness::spec::tools::HarnessV1BuiltinTool;
use crate::harness::spec::{HarnessV1Authentication, HarnessV1DirectAuthentication};
use crate::harness::utils::SandboxChannelReconnectOptions;

use super::VERSION;
use super::cursor_subscription::resolve_cursor_subscription_environment;

/// `CursorAuthenticationMode`.
pub type CursorAuthenticationMode = ACPAuthenticationMode;

/// `CursorHarnessSettings`.
#[derive(Clone, Default)]
pub struct CursorHarnessSettings {
    /// Declares the provider authentication configured in Cursor, or
    /// supplies an isolated environment for Cursor CLI authentication. The
    /// adapter cannot change provider routing and warns for explicit routing
    /// modes.
    pub auth: Option<CursorAuthenticationMode>,
    /// Customizes each credential value before it is forwarded into a
    /// sandbox process. This does not restrict which credentials the harness
    /// adapter can discover, read, or otherwise access in the host process.
    pub credential_forwarding: Option<HarnessV1CredentialForwarding>,
    /// Overrides the sandbox port used by the ACP bridge.
    pub port: Option<u16>,
    /// Override the host endpoint used to connect to the sandbox bridge.
    /// Required together with `port` when using a basic sandbox session.
    pub port_endpoint: Option<HarnessV1PortEndpoint>,
    /// Maximum milliseconds to wait for the ACP bridge to start.
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
    /// MCP server definitions keyed by server name. Each definition uses the
    /// underlying runtime's native MCP server configuration format.
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
}

/// `CURSOR_CLIENT_APP`.
fn cursor_client_app() -> ACPClientApp {
    ACPClientApp {
        name: "ai-sdk-harness-cursor".into(),
        version: VERSION.into(),
    }
}

/// `CURSOR_BUILTIN_TOOLS`.
///
/// The original variants were captured from Cursor CLI 2026.08.11-e8db854.
/// Cursor CLI 2026.10.22 reports file writes with a `Write <path>` title, an
/// `edit` kind, and a path-only raw input. Its CLI output reference
/// identifies the native tool as `writeToolCall`. `mcpToolCall` is dynamic
/// and `truncatedToolCall` is a transport sentinel, so neither is exposed as
/// a built-in.
pub fn cursor_builtin_tools() -> IndexMap<String, HarnessV1BuiltinTool> {
    load_builtin_tool_catalog(include_str!(
        "../../../../assets/harness/cursor/builtin-tools.json"
    ))
}

fn is_explicit_routing(auth: Option<&CursorAuthenticationMode>) -> bool {
    matches!(
        auth,
        Some(HarnessV1Authentication::AiGateway)
            | Some(HarnessV1Authentication::Choice(
                HarnessV1DirectAuthentication::Direct
            ))
    )
}

/// `createCursor(settings)`.
pub fn create_cursor(settings: CursorHarnessSettings) -> HarnessResult<ACPV1Harness> {
    if let Some(warning) = cursor_authentication_warning(settings.auth.as_ref()) {
        eprintln!("{warning}");
    }
    create_acp(cursor_acp_settings(settings))
}

/// `cursor`: the default Cursor harness instance. Equivalent to
/// `create_cursor(CursorHarnessSettings::default())`.
pub fn cursor() -> HarnessResult<ACPV1Harness> {
    create_cursor(CursorHarnessSettings::default())
}

/// [rust] The settings `createCursor` passes to `createACP`, split out so
/// the tests can inspect them as the TS tests inspect the mocked call.
pub(crate) fn cursor_acp_settings(settings: CursorHarnessSettings) -> ACPHarnessSettings {
    let gateway = matches!(settings.auth, Some(HarnessV1Authentication::AiGateway));
    let mut acp = ACPV1Settings::new(
        "cursor",
        ACPSource::InstallCommand(ACPInstallCommandSource {
            command: "curl https://cursor.com/install -fsS | bash".into(),
        }),
        "agent",
        ACPModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        Arc::new(|tool_call: &ACPToolCall| Ok(is_cursor_mcp_tool_call(tool_call))),
        Arc::new(resolve_cursor_host_tool_call),
    );
    acp.version = Some("v1".into());
    acp.non_text_content_types = Some(vec![ACPNonTextContentType::Image]);
    acp.auth = settings.auth;
    acp.resolve_authentication_environment = Some(Arc::new(
        |options: ACPResolveAuthenticationEnvironmentOptions| {
            async move {
                resolve_cursor_subscription_environment(options.auth.as_ref(), options.env).await
            }
            .boxed()
        },
    ));
    acp.credential_forwarding = settings.credential_forwarding;
    acp.mcp_servers = settings.mcp_servers;
    acp.args = Some(
        ["--disable-auto-update", "--disable-project-configs", "acp"]
            .map(String::from)
            .to_vec(),
    );
    acp.write_protected_paths = Some(vec![".cursor".into()]);
    acp.client_capabilities = Some(IndexMap::from([(
        "_meta".to_string(),
        json!({ "parameterizedModelPicker": true }),
    )]));
    acp.credential_env = Some(vec!["CURSOR_API_KEY".into()]);
    acp.credential_brokering = Some(Arc::new(move |options| {
        Ok(cursor_credential_brokering(options, gateway))
    }));

    let mut harness = ACPHarnessSettings::new(acp);
    harness.builtin_tools = Some(cursor_builtin_tools());
    harness.port = settings.port;
    harness.port_endpoint = settings.port_endpoint;
    harness.startup_timeout_ms = settings.startup_timeout_ms;
    harness.host_disconnect_timeout_ms = settings.host_disconnect_timeout_ms;
    harness.reconnect = settings.reconnect;
    harness.keep_alive_ms = settings.keep_alive_ms;
    harness.client_app = Some(cursor_client_app());
    harness
}

fn non_empty<'a>(env: Option<&'a IndexMap<String, String>>, name: &str) -> Option<&'a str> {
    env.and_then(|env| env.get(name))
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

fn cursor_credential_brokering(
    options: ACPCredentialBrokeringOptions<'_>,
    gateway: bool,
) -> Vec<HarnessV1RequestTransformation> {
    let mut transformations = Vec::new();
    if let (Some(key), Some(sandbox_key)) = (
        non_empty(Some(options.env), "CURSOR_API_KEY"),
        non_empty(options.sandbox_env, "CURSOR_API_KEY"),
    ) {
        transformations.push(HarnessV1RequestTransformation {
            r#match: HarnessV1RequestTransformationMatch {
                host: "api2.cursor.sh".into(),
                path: Some(HarnessV1RequestTransformationMatcher::Exact(
                    "/auth/exchange_user_api_key".into(),
                )),
                method: Some(vec!["POST".into()]),
                query_string: None,
                headers: Some(vec![HarnessV1RequestTransformationKeyValueMatcher {
                    key: Some(HarnessV1RequestTransformationMatcher::Exact(
                        "Authorization".into(),
                    )),
                    value: Some(HarnessV1RequestTransformationMatcher::Exact(format!(
                        "Bearer {sandbox_key}"
                    ))),
                }]),
            },
            transform: HarnessV1RequestTransformationTransform {
                headers: IndexMap::from([("Authorization".to_string(), format!("Bearer {key}"))]),
            },
        });
    }
    if let Some(headers) = options.headers {
        let (host, path) = if gateway {
            (
                "ai-gateway.vercel.sh",
                Some(HarnessV1RequestTransformationMatcher::StartsWith(
                    "/cursor/v1".into(),
                )),
            )
        } else {
            ("api2.cursor.sh", None)
        };
        transformations.push(HarnessV1RequestTransformation {
            r#match: HarnessV1RequestTransformationMatch {
                host: host.into(),
                path,
                method: None,
                query_string: None,
                headers: None,
            },
            transform: HarnessV1RequestTransformationTransform {
                headers: headers.clone(),
            },
        });
    }
    transformations
}

/// `warnCursorAuthenticationConfiguration`: the warning `createCursor`
/// prints for the `direct` and `ai-gateway` modes.
///
/// [rust] `console.warn` becomes a line on stderr.
pub(crate) fn cursor_authentication_warning(
    auth: Option<&CursorAuthenticationMode>,
) -> Option<String> {
    if !is_explicit_routing(auth) {
        return None;
    }
    let gateway = matches!(auth, Some(HarnessV1Authentication::AiGateway));
    let name = if gateway { "ai-gateway" } else { "direct" };
    let detail = if gateway {
        "Configure an AI Gateway API key as Cursor's OpenAI API key and set Override OpenAI Base URL to https://ai-gateway.vercel.sh/cursor/v1."
    } else {
        "Configure Cursor to use its direct model-provider routing."
    };
    Some(format!(
        "[cursor] auth: {} cannot configure Cursor provider authentication. {detail} CURSOR_API_KEY is still required for Cursor CLI authentication.",
        JsonValue::String(name.into())
    ))
}

fn is_cursor_mcp_tool_call(tool_call: &ACPToolCall) -> bool {
    let Some(raw_input) = tool_call.raw_input.as_ref().and_then(JsonValue::as_object) else {
        return false;
    };
    raw_input
        .get("providerIdentifier")
        .is_some_and(JsonValue::is_string)
        && raw_input.get("toolName").is_some_and(JsonValue::is_string)
        && raw_input.get("args").is_some_and(JsonValue::is_object)
}

/// Cursor reports host tool calls with a `{ providerIdentifier, toolName,
/// args }` raw input, where `providerIdentifier` is the MCP server name.
fn resolve_cursor_host_tool_call(
    options: ACPResolveHostToolCallOptions<'_>,
) -> Result<Option<ACPResolvedHostToolCall>, BoxError> {
    let Some(raw_input) = options
        .tool_call
        .raw_input
        .as_ref()
        .and_then(JsonValue::as_object)
    else {
        return Ok(None);
    };
    if raw_input
        .get("providerIdentifier")
        .and_then(JsonValue::as_str)
        != Some(options.server_name)
    {
        return Ok(None);
    }
    let Some(tool_name) = raw_input.get("toolName").and_then(JsonValue::as_str) else {
        return Ok(None);
    };
    if !options.tool_names.iter().any(|name| name == tool_name) {
        return Ok(None);
    }
    let Some(args) = raw_input.get("args").and_then(JsonValue::as_object) else {
        return Ok(None);
    };
    Ok(Some(ACPResolvedHostToolCall {
        tool_name: tool_name.into(),
        input: args.clone().into_iter().collect(),
    }))
}
