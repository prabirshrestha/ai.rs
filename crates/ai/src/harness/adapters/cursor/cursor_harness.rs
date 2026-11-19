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
use crate::harness::adapters::zod_schema::{
    Field, array, literal, loose_object, loose_object_json, number, optional, required, string,
    unknown,
};
use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::sdk::tool::Tool;
use crate::harness::spec::HarnessV1CredentialForwarding;
use crate::harness::spec::network_sandbox::{
    HarnessV1PortEndpoint, HarnessV1RequestTransformation,
    HarnessV1RequestTransformationKeyValueMatcher, HarnessV1RequestTransformationMatch,
    HarnessV1RequestTransformationMatcher, HarnessV1RequestTransformationTransform,
};
use crate::harness::spec::tools::{
    CommonToolOptions, HarnessV1BuiltinTool, HarnessV1BuiltinToolName, HarnessV1BuiltinToolUseKind,
    common_tool,
};
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

fn tool(
    title: &str,
    fields: &[Field],
    native_name: &str,
    tool_use_kind: Option<HarnessV1BuiltinToolUseKind>,
) -> HarnessV1BuiltinTool {
    let mut tool = Tool::new(loose_object(fields));
    tool.title = Some(title.into());
    HarnessV1BuiltinTool {
        tool,
        native_name: Some(native_name.into()),
        common_name: None,
        tool_use_kind,
    }
}

fn common(
    common_name: HarnessV1BuiltinToolName,
    native_name: &str,
    tool_use_kind: HarnessV1BuiltinToolUseKind,
    fields: &[Field],
    title: &str,
) -> HarnessV1BuiltinTool {
    let mut tool = common_tool(
        common_name,
        CommonToolOptions {
            native_name: native_name.into(),
            tool_use_kind: Some(tool_use_kind),
            description: None,
            input_schema: loose_object(fields),
        },
    )
    .expect("Cursor built-in schemas extend the standard schemas");
    tool.tool.title = Some(title.into());
    tool
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
    use HarnessV1BuiltinToolName as N;
    use HarnessV1BuiltinToolUseKind::{Bash, Edit, Readonly};
    let path = || [required("path", string())];
    IndexMap::from([
        (
            "bash".into(),
            common(
                N::Bash,
                "shellToolCall",
                Bash,
                &[required("command", string())],
                "Terminal",
            ),
        ),
        (
            "delete".into(),
            tool("Delete", &path(), "deleteToolCall", Some(Edit)),
        ),
        (
            "glob".into(),
            common(
                N::Glob,
                "globToolCall",
                Readonly,
                &[required("pattern", string())],
                "Find",
            ),
        ),
        (
            "grep".into(),
            common(
                N::Grep,
                "grepToolCall",
                Readonly,
                &[required("pattern", string()), optional("path", string())],
                "grep",
            ),
        ),
        (
            "read".into(),
            tool("Read", &path(), "readToolCall", Some(Readonly)),
        ),
        (
            "write".into(),
            tool("Write", &path(), "writeToolCall", Some(Edit)),
        ),
        (
            "updateTodos".into(),
            tool(
                "Update TODOs",
                &[
                    required("_toolName", literal("updateTodos")),
                    optional(
                        "todos",
                        array(loose_object_json(&[
                            required("id", string()),
                            required("content", string()),
                            required("status", string()),
                        ])),
                    ),
                ],
                "updateTodosToolCall",
                None,
            ),
        ),
        (
            "readTodos".into(),
            tool("Read TODOs", &[], "readTodosToolCall", Some(Readonly)),
        ),
        (
            "edit".into(),
            tool("Edit", &path(), "editToolCall", Some(Edit)),
        ),
        (
            "ls".into(),
            tool("List", &path(), "lsToolCall", Some(Readonly)),
        ),
        (
            "readLints".into(),
            tool(
                "Read Lints",
                &[required("paths", array(string()))],
                "readLintsToolCall",
                Some(Readonly),
            ),
        ),
        (
            "semanticSearch".into(),
            tool(
                "Codebase Search",
                &[required("query", string())],
                "semSearchToolCall",
                Some(Readonly),
            ),
        ),
        (
            "createPlan".into(),
            tool(
                "Create Plan",
                &[
                    required("_toolName", literal("createPlan")),
                    optional("name", string()),
                    optional("plan", string()),
                ],
                "createPlanToolCall",
                None,
            ),
        ),
        (
            "webSearch".into(),
            tool(
                "Web Search",
                &[required("searchTerm", string())],
                "webSearchToolCall",
                Some(Readonly),
            ),
        ),
        (
            "task".into(),
            tool(
                "Task",
                &[
                    required("_toolName", literal("task")),
                    optional("prompt", string()),
                    optional("description", string()),
                    optional("subagentType", unknown()),
                ],
                "taskToolCall",
                None,
            ),
        ),
        (
            "listMcpResources".into(),
            tool(
                "List MCP Resources",
                &[optional("server", string())],
                "listMcpResourcesToolCall",
                Some(Readonly),
            ),
        ),
        (
            "readMcpResource".into(),
            tool(
                "Fetch MCP Resource",
                &[
                    optional("server", string()),
                    optional("uri", string()),
                    optional("downloadPath", string()),
                ],
                "readMcpResourceToolCall",
                Some(Readonly),
            ),
        ),
        (
            "applyAgentDiff".into(),
            tool(
                "Apply Agent Diff",
                &[optional("path", string())],
                "applyAgentDiffToolCall",
                Some(Edit),
            ),
        ),
        (
            "fetch".into(),
            tool(
                "Fetch",
                &[required("url", string())],
                "fetchToolCall",
                Some(Readonly),
            ),
        ),
        (
            "switchMode".into(),
            tool(
                "Switch Mode",
                &[
                    required("targetModeId", string()),
                    optional("explanation", string()),
                ],
                "switchModeToolCall",
                None,
            ),
        ),
        (
            "generateImage".into(),
            tool(
                "Generate Image",
                &[
                    required("_toolName", literal("generateImage")),
                    optional("description", string()),
                    optional("filename", string()),
                ],
                "generateImageToolCall",
                None,
            ),
        ),
        (
            "recordScreen".into(),
            tool("Record Screen", &[], "recordScreenToolCall", None),
        ),
        (
            "computerUse".into(),
            tool(
                "Computer Use",
                &[
                    optional("action", string()),
                    optional("coordinate", unknown()),
                    optional("text", string()),
                ],
                "computerUseToolCall",
                Some(Bash),
            ),
        ),
        (
            "writeShellStdin".into(),
            tool(
                "Write to stdin",
                &[optional("shellId", number()), optional("chars", string())],
                "writeShellStdinToolCall",
                Some(Bash),
            ),
        ),
        (
            "reflect".into(),
            tool(
                "Reflect",
                &[optional("reflection", string())],
                "reflectToolCall",
                Some(Readonly),
            ),
        ),
        (
            "setupVmEnvironment".into(),
            tool(
                "Setup VM Environment",
                &[],
                "setupVmEnvironmentToolCall",
                Some(Bash),
            ),
        ),
        (
            "replaceEnv".into(),
            tool("Replace Environment", &[], "replaceEnvToolCall", Some(Bash)),
        ),
        (
            "startGrindExecution".into(),
            tool(
                "Start Grind Execution",
                &[],
                "startGrindExecutionToolCall",
                Some(Bash),
            ),
        ),
        (
            "startGrindPlanning".into(),
            tool(
                "Start Grind Planning",
                &[],
                "startGrindPlanningToolCall",
                Some(Readonly),
            ),
        ),
        (
            "webFetch".into(),
            tool(
                "Web Fetch",
                &[required("url", string())],
                "webFetchToolCall",
                Some(Readonly),
            ),
        ),
        (
            "reportBugfixResults".into(),
            tool(
                "Report Bugfix Results",
                &[],
                "reportBugfixResultsToolCall",
                None,
            ),
        ),
    ])
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
