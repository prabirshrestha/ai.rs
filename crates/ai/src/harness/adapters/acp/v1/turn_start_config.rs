//! Port of `harness-acp/src/v1/acp-v1-turn-start-config.ts`.

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::harness::adapters::acp::acp_auth::ACPAuthenticationProfileIdentity;
use crate::harness::agent::internal::sha256_hex;
use crate::harness::provider::{JsonValue, TypeValidationError};
use crate::harness::spec::diagnostic::HarnessV1DebugConfig;
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::tools::{HarnessV1PermissionMode, HarnessV1ToolSpec};

use super::bridge_protocol::{
    ACPBuiltinToolMapping, ACPColdSessionState, ACPTurnStartConfig, ACPTurnStartConfigVersion,
    parse_acp_serializable_tool_spec,
};
use super::prompt::ACPContentBlock;
use super::settings::{
    ACPInstructionMapping, ACPModelMapping, ACPOutputSchemaMapping, ACPPermissionModeMapping,
    ACPSerializableValue,
};

/// Input of [`create_acp_turn_start_config`].
#[derive(Debug, Clone, Copy)]
pub struct CreateACPTurnStartConfigOptions<'a> {
    pub prompt: &'a [ACPContentBlock],
    pub tools: &'a [HarnessV1ToolSpec],
    pub builtin_tools: &'a [ACPBuiltinToolMapping],
    pub permission_mode: HarnessV1PermissionMode,
    pub permission_mode_mapping: Option<&'a ACPPermissionModeMapping>,
    pub mcp_servers: Option<&'a IndexMap<String, JsonValue>>,
    pub debug: Option<&'a HarnessV1DebugConfig>,
    pub authentication_profile: &'a ACPAuthenticationProfileIdentity,
    pub session_meta: Option<&'a IndexMap<String, ACPSerializableValue>>,
    pub instruction_mapping: Option<&'a ACPInstructionMapping>,
    pub response_format: Option<&'a HarnessV1ResponseFormat>,
    pub output_schema_mapping: Option<&'a ACPOutputSchemaMapping>,
    pub model: Option<&'a str>,
    pub model_mapping: &'a ACPModelMapping,
}

fn to_json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// `createACPTurnStartConfig`.
pub fn create_acp_turn_start_config(
    options: CreateACPTurnStartConfigOptions<'_>,
) -> Result<ACPTurnStartConfig, TypeValidationError> {
    let mut fingerprint_input = Map::new();
    fingerprint_input.insert(
        "authenticationProfile".into(),
        to_json(options.authentication_profile),
    );
    fingerprint_input.insert(
        "sessionMeta".into(),
        options.session_meta.map_or(Value::Null, to_json),
    );
    if let Some(mapping) = options.instruction_mapping {
        fingerprint_input.insert("instructionMapping".into(), to_json(mapping));
    }
    if let Some(mapping) = options.output_schema_mapping {
        fingerprint_input.insert("outputSchemaMapping".into(), to_json(mapping));
    }
    fingerprint_input.insert("modelMapping".into(), to_json(options.model_mapping));
    fingerprint_input.insert("builtinTools".into(), to_json(&options.builtin_tools));
    fingerprint_input.insert(
        "permissionModeMapping".into(),
        options.permission_mode_mapping.map_or(Value::Null, to_json),
    );
    fingerprint_input.insert(
        "mcpServers".into(),
        options.mcp_servers.map_or(Value::Null, to_json),
    );

    let tools = options
        .tools
        .iter()
        .map(parse_acp_serializable_tool_spec)
        .collect::<Result<Vec<_>, _>>()?;
    let model = options.model.map(str::to_string);
    Ok(ACPTurnStartConfig {
        version: ACPTurnStartConfigVersion,
        configuration_fingerprint: sha256_hex(&stable_stringify(&Value::Object(fingerprint_input))),
        prompt: options.prompt.to_vec(),
        tools,
        builtin_tools: options.builtin_tools.to_vec(),
        permission_mode: options.permission_mode,
        model_mapping: model.as_ref().map(|_| options.model_mapping.clone()),
        model,
        response_format: options.response_format.cloned(),
        output_schema_mapping: options.output_schema_mapping.cloned(),
        permission_mode_mapping: options.permission_mode_mapping.cloned(),
        debug: options.debug.cloned(),
    })
}

/// `createACPColdSessionState`.
pub fn create_acp_cold_session_state(
    turn_start_config: &ACPTurnStartConfig,
) -> ACPColdSessionState {
    ACPColdSessionState {
        version: turn_start_config.version,
        configuration_fingerprint: turn_start_config.configuration_fingerprint.clone(),
        tools: turn_start_config.tools.clone(),
        builtin_tools: turn_start_config.builtin_tools.clone(),
        permission_mode: turn_start_config.permission_mode,
        response_format: turn_start_config.response_format.clone(),
        output_schema_mapping: turn_start_config.output_schema_mapping.clone(),
        permission_mode_mapping: turn_start_config.permission_mode_mapping.clone(),
    }
}

/// `stableStringify`: JSON with object keys sorted by code unit.
fn stable_stringify(value: &Value) -> String {
    sort_value(value).to_string()
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(sort_value).collect()),
        Value::Object(record) => {
            let mut entries: Vec<(&String, &Value)> = record.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.encode_utf16().cmp(right.encode_utf16()));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, entry)| (key.clone(), sort_value(entry)))
                    .collect(),
            )
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::adapters::acp::acp_auth::ACPProviderKind;

    fn profile() -> ACPAuthenticationProfileIdentity {
        serde_json::from_value(json!({ "digest": "d", "providerKind": "direct" })).unwrap()
    }

    fn options<'a>(
        profile: &'a ACPAuthenticationProfileIdentity,
        mapping: &'a ACPModelMapping,
        model: Option<&'a str>,
    ) -> CreateACPTurnStartConfigOptions<'a> {
        CreateACPTurnStartConfigOptions {
            prompt: &[],
            tools: &[],
            builtin_tools: &[],
            permission_mode: HarnessV1PermissionMode::AllowAll,
            permission_mode_mapping: None,
            mcp_servers: None,
            debug: None,
            authentication_profile: profile,
            session_meta: None,
            instruction_mapping: None,
            response_format: None,
            output_schema_mapping: None,
            model,
            model_mapping: mapping,
        }
    }

    #[test]
    fn fingerprints_only_the_non_secret_configuration() {
        let profile = profile();
        assert_eq!(profile.provider_kind, ACPProviderKind::Direct);
        let mapping = ACPModelMapping::SessionModel { path: "m".into() };
        let first = create_acp_turn_start_config(options(&profile, &mapping, None)).unwrap();
        let with_model =
            create_acp_turn_start_config(options(&profile, &mapping, Some("model"))).unwrap();
        assert_eq!(
            first.configuration_fingerprint,
            with_model.configuration_fingerprint
        );
        assert_eq!(first.model_mapping, None);
        assert_eq!(with_model.model_mapping, Some(mapping.clone()));

        let other = ACPModelMapping::SessionModel { path: "n".into() };
        assert_ne!(
            create_acp_turn_start_config(options(&profile, &other, None))
                .unwrap()
                .configuration_fingerprint,
            first.configuration_fingerprint
        );
        assert_eq!(first.configuration_fingerprint.len(), 64);
    }

    #[test]
    fn rejects_tools_whose_input_schema_is_not_an_object() {
        let profile = profile();
        let mapping = ACPModelMapping::SessionModel { path: "m".into() };
        let tools = [HarnessV1ToolSpec {
            name: "t".into(),
            description: None,
            input_schema: Some(json!([])),
        }];
        let mut input = options(&profile, &mapping, None);
        input.tools = &tools;
        assert!(create_acp_turn_start_config(input).is_err());
    }
}
