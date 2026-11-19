//! Port of `harness-acp/src/v1/acp-v1-settings.test.ts`.
//!
//! [rust] The "callbacks must be functions" case is enforced by the type
//! system and is not ported.

use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::json;

use super::*;

fn map(entries: &[(&str, &str)]) -> IndexMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn strings(values: &[&str]) -> Option<Vec<String>> {
    Some(values.iter().map(|v| v.to_string()).collect())
}

fn locked_source() -> ACPNpmLockedSource {
    ACPNpmLockedSource {
        package_json: "{\"dependencies\":{\"@example/acp-agent\":\"1.2.3\"}}\n".into(),
        pnpm_lock_yaml: "lockfileVersion: '9.0'\n".into(),
        pnpm_workspace_yaml: None,
    }
}

fn base_settings() -> ACPV1Settings {
    let mut settings = ACPV1Settings::new(
        "example-acp",
        ACPSource::NpmLocked(locked_source()),
        "acp-agent",
        ACPModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        Arc::new(|_| Ok(false)),
        Arc::new(|_| Ok(None)),
    );
    settings.forward_env = strings(&["PROVIDER_API_KEY", "SECOND_PROVIDER_API_KEY"]);
    settings.env = Some(map(&[("PROVIDER_BASE_URL", "https://provider.example")]));
    settings
}

fn brokering() -> Option<ACPCredentialBrokering> {
    Some(Arc::new(|_| Ok(Vec::new())))
}

fn validate(configure: impl FnOnce(&mut ACPV1Settings)) -> Result<(), String> {
    let mut settings = base_settings();
    configure(&mut settings);
    validate_acp_v1_settings(&settings).map_err(|error| error.to_string())
}

#[track_caller]
fn assert_throws(result: Result<(), String>, message: &str) {
    let error = result.expect_err("expected validation to fail");
    assert!(
        error.contains(message),
        "{error:?} does not contain {message:?}"
    );
}

#[test]
fn accepts_locked_and_install_command_sources() {
    validate(|_| {}).unwrap();
    validate(|s| {
        s.source = ACPSource::InstallCommand(ACPInstallCommandSource {
            command: "curl https://example.com/install -fsS | bash".into(),
        })
    })
    .unwrap();
}

#[test]
fn requires_credential_environment_and_brokering_together() {
    assert_throws(
        validate(|s| s.credential_env = strings(&["THIRD_API_KEY"])),
        "credentialEnv and credentialBrokering must be configured together",
    );
    assert_throws(
        validate(|s| s.credential_brokering = brokering()),
        "credentialEnv and credentialBrokering must be configured together",
    );
}

#[test]
fn reserves_the_harness_tools_mcp_server_name() {
    assert_throws(
        validate(|s| {
            s.mcp_servers = Some(IndexMap::from([("ai-sdk-harness-tools".into(), json!({}))]))
        }),
        "is reserved for HarnessAgent tools",
    );
}

#[test]
fn requires_a_kebab_case_harness_id() {
    assert_throws(
        validate(|s| s.harness_id = "Example ACP".into()),
        "stable kebab-case identifier",
    );
}

#[test]
fn rejects_empty_locked_artifacts_and_install_commands() {
    assert_throws(
        validate(|s| {
            s.source = ACPSource::NpmLocked(ACPNpmLockedSource {
                package_json: String::new(),
                ..locked_source()
            })
        }),
        "source.packageJson must not be empty",
    );
    assert_throws(
        validate(|s| {
            s.source = ACPSource::NpmLocked(ACPNpmLockedSource {
                pnpm_lock_yaml: String::new(),
                ..locked_source()
            })
        }),
        "source.pnpmLockYaml must not be empty",
    );
    assert_throws(
        validate(|s| {
            s.source = ACPSource::NpmLocked(ACPNpmLockedSource {
                pnpm_workspace_yaml: Some(String::new()),
                ..locked_source()
            })
        }),
        "source.pnpmWorkspaceYaml must not be empty",
    );
    assert_throws(
        validate(|s| {
            s.source = ACPSource::InstallCommand(ACPInstallCommandSource {
                command: "  \n".into(),
            })
        }),
        "source.command must not be empty",
    );
}

#[test]
fn requires_a_bare_executable_name() {
    assert_throws(
        validate(|s| s.executable = "bin/acp-agent".into()),
        "bare command name without a path",
    );
}

#[test]
fn rejects_invalid_environment_variable_names_and_values() {
    assert_throws(
        validate(|s| s.forward_env = strings(&["not-an-environment-variable"])),
        "environment variable name is invalid",
    );
    assert_throws(
        validate(|s| {
            s.credential_env = strings(&["not-an-environment-variable"]);
            s.credential_brokering = brokering();
        }),
        "environment variable name is invalid",
    );
    assert_throws(
        validate(|s| s.env = Some(map(&[("not-an-environment-variable", "value")]))),
        "environment variable name is invalid",
    );
    assert_throws(
        validate(|s| s.env = Some(map(&[("PROVIDER_BASE_URL", "a\0b")]))),
        "contains NUL",
    );
}

#[test]
fn requires_every_environment_key_to_be_configured_once() {
    assert_throws(
        validate(|s| s.env = Some(map(&[("PROVIDER_API_KEY", "not-secret")]))),
        "cannot be configured in both forwardEnv and env",
    );
    assert_throws(
        validate(|s| {
            s.credential_env = strings(&["PROVIDER_API_KEY"]);
            s.credential_brokering = brokering();
        }),
        "cannot be configured in both forwardEnv and credentialEnv",
    );
    assert_throws(
        validate(|s| {
            s.credential_env = strings(&["PROVIDER_BASE_URL"]);
            s.credential_brokering = brokering();
        }),
        "cannot be configured in both credentialEnv and env",
    );
}

#[test]
fn accepts_relative_write_protected_paths() {
    validate(|s| s.write_protected_paths = strings(&[".github/hooks", ".mcp.json", ".cursor"]))
        .unwrap();
}

#[test]
fn rejects_write_protected_paths() {
    for entry in [
        "", " ", "/abs", "../x", "a/../b", "./a", "a//b", "a/", "a\0b",
    ] {
        assert_throws(
            validate(|s| s.write_protected_paths = strings(&[entry])),
            "ACP writeProtectedPaths entries must be relative paths inside the working directory",
        );
    }
}

#[test]
fn forwards_only_configured_non_empty_values_and_literal_values() {
    assert_eq!(
        resolve_acp_agent_environment(
            &base_settings(),
            &map(&[
                ("PROVIDER_API_KEY", "provider-secret"),
                ("SECOND_PROVIDER_API_KEY", ""),
                ("UNRELATED_SECRET", "unrelated-secret"),
            ]),
            &IndexMap::new(),
        ),
        map(&[
            ("PROVIDER_API_KEY", "provider-secret"),
            ("PROVIDER_BASE_URL", "https://provider.example"),
        ])
    );
}

#[test]
fn reads_credential_environment_keys_from_the_credential_environment() {
    let mut settings = base_settings();
    settings.forward_env = strings(&["SECOND_PROVIDER_API_KEY"]);
    settings.credential_env = strings(&["PROVIDER_API_KEY"]);
    assert_eq!(
        resolve_acp_agent_environment(
            &settings,
            &map(&[
                ("PROVIDER_API_KEY", "ambient-provider-secret"),
                ("SECOND_PROVIDER_API_KEY", "forwarded-value"),
            ]),
            &map(&[("PROVIDER_API_KEY", "programmatic-provider-secret")]),
        ),
        map(&[
            ("SECOND_PROVIDER_API_KEY", "forwarded-value"),
            ("PROVIDER_API_KEY", "programmatic-provider-secret"),
            ("PROVIDER_BASE_URL", "https://provider.example"),
        ])
    );
}
