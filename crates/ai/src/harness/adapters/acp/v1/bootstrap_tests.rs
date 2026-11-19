//! Port of `harness-acp/src/v1/acp-v1-bootstrap.test.ts`.
//!
//! [rust] The TS file mocks `readFile` to observe asset reads; the assets
//! here are injected instead, so the "does not read the bridge manifest"
//! assertions become assertions on the produced files.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use indexmap::IndexMap;

use super::*;
use crate::harness::adapters::acp::v1::settings::{ACPInstallCommandSource, ACPNpmLockedSource};

const MOCK_ASSETS: ACPBridgeAssets = ACPBridgeAssets {
    package_json: "{\"name\":\"harness-acp-bridge\"}\n",
    pnpm_lock_yaml: "lockfileVersion: '9.0'\n",
    index_mjs: "// mock bridge\n",
    host_tool_mcp_mjs: "// mock host tool MCP\n",
};

fn locked_source() -> ACPSource {
    ACPSource::NpmLocked(ACPNpmLockedSource {
        package_json: "{\"name\":\"locked-acp-agent\"}\n".into(),
        pnpm_lock_yaml: "lockfileVersion: '9.0'\n# locked\n".into(),
        pnpm_workspace_yaml: Some("allowBuilds:\n  '@example/acp-agent': true\n".into()),
    })
}

fn install_command(command: &str) -> ACPSource {
    ACPSource::InstallCommand(ACPInstallCommandSource {
        command: command.into(),
    })
}

fn get_install_script(command: &str) -> String {
    create_acp_bootstrap_with_assets("example-acp", &install_command(command), MOCK_ASSETS)
        .get_bootstrap()
        .files
        .iter()
        .find(|file| file.path.ends_with("/install.sh"))
        .unwrap()
        .content
        .clone()
}

fn temporary_directory() -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "ai-sdk-harness-acp-install-{}",
        crate::harness::utils::generate_sandbox_credential_placeholder()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

#[test]
fn installs_a_locked_source_as_the_complete_bridge_installation() {
    let bootstrap = create_acp_bootstrap_with_assets("example-acp", &locked_source(), MOCK_ASSETS)
        .get_bootstrap();
    assert_eq!(bootstrap.bootstrap_dir, ".harness-bootstrap/example-acp");
    let files: IndexMap<&str, &str> = bootstrap
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.content.as_str()))
        .collect();
    assert_eq!(
        files,
        IndexMap::from([
            (
                ".harness-bootstrap/example-acp/bridge.mjs",
                "// mock bridge\n"
            ),
            (
                ".harness-bootstrap/example-acp/host-tool-mcp.mjs",
                "// mock host tool MCP\n"
            ),
            (
                ".harness-bootstrap/example-acp/package.json",
                "{\"name\":\"locked-acp-agent\"}\n"
            ),
            (
                ".harness-bootstrap/example-acp/pnpm-lock.yaml",
                "lockfileVersion: '9.0'\n# locked\n"
            ),
            (
                ".harness-bootstrap/example-acp/pnpm-workspace.yaml",
                "allowBuilds:\n  '@example/acp-agent': true\n"
            ),
        ])
    );
    assert_eq!(
        bootstrap.commands,
        vec![HarnessV1BootstrapCommand {
            command: "pnpm install --frozen-lockfile --prod --store-dir .pnpm-store".into()
        }]
    );
}

#[test]
fn installs_the_bridge_manifest_before_running_an_install_command() {
    let bootstrap = create_acp_bootstrap_with_assets(
        "cursor-acp",
        &install_command("curl https://cursor.com/install -fsS | bash"),
        MOCK_ASSETS,
    )
    .get_bootstrap();
    let mut paths: Vec<&str> = bootstrap
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    paths.sort();
    assert_eq!(
        paths,
        vec![
            ".harness-bootstrap/cursor-acp/bridge.mjs",
            ".harness-bootstrap/cursor-acp/host-tool-mcp.mjs",
            ".harness-bootstrap/cursor-acp/install.sh",
            ".harness-bootstrap/cursor-acp/package.json",
            ".harness-bootstrap/cursor-acp/pnpm-lock.yaml",
        ]
    );
    let content = |suffix: &str| {
        bootstrap
            .files
            .iter()
            .find(|file| file.path.ends_with(suffix))
            .map(|file| file.content.clone())
            .unwrap()
    };
    assert_eq!(
        content("/package.json"),
        "{\"name\":\"harness-acp-bridge\"}\n"
    );
    assert_eq!(content("/pnpm-lock.yaml"), "lockfileVersion: '9.0'\n");
    assert!(content("/install.sh").contains("curl https://cursor.com/install -fsS | bash"));
    assert_eq!(
        bootstrap.commands,
        vec![
            HarnessV1BootstrapCommand {
                command: "pnpm install --frozen-lockfile --prod --store-dir .pnpm-store".into()
            },
            HarnessV1BootstrapCommand {
                command: "bash install.sh".into()
            },
        ]
    );
}

#[test]
fn caches_the_bootstrap() {
    let bootstrap = create_acp_bootstrap("example-acp", &locked_source());
    assert!(Arc::ptr_eq(
        &bootstrap.get_bootstrap(),
        &bootstrap.get_bootstrap()
    ));
}

#[test]
fn embeds_the_vendored_bridge_bundle() {
    let bootstrap = create_acp_bootstrap("example-acp", &install_command("true")).get_bootstrap();
    let bridge = &bootstrap.files[0];
    assert_eq!(bridge.path, ".harness-bootstrap/example-acp/bridge.mjs");
    assert_eq!(bridge.content, ACP_BRIDGE_ASSETS.index_mjs);
    assert!(
        ACP_BRIDGE_ASSETS
            .package_json
            .contains("harness-acp-bridge")
    );
}

#[test]
fn installs_into_a_private_home_below_the_bootstrap_directory() {
    let temporary = temporary_directory();
    let bootstrap_directory = temporary.join(".harness-bootstrap").join("example-acp");
    let ambient_home = temporary.join("ambient-home");
    let script_path = bootstrap_directory.join("install.sh");
    std::fs::create_dir_all(&bootstrap_directory).unwrap();
    std::fs::create_dir_all(&ambient_home).unwrap();
    std::fs::write(
        &script_path,
        get_install_script(
            &[
                "mkdir -p \"$HOME/.local/share/acp-agent\"",
                "printf '#!/usr/bin/env bash\\nexit 0\\n' > \"$HOME/.local/share/acp-agent/acp-agent\"",
                "chmod +x \"$HOME/.local/share/acp-agent/acp-agent\"",
                "ln -s \"$HOME/.local/share/acp-agent/acp-agent\" \"$HOME/.local/bin/acp-agent\"",
                "pwd > \"$HOME/install-working-directory.txt\"",
            ]
            .join("\n"),
        ),
    )
    .unwrap();

    let status = Command::new("bash")
        .arg(&script_path)
        .env("HOME", &ambient_home)
        .status()
        .unwrap();
    assert!(status.success());

    let installation_home = bootstrap_directory.join("home");
    assert_eq!(
        std::fs::canonicalize(installation_home.join(".local/bin/acp-agent")).unwrap(),
        std::fs::canonicalize(installation_home.join(".local/share/acp-agent/acp-agent")).unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(installation_home.join("install-working-directory.txt")).unwrap(),
        format!("{}\n", installation_home.display())
    );
    assert!(!ambient_home.join(".local/bin/acp-agent").exists());
    std::fs::remove_dir_all(&temporary).unwrap();
}

#[test]
fn fails_when_the_download_side_of_an_installer_pipeline_fails() {
    let temporary = temporary_directory();
    let script_path = temporary.join("install.sh");
    std::fs::write(&script_path, get_install_script("false | true")).unwrap();
    let status = Command::new("bash").arg(&script_path).status().unwrap();
    assert_eq!(status.code(), Some(1));
    std::fs::remove_dir_all(&temporary).unwrap();
}

#[test]
fn launches_a_locked_agent_from_the_bridge_installation() {
    assert_eq!(
        resolve_acp_agent_launch(
            "/home/agent/.harness-bootstrap/example-acp",
            &locked_source(),
            "acp-agent",
            Some(&["stdio".to_string()]),
        ),
        ACPAgentLaunch {
            command: "/home/agent/.harness-bootstrap/example-acp/node_modules/.bin/acp-agent"
                .into(),
            args: vec!["stdio".into()],
            home: None,
        }
    );
}

#[test]
fn launches_an_installed_agent_with_its_private_home() {
    assert_eq!(
        resolve_acp_agent_launch(
            "/home/agent/.harness-bootstrap/cursor-acp",
            &install_command("install"),
            "agent",
            None,
        ),
        ACPAgentLaunch {
            command: "/home/agent/.harness-bootstrap/cursor-acp/home/.local/bin/agent".into(),
            args: vec![],
            home: Some("/home/agent/.harness-bootstrap/cursor-acp/home".into()),
        }
    );
}
