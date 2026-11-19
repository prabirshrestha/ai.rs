//! [rust] `createOpencodeServer` and the bridge's `ensureRuntime`: start
//! `opencode serve` in the sandbox and connect to it from this process.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use futures::StreamExt as _;
use indexmap::IndexMap;
use serde_json::{Map, Value};
use tokio::sync::oneshot;

use crate::harness::adapters::host_relay::HostRelay;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::network::{
    HarnessNetworkSandboxSession, HarnessPortEndpoint, HarnessPortProtocol,
};
use crate::harness::sandbox::{
    SandboxProcessControl, SandboxProcessOptions, SandboxRunOptions, SandboxSession,
    SandboxSpawnOptions,
};
use crate::harness::utils::{random_uuid, shell_quote};

use super::client::OpenCodeClient;

const FALLBACK_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
const TAIL_LINES: usize = 40;

/// `OPENCODE_SERVER_USERNAME`'s default.
pub(super) const SERVER_USERNAME: &str = "opencode";

/// How to start the server.
pub(super) struct ServerLaunch<'a> {
    pub sandbox: &'a Arc<dyn SandboxSession>,
    pub network: Option<&'a Arc<dyn HarnessNetworkSandboxSession>>,
    pub work_dir: &'a str,
    /// The `opencode` executable, shell-quoted, or an override command.
    pub command: &'a str,
    /// The bootstrap directory, for `PATH`.
    pub bootstrap_dir: Option<&'a str>,
    pub env: &'a IndexMap<String, String>,
    pub config: &'a Map<String, Value>,
    pub port: Option<u16>,
    pub port_endpoint: Option<&'a HarnessPortEndpoint>,
    pub startup_timeout: Duration,
}

/// A running (or attached) server.
pub(super) struct Server {
    pub client: OpenCodeClient,
    pub url: String,
    pub password: String,
    pub port: Option<u16>,
    pub pid: Option<u32>,
    process: Option<Arc<dyn SandboxProcessControl>>,
    pub relay: Option<HostRelay>,
    pub fingerprint: String,
    /// Tool name prefixes of connected user MCP servers.
    pub mcp_prefixes: Vec<String>,
    tail: Arc<StdMutex<VecDeque<String>>>,
}

/// Where a detached server can be found again.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenCodeServerCoords {
    pub url: String,
    pub password: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    pub fingerprint: String,
}

fn push_tail(tail: &StdMutex<VecDeque<String>>, line: String) {
    let mut tail = tail.lock().unwrap();
    tail.push_back(line);
    while tail.len() > TAIL_LINES {
        tail.pop_front();
    }
}

/// The server URL of a `opencode server listening on <url>` line.
pub(super) fn parse_listening_line(line: &str) -> Option<Result<String, String>> {
    if !line.starts_with("opencode server listening") {
        return None;
    }
    let url = line
        .split_whitespace()
        .skip_while(|word| *word != "on")
        .nth(1)
        .filter(|url| url.starts_with("http://") || url.starts_with("https://"));
    Some(match url {
        Some(url) => Ok(url.to_string()),
        None => Err(format!("Failed to parse server url from output: {line}")),
    })
}

fn random_password() -> String {
    format!("{}{}", random_uuid(), random_uuid()).replace('-', "")
}

impl Server {
    pub fn coords(&self) -> OpenCodeServerCoords {
        OpenCodeServerCoords {
            url: self.url.clone(),
            password: self.password.clone(),
            port: self.port,
            pid: self.pid,
            fingerprint: self.fingerprint.clone(),
        }
    }

    pub fn tail(&self) -> String {
        self.tail
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Reconnect to a detached server.
    pub fn attach(
        coords: &OpenCodeServerCoords,
        work_dir: &str,
        endpoint_headers: Option<&IndexMap<String, String>>,
    ) -> HarnessResult<Self> {
        Ok(Self {
            client: OpenCodeClient::new(
                &coords.url,
                SERVER_USERNAME,
                &coords.password,
                work_dir,
                endpoint_headers,
            )?,
            url: coords.url.clone(),
            password: coords.password.clone(),
            port: coords.port,
            pid: coords.pid,
            process: None,
            relay: None,
            fingerprint: coords.fingerprint.clone(),
            mcp_prefixes: Vec::new(),
            tail: Arc::default(),
        })
    }

    /// `createOpencodeServer`.
    pub async fn start(launch: ServerLaunch<'_>) -> HarnessResult<Self> {
        let (hostname, port) = match (launch.network, launch.port_endpoint) {
            (Some(network), None) => {
                let port = match launch.port.or_else(|| network.ports().first().copied()) {
                    Some(port) => port,
                    None => {
                        return Err(HarnessError::new(
                            "OpenCode requires a sandbox port to serve on; configure the sandbox with a port or set `port`.",
                        )
                        .into());
                    }
                };
                ("0.0.0.0", port)
            }
            (_, Some(_)) => ("0.0.0.0", launch.port.unwrap_or(0)),
            (None, None) => ("127.0.0.1", launch.port.unwrap_or(0)),
        };
        let password = random_password();
        let mut env = launch.env.clone();
        env.insert("OPENCODE_SERVER_PASSWORD".into(), password.clone());
        env.insert("OPENCODE_DISABLE_PROJECT_CONFIG".into(), "1".into());
        env.insert(
            "OPENCODE_CONFIG_CONTENT".into(),
            Value::Object(launch.config.clone()).to_string(),
        );
        let path = match launch.bootstrap_dir {
            Some(dir) => format!(
                "export PATH={}:\"${{PATH:-{FALLBACK_PATH}}}\":{} && ",
                shell_quote(&format!("{dir}/node_modules/.bin")),
                shell_quote(&format!("{dir}/node_modules/@vscode/ripgrep/bin")),
            ),
            None => String::new(),
        };
        let command = format!(
            "cd {} && {path}exec {} serve --hostname={hostname} --port={port}",
            shell_quote(launch.work_dir),
            launch.command,
        );
        let mut process = launch
            .sandbox
            .spawn(SandboxSpawnOptions {
                process: SandboxProcessOptions {
                    command,
                    env: Some(env),
                    ..Default::default()
                },
                stdin: None,
            })
            .await?;
        let control = process.control.clone();
        let pid = process.pid;
        let tail: Arc<StdMutex<VecDeque<String>>> = Arc::default();
        let (url_sender, url_receiver) = oneshot::channel::<Result<String, String>>();
        {
            let mut stdout =
                std::mem::replace(&mut process.stdout, futures::stream::empty().boxed());
            let tail = tail.clone();
            tokio::spawn(async move {
                let mut sender = Some(url_sender);
                let mut buffer = Vec::new();
                while let Some(Ok(chunk)) = stdout.next().await {
                    buffer.extend_from_slice(&chunk);
                    while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                        let line: Vec<u8> = buffer.drain(..=position).collect();
                        let line = String::from_utf8_lossy(&line).trim_end().to_string();
                        if let Some(parsed) = parse_listening_line(&line)
                            && let Some(sender) = sender.take()
                        {
                            let _ = sender.send(parsed);
                        }
                        push_tail(&tail, line);
                    }
                }
            });
        }
        {
            let mut stderr =
                std::mem::replace(&mut process.stderr, futures::stream::empty().boxed());
            let tail = tail.clone();
            tokio::spawn(async move {
                let mut buffer = Vec::new();
                while let Some(Ok(chunk)) = stderr.next().await {
                    buffer.extend_from_slice(&chunk);
                    while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                        let line: Vec<u8> = buffer.drain(..=position).collect();
                        push_tail(&tail, String::from_utf8_lossy(&line).trim_end().to_string());
                    }
                }
                if !buffer.is_empty() {
                    push_tail(&tail, String::from_utf8_lossy(&buffer).into_owned());
                }
            });
        }
        let output = |tail: &Arc<StdMutex<VecDeque<String>>>| {
            let lines: Vec<String> = tail.lock().unwrap().iter().cloned().collect();
            if lines.is_empty() {
                String::new()
            } else {
                format!("\nServer output: {}", lines.join("\n"))
            }
        };
        let waiter = control.clone();
        let printed = tokio::select! {
            url = url_receiver => match url {
                Ok(Ok(url)) => url,
                Ok(Err(message)) => {
                    let _ = control.kill().await;
                    return Err(HarnessError::new(message).into());
                }
                Err(_) => {
                    let status = waiter.wait().await.ok();
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    return Err(HarnessError::new(format!(
                        "OpenCode server exited with code {}{}",
                        status.map_or_else(|| "unknown".to_string(), |s| s.exit_code.to_string()),
                        output(&tail)
                    ))
                    .into());
                }
            },
            _ = tokio::time::sleep(launch.startup_timeout) => {
                let _ = control.kill().await;
                return Err(HarnessError::new(format!(
                    "Timeout waiting for the OpenCode server to start after {}ms{}",
                    launch.startup_timeout.as_millis(),
                    output(&tail)
                ))
                .into());
            }
        };
        let endpoint = match (launch.port_endpoint, launch.network) {
            (Some(endpoint), _) => Some(endpoint.clone()),
            (None, Some(network)) => {
                match network
                    .get_port_endpoint(port, Some(HarnessPortProtocol::Http))
                    .await
                {
                    Ok(endpoint) => Some(endpoint),
                    Err(error) => {
                        let _ = control.kill().await;
                        return Err(error);
                    }
                }
            }
            (None, None) => None,
        };
        let (url, headers) = match endpoint {
            Some(endpoint) => (endpoint.url, endpoint.headers),
            None => (printed, None),
        };
        let client = OpenCodeClient::new(
            &url,
            SERVER_USERNAME,
            &password,
            launch.work_dir,
            headers.as_ref(),
        )?;
        Ok(Self {
            client,
            url,
            password,
            port: (port != 0).then_some(port),
            pid,
            process: Some(control),
            relay: None,
            fingerprint: String::new(),
            mcp_prefixes: Vec::new(),
            tail,
        })
    }

    /// `closeRuntime`.
    pub async fn close(self, sandbox: &dyn SandboxSession) {
        if let Some(relay) = &self.relay {
            relay.close().await;
        }
        match &self.process {
            Some(process) => {
                let _ = process.kill().await;
            }
            None => kill_detached(sandbox, self.pid).await,
        }
    }
}

/// Stop a server another process started.
pub(super) async fn kill_detached(sandbox: &dyn SandboxSession, pid: Option<u32>) {
    let Some(pid) = pid else {
        return;
    };
    let _ = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions::new(format!("kill {pid} 2>/dev/null || true")),
            max_output_length: None,
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_listening_line() {
        assert_eq!(
            parse_listening_line("opencode server listening on http://127.0.0.1:4096"),
            Some(Ok("http://127.0.0.1:4096".into()))
        );
        assert!(matches!(
            parse_listening_line("opencode server listening somewhere"),
            Some(Err(_))
        ));
        assert_eq!(parse_listening_line("starting"), None);
    }
}
