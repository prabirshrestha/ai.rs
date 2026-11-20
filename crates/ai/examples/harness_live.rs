//! Probe a harness CLI installed and signed in on this machine, then run one
//! short turn on it.
//!
//! ```sh
//! cargo run -p ai --example harness_live -- claude
//! cargo run -p ai --example harness_live -- codex --shared
//! cargo run -p ai --example harness_live -- copilot
//! cargo run -p ai --example harness_live -- cursor
//! cargo run -p ai --example harness_live -- acp opencode acp
//! cargo run -p ai --example harness_live -- acp fx acp
//! ```
//!
//! `--no-turn` only probes; `--prompt <text>` changes the turn's prompt.

use std::sync::Arc;

use ai::harness::adapters::acp::{
    AcpClientApp, AcpHarness, AcpHarnessSettings, AcpModelMapping, AcpSource,
};
use ai::harness::adapters::claude_code::{ClaudeCodeHarness, ClaudeCodeHarnessSettings};
use ai::harness::adapters::codex::{CodexHarness, CodexHarnessSettings};
use ai::harness::adapters::cursor::{CursorHarnessSettings, create_cursor};
use ai::harness::adapters::github_copilot::{GitHubCopilotHarnessSettings, create_github_copilot};
use ai::harness::{
    Harness, HarnessSandbox, HarnessSession, HarnessStartOptions, HarnessTurnOptions, HostSandbox,
    PermissionMode,
};
use ai::{AssistantContent, Message};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn harness(name: &str, rest: &[String], shared: bool) -> Result<Box<dyn Harness>, BoxError> {
    Ok(match name {
        "claude" => Box::new(ClaudeCodeHarness::new(ClaudeCodeHarnessSettings {
            executable: Some("claude".into()),
            ..Default::default()
        })),
        "codex" => Box::new(CodexHarness::new(CodexHarnessSettings {
            executable: Some("codex".into()),
            shared_app_server: shared,
            ..Default::default()
        })),
        "copilot" => Box::new(create_github_copilot(GitHubCopilotHarnessSettings {
            executable_override: Some("copilot".into()),
            ..Default::default()
        })?),
        "cursor" => Box::new(create_cursor(CursorHarnessSettings {
            executable_override: Some("agent".into()),
            ..Default::default()
        })?),
        "acp" => {
            let (program, args) = rest.split_first().ok_or("acp needs a command")?;
            let mut settings = AcpHarnessSettings::new(
                format!("acp-{}", program.rsplit('/').next().unwrap_or(program)),
                AcpClientApp {
                    name: "harness-live".into(),
                    version: "0".into(),
                },
                AcpSource::InstallCommand {
                    command: "true".into(),
                },
                program.clone(),
                AcpModelMapping::SessionConfigOption {
                    path: "model".into(),
                },
                Arc::new(|_| None),
            );
            settings.args = args.to_vec();
            settings.executable_override = Some(program.clone());
            Box::new(AcpHarness::new(settings)?)
        }
        other => return Err(format!("unknown harness {other}").into()),
    })
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(message) => Some(
                message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

async fn turn(session: &dyn HarnessSession, prompt: &str) -> Result<(), BoxError> {
    let started = std::time::Instant::now();
    let turn = session
        .prompt(Message::user_text(prompt), HarnessTurnOptions::default())
        .await?;
    let (events, messages) = turn.collect().await?;
    println!(
        "turn on {} in {:.1}s: {} events, reply {:?}",
        session.session_id(),
        started.elapsed().as_secs_f32(),
        events.len(),
        texts(&messages)
    );
    for message in &messages {
        if let Message::Assistant(message) = message
            && let Some(error) = &message.error_message
        {
            println!("  error: {error}");
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |args: &mut Vec<String>, name: &str| {
        let found = args.iter().position(|arg| arg == name);
        found.map(|at| args.remove(at)).is_some()
    };
    let shared = flag(&mut args, "--shared");
    let no_turn = flag(&mut args, "--no-turn");
    let prompt = match args.iter().position(|arg| arg == "--prompt") {
        Some(at) => {
            args.remove(at);
            args.remove(at)
        }
        None => "Reply with the single word: pong".into(),
    };
    let name = args.first().cloned().ok_or("name a harness")?;
    let harness = harness(&name, &args[1..], shared)?;

    let work_dir = std::env::temp_dir().join(format!("harness-live-{}", std::process::id()));
    std::fs::create_dir_all(&work_dir)?;
    let work_dir = work_dir.to_string_lossy().into_owned();
    let sandbox = HarnessSandbox::Basic(Arc::new(HostSandbox::new(work_dir.clone())));
    let mut options = HarnessStartOptions::new("live-1", sandbox, work_dir.clone());
    options.permission_mode = PermissionMode::AllowAll;

    let session = harness.start(options.clone()).await?;
    let started = std::time::Instant::now();
    match session.probe().await {
        Ok(probe) => {
            println!(
                "probe in {:.1}s: signed in {:?}, account {:?}, error {:?}",
                started.elapsed().as_secs_f32(),
                probe.signed_in,
                probe.account,
                probe.error
            );
            for model in &probe.models {
                println!(
                    "  model {} ({}){} {:?}",
                    model.id,
                    model.name,
                    if model.is_default { " default" } else { "" },
                    model.thinking_levels
                );
            }
            let commands: Vec<_> = probe.commands.iter().map(|c| c.name.as_str()).collect();
            println!("  commands {commands:?}");
        }
        Err(error) => println!("probe failed: {error}"),
    }

    if !no_turn {
        turn(session.as_ref(), &prompt).await?;
        if shared {
            options.session_id = "live-2".into();
            let second = harness.start(options).await?;
            turn(second.as_ref(), &prompt).await?;
            turn(session.as_ref(), "Now reply with the single word: ping").await?;
            second.destroy().await?;
        }
    }
    session.destroy().await?;
    let _ = std::fs::remove_dir_all(&work_dir);
    Ok(())
}
