mod args;
mod command;
mod provider;
mod render;
mod tools;

use std::env;
use std::io::{self, IsTerminal, Read, Write};
use std::path::Path;

use ai::{
    Agent, AgentOptions, AgentResult, OAuthLoginCallbacks, Result, providers::github_copilot,
};
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};

use args::{Args, ProviderKind};
use command::{ReplCommand, parse_line};
use provider::{
    ActiveProvider, ProviderSelection, build_provider, detect_provider, model_id_from_env,
    selection_from_env, setup_error,
};
use render::{help_text, subscribe};
use tools::build_bash_tool;

struct Session {
    agent: Agent,
    provider: ActiveProvider,
    setup_error: Option<String>,
    base_url: Option<String>,
    cli_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut session = Session::new(&args)?;
    let _subscription = subscribe(&session.agent);

    match resolve_startup_prompt(&args)? {
        Some(prompt) => session.one_shot(&prompt).await,
        None => session.repl().await,
    }
}

impl Session {
    fn new(args: &Args) -> Result<Self> {
        let base_url = args
            .base_url
            .clone()
            .or_else(|| nonempty_env("OPENAI_BASE_URL"));
        let selection = selection_from_env(args.provider, base_url.clone());
        let kind = detect_provider(&selection);
        let provider = build_provider(kind, selection.openai_base_url.as_deref())?;
        let model_id = model_id_from_env(args.model.as_deref(), kind);
        let model = provider.model(&model_id)?;
        let setup_error = setup_error(kind, &selection);
        let cwd = env::current_dir()?;

        let agent = Agent::new(
            AgentOptions::builder(model)
                .system_prompt(system_prompt(&cwd))
                .tool(build_bash_tool()?)
                .build(),
        );

        Ok(Self {
            agent,
            provider,
            setup_error,
            base_url,
            cli_model: args.model.clone(),
        })
    }

    async fn one_shot(&self, prompt: &str) -> Result<()> {
        if let Some(error) = &self.setup_error {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
        if let Err(error) = run_prompt(&self.agent, prompt).await {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
        Ok(())
    }

    async fn repl(&mut self) -> Result<()> {
        self.print_banner().await;

        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();
        loop {
            print!("zs> ");
            let _ = io::stdout().flush();

            let Some(line) = lines.next_line().await? else {
                break;
            };
            match parse_line(&line) {
                None => continue,
                Some(ReplCommand::Exit) => break,
                Some(ReplCommand::Help) => println!("{}", help_text()),
                Some(ReplCommand::Clear) => {
                    self.agent.reset().await;
                    println!("context cleared");
                }
                Some(ReplCommand::Model { id }) => self.handle_model(id).await,
                Some(ReplCommand::Provider { name }) => self.handle_provider(name).await,
                Some(ReplCommand::Login { enterprise_domain }) => {
                    self.handle_login(enterprise_domain).await;
                }
                Some(ReplCommand::Unknown { name }) => {
                    if name.is_empty() {
                        eprintln!("unknown command\ntype /help for commands");
                    } else {
                        eprintln!("unknown command: /{name}\ntype /help for commands");
                    }
                }
                Some(ReplCommand::Prompt(prompt)) => {
                    if let Some(error) = &self.setup_error {
                        eprintln!("error: {error}");
                        continue;
                    }
                    if let Err(error) = run_prompt(&self.agent, &prompt).await {
                        eprintln!("error: {error}");
                    }
                    println!();
                }
            }
        }

        println!();
        Ok(())
    }

    async fn print_banner(&self) {
        let model = self.agent.state().await.model;
        println!("zs — small agent CLI powered by ai.rs");
        println!(
            "provider: {}    model: {}",
            self.provider.kind().as_str(),
            model.id
        );
        println!(
            "base URL: {}",
            self.base_url.as_deref().unwrap_or("provider default")
        );
        if let Some(error) = &self.setup_error {
            println!("{error}");
        }
        println!("type a prompt, or /help");
    }

    async fn handle_model(&self, id: Option<String>) {
        match id {
            None => {
                let model = self.agent.state().await.model;
                println!("model: {} ({})", model.id, model.provider);
            }
            Some(model_id) => match self.provider.model(&model_id) {
                Ok(model) => {
                    self.agent.set_model(model).await;
                    println!("switched to {model_id} ({})", self.provider.kind().as_str());
                }
                Err(error) => eprintln!("error: {error}"),
            },
        }
    }

    async fn handle_provider(&mut self, name: Option<String>) {
        match name {
            None => {
                let model = self.agent.state().await.model;
                println!(
                    "provider: {}    model: {}",
                    self.provider.kind().as_str(),
                    model.id
                );
            }
            Some(name) => {
                let Some(kind) = ProviderKind::parse_name(&name) else {
                    eprintln!("unknown provider: {name}");
                    eprintln!("use openai, anthropic, or copilot");
                    return;
                };
                if let Err(error) = self.switch_provider(kind).await {
                    eprintln!("error: {error}");
                }
            }
        }
    }

    async fn switch_provider(&mut self, kind: ProviderKind) -> Result<()> {
        let selection = self.selection_for(kind);
        let provider = build_provider(kind, selection.openai_base_url.as_deref())?;
        let model_id = model_id_from_env(self.cli_model.as_deref(), kind);
        let model = provider.model(&model_id)?;
        self.agent.set_model(model).await;
        self.provider = provider;
        self.setup_error = setup_error(kind, &selection);
        println!("switched to {} / {model_id}", kind.as_str());
        if let Some(error) = &self.setup_error {
            println!("{error}");
        }
        Ok(())
    }

    async fn handle_login(&mut self, enterprise_domain: Option<String>) {
        match login_github_copilot(enterprise_domain).await {
            Ok((model_id, provider)) => match provider.model(&model_id) {
                Ok(model) => {
                    self.agent.set_model(model).await;
                    self.provider = provider;
                    self.setup_error = None;
                    println!("logged into GitHub Copilot; switched to {model_id}");
                }
                Err(error) => eprintln!("error: {error}"),
            },
            Err(error) => eprintln!("error: {error}"),
        }
    }

    fn selection_for(&self, requested: ProviderKind) -> ProviderSelection {
        let mut selection = selection_from_env(Some(requested), self.base_url.clone());
        if matches!(self.provider, ActiveProvider::GitHubCopilot(_)) {
            selection.copilot_configured = true;
        }
        selection
    }
}

async fn run_prompt(agent: &Agent, prompt: &str) -> AgentResult<()> {
    tokio::select! {
        result = agent.prompt_text(prompt, Vec::new()) => result,
        _ = tokio::signal::ctrl_c() => {
            agent.abort().await;
            eprintln!("\naborted");
            Ok(())
        }
    }
}

async fn login_github_copilot(
    enterprise_domain: Option<String>,
) -> Result<(String, ActiveProvider)> {
    let callbacks = OAuthLoginCallbacks::builder()
        .on_prompt(move |_| {
            let enterprise_domain = enterprise_domain.clone().unwrap_or_default();
            async move { Ok(enterprise_domain) }
        })
        .on_device_code(|info| {
            println!(
                "Open {} and enter code {}",
                info.verification_uri, info.user_code
            );
            if let Some(expires_in_seconds) = info.expires_in_seconds {
                println!("code expires in {expires_in_seconds} seconds");
            }
        })
        .on_progress(|message| println!("{message}"))
        .build();

    let credentials = github_copilot::oauth().login(callbacks).await?;
    let model_id = model_id_from_env(None, ProviderKind::Copilot);
    let base_url = github_copilot::base_url_for_credentials(&credentials);
    let copilot = github_copilot::builder()
        .api_key(credentials.access)
        .base_url(base_url)
        .build()?;

    Ok((model_id, ActiveProvider::GitHubCopilot(copilot)))
}

fn resolve_startup_prompt(args: &Args) -> Result<Option<String>> {
    if let Some(prompt) = args.prompt_text() {
        return Ok(Some(prompt));
    }
    if io::stdin().is_terminal() {
        return Ok(None);
    }

    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    let prompt = buf.trim();
    if prompt.is_empty() {
        Ok(None)
    } else {
        Ok(Some(prompt.to_string()))
    }
}

fn system_prompt(cwd: &Path) -> String {
    format!(
        r#"You are zs, a small coding-agent CLI powered by ai.rs.

Available tools:
- bash: Run shell commands in the current working directory

Guidelines:
- Use bash for file operations like ls, rg, find, and reading or writing files
- Be concise in your responses
- Show file paths clearly when working with files
- Do not run destructive commands unless the user explicitly asks.

Current working directory: {}"#,
        cwd.display()
    )
}

fn nonempty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
