use std::env;
use std::io::Write;

use ai::{
    Agent, AgentError, AgentEvent, AgentOptions, AgentToolBuilder, AgentToolResult,
    AssistantContent, AssistantMessage, AssistantMessageEvent, DynAgentTool, Message, Model,
    OAuthLoginCallbacks, Result,
    providers::{github_copilot, openai},
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

#[tokio::main]
async fn main() -> Result<()> {
    let (agent, mut active_provider, mut provider_setup_error) = build_agent()?;

    let _subscription = agent.subscribe(|event, _token| async move {
        match event {
            AgentEvent::MessageUpdate {
                assistant_message_event:
                    AssistantMessageEvent::TextDelta { delta, .. }
                    | AssistantMessageEvent::ThinkingDelta { delta, .. },
                ..
            } => {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            AgentEvent::MessageUpdate {
                assistant_message_event: AssistantMessageEvent::Error { error, .. },
                ..
            } => {
                eprintln!(
                    "\nerror: {}",
                    error
                        .error_message
                        .as_deref()
                        .unwrap_or("assistant stream failed")
                );
            }
            AgentEvent::MessageEnd { message } => {
                if let Message::Assistant(message) = message {
                    if let Some(error) = &message.error_message {
                        eprintln!(
                            "\nerror from {} ({}, {}): {error}",
                            message.model, message.provider, message.api
                        );
                    } else if assistant_visible_content(&message).trim().is_empty() {
                        eprintln!(
                            "\nwarning: empty response from {} ({}, {})",
                            message.model, message.provider, message.api
                        );
                    }
                }
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } => {
                println!("\n\n{tool_name}({args})");
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                is_error,
                ..
            } => {
                println!("{tool_name} {}", if is_error { "error" } else { "done" });
            }
            _ => {}
        }
        Ok(())
    });

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    loop {
        print!("\n> ");
        let _ = std::io::stdout().flush();

        let Some(line) = lines.next_line().await? else {
            break;
        };
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        let slash_command = prompt
            .strip_prefix('/')
            .map(|command| command.trim_start_matches('/'));

        match slash_command {
            Some("exit" | "quit") => break,
            Some("clear") => {
                agent.reset().await;
                println!("context cleared");
            }
            Some(command) if command == "model" || command.starts_with("model ") => {
                let model_id = command.strip_prefix("model").map(str::trim).unwrap_or("");
                if model_id.is_empty() {
                    let model = agent.state().await.model;
                    println!("model: {} ({})", model.id, model.provider);
                } else {
                    match switch_model(&agent, &active_provider, model_id).await {
                        Ok(provider) => println!("switched to {model_id} ({provider})"),
                        Err(error) => eprintln!("\nerror: {error}"),
                    }
                }
            }
            Some(command) if command == "login" || command.starts_with("login ") => {
                let enterprise_domain = command
                    .strip_prefix("login")
                    .map(str::trim)
                    .filter(|domain| !domain.is_empty())
                    .map(str::to_string);
                match login_github_copilot_and_swap(&agent, enterprise_domain).await {
                    Ok((model_id, provider)) => {
                        active_provider = provider;
                        provider_setup_error = None;
                        println!("logged into GitHub Copilot; switched to {model_id}");
                    }
                    Err(error) => eprintln!("\nerror: {error}"),
                }
            }
            _ => {
                if let Some(error) = &provider_setup_error {
                    eprintln!("\nerror: {error}");
                    println!();
                    continue;
                }
                if let Err(error) = agent.prompt_text(prompt, Vec::new()).await {
                    eprintln!("\nerror: {error}");
                }
                println!();
            }
        }
    }

    println!();
    Ok(())
}

fn build_agent() -> Result<(Agent, ActiveProvider, Option<String>)> {
    let cwd = env::current_dir()?;
    let base_url = env::var("OPENAI_BASE_URL").ok();
    let api_key = openai_api_key();
    let provider_setup_error = openai_setup_error(base_url.as_deref(), api_key.as_deref());
    let model_id = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.5".to_string());
    let provider = ActiveProvider::OpenAi(build_openai_provider(
        base_url.as_deref(),
        api_key.as_deref(),
    )?);
    let model = provider.model(&model_id)?;

    let system_prompt = format!(
        r#"You are an expert coding assistant operating inside simple-coding-agent, a minimal coding agent harness.

Available tools:
- bash: Run shell commands in the current working directory

Guidelines:
- Use bash for file operations like ls, rg, find
- Be concise in your responses
- Show file paths clearly when working with files
- Do not run destructive commands unless the user explicitly asks.

Current working directory: {}"#,
        cwd.display()
    );

    let agent = Agent::new(
        AgentOptions::builder(model)
            .system_prompt(system_prompt)
            .tool(build_bash_tool()?)
            .build(),
    );

    println!("simple coding agent example powered by ai.rs");
    println!("model: {model_id}");
    println!(
        "base URL: {}",
        base_url.as_deref().unwrap_or("OpenAI default")
    );
    if let Some(error) = &provider_setup_error {
        println!("{error}");
    }
    println!(
        "type a prompt, /model [name] to view or switch models on the active provider, /login to use Copilot, /clear to reset context, or /exit to quit"
    );

    Ok((agent, provider, provider_setup_error))
}

fn build_bash_tool() -> Result<DynAgentTool> {
    AgentToolBuilder::new("bash")
        .description(
            "Run a bash command in the current working directory and return stdout, stderr, and exit status.",
        )
        .parameters(json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run in the agent process's current working directory."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        }))
        .execute(|args| async move {
            let command = args
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| AgentError::Other("missing string argument: command".to_string()))?;

            // For a production agent, add a timeout and cap output before
            // returning stdout/stderr to the model.
            let output = Command::new("bash")
                .arg("-lc")
                .arg(command)
                .output()
                .await
                .map_err(|error| AgentError::Other(format!("failed to run bash: {error}")))?;

            let mut text = format!("exit status: {}\n", output.status);
            text.push_str("\nstdout:\n");
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str("\n\nstderr:\n");
            text.push_str(&String::from_utf8_lossy(&output.stderr));

            Ok(AgentToolResult::text(text))
        })
        .build()
}

#[derive(Clone)]
enum ActiveProvider {
    OpenAi(openai::OpenAi),
    GitHubCopilot(github_copilot::GitHubCopilot),
}

impl ActiveProvider {
    fn id(&self) -> &'static str {
        match self {
            Self::OpenAi(_) => "openai",
            Self::GitHubCopilot(_) => "github-copilot",
        }
    }

    fn model(&self, model_id: &str) -> Result<Model> {
        match self {
            Self::OpenAi(provider) => provider.model(model_id).build(),
            Self::GitHubCopilot(provider) => provider.model(model_id).build(),
        }
    }
}

fn build_openai_provider(base_url: Option<&str>, api_key: Option<&str>) -> Result<openai::OpenAi> {
    match base_url {
        Some(base_url) => openai::builder()
            .api_key(api_key)
            .base_url(base_url)
            .chat_completions()
            .build(),
        None => openai::builder().api_key(api_key).build(),
    }
}

fn openai_api_key() -> Option<String> {
    env::var("OPENAI_API_KEY")
        .ok()
        .map(|api_key| api_key.trim().to_string())
        .filter(|api_key| !api_key.is_empty())
}

fn openai_setup_error(base_url: Option<&str>, api_key: Option<&str>) -> Option<String> {
    if base_url.is_some() {
        return None;
    }

    match api_key {
        Some(api_key) if looks_like_github_token(api_key) => Some(
            "OPENAI_API_KEY looks like a GitHub token. Set OPENAI_API_KEY to an OpenAI key, unset it and run /login for Copilot, or set OPENAI_BASE_URL for a local OpenAI-compatible server."
                .to_string(),
        ),
        Some(_) => None,
        None => Some(
            "no OPENAI_API_KEY found; set it before prompting, run /login for Copilot, or set OPENAI_BASE_URL for a local OpenAI-compatible server"
                .to_string(),
        ),
    }
}

fn looks_like_github_token(api_key: &str) -> bool {
    ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"]
        .iter()
        .any(|prefix| api_key.starts_with(prefix))
}

async fn switch_model(agent: &Agent, provider: &ActiveProvider, model_id: &str) -> Result<String> {
    let model = provider.model(model_id)?;
    let provider_id = provider.id().to_string();

    agent.set_model(model).await;

    Ok(provider_id)
}

async fn login_github_copilot_and_swap(
    agent: &Agent,
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
    let model_id = env::var("COPILOT_MODEL").unwrap_or_else(|_| "gpt-5.5".to_string());
    let base_url = github_copilot::base_url_for_credentials(&credentials);
    let copilot = github_copilot::builder()
        .api_key(credentials.access)
        .base_url(base_url)
        .build()?;
    let provider = ActiveProvider::GitHubCopilot(copilot);
    let model = provider.model(&model_id)?;

    agent.set_model(model).await;

    Ok((model_id, provider))
}

fn assistant_visible_content(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            AssistantContent::Thinking(thinking) => Some(thinking.thinking.as_str()),
            AssistantContent::ToolCall(_) => Some("<tool_call>"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{looks_like_github_token, openai_setup_error};

    #[test]
    fn openai_setup_accepts_local_base_url_without_key() {
        assert_eq!(
            openai_setup_error(Some("http://localhost:11434/v1"), None),
            None
        );
    }

    #[test]
    fn openai_setup_rejects_missing_and_github_tokens() {
        assert!(
            openai_setup_error(None, None)
                .expect("missing key should report setup instructions")
                .contains("no OPENAI_API_KEY found")
        );
        assert!(looks_like_github_token("ghu_abc"));
        assert!(
            openai_setup_error(None, Some("ghu_abc"))
                .expect("GitHub token should report setup instructions")
                .contains("looks like a GitHub token")
        );
    }

    #[test]
    fn openai_setup_accepts_openai_shaped_key() {
        assert_eq!(openai_setup_error(None, Some("sk-test")), None);
    }
}
