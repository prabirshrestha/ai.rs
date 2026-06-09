use std::env;
use std::io::Write;

use ai::{
    Agent, AgentError, AgentEvent, AgentOptions, AgentToolBuilder, AgentToolResult,
    AssistantMessageEvent, DynAgentTool, Result, providers::openai,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = build_agent()?;

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
                assistant_message_event: AssistantMessageEvent::Error { .. },
                ..
            } => {
                eprintln!("\nerror: assistant stream failed");
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

        match prompt {
            "/exit" | "/quit" => break,
            "/clear" => {
                agent.reset().await;
                println!("context cleared");
            }
            _ => {
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

fn build_agent() -> Result<Agent> {
    let cwd = env::current_dir()?;
    let base_url = env::var("OPENAI_BASE_URL").ok();
    let model_id = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.5".to_string());
    let openai = match base_url.as_deref() {
        Some(base_url) => openai::builder()
            .base_url(base_url)
            .chat_completions()
            .build()?,
        None => openai::from_env()?,
    };
    let model = openai.model(&model_id).build()?;

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
    println!("type a prompt, /clear to reset context, or /exit to quit");

    Ok(agent)
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
                    "description": "The bash command to run in the agent process current working directory."
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
