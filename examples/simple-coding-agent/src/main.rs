//! simple-coding-agent, written the Pi 1.0 way: one `Models` registry for
//! every provider, models picked by provider and id, `stream_simple_fn`
//! set once, and `/login` storing the GitHub Copilot credential in the same
//! registry.

use std::env;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use ai::{
    Agent, AgentError, AgentEvent, AgentOptions, AgentToolBuilder, AgentToolResult, AnyModel,
    ApiKeyAuth, ApiKeyAuthInput, AssistantMessageEvent, AuthEvent, AuthInteraction, AuthPrompt,
    AuthResult, AuthType, CreateModelsOptions, CreateProviderOptions, DynAgentTool,
    InMemoryCredentialStore, LoginOptions, Message, Model, ModelAuth, ModelInput, ProviderApi,
    ProviderAuth, ProviderHeaders, Result, api::openai_completions::openai_completions_api,
    create_provider, providers::all::builtin_models, stream_simple_fn,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

const BASH_TOOL_TIMEOUT: Duration = Duration::from_secs(60);
const BASH_TOOL_OUTPUT_LIMIT: usize = 16 * 1024;
/// Provider registered for `OPENAI_BASE_URL`.
const LOCAL_PROVIDER: &str = "openai-compatible";

#[tokio::main]
async fn main() -> Result<()> {
    // 1. One registry for every provider. Keys come from the credential
    //    store, then env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    //    COPILOT_GITHUB_TOKEN, ...). Swap InMemoryCredentialStore for a
    //    persistent `CredentialStore` to keep logins.
    let models = builtin_models(CreateModelsOptions {
        credentials: Some(Arc::new(InMemoryCredentialStore::new())),
        ..Default::default()
    });

    // An OpenAI-compatible server (llama.cpp, Ollama, vLLM, ...) is one more
    // provider in the same registry.
    let base_url = env::var("OPENAI_BASE_URL").ok();
    let default_provider = if base_url.is_some() {
        LOCAL_PROVIDER
    } else {
        "openai"
    };

    // 2. Models are picked by provider + id from that registry.
    let provider = env::var("PI_PROVIDER").unwrap_or_else(|_| default_provider.to_string());
    let model_id = env::var("PI_MODEL").unwrap_or_else(|_| "gpt-6.1-sol".to_string());
    if let Some(base_url) = &base_url {
        models.set_provider(create_provider(local_provider_options(
            base_url, &model_id,
        ))?);
    }
    let Some(model) = models.get_model(&provider, &model_id) else {
        return Err(ai::Error::message(format!(
            "Unknown model {provider}/{model_id}"
        )));
    };

    // 3. The agent. The system prompt and tools become the leading system
    //    message of the transcript; the stream function is the registry's
    //    `stream_simple`, set once.
    let agent = Agent::new(
        AgentOptions::builder(model)
            .system_prompt(format!(
                "You are an expert coding assistant. Use the bash tool to inspect and change files. Do not run destructive commands unless the user explicitly asks. Current directory: {}",
                env::current_dir()?.display()
            ))
            .tool(build_bash_tool()?)
            .stream_fn(stream_simple_fn(models.clone()))
            .build(),
    );

    let _subscription = agent.subscribe(|event, _| async move {
        match event {
            AgentEvent::MessageUpdate {
                assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
                ..
            } => {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            AgentEvent::MessageEnd {
                message: Message::Assistant(message),
            } => {
                if let Some(error) = &message.error_message {
                    eprintln!("\nerror: {error}");
                }
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } => println!("\n{tool_name}({args})"),
            _ => {}
        }
        Ok(())
    });

    println!("model: {provider}/{model_id}");
    println!(
        "type a prompt, /model provider/id to switch models, /login [enterprise-domain] for GitHub Copilot, /clear or /exit"
    );

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!("\n> ");
        let _ = std::io::stdout().flush();
        let Some(line) = lines.next_line().await? else {
            break;
        };
        let line = line.trim();
        match line {
            "" => continue,
            "/exit" => break,
            "/clear" => {
                if let Err(error) = agent.reset() {
                    eprintln!("error: {error}");
                }
            }
            // /model <provider>/<id>: switch model; same registry, same
            // stream function.
            "/model" => {
                let model = agent.state().model;
                println!("model: {}/{}", model.provider, model.id);
            }
            _ if line.starts_with("/model ") => {
                let (provider, id) = line["/model ".len()..]
                    .trim()
                    .split_once('/')
                    .unwrap_or_default();
                match models.get_model(provider, id) {
                    Some(next) => agent.set_model(next),
                    None => println!("unknown model"),
                }
            }
            // /login: OAuth through the same registry; Models stores and
            // refreshes the credential. No new registry, no stream function
            // swap.
            _ if line == "/login" || line.starts_with("/login ") => {
                let enterprise_domain = line["/login".len()..].trim().to_string();
                let interaction = Arc::new(Terminal { enterprise_domain });
                match models
                    .login(
                        "github-copilot",
                        AuthType::OAuth,
                        interaction,
                        LoginOptions::default(),
                    )
                    .await
                {
                    Ok(_) => {
                        let id =
                            env::var("COPILOT_MODEL").unwrap_or_else(|_| "gpt-6.1-sol".to_string());
                        match models.get_model("github-copilot", &id) {
                            Some(next) => {
                                agent.set_model(next);
                                println!("logged into GitHub Copilot; model: github-copilot/{id}");
                            }
                            None => println!("logged in, but github-copilot/{id} is unknown"),
                        }
                    }
                    Err(error) => eprintln!("error: {error}"),
                }
            }
            _ => {
                if let Err(error) = agent.prompt_text(line, Vec::new()).await {
                    eprintln!("\nerror: {error}");
                }
            }
        }
    }
    Ok(())
}

/// `/login` interaction: answers the GitHub Enterprise domain prompt with the
/// command's argument (blank for github.com) and prints the device code and
/// progress.
struct Terminal {
    enterprise_domain: String,
}

#[async_trait::async_trait]
impl AuthInteraction for Terminal {
    async fn prompt(&self, _prompt: AuthPrompt) -> Result<String> {
        Ok(self.enterprise_domain.clone())
    }

    fn notify(&self, event: AuthEvent) {
        match event {
            AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                ..
            } => println!("Open {verification_uri} and enter {user_code}"),
            AuthEvent::Progress { message } | AuthEvent::Info { message, .. } => {
                println!("{message}")
            }
            AuthEvent::AuthUrl { url, .. } => println!("Open {url}"),
        }
    }
}

/// A provider for an OpenAI-compatible server at `base_url` serving one chat
/// model through Chat Completions, without an API key.
fn local_provider_options(base_url: &str, model_id: &str) -> CreateProviderOptions {
    let model = Model {
        id: model_id.to_string(),
        name: model_id.to_string(),
        api: "openai-completions".to_string(),
        provider: LOCAL_PROVIDER.to_string(),
        base_url: base_url.to_string(),
        input: vec![ModelInput::Text],
        context_window: 128_000,
        max_tokens: 16_384,
        ..Default::default()
    };
    CreateProviderOptions {
        id: LOCAL_PROVIDER.to_string(),
        name: Some("OpenAI-compatible server".to_string()),
        base_url: Some(base_url.to_string()),
        auth: ProviderAuth {
            api_key: Some(Arc::new(Keyless)),
            oauth: None,
        },
        models: vec![AnyModel::Chat(model)],
        api: Some(ProviderApi::Single(openai_completions_api())),
        ..Default::default()
    }
}

/// Auth for servers that need no key: Chat Completions requires a key or an
/// `Authorization` header, so this passes a placeholder key and suppresses
/// the header it would produce.
struct Keyless;

#[async_trait::async_trait]
impl ApiKeyAuth for Keyless {
    fn name(&self) -> &str {
        "No API key"
    }

    async fn resolve(&self, _input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
        let mut headers = ProviderHeaders::new();
        headers.insert("Authorization", None::<String>);
        Ok(Some(AuthResult {
            auth: ModelAuth {
                api_key: Some("unused".to_string()),
                headers: Some(headers),
                ..Default::default()
            },
            env: None,
            source: Some("keyless".to_string()),
        }))
    }
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

            let output = tokio::time::timeout(
                BASH_TOOL_TIMEOUT,
                Command::new("bash")
                    .kill_on_drop(true)
                    .arg("-lc")
                    .arg(command)
                    .output(),
            )
            .await
            .map_err(|_| AgentError::Other(format!("bash command timed out after {BASH_TOOL_TIMEOUT:?}")))?
            .map_err(|error| AgentError::Other(format!("failed to run bash: {error}")))?;

            Ok(AgentToolResult::text(format_bash_output(
                output.status.to_string(),
                &output.stdout,
                &output.stderr,
                BASH_TOOL_OUTPUT_LIMIT,
            )))
        })
        .build()
}

fn format_bash_output(
    status: impl std::fmt::Display,
    stdout: &[u8],
    stderr: &[u8],
    limit: usize,
) -> String {
    let mut text = format!("exit status: {status}\n");
    text.push_str("\nstdout:\n");
    append_limited_utf8(&mut text, stdout, limit);
    text.push_str("\n\nstderr:\n");
    append_limited_utf8(&mut text, stderr, limit);
    text
}

fn append_limited_utf8(text: &mut String, bytes: &[u8], limit: usize) {
    let shown = bytes.len().min(limit);
    text.push_str(&String::from_utf8_lossy(&bytes[..shown]));
    if bytes.len() > shown {
        text.push_str(&format!("\n[truncated {} bytes]", bytes.len() - shown));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bash_output_is_truncated_per_stream() {
        let text = format_bash_output("exit 0", b"abcdef", b"12345", 3);

        assert!(text.contains("stdout:\nabc\n[truncated 3 bytes]"));
        assert!(text.contains("stderr:\n123\n[truncated 2 bytes]"));
        assert!(!text.contains("def"));
        assert!(!text.contains("45"));
    }

    #[test]
    fn bash_output_keeps_short_streams_intact() {
        let text = format_bash_output("exit 0", b"ok", b"", 16);

        assert!(text.contains("exit status: exit 0"));
        assert!(text.contains("stdout:\nok"));
        assert!(text.ends_with("stderr:\n"));
    }

    #[tokio::test]
    async fn local_provider_sends_no_authorization_header() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let head = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0; 16 * 1024];
            let read = socket.read(&mut buffer).await.unwrap();
            let _ = socket
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await;
            String::from_utf8_lossy(&buffer[..read]).to_lowercase()
        });

        let models = builtin_models(Default::default());
        models.set_provider(create_provider(local_provider_options(&base_url, "gemma")).unwrap());
        let model = models.get_model(LOCAL_PROVIDER, "gemma").unwrap();
        assert_eq!(model.api, "openai-completions");
        let context = ai::Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        models
            .complete_simple(
                &model,
                &context,
                ai::SimpleStreamOptions {
                    stream: ai::StreamOptions {
                        max_retries: Some(0),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .await;

        let head = head.await.unwrap();
        assert!(head.starts_with("post /v1/chat/completions"), "{head}");
        assert!(!head.contains("authorization:"), "{head}");
    }

    #[test]
    fn default_models_exist_in_the_builtin_catalogs() {
        let models = builtin_models(Default::default());
        assert!(models.get_model("openai", "gpt-6.1-sol").is_some());
        assert!(models.get_model("github-copilot", "gpt-6.1-sol").is_some());
    }
}
