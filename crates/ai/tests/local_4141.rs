use ai::{
    Agent, AgentOptions, AgentTool, AgentToolResult, Context, Message, Model, ModelCost,
    ModelInput, ModelThinkingLevel, SimpleStreamOptions, StopReason, StreamOptions, Tool,
    ToolResultContent, complete_simple,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

const LOCAL_BASE_URL: &str = "http://localhost:4141/v1";
const DEFAULT_CHAT_COMPLETIONS_MODEL: &str = "gpt-5.2";

fn local_model(api: &str, model_id: &str, reasoning: bool) -> Model {
    Model {
        id: model_id.to_string(),
        name: model_id.to_string(),
        api: api.to_string(),
        provider: "openai".to_string(),
        base_url: LOCAL_BASE_URL.to_string(),
        reasoning,
        input: vec![ModelInput::Text, ModelInput::Image],
        cost: ModelCost::default(),
        context_window: 1_000_000,
        max_tokens: 128,
        ..Default::default()
    }
}

fn local_options(reasoning: Option<ModelThinkingLevel>) -> SimpleStreamOptions {
    SimpleStreamOptions {
        stream: StreamOptions {
            api_key: Some("local-test".to_string()),
            timeout_ms: Some(60_000),
            max_tokens: Some(64),
            ..Default::default()
        },
        reasoning,
        thinking_budgets: None,
        tool_choice: None,
    }
}

struct LocalEchoTool;

#[async_trait]
impl AgentTool for LocalEchoTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "echo_check".to_string(),
            description: "Returns the local tool check marker.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            }),
        }
    }

    fn label(&self) -> &str {
        "Echo check"
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        _args: Value,
        _cancellation_token: Option<CancellationToken>,
        _on_update: Option<ai::AgentToolUpdateCallback>,
    ) -> ai::AgentResult<AgentToolResult> {
        let mut result = AgentToolResult::text("local-tool-ok");
        result.terminate = true;
        Ok(result)
    }
}

async fn local_4141_available() -> bool {
    reqwest::Client::new()
        .get(format!("{LOCAL_BASE_URL}/models"))
        .timeout(std::time::Duration::from_millis(500))
        .send()
        .await
        .is_ok()
}

async fn skip_unless_available(test_name: &str) -> bool {
    if local_4141_available().await {
        return false;
    }
    if std::env::var("PI_REQUIRE_LOCAL_4141").ok().as_deref() == Some("1") {
        panic!("{test_name} requires {LOCAL_BASE_URL}, but it is not accepting connections");
    }
    eprintln!("skipping {test_name}: {LOCAL_BASE_URL} is not accepting connections");
    true
}

fn require_local_4141() -> bool {
    std::env::var("PI_REQUIRE_LOCAL_4141").ok().as_deref() == Some("1")
}

#[tokio::test]
async fn local_openai_responses_gpt55_low_effort() -> ai::Result<()> {
    if skip_unless_available("local_openai_responses_gpt55_low_effort").await {
        return Ok(());
    }

    let message = complete_simple(
        local_model("openai-responses", "gpt-5.5", true),
        Context {
            system_prompt: Some("Reply with a short plain sentence.".to_string()),
            messages: vec![Message::user_text("Say port check ok.")],
            tools: Vec::new(),
        },
        Some(local_options(Some(ModelThinkingLevel::Low))),
    )
    .await?;

    assert_ne!(message.stop_reason, StopReason::Error, "{message:#?}");
    assert!(
        !message.content.is_empty(),
        "expected at least one content block"
    );
    Ok(())
}

#[tokio::test]
async fn local_openai_chat_completions_streaming() -> ai::Result<()> {
    if skip_unless_available("local_openai_chat_completions_streaming").await {
        return Ok(());
    }

    let chat_model = std::env::var("PI_LOCAL_CHAT_COMPLETIONS_MODEL")
        .unwrap_or_else(|_| DEFAULT_CHAT_COMPLETIONS_MODEL.to_string());
    let chat_model_overridden = std::env::var("PI_LOCAL_CHAT_COMPLETIONS_MODEL").is_ok();

    let message = complete_simple(
        local_model(
            "openai-completions",
            &chat_model,
            chat_model.starts_with("gpt-5"),
        ),
        Context {
            system_prompt: Some("Reply with a short plain sentence.".to_string()),
            messages: vec![Message::user_text("Say chat completion check ok.")],
            tools: Vec::new(),
        },
        Some(local_options(
            chat_model
                .starts_with("gpt-5")
                .then_some(ModelThinkingLevel::Low),
        )),
    )
    .await?;

    if message.stop_reason == StopReason::Error
        && !chat_model_overridden
        && !require_local_4141()
        && message
            .error_message
            .as_deref()
            .is_some_and(|error| error.contains("unsupported_api_for_model"))
    {
        eprintln!(
            "skipping local_openai_chat_completions_streaming: default chat model {chat_model:?} is not supported by {LOCAL_BASE_URL}; set PI_LOCAL_CHAT_COMPLETIONS_MODEL"
        );
        return Ok(());
    }

    assert_ne!(message.stop_reason, StopReason::Error, "{message:#?}");
    assert!(
        !message.content.is_empty(),
        "expected at least one content block"
    );
    Ok(())
}

#[tokio::test]
async fn local_agent_openai_responses_gpt55_low_effort() -> ai::AgentResult<()> {
    if skip_unless_available("local_agent_openai_responses_gpt55_low_effort").await {
        return Ok(());
    }

    let mut options = AgentOptions::new(local_model("openai-responses", "gpt-5.5", true));
    options.initial_state.system_prompt = Some("Reply with a short plain sentence.".to_string());
    options.options = local_options(Some(ModelThinkingLevel::Low));
    let agent = Agent::new(options);

    agent
        .prompt_text("Say agent responses check ok.", Vec::new())
        .await?;

    let state = agent.state().await;
    let Some(Message::Assistant(message)) = state.messages.last() else {
        panic!(
            "expected final assistant message, got {:#?}",
            state.messages
        );
    };
    assert_ne!(message.stop_reason, StopReason::Error, "{message:#?}");
    assert!(
        !message.content.is_empty(),
        "expected at least one content block"
    );
    Ok(())
}

#[tokio::test]
async fn local_agent_openai_chat_completions_streaming() -> ai::AgentResult<()> {
    if skip_unless_available("local_agent_openai_chat_completions_streaming").await {
        return Ok(());
    }

    let chat_model = std::env::var("PI_LOCAL_CHAT_COMPLETIONS_MODEL")
        .unwrap_or_else(|_| DEFAULT_CHAT_COMPLETIONS_MODEL.to_string());
    let chat_model_overridden = std::env::var("PI_LOCAL_CHAT_COMPLETIONS_MODEL").is_ok();

    let mut options = AgentOptions::new(local_model(
        "openai-completions",
        &chat_model,
        chat_model.starts_with("gpt-5"),
    ));
    options.initial_state.system_prompt = Some("Reply with a short plain sentence.".to_string());
    options.options = local_options(
        chat_model
            .starts_with("gpt-5")
            .then_some(ModelThinkingLevel::Low),
    );
    let agent = Agent::new(options);

    agent
        .prompt_text("Say agent chat completion check ok.", Vec::new())
        .await?;

    let state = agent.state().await;
    let Some(Message::Assistant(message)) = state.messages.last() else {
        panic!(
            "expected final assistant message, got {:#?}",
            state.messages
        );
    };
    if message.stop_reason == StopReason::Error
        && !chat_model_overridden
        && !require_local_4141()
        && message
            .error_message
            .as_deref()
            .is_some_and(|error| error.contains("unsupported_api_for_model"))
    {
        eprintln!(
            "skipping local_agent_openai_chat_completions_streaming: default chat model {chat_model:?} is not supported by {LOCAL_BASE_URL}; set PI_LOCAL_CHAT_COMPLETIONS_MODEL"
        );
        return Ok(());
    }

    assert_ne!(message.stop_reason, StopReason::Error, "{message:#?}");
    assert!(
        !message.content.is_empty(),
        "expected at least one content block"
    );
    Ok(())
}

#[tokio::test]
async fn local_agent_openai_chat_completions_tool_call() -> ai::AgentResult<()> {
    if skip_unless_available("local_agent_openai_chat_completions_tool_call").await {
        return Ok(());
    }

    let chat_model = std::env::var("PI_LOCAL_CHAT_COMPLETIONS_MODEL")
        .unwrap_or_else(|_| DEFAULT_CHAT_COMPLETIONS_MODEL.to_string());

    let mut options = AgentOptions::new(local_model(
        "openai-completions",
        &chat_model,
        chat_model.starts_with("gpt-5"),
    ));
    options.initial_state.system_prompt = Some("Use tools when they are selected.".to_string());
    options.initial_state.tools = vec![std::sync::Arc::new(LocalEchoTool)];
    options.options = local_options(
        chat_model
            .starts_with("gpt-5")
            .then_some(ModelThinkingLevel::Low),
    );
    options.options.tool_choice = Some(json!({
        "type": "function",
        "function": { "name": "echo_check" }
    }));
    let agent = Agent::new(options);

    agent
        .prompt_text("Call the selected tool once.", Vec::new())
        .await?;

    let state = agent.state().await;
    let tool_result = state
        .messages
        .iter()
        .find_map(|message| match message {
            Message::ToolResult(result) if result.tool_name == "echo_check" => Some(result),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected echo_check tool result, got {:#?}", state.messages));

    assert!(!tool_result.is_error, "{tool_result:#?}");
    assert!(tool_result.content.iter().any(|content| matches!(
        content,
        ToolResultContent::Text(text) if text.text == "local-tool-ok"
    )));
    Ok(())
}
