# ai

Rust port of the focused `pi` AI package surface.

This crate intentionally keeps the active provider API surface narrow:

- OpenAI Chat Completions: `openai-completions`
- OpenAI Responses: `openai-responses`
- Anthropic Messages: `anthropic-messages`

GitHub Copilot is supported as an OAuth-backed provider through the OpenAI and
Anthropic-compatible routes above. The agent and agent-loop runtime live in this
same `ai` crate; there is no separate agent crate and no TypeScript harness in
this port.

The crate is still work in progress and the API may change.

## Install

```sh
cargo add ai
```

The crate uses Tokio streams and `reqwest` internally. Applications normally use
Tokio as their async runtime.

## Environment Keys

The built-in environment lookup is scoped to the focused providers:

| Provider | Environment variables |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_OAUTH_TOKEN`, then `ANTHROPIC_API_KEY` |
| `github-copilot` | `COPILOT_GITHUB_TOKEN` |

You can also pass an explicit API key in `StreamOptions`.

## Chat Completions

```rust
use ai::{complete_simple, Context, Message, SimpleStreamOptions};

#[tokio::main]
async fn main() -> ai::Result<()> {
    let model = ai::get_model("openai", "gpt-4o-mini").expect("model");
    let message = complete_simple(
        model,
        Context {
            system_prompt: Some("Reply in one short sentence.".to_string()),
            messages: vec![Message::user_text("Say hello.")],
            tools: Vec::new(),
        },
        Some(SimpleStreamOptions::default()),
    )
    .await?;

    println!("{message:?}");
    Ok(())
}
```

## Responses

```rust
use ai::{complete_simple, Context, Message, Model, ModelThinkingLevel, SimpleStreamOptions};

#[tokio::main]
async fn main() -> ai::Result<()> {
    let mut options = SimpleStreamOptions::default();
    options.reasoning = Some(ModelThinkingLevel::Low);

    let model = Model {
        id: "gpt-5.5".to_string(),
        name: "GPT 5.5".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        reasoning: true,
        ..Default::default()
    };

    let message = complete_simple(
        model,
        Context {
            messages: vec![Message::user_text("Summarize Rust ownership.")],
            ..Default::default()
        },
        Some(options),
    )
    .await?;

    println!("{message:?}");
    Ok(())
}
```

## Anthropic

```rust
use ai::{complete_simple, Context, Message, Model, SimpleStreamOptions};

#[tokio::main]
async fn main() -> ai::Result<()> {
    let model = Model {
        id: "claude-sonnet-4-5".to_string(),
        name: "Claude Sonnet".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        ..Default::default()
    };

    let message = complete_simple(
        model,
        Context {
            messages: vec![Message::user_text("Write a short status update.")],
            ..Default::default()
        },
        Some(SimpleStreamOptions::default()),
    )
    .await?;

    println!("{message:?}");
    Ok(())
}
```

## GitHub Copilot

Copilot OAuth is available through `login_github_copilot`,
`refresh_github_copilot_token`, and the OAuth registry helpers. After login, the
Copilot access token is used as the API key for models whose provider is
`github-copilot`.

Copilot dynamic request headers are shared by the OpenAI Chat Completions,
OpenAI Responses, and Anthropic routes. The helper functions are public:

- `build_copilot_dynamic_headers`
- `infer_copilot_initiator`
- `has_copilot_vision_input`
- `get_github_copilot_base_url`

## Agent Runtime

The agent API is part of this crate:

- `Agent`
- `AgentOptions`
- `agent_loop`
- `agent_loop_continue`
- `run_agent_loop`
- `run_agent_loop_continue`

Tools implement `AgentTool`; the loop supports sequential or parallel tool
execution, cancellation, `before_tool_call`, `after_tool_call`,
`should_stop_after_turn`, and `prepare_next_turn` hooks.

## Scope

Included in this port:

- OpenAI Chat Completions provider behavior
- OpenAI Responses provider behavior
- Anthropic Messages provider behavior
- GitHub Copilot dynamic headers and OAuth flow
- Anthropic OAuth flow
- Agent and agent-loop runtime inside the `ai` crate

Not included in the active built-in provider surface:

- Mistral Conversations
- Google Generative AI or Google Vertex
- Azure OpenAI Responses
- OpenAI Codex Responses
- Bedrock or other broad provider-specific APIs
- TypeScript agent harness
- Separate Rust agent crate

The generated model catalog is loaded from upstream metadata, then filtered to
models that use the three API routes listed above.

## Verification

```sh
cargo fmt -p ai --check
cargo test -p ai --quiet
```

Optional local integration tests use an OpenAI-compatible server at
`http://localhost:4141/v1`:

```sh
PI_REQUIRE_LOCAL_4141=1 cargo test -p ai --test local_4141 --quiet
```

## License

MIT
