# simple-coding-agent

A tiny interactive coding agent backed by the `ai` crate, written the Pi 1.0
way. It exposes one tool, `bash` (60-second timeout, output capped at 16 KiB
per stream).

There is one `Models` registry (`builtin_models` with an in-memory credential
store) for every provider. The model is picked from it by provider and id,
and the agent's stream function is set once to `stream_simple_fn(models)`.
Keys come from the credential store first, then from the provider's env vars
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COPILOT_GITHUB_TOKEN`, ...).

Run from the workspace root:

```bash
export OPENAI_API_KEY=sk-...
cargo run -p simple-coding-agent
```

Environment:

- `PI_PROVIDER`: provider id, default `openai`.
- `PI_MODEL`: model id, default `gpt-6.1-sol`.
- `COPILOT_MODEL`: model `/login` switches to, default `gpt-6.1-sol`.
- `OPENAI_BASE_URL`: registers an `openai-compatible` provider for a local
  OpenAI-compatible server (Chat Completions, no API key) serving
  `PI_MODEL`, and makes it the default provider.

For Anthropic:

```bash
ANTHROPIC_API_KEY=sk-ant-... PI_PROVIDER=anthropic PI_MODEL=claude-sonnet-4-6 \
cargo run -p simple-coding-agent
```

For Ollama:

```bash
ollama pull gemma4:12b
OPENAI_BASE_URL=http://localhost:11434/v1 PI_MODEL=gemma4:12b \
cargo run -p simple-coding-agent
```

For GitHub Copilot, start the REPL and run `/login`.

Commands inside the REPL:

- `/model provider/id`: switch models (for example
  `/model anthropic/claude-sonnet-4-6`) while keeping the conversation.
  `/model` alone prints the current model.
- `/login`: log into GitHub Copilot with the device-code flow. The
  credential is stored in the registry, which refreshes it; the agent
  switches to `github-copilot/$COPILOT_MODEL` and keeps the conversation.
- `/login <enterprise-domain>`: the same for a GitHub Enterprise domain.
- `/clear`: reset the conversation.
- `/exit`: exit.
