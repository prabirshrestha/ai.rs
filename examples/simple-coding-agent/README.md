# simple-coding-agent

A tiny interactive coding agent backed by the `ai` crate. It exposes one tool:
`bash`.

Run from the workspace root:

```bash
cargo run -p simple-coding-agent
```

For Ollama:

```bash
ollama pull gemma4:12b
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_MODEL=gemma4:12b \
cargo run -p simple-coding-agent
```

Commands inside the REPL:

- `/clear`: reset conversation context.
- `/model [name]`: show the current model, or switch models on the active
  provider while preserving conversation context.
- `/login`: log into GitHub Copilot with the device-code flow and switch the
  agent to Copilot while preserving conversation context. Set `COPILOT_MODEL` to
  override the default `gpt-5.5` model.
- `/login <enterprise-domain>`: log into GitHub Copilot for a GitHub Enterprise
  domain.
- `/exit` or `/quit`: exit.
