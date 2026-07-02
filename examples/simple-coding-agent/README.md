# simple-coding-agent

A tiny interactive coding agent backed by the `ai` crate. It exposes one tool:
`bash`.

Run from the workspace root:

```bash
export OPENAI_API_KEY=sk-...
cargo run -p simple-coding-agent
```

If you start without `OPENAI_API_KEY`, the REPL still opens so you can run
`/login` for GitHub Copilot or point the example at a local OpenAI-compatible
server before prompting.

`OPENAI_API_KEY` must be an OpenAI key, not a GitHub or Copilot token. To use
Copilot, leave `OPENAI_API_KEY` unset and run `/login` in the REPL.

For Ollama:

```bash
ollama pull gemma4:12b
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_MODEL=gemma4:12b \
cargo run -p simple-coding-agent
```

For GitHub Copilot, start the REPL and run:

```text
/login
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
