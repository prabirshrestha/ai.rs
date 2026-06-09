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
- `/exit` or `/quit`: exit.
