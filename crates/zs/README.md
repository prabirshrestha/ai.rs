# zs

A small Codex/Grok-style coding-agent CLI backed by the `ai` crate. It streams
replies, runs an agent loop, and exposes one tool: `bash`.

The binary and crates.io package are both named `zs` — two left-hand QWERTY
keys. `qw` is already published by another crates.io user.

Install from crates.io (once published):

```bash
cargo install zs
```

Run from this workspace:

```bash
export OPENAI_API_KEY=sk-...
cargo run -p zs
```

Pass a prompt to run one turn and exit:

```bash
cargo run -p zs -- "Summarize this repository in one paragraph."
```

If you start without credentials, the REPL still opens so you can run `/login`
for GitHub Copilot or point `zs` at a local OpenAI-compatible server before
prompting.

`OPENAI_API_KEY` must be an OpenAI key, not a GitHub or Copilot token. To use
Copilot, leave `OPENAI_API_KEY` unset and run `/login` in the REPL, or set
`COPILOT_GITHUB_TOKEN`.

For Ollama:

```bash
ollama pull gemma4:12b
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_MODEL=gemma4:12b \
cargo run -p zs
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cargo run -p zs -- --provider anthropic
```

## Flags

- `--provider <openai|anthropic|copilot>`: choose a provider. Also `ZS_PROVIDER`.
- `--model <id>`: choose a model id. Also `ZS_MODEL`, or provider-specific
  `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `COPILOT_MODEL`.
- `--base-url <url>`: OpenAI-compatible base URL. Also `OPENAI_BASE_URL`.

If stdin is not a terminal and no prompt arguments are given, `zs` reads the
prompt from stdin.

## REPL commands

- `/help`: list commands.
- `/clear`: reset conversation context.
- `/model [name]`: show the current model, or switch models on the active
  provider while preserving conversation context.
- `/provider [name]`: show the current provider, or switch to `openai`,
  `anthropic`, or `copilot` while preserving conversation context.
- `/login`: log into GitHub Copilot with the device-code flow and switch the
  agent to Copilot while preserving conversation context.
- `/login <enterprise-domain>`: log into GitHub Copilot for a GitHub Enterprise
  domain.
- `/exit` or `/quit`: exit.
