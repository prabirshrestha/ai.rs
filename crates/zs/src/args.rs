use clap::{Parser, ValueEnum};

/// A small Codex/Grok-style agent CLI powered by ai.rs.
#[derive(Debug, Parser)]
#[command(
    name = "zs",
    version,
    after_help = "\
Examples:
  zs
  zs \"summarize this repo\"
  zs --provider anthropic --model claude-sonnet-4-5
  OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_MODEL=gemma4:12b zs
"
)]
pub struct Args {
    /// Provider: openai, anthropic, or copilot.
    #[arg(short, long, value_enum)]
    pub provider: Option<ProviderKind>,

    /// Model id. Defaults depend on the provider.
    #[arg(short, long)]
    pub model: Option<String>,

    /// OpenAI-compatible base URL.
    #[arg(long)]
    pub base_url: Option<String>,

    /// Prompt to send. If omitted, starts an interactive REPL.
    pub prompt: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ProviderKind {
    #[value(
        name = "openai",
        alias = "openai-responses",
        alias = "openai-completions"
    )]
    OpenAi,
    #[value(alias = "claude")]
    Anthropic,
    #[value(alias = "github-copilot", alias = "github_copilot")]
    Copilot,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Copilot => "copilot",
        }
    }

    pub fn parse_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "openai" | "openai-responses" | "openai-completions" => Some(Self::OpenAi),
            "anthropic" | "claude" => Some(Self::Anthropic),
            "copilot" | "github-copilot" | "github_copilot" => Some(Self::Copilot),
            _ => None,
        }
    }
}

impl Args {
    pub fn prompt_text(&self) -> Option<String> {
        let prompt = self.prompt.join(" ");
        let prompt = prompt.trim();
        if prompt.is_empty() {
            None
        } else {
            Some(prompt.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, ProviderKind};

    #[test]
    fn parses_one_shot_prompt_and_flags() {
        let args = Args::try_parse_from([
            "zs",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-5",
            "hello",
            "world",
        ])
        .expect("args");

        assert_eq!(args.provider, Some(ProviderKind::Anthropic));
        assert_eq!(args.model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(args.prompt_text().as_deref(), Some("hello world"));
    }

    #[test]
    fn parses_copilot_aliases() {
        let args = Args::try_parse_from(["zs", "--provider", "github-copilot"]).expect("args");
        assert_eq!(args.provider, Some(ProviderKind::Copilot));
    }

    #[test]
    fn parses_openai_provider_name() {
        let args = Args::try_parse_from(["zs", "--provider", "openai"]).expect("args");
        assert_eq!(args.provider, Some(ProviderKind::OpenAi));
    }

    #[test]
    fn empty_prompt_starts_repl() {
        let args = Args::try_parse_from(["zs"]).expect("args");
        assert!(args.prompt_text().is_none());
        assert_eq!(
            ProviderKind::parse_name("Claude"),
            Some(ProviderKind::Anthropic)
        );
        assert_eq!(ProviderKind::parse_name("nope"), None);
    }
}
