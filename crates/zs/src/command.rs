#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    Help,
    Clear,
    Exit,
    Model { id: Option<String> },
    Provider { name: Option<String> },
    Login { enterprise_domain: Option<String> },
    Unknown { name: String },
    Prompt(String),
}

pub fn parse_line(line: &str) -> Option<ReplCommand> {
    let prompt = line.trim();
    if prompt.is_empty() {
        return None;
    }

    let Some(command) = prompt.strip_prefix('/') else {
        return Some(ReplCommand::Prompt(prompt.to_string()));
    };
    let command = command.trim_start_matches('/');
    let (name, rest) = split_command(command);

    match name {
        "help" | "?" => Some(ReplCommand::Help),
        "clear" => Some(ReplCommand::Clear),
        "exit" | "quit" => Some(ReplCommand::Exit),
        "model" => Some(ReplCommand::Model {
            id: optional_arg(rest),
        }),
        "provider" => Some(ReplCommand::Provider {
            name: optional_arg(rest),
        }),
        "login" => Some(ReplCommand::Login {
            enterprise_domain: optional_arg(rest),
        }),
        "" => Some(ReplCommand::Unknown {
            name: String::new(),
        }),
        other => Some(ReplCommand::Unknown {
            name: other.to_string(),
        }),
    }
}

fn split_command(command: &str) -> (&str, &str) {
    match command.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest),
        None => (command, ""),
    }
}

fn optional_arg(rest: &str) -> Option<String> {
    let rest = rest.trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{ReplCommand, parse_line};

    #[test]
    fn parses_slash_commands() {
        assert_eq!(parse_line("   "), None);
        assert_eq!(
            parse_line("hello"),
            Some(ReplCommand::Prompt("hello".into()))
        );
        assert_eq!(parse_line("/help"), Some(ReplCommand::Help));
        assert_eq!(parse_line("//quit"), Some(ReplCommand::Exit));
        assert_eq!(
            parse_line("/model gpt-5.5"),
            Some(ReplCommand::Model {
                id: Some("gpt-5.5".into())
            })
        );
        assert_eq!(
            parse_line("/provider"),
            Some(ReplCommand::Provider { name: None })
        );
        assert_eq!(
            parse_line("/login github.example.com"),
            Some(ReplCommand::Login {
                enterprise_domain: Some("github.example.com".into())
            })
        );
        assert_eq!(
            parse_line("/nope"),
            Some(ReplCommand::Unknown {
                name: "nope".into()
            })
        );
    }
}
