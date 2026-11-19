//! Pi's system prompt: a port of `buildSystemPromptSections` and
//! `formatSkillsForPrompt` from `pi-coding-agent`, as the harness configures
//! them (no context files, no prompt templates, harness tool definitions
//! without prompt snippets).
//!
//! [rust] The `docs` section is left out: it points at Pi's documentation on
//! the host, which does not exist here.

/// A skill listed to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiPromptSkill {
    pub name: String,
    pub description: String,
    /// Absolute sandbox path of the skill's `SKILL.md`.
    pub file_path: String,
}

/// The inputs of [`build_pi_system_prompt`].
#[derive(Debug, Clone, Default)]
pub struct PiSystemPromptOptions {
    /// Names of the tools the model sees.
    pub selected_tools: Vec<String>,
    /// The turn's instructions (`appendSystemPrompt`).
    pub append_system_prompt: Option<String>,
    pub skills: Vec<PiPromptSkill>,
    pub cwd: String,
}

pub const PI_PREAMBLE: &str = "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.";

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// `formatSkillsForPrompt(skills, fileReadTool)`.
pub fn format_skills_for_prompt(skills: &[PiPromptSkill], file_read_tool: &str) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "\n\nThe following skills provide specialized instructions for specific tasks.".to_string(),
        match file_read_tool {
            "read" => "Use the read tool to load a skill's file when the task matches its description.",
            "bash" => "Use bash to load a skill's file when the task matches its description.",
            _ => "Load a skill's file when the task matches its description.",
        }
        .to_string(),
        "Resolve relative paths against the skill's directory (parent of SKILL.md). Use absolute paths in tool calls.".to_string(),
        String::new(),
        "<available_skills>".to_string(),
    ];
    for skill in skills {
        lines.push("  <skill>".into());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path)
        ));
        lines.push("  </skill>".into());
    }
    lines.push("</available_skills>".into());
    lines.join("\n")
}

fn build_rules(selected_tools: &[String]) -> String {
    let has = |name: &str| selected_tools.iter().any(|tool| tool == name);
    let mut rules: Vec<&str> = Vec::new();
    if has("bash") && !has("grep") && !has("find") && !has("ls") {
        rules.push("Use bash for file operations like ls, rg, find");
    }
    rules.push("Be concise in your responses");
    rules.push("Show file paths clearly when working with files");
    rules
        .iter()
        .map(|rule| format!("- {rule}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `buildSystemPromptSections(input)`, in order.
pub fn build_pi_system_prompt_sections(options: &PiSystemPromptOptions) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = vec![
        ("preamble".into(), PI_PREAMBLE.into()),
        // Harness tool definitions carry no prompt snippets.
        (
            "tools".into(),
            "(none)\n\nAdditional tools may be available.".into(),
        ),
        ("rules".into(), build_rules(&options.selected_tools)),
    ];
    if let Some(text) = options
        .append_system_prompt
        .as_ref()
        .filter(|text| !text.is_empty())
    {
        sections.push(("addendum".into(), text.clone()));
    }
    let reader = ["read", "bash"]
        .into_iter()
        .find(|tool| options.selected_tools.iter().any(|name| name == tool));
    if let Some(reader) = reader {
        let skills = format_skills_for_prompt(&options.skills, reader);
        let skills = skills.trim();
        if !skills.is_empty() {
            sections.push(("skills".into(), skills.to_string()));
        }
    }
    sections.push(("cwd".into(), options.cwd.replace('\\', "/")));
    sections
        .into_iter()
        .map(|(name, content)| {
            if name == "preamble" {
                (name, content)
            } else {
                let wrapped = format!("<{name}>\n{content}\n</{name}>");
                (name, wrapped)
            }
        })
        .collect()
}

/// `buildSystemPrompt(input)`: the sections joined by blank lines.
pub fn build_pi_system_prompt(options: &PiSystemPromptOptions) -> String {
    build_pi_system_prompt_sections(options)
        .into_iter()
        .map(|(_, text)| text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn builds_the_sections_in_order() {
        let prompt = build_pi_system_prompt(&PiSystemPromptOptions {
            selected_tools: tools(&["read", "write", "edit", "bash", "grep", "find", "ls"]),
            append_system_prompt: Some("Be brief.".into()),
            skills: vec![PiPromptSkill {
                name: "notes".into(),
                description: "Use <when> writing".into(),
                file_path: "/home/u/.agents/skills/notes/SKILL.md".into(),
            }],
            cwd: "/work".into(),
        });
        assert_eq!(
            prompt,
            format!(
                "{PI_PREAMBLE}\n\n<tools>\n(none)\n\nAdditional tools may be available.\n</tools>\n\n<rules>\n- Be concise in your responses\n- Show file paths clearly when working with files\n</rules>\n\n<addendum>\nBe brief.\n</addendum>\n\n<skills>\nThe following skills provide specialized instructions for specific tasks.\nUse the read tool to load a skill's file when the task matches its description.\nResolve relative paths against the skill's directory (parent of SKILL.md). Use absolute paths in tool calls.\n\n<available_skills>\n  <skill>\n    <name>notes</name>\n    <description>Use &lt;when&gt; writing</description>\n    <location>/home/u/.agents/skills/notes/SKILL.md</location>\n  </skill>\n</available_skills>\n</skills>\n\n<cwd>\n/work\n</cwd>"
            )
        );
    }

    #[test]
    fn suggests_bash_for_files_and_skills_without_file_tools() {
        let prompt = build_pi_system_prompt(&PiSystemPromptOptions {
            selected_tools: tools(&["bash"]),
            skills: vec![PiPromptSkill {
                name: "s".into(),
                description: "d".into(),
                file_path: "/s/SKILL.md".into(),
            }],
            cwd: "/w".into(),
            ..Default::default()
        });
        assert!(prompt.contains("- Use bash for file operations like ls, rg, find\n"));
        assert!(prompt.contains("Use bash to load a skill's file"));
    }

    #[test]
    fn omits_skills_without_a_reader() {
        let prompt = build_pi_system_prompt(&PiSystemPromptOptions {
            selected_tools: tools(&["grep"]),
            skills: vec![PiPromptSkill {
                name: "s".into(),
                description: "d".into(),
                file_path: "/s/SKILL.md".into(),
            }],
            cwd: "/w".into(),
            ..Default::default()
        });
        assert!(!prompt.contains("<skills>"));
    }
}
