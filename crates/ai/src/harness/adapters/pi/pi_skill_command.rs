//! Port of `harness-pi/src/pi-skill-command.ts`.

use tokio_util::sync::CancellationToken;

use crate::harness::error::HarnessResult;
use crate::harness::sandbox::{SandboxReadTextFileOptions, SandboxSession};

const SKILL_COMMAND_PREFIX: &str = "/skill:";

/// A skill `/skill:<name>` can expand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiCommandSkill {
    pub name: String,
    pub file_path: String,
    pub base_dir: String,
}

/// `stripFrontmatter(content)` from `pi-coding-agent`.
pub fn strip_frontmatter(content: &str) -> String {
    let normalized = content
        .strip_prefix('\u{feff}')
        .unwrap_or(content)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if !normalized.starts_with("---") {
        return normalized;
    }
    match normalized[3..].find("\n---") {
        Some(offset) => normalized[3 + offset + 4..].trim().to_string(),
        None => normalized,
    }
}

/// `expandPiSkillCommand({ text, skills, sandbox, abortSignal })`.
///
/// Expand a message that starts with `/skill:<name>` from the skill's file
/// in the sandbox, in the format Pi uses. When the file cannot be read, a
/// leading space keeps the text from being taken as a command. Unknown skills
/// are returned unchanged.
pub async fn expand_pi_skill_command(
    text: &str,
    skills: &[PiCommandSkill],
    sandbox: &dyn SandboxSession,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<String> {
    let Some(rest) = text.strip_prefix(SKILL_COMMAND_PREFIX) else {
        return Ok(text.to_string());
    };
    let (skill_name, args) = match rest.split_once(' ') {
        Some((name, args)) => (name, args.trim()),
        None => (rest, ""),
    };
    let Some(skill) = skills.iter().find(|skill| skill.name == skill_name) else {
        return Ok(text.to_string());
    };
    let content = sandbox
        .read_text_file(SandboxReadTextFileOptions {
            path: skill.file_path.clone(),
            abort_signal: abort_signal.clone(),
            ..Default::default()
        })
        .await;
    let content = match content {
        Ok(Some(content)) => content,
        Ok(None) => return Ok(format!(" {text}")),
        Err(error) => {
            if abort_signal.is_some_and(|signal| signal.is_cancelled()) {
                return Err(error);
            }
            return Ok(format!(" {text}"));
        }
    };
    let body = strip_frontmatter(&content);
    let body = body.trim();
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{body}\n</skill>",
        skill.name, skill.file_path, skill.base_dir
    );
    Ok(if args.is_empty() {
        block
    } else {
        format!("{block}\n\n{args}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::adapters::pi::pi_test_support::LocalSandbox;
    use crate::harness::adapters::test_temp_dir::mkdtemp;

    const SKILL_FILE: &str = "---\nname: release-notes\ndescription: Use when writing release notes.\n---\n\n# Release notes\n\nUse three sections.\n";

    fn skill(root: &str) -> PiCommandSkill {
        PiCommandSkill {
            name: "release-notes".into(),
            file_path: format!("{root}/release-notes/SKILL.md"),
            base_dir: format!("{root}/release-notes"),
        }
    }

    #[tokio::test]
    async fn leaves_other_text_unchanged() {
        let temp = mkdtemp("pi-skill-");
        let root = temp.path_string();
        let sandbox = LocalSandbox::new(root.clone());
        for text in [
            "",
            "hello",
            "/skills",
            "see /skill:release-notes",
            " /skill:x",
            "/skill:other",
        ] {
            assert_eq!(
                expand_pi_skill_command(text, &[skill(&root)], &sandbox, None)
                    .await
                    .unwrap(),
                text
            );
        }
    }

    #[tokio::test]
    async fn expands_known_skills_from_the_sandbox() {
        let temp = mkdtemp("pi-skill-");
        let root = temp.path_string();
        std::fs::create_dir_all(format!("{root}/release-notes")).unwrap();
        std::fs::write(format!("{root}/release-notes/SKILL.md"), SKILL_FILE).unwrap();
        let sandbox = LocalSandbox::new(root.clone());
        let block = format!(
            "<skill name=\"release-notes\" location=\"{root}/release-notes/SKILL.md\">\nReferences are relative to {root}/release-notes.\n\n# Release notes\n\nUse three sections.\n</skill>"
        );
        assert_eq!(
            expand_pi_skill_command(
                "/skill:release-notes Draft notes for v2.",
                &[skill(&root)],
                &sandbox,
                None
            )
            .await
            .unwrap(),
            format!("{block}\n\nDraft notes for v2.")
        );
        assert_eq!(
            expand_pi_skill_command("/skill:release-notes", &[skill(&root)], &sandbox, None)
                .await
                .unwrap(),
            block
        );
    }

    #[tokio::test]
    async fn adds_a_leading_space_when_the_file_is_missing() {
        let temp = mkdtemp("pi-skill-");
        let root = temp.path_string();
        let sandbox = LocalSandbox::new(root.clone());
        assert_eq!(
            expand_pi_skill_command("/skill:release-notes go", &[skill(&root)], &sandbox, None)
                .await
                .unwrap(),
            " /skill:release-notes go"
        );
    }

    #[test]
    fn strips_frontmatter() {
        assert_eq!(
            strip_frontmatter(SKILL_FILE),
            "# Release notes\n\nUse three sections."
        );
        assert_eq!(strip_frontmatter("no frontmatter"), "no frontmatter");
        assert_eq!(strip_frontmatter("---\nunclosed"), "---\nunclosed");
    }
}
