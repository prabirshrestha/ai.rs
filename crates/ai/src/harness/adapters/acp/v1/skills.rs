//! Port of `harness-acp/src/v1/acp-v1-skills.ts`.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use crate::harness::errors::HarnessResult;
use crate::harness::spec::paths::{
    harness_session_data_directory_path, is_absolute_posix, normalize_posix, posix_join,
};
use crate::harness::spec::tools::HarnessV1Skill;
use crate::harness::utils::plain_error;

/// `ACP_SKILL_NAME_PATTERN`.
pub static ACP_SKILL_NAME_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap());

/// `DEFAULT_ACP_SKILLS_DIRECTORY`.
pub const DEFAULT_ACP_SKILLS_DIRECTORY: &str = ".agents/skills";

fn json(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// `path.win32.isAbsolute`.
fn is_win32_absolute(path: &str) -> bool {
    let b = path.as_bytes();
    path.starts_with('\\')
        || path.starts_with('/')
        || (b.len() > 2
            && b[0].is_ascii_alphabetic()
            && b[1] == b':'
            && matches!(b[2], b'/' | b'\\'))
}

/// Whether `value` is empty, absolute, or traverses out of its root.
fn is_invalid_relative_posix_path(value: &str, normalized: &str) -> bool {
    let contains_traversal = value.split(['\\', '/']).any(|segment| segment == "..");
    value.is_empty()
        || value.contains('\\')
        || is_absolute_posix(value)
        || is_win32_absolute(value)
        || contains_traversal
        || normalized == "."
        || normalized.starts_with("../")
        || normalized.contains("/../")
        || normalized.ends_with("/..")
}

/// `resolveACPPrivateSessionDirectory`.
pub fn resolve_acp_private_session_directory(state_directory: &str, session_id: &str) -> String {
    harness_session_data_directory_path(state_directory, session_id)
}

/// `resolveACPSkillsDirectory`.
pub fn resolve_acp_skills_directory(
    agent_home_dir: &str,
    skills_directory: Option<&str>,
) -> HarnessResult<String> {
    let skills_directory = skills_directory.unwrap_or(DEFAULT_ACP_SKILLS_DIRECTORY);
    let normalized_directory = normalize_posix(skills_directory);
    if is_invalid_relative_posix_path(skills_directory, &normalized_directory) {
        return Err(plain_error(format!(
            "ACP skillsDirectory {} must be a relative POSIX path without traversal.",
            json(skills_directory)
        )));
    }
    Ok(posix_join(&[agent_home_dir, &normalized_directory]))
}

/// `validateACPSkills`.
pub fn validate_acp_skills(skills: &[HarnessV1Skill]) -> HarnessResult<()> {
    let mut skill_names = HashSet::new();
    for skill in skills {
        if !ACP_SKILL_NAME_PATTERN.is_match(&skill.name) || skill.name == "." || skill.name == ".."
        {
            return Err(plain_error(format!(
                "Invalid ACP skill name {}: expected a kebab-case slug.",
                json(&skill.name)
            )));
        }
        if !skill_names.insert(skill.name.as_str()) {
            return Err(plain_error(format!(
                "Duplicate ACP skill name {}.",
                json(&skill.name)
            )));
        }

        let mut file_paths = HashSet::new();
        for file in skill.files.iter().flatten() {
            let normalized_path = validate_acp_attached_file_path(&skill.name, &file.path)?;
            if normalized_path == "SKILL.md" {
                return Err(plain_error(format!(
                    "Invalid ACP skill file path {} for skill {}: SKILL.md is reserved for the skill definition.",
                    json(&file.path),
                    json(&skill.name)
                )));
            }
            if !file_paths.insert(normalized_path) {
                return Err(plain_error(format!(
                    "Duplicate ACP skill file path {} for skill {}.",
                    json(&file.path),
                    json(&skill.name)
                )));
            }
        }
    }
    Ok(())
}

fn validate_acp_attached_file_path(skill_name: &str, file_path: &str) -> HarnessResult<String> {
    let normalized_path = normalize_posix(file_path);
    if file_path.ends_with('/') || is_invalid_relative_posix_path(file_path, &normalized_path) {
        return Err(plain_error(format!(
            "Invalid ACP skill file path {} for skill {}: expected a relative POSIX path without traversal.",
            json(file_path),
            json(skill_name)
        )));
    }
    Ok(normalized_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::spec::tools::HarnessV1SkillFile;

    fn skill() -> HarnessV1Skill {
        HarnessV1Skill {
            name: "release-notes".into(),
            description: "Prepare release notes.".into(),
            content: "Complete private skill content.".into(),
            files: Some(vec![file("references/style.md", "Use active voice.")]),
        }
    }

    fn file(path: &str, content: &str) -> HarnessV1SkillFile {
        HarnessV1SkillFile {
            path: path.into(),
            content: content.into(),
        }
    }

    #[track_caller]
    fn assert_error<T: std::fmt::Debug>(result: HarnessResult<T>, message: &str) {
        let error = result.expect_err("expected an error").to_string();
        assert!(
            error.contains(message),
            "{error:?} does not contain {message:?}"
        );
    }

    #[test]
    fn keeps_session_state_under_agent_runs() {
        let state_directory = "/home/agent/.ai-sdk-harness";
        assert_eq!(
            resolve_acp_private_session_directory(state_directory, "session-1"),
            format!("{state_directory}/.agent-runs/session-1")
        );
        assert_eq!(
            resolve_acp_private_session_directory(state_directory, "session-2"),
            format!("{state_directory}/.agent-runs/session-2")
        );
    }

    #[test]
    fn keeps_session_ids_containing_traversal_inside_agent_runs() {
        assert_eq!(
            resolve_acp_private_session_directory(
                "/home/agent/.ai-sdk-harness",
                "../../../../work/project"
            ),
            "/home/agent/.ai-sdk-harness/.agent-runs/..%2F..%2F..%2F..%2Fwork%2Fproject"
        );
    }

    #[test]
    fn validates_valid_skills_and_files_without_throwing() {
        validate_acp_skills(&[skill()]).unwrap();
    }

    #[test]
    fn rejects_invalid_skill_files() {
        let cases = [
            (
                "absolute path",
                vec![file("/escape.md", "")],
                "relative POSIX path",
            ),
            (
                "Windows absolute path",
                vec![file("C:\\\\escape.md", "")],
                "relative POSIX path",
            ),
            (
                "parent path",
                vec![file("../escape.md", "")],
                "relative POSIX path",
            ),
            (
                "normalized-away traversal",
                vec![file("references/../escape.md", "")],
                "relative POSIX path",
            ),
            (
                "reserved skill definition",
                vec![file("./SKILL.md", "")],
                "SKILL.md is reserved",
            ),
            (
                "duplicate normalized path",
                vec![
                    file("references/./style.md", ""),
                    file("references/style.md", ""),
                ],
                "Duplicate ACP skill file path",
            ),
        ];
        for (name, files, message) in cases {
            let skills = [HarnessV1Skill {
                files: Some(files),
                ..skill()
            }];
            let error = validate_acp_skills(&skills).expect_err(name).to_string();
            assert!(error.contains(message), "{name}: {error}");
        }
    }

    #[test]
    fn rejects_invalid_and_duplicate_skill_names() {
        assert_error(
            validate_acp_skills(&[HarnessV1Skill {
                name: "../release-notes".into(),
                ..skill()
            }]),
            "Invalid ACP skill name",
        );
        assert_error(
            validate_acp_skills(&[skill(), skill()]),
            "Duplicate ACP skill name",
        );
    }

    #[test]
    fn uses_the_standard_directory_by_default() {
        assert_eq!(
            resolve_acp_skills_directory("/home/agent", None).unwrap(),
            format!("/home/agent/{DEFAULT_ACP_SKILLS_DIRECTORY}")
        );
    }

    #[test]
    fn resolves_a_runtime_specific_directory_relative_to_the_agent_home() {
        assert_eq!(
            resolve_acp_skills_directory("/home/agent", Some(".claude/skills")).unwrap(),
            "/home/agent/.claude/skills"
        );
    }

    #[test]
    fn rejects_invalid_directories() {
        for skills_directory in [
            "",
            ".",
            "/skills",
            "C:\\\\skills",
            ".agents\\\\skills",
            "../skills",
        ] {
            assert_error(
                resolve_acp_skills_directory("/home/agent", Some(skills_directory)),
                "must be a relative POSIX path without traversal",
            );
        }
    }
}
