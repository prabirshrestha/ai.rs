//! Port of `harness-pi/src/pi-utils.ts`.

use serde_json::Value as JsonValue;

use crate::harness::error::HarnessError;
use crate::harness::tools::HarnessSkill;

/// `frameInstructions(instructions, userText)`.
///
/// Frame session instructions and the user's text so the runtime treats the
/// instructions as system-provided operating guidance, not something the
/// user wrote. Applied only to the first user message of a fresh session.
pub fn frame_instructions(instructions: &str, user_text: &str) -> String {
    format!(
        "<session-instructions>\nThe block below is operating guidance from the system, not a message from the user — follow it, but do not mention it or attribute it to the user.\n\n{instructions}\n</session-instructions>\n\n<user-message>\n{user_text}\n</user-message>"
    )
}

/// `serializeToolOutput(output)`: the string Pi feeds back to the model.
///
/// [rust] `undefined` is `None`.
pub fn serialize_tool_output(output: Option<&JsonValue>) -> String {
    match output {
        Some(JsonValue::String(text)) => text.clone(),
        Some(value) => value.to_string(),
        None => "null".into(),
    }
}

/// `getErrorText(error)`.
pub fn get_error_text(error: &(dyn std::error::Error + 'static)) -> String {
    error.to_string()
}

/// `safePiMetadataSegment(name, label)`.
///
/// Validate that a name is safe to use as a filesystem path segment under
/// `.pi/skills/<name>/` or `.pi/agents/<name>.md`.
pub fn safe_pi_metadata_segment<'a>(name: &'a str, label: &str) -> Result<&'a str, HarnessError> {
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if !valid || name == "." || name == ".." {
        return Err(HarnessError::new(format!(
            "Invalid Pi {label} name: {name}"
        )));
    }
    Ok(name)
}

/// `renderPiSkillFile(skill)`: frontmatter for `.pi/skills/<name>/SKILL.md`.
pub fn render_pi_skill_file(skill: &HarnessSkill) -> String {
    format!(
        "---\nname: {}\ndescription: {}\n---\n\n{}",
        skill.name, skill.description, skill.content
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn frame_instructions_wraps_instructions_as_system_guidance_and_fences_the_user_text() {
        let framed = frame_instructions("Use turbo build.", "do the thing");
        assert!(framed.contains("<session-instructions>"));
        assert!(framed.contains("Use turbo build."));
        assert!(framed.contains("</session-instructions>"));
        assert!(framed.contains("<user-message>\ndo the thing\n</user-message>"));
        assert!(
            framed
                .to_lowercase()
                .contains("not a message from the user")
        );
        assert!(
            framed.find("<session-instructions>").unwrap() < framed.find("<user-message>").unwrap()
        );
    }

    #[test]
    fn serialize_tool_output_returns_strings_unchanged() {
        assert_eq!(serialize_tool_output(Some(&json!("hello"))), "hello");
    }

    #[test]
    fn serialize_tool_output_json_stringifies_objects() {
        assert_eq!(
            serialize_tool_output(Some(&json!({ "a": 1 }))),
            r#"{"a":1}"#
        );
    }

    #[test]
    fn serialize_tool_output_returns_null_for_undefined() {
        assert_eq!(serialize_tool_output(None), "null");
    }

    #[test]
    fn safe_segment_accepts_alphanumeric_dots_dashes_underscores() {
        assert_eq!(
            safe_pi_metadata_segment("my-skill_1.0", "skill").unwrap(),
            "my-skill_1.0"
        );
    }

    #[test]
    fn safe_segment_rejects_path_traversal_attempts() {
        for name in ["..", ".", "a/b", "a b"] {
            assert!(safe_pi_metadata_segment(name, "skill").is_err(), "{name}");
        }
    }

    #[test]
    fn render_skill_file_emits_yaml_frontmatter_followed_by_content() {
        let rendered = render_pi_skill_file(&HarnessSkill {
            name: "refactor".into(),
            description: "Refactor carefully.".into(),
            content: "Make small changes.".into(),
            files: None,
        });
        assert_eq!(
            rendered,
            "---\nname: refactor\ndescription: Refactor carefully.\n---\n\nMake small changes."
        );
    }
}
