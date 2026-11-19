//! Port of `harness-pi/src/pi-utils.ts`.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::harness::errors::HarnessError;
use crate::harness::spec::harness::HarnessV1Prompt;
use crate::harness::spec::tools::HarnessV1Skill;
use crate::harness::utils::{
    HarnessPromptContent, HarnessPromptContentPart, extract_harness_prompt_content,
};

const HARNESS_ID: &str = "pi";

/// One image of [`PiPromptContent`]: Pi's `ImageContent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "image", rename_all = "camelCase")]
pub struct PiPromptImage {
    pub data: String,
    pub mime_type: String,
}

/// `PiPromptContent`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PiPromptContent {
    pub text: String,
    pub images: Vec<PiPromptImage>,
}

/// `extractPiPromptContent({ prompt })`.
pub fn extract_pi_prompt_content(prompt: &HarnessV1Prompt) -> Result<PiPromptContent, HarnessError> {
    let parts = match extract_harness_prompt_content(prompt)? {
        HarnessPromptContent::Text(text) => {
            return Ok(PiPromptContent {
                text,
                images: Vec::new(),
            });
        }
        HarnessPromptContent::Parts(parts) => parts,
    };
    let mut text_parts = Vec::new();
    let mut images = Vec::new();
    for part in parts {
        match part {
            HarnessPromptContentPart::Text { text } => {
                if !images.is_empty() {
                    return Err(HarnessError::capability_unsupported(
                        "Harness 'pi' cannot preserve text that follows an inline image in a user message.",
                        Some(HARNESS_ID.into()),
                    ));
                }
                text_parts.push(text);
            }
            HarnessPromptContentPart::Image { data, media_type } => images.push(PiPromptImage {
                data,
                mime_type: media_type,
            }),
        }
    }
    Ok(PiPromptContent {
        text: text_parts.join("\n\n"),
        images,
    })
}

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
        return Err(HarnessError::new(format!("Invalid Pi {label} name: {name}")));
    }
    Ok(name)
}

/// `renderPiSkillFile(skill)`: frontmatter for `.pi/skills/<name>/SKILL.md`.
pub fn render_pi_skill_file(skill: &HarnessV1Skill) -> String {
    format!(
        "---\nname: {}\ndescription: {}\n---\n\n{}",
        skill.name, skill.description, skill.content
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::harness::provider::DataContent;
    use crate::harness::sdk::messages::{
        FilePart, FilePartData, ImagePart, ImagePartImage, TextPart, UserContent,
        UserContentPart, UserModelMessage,
    };

    fn message(content: UserContent) -> HarnessV1Prompt {
        HarnessV1Prompt::Message(UserModelMessage {
            content,
            provider_options: None,
        })
    }

    fn text(text: &str) -> UserContentPart {
        UserContentPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })
    }

    fn image(bytes: Vec<u8>) -> UserContentPart {
        UserContentPart::Image(ImagePart {
            image: ImagePartImage::Content(DataContent::Bytes(bytes)),
            media_type: Some("image/png".into()),
            provider_options: None,
        })
    }

    #[test]
    fn frame_instructions_wraps_instructions_as_system_guidance_and_fences_the_user_text() {
        let framed = frame_instructions("Use turbo build.", "do the thing");
        assert!(framed.contains("<session-instructions>"));
        assert!(framed.contains("Use turbo build."));
        assert!(framed.contains("</session-instructions>"));
        assert!(framed.contains("<user-message>\ndo the thing\n</user-message>"));
        assert!(framed.to_lowercase().contains("not a message from the user"));
        assert!(
            framed.find("<session-instructions>").unwrap()
                < framed.find("<user-message>").unwrap()
        );
    }

    #[test]
    fn extract_returns_the_string_prompt_as_text() {
        assert_eq!(
            extract_pi_prompt_content(&"hello there".into()).unwrap(),
            PiPromptContent {
                text: "hello there".into(),
                images: vec![]
            }
        );
    }

    #[test]
    fn extract_joins_multiple_text_parts_with_double_newlines() {
        let prompt = message(UserContent::Parts(vec![text("first"), text("second")]));
        assert_eq!(
            extract_pi_prompt_content(&prompt).unwrap(),
            PiPromptContent {
                text: "first\n\nsecond".into(),
                images: vec![]
            }
        );
    }

    #[test]
    fn extract_handles_a_single_string_content() {
        let prompt = message(UserContent::Text("just text".into()));
        assert_eq!(extract_pi_prompt_content(&prompt).unwrap().text, "just text");
    }

    #[test]
    fn extract_extracts_inline_image_data() {
        let prompt = message(UserContent::Parts(vec![
            text("Describe this."),
            image(vec![1, 2, 3]),
        ]));
        assert_eq!(
            extract_pi_prompt_content(&prompt).unwrap(),
            PiPromptContent {
                text: "Describe this.".into(),
                images: vec![PiPromptImage {
                    data: "AQID".into(),
                    mime_type: "image/png".into()
                }]
            }
        );
        assert_eq!(
            serde_json::to_value(&PiPromptImage {
                data: "AQID".into(),
                mime_type: "image/png".into()
            })
            .unwrap(),
            json!({ "type": "image", "data": "AQID", "mimeType": "image/png" })
        );
    }

    #[test]
    fn extract_rejects_text_parts_after_an_inline_image() {
        let prompt = message(UserContent::Parts(vec![image(vec![1, 2]), text("after")]));
        let error = extract_pi_prompt_content(&prompt).unwrap_err();
        assert!(HarnessError::is_capability_unsupported(&error));
    }

    #[test]
    fn extract_throws_for_unsupported_user_content_parts() {
        let prompt = message(UserContent::Parts(vec![UserContentPart::File(FilePart {
            data: FilePartData::Content(DataContent::Bytes(vec![1, 2])),
            filename: None,
            media_type: "text/plain".into(),
            provider_options: None,
        })]));
        let error = extract_pi_prompt_content(&prompt).unwrap_err();
        assert!(HarnessError::is_capability_unsupported(&error));
    }

    #[test]
    fn serialize_tool_output_returns_strings_unchanged() {
        assert_eq!(serialize_tool_output(Some(&json!("hello"))), "hello");
    }

    #[test]
    fn serialize_tool_output_json_stringifies_objects() {
        assert_eq!(serialize_tool_output(Some(&json!({ "a": 1 }))), r#"{"a":1}"#);
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
        let rendered = render_pi_skill_file(&HarnessV1Skill {
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
