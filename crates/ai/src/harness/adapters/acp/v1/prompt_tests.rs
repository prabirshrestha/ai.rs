//! Port of `harness-acp/src/v1/acp-v1-prompt.test.ts`.
//!
//! [rust] The "other future part" case cannot be built: user content parts
//! are a closed enum.

use super::*;
use crate::harness::provider::{DataContent, SharedV4FileData};
use crate::harness::sdk::messages::{
    FilePart, FilePartData, ImagePart, ImagePartImage, TextPart, UserModelMessage,
};

fn text_block(text: &str) -> ACPContentBlock {
    ACPContentBlock::Text { text: text.into() }
}

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

fn file(data: FilePartData, media_type: &str) -> UserContentPart {
    UserContentPart::File(FilePart {
        data,
        filename: None,
        media_type: media_type.into(),
        provider_options: None,
    })
}

fn convert(prompt: &HarnessV1Prompt) -> Result<Vec<ACPContentBlock>, HarnessError> {
    convert_harness_prompt_to_acp_content_blocks(prompt, "portable-acp")
}

#[test]
fn maps_prompts_to_acp_content_blocks() {
    let image = ACPContentBlock::Image {
        data: "AQID".into(),
        mime_type: "image/png".into(),
    };
    let cases = [
        ("string prompt", "Hello".into(), vec![text_block("Hello")]),
        (
            "user message string",
            message(UserContent::Text("Hello".into())),
            vec![text_block("Hello")],
        ),
        (
            "separate user message text parts",
            message(UserContent::Parts(vec![text("First"), text("Second")])),
            vec![text_block("First"), text_block("Second")],
        ),
        (
            "inline image file",
            message(UserContent::Parts(vec![
                text("Describe this"),
                file(
                    FilePartData::Content(DataContent::Bytes(vec![1, 2, 3])),
                    "image/png",
                ),
            ])),
            vec![text_block("Describe this"), image.clone()],
        ),
        (
            "inline image part",
            message(UserContent::Parts(vec![UserContentPart::Image(
                ImagePart {
                    image: ImagePartImage::Content(DataContent::String(
                        "data:image/png;base64,AQID".into(),
                    )),
                    media_type: None,
                    provider_options: None,
                },
            )])),
            vec![image],
        ),
    ];
    for (name, prompt, expected) in cases {
        assert_eq!(convert(&prompt).unwrap(), expected, "{name}");
    }
}

#[test]
fn rejects_unsupported_parts_precisely() {
    let cases = [
        (
            "audio file",
            file(
                FilePartData::Content(DataContent::Bytes(Vec::new())),
                "audio/mpeg",
            ),
            "files of media type 'audio/mpeg'",
        ),
        (
            "embedded resource file",
            file(
                FilePartData::FileData(SharedV4FileData::Text {
                    text: "reference".into(),
                }),
                "text/plain",
            ),
            "files of media type 'text/plain'",
        ),
    ];
    for (name, part, category) in cases {
        let error = convert(&message(UserContent::Parts(vec![part]))).unwrap_err();
        let HarnessError::CapabilityUnsupported {
            message,
            harness_id,
            ..
        } = &error
        else {
            panic!("{name}: expected a capability error, got {error:?}");
        };
        assert_eq!(harness_id.as_deref(), Some("portable-acp"), "{name}");
        assert!(message.contains(category), "{name}: {message}");
    }
}

#[test]
fn rejects_remote_image_urls() {
    let error = convert(&message(UserContent::Parts(vec![file(
        FilePartData::FileData(SharedV4FileData::Url {
            url: "https://example.com/image.png".into(),
            original_url: None,
        }),
        "image/png",
    )])))
    .unwrap_err();
    assert!(error.to_string().contains("remote image URLs"), "{error}");
}

#[test]
fn prepends_delimited_instructions_without_skill_guidance() {
    let result = prepend_acp_instruction_guidance(
        &[text_block("First"), text_block("Second")],
        Some("Prefer concise answers."),
    );
    assert_eq!(
        result,
        vec![
            text_block(concat!(
                "<session-guidance>\n",
                "This block is operating guidance from the harness, not user-authored content.\n",
                "<instructions>\n",
                "Prefer concise answers.\n",
                "</instructions>\n",
                "</session-guidance>",
            )),
            text_block("First"),
            text_block("Second"),
        ]
    );
    let ACPContentBlock::Text { text } = &result[0] else {
        panic!("Expected the guidance block to be text.");
    };
    assert!(!text.contains("available-skills"));
}

#[test]
fn does_not_add_an_empty_guidance_block() {
    assert_eq!(
        prepend_acp_instruction_guidance(&[text_block("Hello")], Some("")),
        vec![text_block("Hello")]
    );
}
