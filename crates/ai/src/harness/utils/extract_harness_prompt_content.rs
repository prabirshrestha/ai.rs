//! Port of `extract-harness-prompt-content.ts`: normalize a harness prompt
//! into text or text-and-inline-image parts.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::harness::errors::HarnessError;
use crate::harness::provider::{DataContent, SharedV4FileData};
use crate::harness::sdk::messages::{FilePartData, ImagePartImage, UserContent, UserContentPart};
use crate::harness::spec::harness::HarnessV1Prompt;

/// `HarnessPromptContentPart`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "lowercase",
    rename_all_fields = "camelCase"
)]
pub enum HarnessPromptContentPart {
    Text { text: String },
    Image { data: String, media_type: String },
}

/// `HarnessPromptContent` (`harnessPromptContentSchema`): a string or parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HarnessPromptContent {
    Text(String),
    Parts(Vec<HarnessPromptContentPart>),
}

fn unsupported(description: &str) -> HarnessError {
    HarnessError::capability_unsupported(
        format!(
            "The harness layer does not support {description} in user messages. Pass text parts and inline images as bytes, base64 strings, or data URLs."
        ),
        None,
    )
}

/// `getTopLevelMediaType`.
pub(crate) fn get_top_level_media_type(media_type: &str) -> &str {
    media_type.split_once('/').map_or(media_type, |(t, _)| t)
}

/// `isFullMediaType`.
pub(crate) fn is_full_media_type(media_type: &str) -> bool {
    media_type
        .split_once('/')
        .is_some_and(|(_, sub)| !sub.is_empty() && sub != "*")
}

/// `detectMediaType({ data, topLevelType: 'image' })` for base64 data.
pub(crate) fn detect_image_media_type(base64_data: &str) -> Option<&'static str> {
    // Decode enough of the prefix for the longest signature.
    let prefix: String = base64_data.chars().take(24).collect();
    let usable = prefix.len() / 4 * 4;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&prefix[..usable])
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(base64_data))
        .ok()?;
    const N: Option<u8> = None;
    let s = |b: u8| Some(b);
    let signatures: &[(&str, &[Option<u8>])] = &[
        (
            "image/gif",
            &[s(0x47), s(0x49), s(0x46), s(0x38), s(0x37), s(0x61)],
        ),
        (
            "image/gif",
            &[s(0x47), s(0x49), s(0x46), s(0x38), s(0x39), s(0x61)],
        ),
        ("image/png", &[s(0x89), s(0x50), s(0x4e), s(0x47)]),
        ("image/jpeg", &[s(0xff), s(0xd8)]),
        (
            "image/webp",
            &[
                s(0x52),
                s(0x49),
                s(0x46),
                s(0x46),
                N,
                N,
                N,
                N,
                s(0x57),
                s(0x45),
                s(0x42),
                s(0x50),
            ],
        ),
        (
            "image/bmp",
            &[s(0x42), s(0x4d), N, N, N, N, s(0), s(0), s(0), s(0)],
        ),
        ("image/tiff", &[s(0x49), s(0x49), s(0x2a), s(0)]),
        ("image/tiff", &[s(0x4d), s(0x4d), s(0), s(0x2a)]),
        (
            "image/avif",
            &[
                s(0),
                s(0),
                s(0),
                N,
                s(0x66),
                s(0x74),
                s(0x79),
                s(0x70),
                s(0x61),
                s(0x76),
                s(0x69),
                s(0x66),
            ],
        ),
        (
            "image/heic",
            &[
                s(0),
                s(0),
                s(0),
                N,
                s(0x66),
                s(0x74),
                s(0x79),
                s(0x70),
                s(0x68),
                s(0x65),
                s(0x69),
                s(0x63),
            ],
        ),
    ];
    signatures.iter().find_map(|(media_type, prefix)| {
        (bytes.len() >= prefix.len()
            && prefix
                .iter()
                .enumerate()
                .all(|(i, b)| b.is_none_or(|b| bytes[i] == b)))
        .then_some(*media_type)
    })
}

struct InlineImage {
    data: String,
    media_type: Option<String>,
}

fn has_url_scheme(s: &str) -> bool {
    let Some((scheme, _)) = s.split_once(':') else {
        return false;
    };
    let mut chars = scheme.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

fn data_url_image(url: &str) -> Result<InlineImage, HarnessError> {
    if !url.to_ascii_lowercase().starts_with("data:") {
        return Err(unsupported("remote image URLs"));
    }
    let mut pieces = url.split(',');
    let header = pieces.next().unwrap_or_default();
    let data = pieces.next();
    let Some(data) = data.filter(|_| header.ends_with(";base64")) else {
        return Err(unsupported("data URLs that are not base64 encoded"));
    };
    let media_type = header["data:".len()..]
        .split(';')
        .next()
        .unwrap_or_default();
    Ok(InlineImage {
        data: data.to_string(),
        media_type: (!media_type.is_empty()).then(|| media_type.to_string()),
    })
}

fn bytes_image(bytes: &[u8]) -> InlineImage {
    InlineImage {
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
        media_type: None,
    }
}

fn content_image(data: &DataContent) -> Result<InlineImage, HarnessError> {
    match data {
        DataContent::Bytes(b) => Ok(bytes_image(b)),
        DataContent::String(s) if has_url_scheme(s) => data_url_image(s),
        DataContent::String(s) => Ok(InlineImage {
            data: s.clone(),
            media_type: None,
        }),
    }
}

fn file_data_image(data: &FilePartData) -> Result<InlineImage, HarnessError> {
    match data {
        FilePartData::Content(c) => content_image(c),
        FilePartData::Reference(_) => Err(unsupported("provider references")),
        FilePartData::FileData(f) => match f {
            SharedV4FileData::Data { data } => match data {
                DataContent::String(s) => Ok(InlineImage {
                    data: s.clone(),
                    media_type: None,
                }),
                DataContent::Bytes(b) => Ok(bytes_image(b)),
            },
            SharedV4FileData::Url { url, .. } => data_url_image(url),
            SharedV4FileData::Reference { .. } => Err(unsupported("provider references")),
            SharedV4FileData::Text { .. } => Err(unsupported("text data for images")),
        },
    }
}

fn convert_user_image(
    image: InlineImage,
    media_type: &str,
) -> Result<HarnessPromptContentPart, HarnessError> {
    let resolved = if is_full_media_type(media_type) {
        Some(media_type.to_string())
    } else {
        image
            .media_type
            .clone()
            .or_else(|| detect_image_media_type(&image.data).map(str::to_string))
    };
    let Some(media_type) = resolved else {
        return Err(unsupported("images whose media type cannot be detected"));
    };
    Ok(HarnessPromptContentPart::Image {
        data: image.data,
        media_type,
    })
}

/// `extractHarnessPromptContent({ prompt })`.
pub fn extract_harness_prompt_content(
    prompt: &HarnessV1Prompt,
) -> Result<HarnessPromptContent, HarnessError> {
    let message = match prompt {
        HarnessV1Prompt::Text(s) => return Ok(HarnessPromptContent::Text(s.clone())),
        HarnessV1Prompt::Message(m) => m,
    };
    let parts = match &message.content {
        UserContent::Text(s) => return Ok(HarnessPromptContent::Text(s.clone())),
        UserContent::Parts(parts) => parts,
    };
    if parts.iter().all(|p| matches!(p, UserContentPart::Text(_))) {
        let texts: Vec<&str> = parts
            .iter()
            .filter_map(|p| match p {
                UserContentPart::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        return Ok(HarnessPromptContent::Text(texts.join("\n\n")));
    }
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        out.push(match part {
            UserContentPart::Text(t) => HarnessPromptContentPart::Text {
                text: t.text.clone(),
            },
            UserContentPart::Image(i) => {
                let image = match &i.image {
                    ImagePartImage::Content(c) => content_image(c)?,
                    ImagePartImage::Reference(_) => return Err(unsupported("provider references")),
                };
                convert_user_image(image, i.media_type.as_deref().unwrap_or("image"))?
            }
            UserContentPart::File(f) => {
                if get_top_level_media_type(&f.media_type) != "image" {
                    return Err(unsupported(&format!(
                        "files of media type '{}'",
                        f.media_type
                    )));
                }
                convert_user_image(file_data_image(&f.data)?, &f.media_type)?
            }
        });
    }
    Ok(HarnessPromptContent::Parts(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sdk::messages::{FilePart, ImagePart, TextPart, UserModelMessage};
    use serde_json::json;

    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    fn msg(parts: Vec<UserContentPart>) -> HarnessV1Prompt {
        HarnessV1Prompt::Message(UserModelMessage {
            content: UserContent::Parts(parts),
            provider_options: None,
        })
    }

    fn text(t: &str) -> UserContentPart {
        UserContentPart::Text(TextPart {
            text: t.into(),
            provider_options: None,
        })
    }

    fn file(media_type: &str, data: FilePartData) -> UserContentPart {
        UserContentPart::File(FilePart {
            data,
            filename: None,
            media_type: media_type.into(),
            provider_options: None,
        })
    }

    // extract-harness-prompt-content.test.ts
    #[test]
    fn schema_accepts_normalized_text_and_image_prompts() {
        for v in [
            json!("hello"),
            json!([{ "type": "text", "text": "hello" }]),
            json!([{ "type": "image", "data": "iVBORw0KGgo=", "mediaType": "image/png" }]),
        ] {
            serde_json::from_value::<HarnessPromptContent>(v).unwrap();
        }
    }

    #[test]
    fn schema_rejects_unsupported_prompt_content_parts() {
        assert!(
            serde_json::from_value::<HarnessPromptContent>(
                json!([{ "type": "audio", "data": "audio-data" }])
            )
            .is_err()
        );
    }

    #[test]
    fn preserves_string_prompts_and_string_content() {
        assert_eq!(
            extract_harness_prompt_content(&"hello".into()).unwrap(),
            HarnessPromptContent::Text("hello".into())
        );
        let p = HarnessV1Prompt::Message(UserModelMessage {
            content: UserContent::Text("hello".into()),
            provider_options: None,
        });
        assert_eq!(
            extract_harness_prompt_content(&p).unwrap(),
            HarnessPromptContent::Text("hello".into())
        );
    }

    #[test]
    fn joins_text_only_parts_with_blank_lines() {
        assert_eq!(
            extract_harness_prompt_content(&msg(vec![text("First"), text("Second")])).unwrap(),
            HarnessPromptContent::Text("First\n\nSecond".into())
        );
    }

    #[test]
    fn normalizes_inline_images_and_preserves_content_order() {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(PNG)
            .unwrap();
        let prompt = msg(vec![
            text("Before"),
            file(
                "image/png",
                FilePartData::FileData(SharedV4FileData::Url {
                    url: format!("data:image/png;base64,{PNG}"),
                    original_url: None,
                }),
            ),
            file(
                "image",
                FilePartData::FileData(SharedV4FileData::Data {
                    data: DataContent::Bytes(bytes),
                }),
            ),
            file(
                "image/png",
                FilePartData::Content(DataContent::String(PNG.into())),
            ),
            UserContentPart::Image(ImagePart {
                image: ImagePartImage::Content(DataContent::String(PNG.into())),
                media_type: None,
                provider_options: None,
            }),
            text("After"),
        ]);
        let image = || HarnessPromptContentPart::Image {
            data: PNG.into(),
            media_type: "image/png".into(),
        };
        assert_eq!(
            extract_harness_prompt_content(&prompt).unwrap(),
            HarnessPromptContent::Parts(vec![
                HarnessPromptContentPart::Text {
                    text: "Before".into()
                },
                image(),
                image(),
                image(),
                image(),
                HarnessPromptContentPart::Text {
                    text: "After".into()
                },
            ])
        );
    }

    #[test]
    fn rejects_unsupported_images_with_a_harness_capability_error() {
        let cases = [
            file(
                "image/png",
                FilePartData::FileData(SharedV4FileData::Url {
                    url: "https://example.com/a.png".into(),
                    original_url: None,
                }),
            ),
            file(
                "image/png",
                FilePartData::FileData(SharedV4FileData::Reference {
                    reference: [("test".to_string(), "file_1".to_string())]
                        .into_iter()
                        .collect(),
                }),
            ),
            file(
                "application/pdf",
                FilePartData::FileData(SharedV4FileData::Data {
                    data: DataContent::String(PNG.into()),
                }),
            ),
            file(
                "image/svg+xml",
                FilePartData::Content(DataContent::String(
                    "data:image/svg+xml,%3Csvg%3E%3C/svg%3E".into(),
                )),
            ),
            file(
                "image",
                FilePartData::FileData(SharedV4FileData::Data {
                    data: DataContent::String("aGVsbG8=".into()),
                }),
            ),
        ];
        for part in cases {
            let err = extract_harness_prompt_content(&msg(vec![part])).unwrap_err();
            assert!(matches!(err, HarnessError::CapabilityUnsupported { .. }));
            assert!(err.message().contains("The harness layer does not support"));
        }
    }
}
