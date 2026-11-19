//! Port of `normalize-harness-tool-model-output.ts` and
//! `convert-harness-tool-model-output.ts`.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use super::extract_harness_prompt_content::{
    detect_image_media_type, get_top_level_media_type, is_full_media_type,
};
use crate::harness::errors::HarnessError;
use crate::harness::provider::{DataContent, SharedV4FileData};
use crate::harness::sdk::messages::{ToolResultContentPart, ToolResultOutput};

fn part_type(part: &ToolResultContentPart) -> &'static str {
    match part {
        ToolResultContentPart::Text { .. } => "text",
        ToolResultContentPart::File { .. } => "file",
        ToolResultContentPart::FileData { .. } => "file-data",
        ToolResultContentPart::FileUrl { .. } => "file-url",
        ToolResultContentPart::FileId { .. } => "file-id",
        ToolResultContentPart::FileReference { .. } => "file-reference",
        ToolResultContentPart::ImageData { .. } => "image-data",
        ToolResultContentPart::ImageUrl { .. } => "image-url",
        ToolResultContentPart::ImageFileId { .. } => "image-file-id",
        ToolResultContentPart::ImageFileReference { .. } => "image-file-reference",
        ToolResultContentPart::Custom { .. } => "custom",
    }
}

fn file_data_type(data: &SharedV4FileData) -> &'static str {
    match data {
        SharedV4FileData::Data { .. } => "data",
        SharedV4FileData::Url { .. } => "url",
        SharedV4FileData::Reference { .. } => "reference",
        SharedV4FileData::Text { .. } => "text",
    }
}

fn normalize_part(part: &ToolResultContentPart) -> Result<ToolResultContentPart, String> {
    if let ToolResultContentPart::Text { .. } = part {
        return Ok(part.clone());
    }
    // Legacy `image-data` / `file-data` become `file` with inline data.
    let (data, media_type, filename, provider_options) = match part {
        ToolResultContentPart::ImageData {
            data,
            media_type,
            provider_options,
        } => (
            SharedV4FileData::Data {
                data: DataContent::String(data.clone()),
            },
            media_type.clone(),
            None,
            provider_options.clone(),
        ),
        ToolResultContentPart::FileData {
            data,
            media_type,
            filename,
            provider_options,
        } => (
            SharedV4FileData::Data {
                data: DataContent::String(data.clone()),
            },
            media_type.clone(),
            filename.clone(),
            provider_options.clone(),
        ),
        ToolResultContentPart::File {
            data,
            media_type,
            filename,
            provider_options,
        } => (
            data.clone(),
            media_type.clone(),
            filename.clone(),
            provider_options.clone(),
        ),
        other => return Err(format!("Unsupported content '{}'.", part_type(other))),
    };
    let inline = match &data {
        SharedV4FileData::Data { data } if get_top_level_media_type(&media_type) == "image" => data,
        _ => {
            return Err(format!(
                "Unsupported content '{}' ({media_type}, {}).",
                part_type(part),
                file_data_type(&data)
            ));
        }
    };
    let data = match inline {
        DataContent::String(s) => s.clone(),
        DataContent::Bytes(b) => base64::engine::general_purpose::STANDARD.encode(b),
    };
    let media_type = if is_full_media_type(&media_type) {
        media_type
    } else {
        detect_image_media_type(&data)
            .ok_or_else(|| "Cannot detect the inline image media type.".to_string())?
            .to_string()
    };
    Ok(ToolResultContentPart::File {
        data: SharedV4FileData::Data {
            data: DataContent::String(data),
        },
        media_type,
        filename,
        provider_options,
    })
}

/// `normalizeHarnessToolModelOutput({ output })`: keep text, turn every
/// inline image into a `file` part with base64 data and a full media type.
/// Anything else is `HarnessCapabilityUnsupportedError`.
pub fn normalize_harness_tool_model_output(
    output: &ToolResultOutput,
) -> Result<ToolResultOutput, HarnessError> {
    let ToolResultOutput::Content { value } = output else {
        return Ok(output.clone());
    };
    value
        .iter()
        .map(normalize_part)
        .collect::<Result<Vec<_>, _>>()
        .map(|value| ToolResultOutput::Content { value })
        .map_err(|cause| {
            HarnessError::capability_unsupported(
                format!(
                    "Harnesses support only text and inline images in tool model output. {cause}"
                ),
                None,
            )
            .caused_by(cause)
        })
}

/// `HarnessToolModelOutputContent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "lowercase",
    rename_all_fields = "camelCase"
)]
pub enum HarnessToolModelOutputContent {
    Text { text: String },
    Image { data: String, media_type: String },
}

/// Result of [`convert_harness_tool_model_output`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedHarnessToolModelOutput {
    pub content: Vec<HarnessToolModelOutputContent>,
    pub is_error: bool,
}

/// `convertHarnessToolModelOutput({ output })`.
///
/// [rust] TS raises the bridge-side `HarnessBridgeCapabilityUnsupportedError`,
/// which the host translates to `HarnessCapabilityUnsupportedError`. Rust
/// only runs on the host, so it returns the translated error directly.
pub fn convert_harness_tool_model_output(
    output: &ToolResultOutput,
) -> Result<ConvertedHarnessToolModelOutput, HarnessError> {
    let text = |text: String, is_error: bool| ConvertedHarnessToolModelOutput {
        content: vec![HarnessToolModelOutputContent::Text { text }],
        is_error,
    };
    Ok(match output {
        ToolResultOutput::Text { value, .. } => text(value.clone(), false),
        ToolResultOutput::ErrorText { value, .. } => text(value.clone(), true),
        ToolResultOutput::Json { value, .. } => text(value.to_string(), false),
        ToolResultOutput::ErrorJson { value, .. } => text(value.to_string(), true),
        ToolResultOutput::ExecutionDenied { reason, .. } => text(
            reason.clone().unwrap_or_else(|| "Tool execution denied.".into()),
            true,
        ),
        ToolResultOutput::Content { value } => ConvertedHarnessToolModelOutput {
            content: value
                .iter()
                .map(|part| match part {
                    ToolResultContentPart::Text { text, .. } => {
                        Ok(HarnessToolModelOutputContent::Text { text: text.clone() })
                    }
                    ToolResultContentPart::File {
                        data: SharedV4FileData::Data {
                            data: DataContent::String(data),
                        },
                        media_type,
                        ..
                    } if media_type.starts_with("image/") && media_type != "image/*" => {
                        Ok(HarnessToolModelOutputContent::Image {
                            data: data.clone(),
                            media_type: media_type.clone(),
                        })
                    }
                    other => Err(HarnessError::capability_unsupported(
                        format!(
                            "Harnesses support only text and inline images in tool model output; unsupported content '{}'.",
                            part_type(other)
                        ),
                        None,
                    )),
                })
                .collect::<Result<_, _>>()?,
            is_error: false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BYTES: [u8; 4] = [137, 80, 78, 71];
    const BASE64: &str = "iVBORw==";

    fn parse(v: serde_json::Value) -> ToolResultOutput {
        serde_json::from_value(v).unwrap()
    }

    // normalize-harness-tool-model-output.test.ts
    #[test]
    fn normalizes_inline_data() {
        for data in [
            DataContent::String(BASE64.into()),
            DataContent::Bytes(BYTES.to_vec()),
        ] {
            let input = ToolResultOutput::Content {
                value: vec![
                    ToolResultContentPart::Text {
                        text: "before".into(),
                        provider_options: None,
                    },
                    ToolResultContentPart::File {
                        data: SharedV4FileData::Data { data: data.clone() },
                        media_type: "image".into(),
                        filename: None,
                        provider_options: Some(
                            serde_json::from_value(json!({ "test": { "detail": "high" } }))
                                .unwrap(),
                        ),
                    },
                    ToolResultContentPart::Text {
                        text: "after".into(),
                        provider_options: None,
                    },
                ],
            };
            let out = normalize_harness_tool_model_output(&input).unwrap();
            assert_eq!(
                serde_json::to_value(&out).unwrap(),
                json!({ "type": "content", "value": [
                    { "type": "text", "text": "before" },
                    { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": BASE64 },
                      "providerOptions": { "test": { "detail": "high" } } },
                    { "type": "text", "text": "after" }
                ] })
            );
            assert!(!serde_json::to_string(&out).unwrap().contains("137"));
        }
    }

    #[test]
    fn normalizes_legacy_image_data_and_file_data() {
        for ty in ["image-data", "file-data"] {
            let out = normalize_harness_tool_model_output(&parse(json!({
                "type": "content", "value": [{ "type": ty, "mediaType": "image/*", "data": BASE64 }]
            })))
            .unwrap();
            assert_eq!(
                serde_json::to_value(out).unwrap(),
                json!({ "type": "content", "value": [
                    { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": BASE64 } }
                ] })
            );
        }
    }

    #[test]
    fn rejects_unsupported_parts() {
        for part in [
            json!({ "type": "image-url", "url": "https://example.com/image.png" }),
            json!({ "type": "image-file-id", "fileId": "image-id" }),
            json!({ "type": "file", "mediaType": "image/png", "data": { "type": "url", "url": "https://example.com/image.png" } }),
            json!({ "type": "file", "mediaType": "image/png", "data": { "type": "reference", "reference": { "test": "image-id" } } }),
            json!({ "type": "file", "mediaType": "audio/wav", "data": { "type": "data", "data": BASE64 } }),
            json!({ "type": "file", "mediaType": "image", "data": { "type": "data", "data": "AAAA" } }),
            json!({ "type": "custom", "providerOptions": {} }),
        ] {
            let err = normalize_harness_tool_model_output(&parse(
                json!({ "type": "content", "value": [part] }),
            ))
            .unwrap_err();
            assert!(matches!(err, HarnessError::CapabilityUnsupported { .. }));
        }
    }

    #[test]
    fn does_not_mutate_the_callback_output() {
        let input = ToolResultOutput::Content {
            value: vec![ToolResultContentPart::File {
                data: SharedV4FileData::Data {
                    data: DataContent::Bytes(BYTES.to_vec()),
                },
                media_type: "image".into(),
                filename: None,
                provider_options: None,
            }],
        };
        let before = input.clone();
        normalize_harness_tool_model_output(&input).unwrap();
        assert_eq!(input, before);
    }

    // convert-harness-tool-model-output.test.ts
    #[test]
    fn converts_scalar_outputs() {
        for (output, text, is_error) in [
            (json!({ "type": "text", "value": "hello" }), "hello", false),
            (
                json!({ "type": "json", "value": { "ok": true } }),
                "{\"ok\":true}",
                false,
            ),
            (
                json!({ "type": "error-text", "value": "failed" }),
                "failed",
                true,
            ),
            (
                json!({ "type": "error-json", "value": { "error": "failed" } }),
                "{\"error\":\"failed\"}",
                true,
            ),
            (
                json!({ "type": "execution-denied", "reason": "no permission" }),
                "no permission",
                true,
            ),
            (
                json!({ "type": "execution-denied" }),
                "Tool execution denied.",
                true,
            ),
        ] {
            assert_eq!(
                convert_harness_tool_model_output(&parse(output)).unwrap(),
                ConvertedHarnessToolModelOutput {
                    content: vec![HarnessToolModelOutputContent::Text { text: text.into() }],
                    is_error,
                }
            );
        }
    }

    #[test]
    fn preserves_ordered_text_and_multiple_images() {
        let out = convert_harness_tool_model_output(&parse(json!({ "type": "content", "value": [
            { "type": "text", "text": "before" },
            { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": "png" } },
            { "type": "text", "text": "after" },
            { "type": "file", "mediaType": "image/jpeg", "data": { "type": "data", "data": "jpeg" } }
        ] })))
        .unwrap();
        assert_eq!(
            serde_json::to_value(&out.content).unwrap(),
            json!([
                { "type": "text", "text": "before" },
                { "type": "image", "mediaType": "image/png", "data": "png" },
                { "type": "text", "text": "after" },
                { "type": "image", "mediaType": "image/jpeg", "data": "jpeg" }
            ])
        );
        assert!(!out.is_error);
    }
}
