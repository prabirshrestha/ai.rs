//! Port of `harness-acp/src/v1/acp-v1-prompt.ts`.

use serde::{Deserialize, Serialize};

use crate::harness::errors::HarnessError;
use crate::harness::sdk::messages::{UserContent, UserContentPart};
use crate::harness::spec::harness::HarnessV1Prompt;
use crate::harness::utils::{
    HarnessPromptContent, HarnessPromptContentPart, extract_harness_prompt_content,
};

/// `ACPContentBlock` (`acpContentBlockSchema`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "lowercase",
    rename_all_fields = "camelCase"
)]
pub enum ACPContentBlock {
    Text { text: String },
    Image { data: String, mime_type: String },
}

/// `convertHarnessPromptToACPContentBlocks`.
pub fn convert_harness_prompt_to_acp_content_blocks(
    prompt: &HarnessV1Prompt,
    harness_id: &str,
) -> Result<Vec<ACPContentBlock>, HarnessError> {
    let message = match prompt {
        HarnessV1Prompt::Text(text) => {
            return Ok(vec![ACPContentBlock::Text { text: text.clone() }]);
        }
        HarnessV1Prompt::Message(message) => message,
    };
    let parts = match &message.content {
        UserContent::Text(text) => return Ok(vec![ACPContentBlock::Text { text: text.clone() }]),
        UserContent::Parts(parts) => parts,
    };
    if parts
        .iter()
        .all(|part| matches!(part, UserContentPart::Text(_)))
    {
        return Ok(parts
            .iter()
            .filter_map(|part| match part {
                UserContentPart::Text(part) => Some(ACPContentBlock::Text {
                    text: part.text.clone(),
                }),
                _ => None,
            })
            .collect());
    }

    match extract_harness_prompt_content(prompt) {
        Ok(HarnessPromptContent::Parts(content)) => Ok(content
            .into_iter()
            .map(|part| match part {
                HarnessPromptContentPart::Text { text } => ACPContentBlock::Text { text },
                HarnessPromptContentPart::Image { data, media_type } => ACPContentBlock::Image {
                    data,
                    mime_type: media_type,
                },
            })
            .collect()),
        // Unreachable: mixed parts never extract to a string.
        Ok(HarnessPromptContent::Text(text)) => Ok(vec![ACPContentBlock::Text { text }]),
        Err(cause @ HarnessError::CapabilityUnsupported { .. }) => {
            Err(HarnessError::CapabilityUnsupported {
                message: cause.message().to_string(),
                harness_id: Some(harness_id.to_string()),
                cause: Some(Box::new(cause)),
            })
        }
        Err(error) => Err(error),
    }
}

/// `prependACPInstructionGuidance`.
pub fn prepend_acp_instruction_guidance(
    prompt: &[ACPContentBlock],
    instructions: Option<&str>,
) -> Vec<ACPContentBlock> {
    let Some(instructions) = instructions.filter(|instructions| !instructions.is_empty()) else {
        return prompt.to_vec();
    };
    let lines = [
        "<session-guidance>",
        "This block is operating guidance from the harness, not user-authored content.",
        "<instructions>",
        instructions,
        "</instructions>",
        "</session-guidance>",
    ];
    let mut blocks = vec![ACPContentBlock::Text {
        text: lines.join("\n"),
    }];
    blocks.extend_from_slice(prompt);
    blocks
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;
