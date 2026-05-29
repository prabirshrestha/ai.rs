use ai::{Message, TextContent, UserContent, UserMessage, UserMessageContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    pub command: String,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: u64,
    #[serde(default)]
    pub exclude_from_context: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HarnessMessageContent {
    Text(String),
    Parts(Vec<UserContent>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    pub custom_type: String,
    pub content: HarnessMessageContent,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    pub summary: String,
    pub from_id: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    pub summary: String,
    pub tokens_before: u32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum HarnessMessage {
    #[serde(rename = "llm")]
    Llm(Message),
    #[serde(rename = "bashExecution")]
    BashExecution(BashExecutionMessage),
    #[serde(rename = "custom")]
    Custom(CustomMessage),
    #[serde(rename = "branchSummary")]
    BranchSummary(BranchSummaryMessage),
    #[serde(rename = "compactionSummary")]
    CompactionSummary(CompactionSummaryMessage),
}

impl From<Message> for HarnessMessage {
    fn from(value: Message) -> Self {
        Self::Llm(value)
    }
}

pub fn bash_execution_to_text(message: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", message.command);
    if message.output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str("```\n");
        text.push_str(&message.output);
        text.push_str("\n```");
    }
    if message.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(exit_code) = message.exit_code
        && exit_code != 0
    {
        text.push_str(&format!("\n\nCommand exited with code {exit_code}"));
    }
    if message.truncated
        && let Some(full_output_path) = message.full_output_path.as_deref()
    {
        text.push_str(&format!(
            "\n\n[Output truncated. Full output: {full_output_path}]"
        ));
    }
    text
}

pub fn create_branch_summary_message(
    summary: impl Into<String>,
    from_id: impl Into<String>,
    timestamp: &str,
) -> BranchSummaryMessage {
    BranchSummaryMessage {
        summary: summary.into(),
        from_id: from_id.into(),
        timestamp: timestamp_to_millis(timestamp),
    }
}

pub fn create_compaction_summary_message(
    summary: impl Into<String>,
    tokens_before: u32,
    timestamp: &str,
) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        summary: summary.into(),
        tokens_before,
        timestamp: timestamp_to_millis(timestamp),
    }
}

pub fn create_custom_message(
    custom_type: impl Into<String>,
    content: HarnessMessageContent,
    display: bool,
    details: Option<Value>,
    timestamp: &str,
) -> CustomMessage {
    CustomMessage {
        custom_type: custom_type.into(),
        content,
        display,
        details,
        timestamp: timestamp_to_millis(timestamp),
    }
}

pub fn convert_harness_messages_to_llm(messages: &[HarnessMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|message| match message {
            HarnessMessage::Llm(message) => Some(message.clone()),
            HarnessMessage::BashExecution(message) if message.exclude_from_context => None,
            HarnessMessage::BashExecution(message) => Some(user_parts_message(
                vec![UserContent::text(bash_execution_to_text(message))],
                message.timestamp,
            )),
            HarnessMessage::Custom(message) => Some(match &message.content {
                HarnessMessageContent::Text(text) => Message::User(UserMessage {
                    content: UserMessageContent::Parts(vec![UserContent::text(text.clone())]),
                    timestamp: message.timestamp,
                }),
                HarnessMessageContent::Parts(parts) => Message::User(UserMessage {
                    content: UserMessageContent::Parts(parts.clone()),
                    timestamp: message.timestamp,
                }),
            }),
            HarnessMessage::BranchSummary(message) => Some(user_parts_message(
                vec![UserContent::Text(TextContent {
                    text: format!(
                        "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
                        message.summary
                    ),
                    text_signature: None,
                })],
                message.timestamp,
            )),
            HarnessMessage::CompactionSummary(message) => Some(user_parts_message(
                vec![UserContent::Text(TextContent {
                    text: format!(
                        "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
                        message.summary
                    ),
                    text_signature: None,
                })],
                message.timestamp,
            )),
        })
        .collect()
}

fn user_parts_message(content: Vec<UserContent>, timestamp: u64) -> Message {
    Message::User(UserMessage {
        content: UserMessageContent::Parts(content),
        timestamp,
    })
}

fn timestamp_to_millis(timestamp: &str) -> u64 {
    if let Ok(value) = timestamp.parse::<u64>() {
        return value;
    }
    time::OffsetDateTime::parse(timestamp, &time::format_description::well_known::Rfc3339)
        .map(|value| value.unix_timestamp_nanos().max(0) as u64 / 1_000_000)
        .unwrap_or_default()
}
