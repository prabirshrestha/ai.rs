//! Port of `utils/transcript.ts`.
//!
//! Pi's replay helpers accept any `{ role: string }[]` so agent transcripts
//! with custom roles can be passed unfiltered. The Rust [`Message`] enum has
//! no custom roles, so the helpers take `&[Message]`.

use indexmap::IndexMap;

use crate::types::{
    Context, Message, SystemMessage, SystemMessageContent, Tool, ToolReference, TranscriptContext,
};
use crate::utils::text::{content_text, get_system_message_text};

/// Build the leading system message for a prompt and tool set. Returns
/// `None` when both are empty, so an empty transcript stays empty.
pub fn create_initial_system_message(
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
) -> Option<SystemMessage> {
    let has_system_prompt = system_prompt.is_some_and(|prompt| !prompt.is_empty());
    let has_tools = tools.is_some_and(|tools| !tools.is_empty());
    if !has_system_prompt && !has_tools {
        return None;
    }
    Some(SystemMessage {
        content: SystemMessageContent::Text(system_prompt.unwrap_or_default().to_string()),
        sections: None,
        tools_added: has_tools.then(|| tools.unwrap_or_default().to_vec()),
        tools_removed: None,
        timestamp: 0,
    })
}

/// Fold `Context::system_prompt` and `Context::tools` into a leading system
/// message. This is the only entry point that produces a
/// [`TranscriptContext`]; every provider-facing function expects the result.
pub fn normalize_context(context: &Context) -> TranscriptContext {
    let initial_message =
        create_initial_system_message(context.system_prompt.as_deref(), context.tools.as_deref());
    let messages = match initial_message {
        Some(initial_message) => std::iter::once(Message::System(initial_message))
            .chain(context.messages.iter().cloned())
            .collect(),
        None => context.messages.clone(),
    };
    TranscriptContext::from_messages(messages)
}

fn as_system_message(message: &Message) -> Option<&SystemMessage> {
    match message {
        Message::System(message) => Some(message),
        _ => None,
    }
}

/// Return the leading system message, if the transcript starts with one.
pub fn get_initial_system_message(messages: &[Message]) -> Option<&SystemMessage> {
    messages.first().and_then(as_system_message)
}

/// Drop the leading system message for APIs that carry the prompt outside the message list.
pub fn without_initial_system_message(messages: &[Message]) -> &[Message] {
    if get_initial_system_message(messages).is_some() {
        &messages[1..]
    } else {
        messages
    }
}

/// Resolve the tools available after applying every transcript delta in order.
pub fn get_current_tools(messages: &[Message]) -> Vec<Tool> {
    let mut tools: IndexMap<String, Tool> = IndexMap::new();
    for message in messages.iter().filter_map(as_system_message) {
        for tool in message.tools_removed.iter().flatten() {
            tools.shift_remove(&tool.name);
        }
        for tool in message.tools_added.iter().flatten() {
            // `Map.set` keeps an existing key's position.
            tools.insert(tool.name.clone(), tool.clone());
        }
    }
    tools.into_values().collect()
}

/// Replay every system message into one leading system message holding the
/// current prompt and tools. Later `content` is appended to the base prompt,
/// `sections` are patched by name, and tools are resolved with
/// [`get_current_tools`].
pub fn get_current_system_message(messages: &[Message]) -> Option<SystemMessage> {
    let mut content = Vec::new();
    let mut sections: IndexMap<String, String> = IndexMap::new();
    let mut timestamp = None;
    for message in messages.iter().filter_map(as_system_message) {
        timestamp.get_or_insert(message.timestamp);
        let text = content_text(&message.content);
        if !text.is_empty() {
            content.push(text);
        }
        for (name, value) in message.sections.iter().flatten() {
            match value {
                None => {
                    sections.shift_remove(name);
                }
                Some(value) => {
                    sections.insert(name.clone(), value.clone());
                }
            }
        }
    }
    let tools = get_current_tools(messages);
    if timestamp.is_none() && tools.is_empty() {
        return None;
    }
    Some(SystemMessage {
        content: SystemMessageContent::Text(content.join("\n\n")),
        sections: (!sections.is_empty()).then(|| {
            sections
                .into_iter()
                .map(|(name, value)| (name, Some(value)))
                .collect()
        }),
        tools_added: (!tools.is_empty()).then_some(tools),
        tools_removed: None,
        timestamp: timestamp.unwrap_or(0),
    })
}

/// Render the current system prompt text after replaying every system message.
pub fn get_current_system_prompt(messages: &[Message]) -> String {
    get_current_system_message(messages)
        .map(|message| get_system_message_text(&message))
        .unwrap_or_default()
}

/// Rebuild the transcript for APIs without mid-conversation system messages:
/// the replayed system message leads, and every later system message is dropped.
pub fn collapse_system_messages(context: &TranscriptContext) -> TranscriptContext {
    let head = get_current_system_message(&context.messages);
    let messages = context
        .messages
        .iter()
        .filter(|message| !matches!(message, Message::System(_)))
        .cloned();
    TranscriptContext::from_messages(match head {
        Some(head) => std::iter::once(Message::System(head))
            .chain(messages)
            .collect(),
        None => messages.collect(),
    })
}

/// Keep later system messages in place when the model accepts them; otherwise collapse them.
pub fn resolve_transcript(
    context: &TranscriptContext,
    supports_mid_convo_system_messages: Option<bool>,
) -> TranscriptContext {
    if supports_mid_convo_system_messages == Some(true) {
        context.clone()
    } else {
        collapse_system_messages(context)
    }
}

/// Strip executable and display-only fields from a tool before transcript
/// comparison or persistence. Rust tools carry no executable fields, so this
/// is a structural copy.
pub fn to_tool_declaration(tool: &Tool) -> Tool {
    Tool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
        constrained_sampling: tool.constrained_sampling.clone(),
    }
}

/// Whether two tools declare the same interface to the model, compared on
/// their serialized declarations as in Pi.
pub fn declarations_equal(left: &Tool, right: &Tool) -> bool {
    serde_json::to_string(&to_tool_declaration(left)).ok()
        == serde_json::to_string(&to_tool_declaration(right)).ok()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolStateChanges {
    pub tools_added: Vec<Tool>,
    pub tools_removed: Vec<ToolReference>,
}

/// Compare two complete tool states. A changed definition is a removal
/// followed by an addition.
pub fn get_tool_state_changes(previous: &[Tool], current: &[Tool]) -> ToolStateChanges {
    let previous_tools: IndexMap<&str, &Tool> = previous
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();
    let current_tools: IndexMap<&str, &Tool> = current
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();
    ToolStateChanges {
        tools_added: current
            .iter()
            .filter(|tool| {
                previous_tools
                    .get(tool.name.as_str())
                    .is_none_or(|previous_tool| !declarations_equal(previous_tool, tool))
            })
            .map(to_tool_declaration)
            .collect(),
        tools_removed: previous
            .iter()
            .filter(|tool| {
                current_tools
                    .get(tool.name.as_str())
                    .is_none_or(|current_tool| !declarations_equal(tool, current_tool))
            })
            .map(|tool| ToolReference {
                name: tool.name.clone(),
            })
            .collect(),
    }
}

/// Every definition referenced by transcript tool state, in first-declaration order.
pub fn get_declared_tools(messages: &[Message]) -> Vec<Tool> {
    let mut definitions: IndexMap<String, Tool> = IndexMap::new();
    for message in messages.iter().filter_map(as_system_message) {
        for tool in message.tools_added.iter().flatten() {
            definitions.insert(tool.name.clone(), tool.clone());
        }
    }
    definitions.into_values().collect()
}

/// Whether a tool name was declared twice with different definitions.
#[deprecated(note = "No built-in transport needs this anymore; kept for API compatibility.")]
pub fn has_tool_redefinitions(messages: &[Message]) -> bool {
    let mut declared: IndexMap<&str, &Tool> = IndexMap::new();
    for message in messages.iter().filter_map(as_system_message) {
        for tool in message.tools_added.iter().flatten() {
            if let Some(previous) = declared.get(tool.name.as_str())
                && !declarations_equal(previous, tool)
            {
                return true;
            }
            declared.insert(tool.name.as_str(), tool);
        }
    }
    false
}

/// Whether tool history contains a removal or same-name redeclaration that
/// an addition-only transport cannot replay.
pub fn has_non_additive_tool_changes(messages: &[Message]) -> bool {
    let mut declared = std::collections::HashSet::new();
    for message in messages.iter().filter_map(as_system_message) {
        if message
            .tools_removed
            .as_ref()
            .is_some_and(|removed| !removed.is_empty())
        {
            return true;
        }
        for tool in message.tools_added.iter().flatten() {
            if !declared.insert(tool.name.as_str()) {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptTools {
    /// Tools sent in the top-level request field.
    pub request_tools: Vec<Tool>,
    /// Whether later system messages carry their own `tools_added` as
    /// in-place additions. When false, `request_tools` already holds the
    /// complete current tool set.
    pub anchors_additions: bool,
}

/// Split tool declarations between the top-level request field and in-place additions.
pub fn resolve_transcript_tools(
    messages: &[Message],
    supports_tool_additions: bool,
) -> TranscriptTools {
    let anchors_additions = supports_tool_additions && !has_non_additive_tool_changes(messages);
    TranscriptTools {
        request_tools: if anchors_additions {
            get_initial_system_message(messages)
                .and_then(|message| message.tools_added.clone())
                .unwrap_or_default()
        } else {
            get_current_tools(messages)
        },
        anchors_additions,
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::{
        AssistantContent, AssistantMessage, ConstrainedSampling, Model, UserMessage,
    };
    use crate::utils::text::render_system_message_update;

    fn tool(name: &str) -> Tool {
        tool_with(name, &format!("{name} tool"))
    }

    fn tool_with(name: &str, description: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: description.to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
            constrained_sampling: None,
        }
    }

    fn sections(entries: &[(&str, Option<&str>)]) -> Option<IndexMap<String, Option<String>>> {
        Some(
            entries
                .iter()
                .map(|(name, value)| (name.to_string(), value.map(str::to_string)))
                .collect(),
        )
    }

    fn system(content: &str, timestamp: u64) -> SystemMessage {
        SystemMessage {
            content: content.into(),
            timestamp,
            ..Default::default()
        }
    }

    fn user(content: &str, timestamp: u64) -> Message {
        Message::User(UserMessage {
            content: content.into(),
            timestamp,
        })
    }

    fn transcript() -> TranscriptContext {
        let mut assistant = AssistantMessage::empty_for(&Model::default());
        assistant.content = vec![AssistantContent::text("ok")];
        assistant.timestamp = 13;
        normalize_context(&Context {
            messages: vec![
                Message::System(SystemMessage {
                    sections: sections(&[("a", Some("<a>1</a>")), ("b", Some("<b>1</b>"))]),
                    tools_added: Some(vec![tool("first")]),
                    ..system("base", 10)
                }),
                user("hello", 11),
                Message::System(system("also do this", 12)),
                Message::Assistant(assistant),
                Message::System(SystemMessage {
                    sections: sections(&[
                        ("a", Some("<a>2</a>")),
                        ("b", None),
                        ("c", Some("<c>1</c>")),
                    ]),
                    tools_removed: Some(vec![ToolReference {
                        name: "first".to_string(),
                    }]),
                    tools_added: Some(vec![tool("second")]),
                    ..system("", 14)
                }),
            ],
            ..Default::default()
        })
    }

    fn roles(messages: &[Message]) -> Vec<&'static str> {
        messages.iter().map(Message::role).collect()
    }

    #[test]
    fn replays_content_sections_and_tools_into_one_leading_message() {
        let transcript = transcript();
        let current = get_current_system_message(&transcript.messages).unwrap();
        assert_eq!(
            current,
            SystemMessage {
                content: "base\n\nalso do this".into(),
                sections: sections(&[("a", Some("<a>2</a>")), ("c", Some("<c>1</c>"))]),
                tools_added: Some(vec![tool("second")]),
                tools_removed: None,
                timestamp: 10,
            }
        );
        assert_eq!(
            get_current_system_prompt(&transcript.messages),
            "base\n\nalso do this\n\n<a>2</a>\n\n<c>1</c>"
        );
    }

    #[test]
    fn collapse_keeps_only_non_system_messages_after_the_replayed_head() {
        let collapsed = collapse_system_messages(&transcript());
        assert_eq!(
            roles(&collapsed.messages),
            vec!["system", "user", "assistant"]
        );
        assert_eq!(collapse_system_messages(&collapsed), collapsed);
    }

    #[test]
    fn replay_of_a_transcript_without_system_messages_is_empty() {
        let context = normalize_context(&Context {
            messages: vec![user("hi", 1)],
            ..Default::default()
        });
        assert_eq!(get_current_system_message(&context.messages), None);
        assert_eq!(get_current_system_prompt(&context.messages), "");
        assert_eq!(
            collapse_system_messages(&context).messages,
            context.messages
        );
    }

    #[test]
    fn a_late_full_patch_on_a_transcript_without_a_leading_message_replays_as_the_prompt() {
        let context = normalize_context(&Context {
            messages: vec![
                user("old session", 1),
                Message::System(SystemMessage {
                    sections: sections(&[("preamble", Some("You are pi."))]),
                    tools_added: Some(vec![tool("x")]),
                    ..system("", 2)
                }),
            ],
            ..Default::default()
        });
        assert_eq!(get_current_system_prompt(&context.messages), "You are pi.");
        let Message::System(head) = &collapse_system_messages(&context).messages[0] else {
            panic!("expected system head");
        };
        assert_eq!(head.tools_added, Some(vec![tool("x")]));
    }

    #[test]
    fn renders_complete_prompts_and_framed_updates() {
        let transcript = transcript();
        let (Message::System(leading), Message::System(update)) =
            (&transcript.messages[0], &transcript.messages[4])
        else {
            panic!("expected system messages");
        };
        assert_eq!(
            get_system_message_text(leading),
            "base\n\n<a>1</a>\n\n<b>1</b>"
        );
        assert_eq!(
            render_system_message_update(update),
            [
                "Updated system prompt section \"a\":\n\n<a>2</a>",
                "Removed system prompt section \"b\".",
                "Updated system prompt section \"c\":\n\n<c>1</c>",
            ]
            .join("\n\n")
        );
    }

    #[test]
    fn normalizes_the_legacy_prompt_and_tool_fields_into_a_leading_system_message() {
        let messages = vec![user("hi", 1)];
        assert_eq!(
            normalize_context(&Context {
                messages: messages.clone(),
                ..Default::default()
            })
            .messages,
            messages
        );
        assert_eq!(
            normalize_context(&Context {
                system_prompt: Some(String::new()),
                tools: Some(Vec::new()),
                messages: messages.clone(),
            })
            .messages,
            messages
        );
        let normalized = normalize_context(&Context {
            system_prompt: Some("be brief".to_string()),
            tools: Some(vec![tool("a")]),
            messages: messages.clone(),
        });
        assert_eq!(
            normalized.messages,
            vec![
                Message::System(SystemMessage {
                    tools_added: Some(vec![tool("a")]),
                    ..system("be brief", 0)
                }),
                messages[0].clone(),
            ]
        );
    }

    #[test]
    fn compares_tool_declarations_without_executable_or_undefined_fields() {
        assert!(declarations_equal(&tool("a"), &tool("a")));
        assert!(!declarations_equal(&tool("a"), &tool_with("a", "changed")));
        assert!(!declarations_equal(
            &tool("a"),
            &Tool {
                constrained_sampling: Some(ConstrainedSampling::Disabled),
                ..tool("a")
            }
        ));
    }

    #[test]
    fn tool_state_changes_treat_changed_definitions_as_removal_plus_addition() {
        let changes = get_tool_state_changes(
            &[tool("a"), tool("b")],
            &[tool_with("b", "changed"), tool("c")],
        );
        assert_eq!(
            changes,
            ToolStateChanges {
                tools_added: vec![tool_with("b", "changed"), tool("c")],
                tools_removed: vec![
                    ToolReference {
                        name: "a".to_string()
                    },
                    ToolReference {
                        name: "b".to_string()
                    },
                ],
            }
        );
        assert_eq!(
            get_tool_state_changes(&[tool("a")], &[tool("a")]),
            ToolStateChanges::default()
        );
    }

    #[test]
    fn detects_non_additive_tool_history_and_redefinitions() {
        let transcript = transcript();
        assert!(has_non_additive_tool_changes(&transcript.messages));
        assert!(!has_tool_redefinitions(&transcript.messages));
        let additive = vec![
            Message::System(SystemMessage {
                tools_added: Some(vec![tool("a")]),
                ..system("", 1)
            }),
            Message::System(SystemMessage {
                tools_added: Some(vec![tool("b")]),
                ..system("", 2)
            }),
        ];
        assert!(!has_non_additive_tool_changes(&additive));
        let redeclared = vec![
            Message::System(SystemMessage {
                tools_added: Some(vec![tool("a")]),
                ..system("", 1)
            }),
            Message::System(SystemMessage {
                tools_added: Some(vec![tool_with("a", "changed")]),
                ..system("", 2)
            }),
        ];
        assert!(has_non_additive_tool_changes(&redeclared));
        assert!(has_tool_redefinitions(&redeclared));
    }

    #[test]
    fn resolves_transcript_tools_for_anchoring_transports() {
        let transcript = transcript();
        let resolved = resolve_transcript_tools(&transcript.messages, true);
        assert!(!resolved.anchors_additions);
        assert_eq!(resolved.request_tools, vec![tool("second")]);

        let additive = vec![
            Message::System(SystemMessage {
                tools_added: Some(vec![tool("a")]),
                ..system("base", 1)
            }),
            Message::System(SystemMessage {
                tools_added: Some(vec![tool("b")]),
                ..system("", 2)
            }),
        ];
        let anchored = resolve_transcript_tools(&additive, true);
        assert!(anchored.anchors_additions);
        assert_eq!(anchored.request_tools, vec![tool("a")]);
        assert_eq!(get_declared_tools(&additive), vec![tool("a"), tool("b")]);
        assert_eq!(
            resolve_transcript_tools(&additive, false).request_tools,
            vec![tool("a"), tool("b")]
        );
        assert_eq!(without_initial_system_message(&additive).len(), 1);
    }
}
