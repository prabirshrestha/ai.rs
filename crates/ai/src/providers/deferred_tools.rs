use std::collections::{HashMap, HashSet};

use crate::types::{AssistantContent, Context, Message, Tool};

pub(crate) struct DeferredToolPlacement {
    pub immediate: Vec<Tool>,
    pub deferred: Vec<(String, Tool)>,
}

/// Split current tools into prefix and transcript-loaded definitions.
pub(crate) fn split_deferred_tools(
    context: &Context,
    enabled: bool,
    normalize_name: impl Fn(&str) -> String,
) -> DeferredToolPlacement {
    let mut names = Vec::new();
    let mut unique_tools = HashMap::new();
    for tool in &context.tools {
        let name = normalize_name(&tool.name);
        if !unique_tools.contains_key(&name) {
            names.push(name.clone());
        }
        unique_tools.insert(name, tool.clone());
    }
    if !enabled {
        return DeferredToolPlacement {
            immediate: names
                .into_iter()
                .filter_map(|name| unique_tools.remove(&name))
                .collect(),
            deferred: Vec::new(),
        };
    }

    let mut deferred_names = HashSet::new();
    let mut used_names = HashSet::new();
    for message in &context.messages {
        match message {
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let AssistantContent::ToolCall(tool_call) = block {
                        used_names.insert(normalize_name(&tool_call.name));
                    }
                }
            }
            Message::ToolResult(result) => {
                for name in &result.added_tool_names {
                    let normalized_name = normalize_name(name);
                    if !used_names.contains(&normalized_name) {
                        deferred_names.insert(normalized_name);
                    }
                }
            }
            Message::User(_) | Message::Custom(_) => {}
        }
    }

    let mut immediate = Vec::new();
    let mut deferred = Vec::new();
    for name in names {
        let Some(tool) = unique_tools.remove(&name) else {
            continue;
        };
        if deferred_names.contains(&name) {
            deferred.push((name, tool));
        } else {
            immediate.push(tool);
        }
    }
    DeferredToolPlacement {
        immediate,
        deferred,
    }
}
