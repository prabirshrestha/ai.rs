//! [rust] An adapter's builtin tool catalog (`builtinTools`), kept as the
//! `builtin_tools.json` data file dumped from the TS adapter: each tool's
//! description, native and common names, kind and input schema.

#![allow(dead_code)]

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use crate::Tool;
use crate::harness::tools::{BuiltinTool, BuiltinToolKind, BuiltinToolName};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CatalogEntry {
    description: Option<String>,
    native_name: Option<String>,
    common_name: Option<BuiltinToolName>,
    tool_use_kind: Option<BuiltinToolKind>,
    input_schema: Value,
}

/// Parse a catalog, keeping its order. Keys are the harness names.
pub fn load_builtin_tool_catalog(json: &str) -> IndexMap<String, BuiltinTool> {
    let entries: IndexMap<String, CatalogEntry> =
        serde_json::from_str(json).expect("the builtin tool catalog is valid");
    entries
        .into_iter()
        .map(|(key, entry)| {
            let tool = Tool {
                name: key.clone(),
                description: entry.description.unwrap_or_default(),
                parameters: entry.input_schema,
                constrained_sampling: None,
            };
            (
                key,
                BuiltinTool {
                    tool,
                    native_name: entry.native_name,
                    common_name: entry.common_name,
                    kind: entry.tool_use_kind,
                },
            )
        })
        .collect()
}

/// The native name of a catalog entry (its key when it has none).
pub fn native_name<'a>(key: &'a str, tool: &'a BuiltinTool) -> &'a str {
    tool.native_name.as_deref().unwrap_or(key)
}

/// The harness name for a native tool name.
pub fn harness_name_for_native(catalog: &IndexMap<String, BuiltinTool>, native: &str) -> String {
    catalog
        .iter()
        .find(|(key, tool)| native_name(key, tool) == native)
        .map_or_else(|| native.to_string(), |(key, _)| key.clone())
}

/// The native name for a harness name.
pub fn native_for_harness_name(catalog: &IndexMap<String, BuiltinTool>, name: &str) -> String {
    catalog.get(name).map_or_else(
        || name.to_string(),
        |tool| native_name(name, tool).to_string(),
    )
}
