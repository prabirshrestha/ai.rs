//! Port of `agent/internal/tool-filtering.ts`.

use indexmap::IndexSet;

use crate::harness::errors::{HarnessError, HarnessResult};
use crate::harness::sdk::parse_tool_call::NoSuchToolError;
use crate::harness::sdk::tool::ToolSet;
use crate::harness::spec::harness::HarnessV1;
use crate::harness::spec::tools::HarnessV1BuiltinToolFiltering;
use crate::harness::utils::plain_error;

/// `ResolvedHarnessAgentToolFiltering`.
#[derive(Debug, Clone)]
pub struct ResolvedHarnessAgentToolFiltering {
    pub active_user_tools: ToolSet,
    pub builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
}

/// `resolveHarnessAgentToolFiltering`.
///
/// `all_tool_names` are the keys of the merged user + built-in tool set.
pub fn resolve_harness_agent_tool_filtering(
    harness: &dyn HarnessV1,
    user_tools: &ToolSet,
    all_tool_names: &[String],
    active_tools: Option<&[String]>,
    inactive_tools: Option<&[String]>,
) -> HarnessResult<ResolvedHarnessAgentToolFiltering> {
    if active_tools.is_some() && inactive_tools.is_some() {
        return Err(plain_error(
            "HarnessAgent: pass either `activeTools` or `inactiveTools`, not both.",
        ));
    }
    let active = dedupe(active_tools);
    let inactive = dedupe(inactive_tools);
    if let Some(requested) = active.as_ref().or(inactive.as_ref()) {
        for name in requested {
            if !all_tool_names.contains(name) {
                return Err(Box::new(NoSuchToolError::new(
                    name,
                    Some(all_tool_names.to_vec()),
                )));
            }
        }
    }

    let includes = |set: &IndexSet<String>, name: &str| set.contains(name);
    let active_user_tools: ToolSet = user_tools
        .iter()
        .filter(|(name, _)| match (&active, &inactive) {
            (Some(a), _) => includes(a, name),
            (None, Some(i)) => !includes(i, name),
            (None, None) => true,
        })
        .map(|(n, t)| (n.clone(), t.clone()))
        .collect();

    let builtin_names: Vec<&String> = harness.builtin_tools().keys().collect();
    let disabled: Vec<String> = builtin_names
        .iter()
        .filter(|name| match (&active, &inactive) {
            (Some(a), _) => !includes(a, name),
            (None, Some(i)) => includes(i, name),
            (None, None) => false,
        })
        .map(|n| n.to_string())
        .collect();

    let builtin_tool_filtering = if disabled.is_empty() {
        None
    } else if let Some(a) = &active {
        Some(HarnessV1BuiltinToolFiltering::Allow {
            tool_names: builtin_names
                .iter()
                .filter(|n| includes(a, n))
                .map(|n| n.to_string())
                .collect(),
        })
    } else {
        Some(HarnessV1BuiltinToolFiltering::Deny {
            tool_names: disabled,
        })
    };

    if builtin_tool_filtering.is_some()
        && !harness.supports_builtin_tool_filtering()
        && !harness.supports_builtin_tool_approvals()
    {
        return Err(HarnessError::capability_unsupported(
            format!(
                "Harness '{}' does not support built-in tool filtering controls.",
                harness.harness_id()
            ),
            Some(harness.harness_id().to_string()),
        )
        .into());
    }

    Ok(ResolvedHarnessAgentToolFiltering {
        active_user_tools,
        builtin_tool_filtering,
    })
}

fn dedupe(names: Option<&[String]>) -> Option<IndexSet<String>> {
    names.map(|n| n.iter().cloned().collect())
}
