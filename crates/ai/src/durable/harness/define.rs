//! Port of durable `src/harness/define.ts`.

use std::sync::Arc;

use crate::chord::{Context, JsonValue};
use crate::durable::errors::Result;
use crate::durable::tasks::Task;

use super::tool::ToolExecutionApi;
use super::types::{
    Extension, ExtensionDefinition, HookRegistration, PromptInput, PromptSection,
    ToolExecutionResult, ToolRegistration, Wrap,
};

/// Seal an extension definition; the result compares by identity.
pub fn define_extension(extension: ExtensionDefinition) -> Extension {
    Extension(Arc::new(extension))
}

/// A tool with default options: `replay` unsafe, the settings' execution mode, no limits.
pub fn define_tool<F, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    parameters: JsonValue,
    execute: F,
) -> ToolRegistration
where
    F: Fn(JsonValue, ToolExecutionApi, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolExecutionResult>> + Send + 'static,
{
    ToolRegistration {
        name: name.into(),
        description: description.into(),
        parameters,
        replay: None,
        execution_mode: None,
        prepare_arguments: None,
        output_limits: None,
        execute: Arc::new(move |args, api, context| Box::pin(execute(args, api, context))),
    }
}

/// A prompt section; tagged unless `tag` is `Some(false)`.
pub fn section<F, Fut>(key: impl Into<String>, render: F, tag: Option<bool>) -> PromptSection
where
    F: Fn(PromptInput, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Option<String>>> + Send + 'static,
{
    PromptSection {
        key: key.into(),
        render: Arc::new(move |input, context| Box::pin(render(input, context))),
        tag,
    }
}

/// A section rendering fixed text.
pub fn text_section(key: impl Into<String>, text: impl Into<String>) -> PromptSection {
    let text = text.into();
    section(
        key,
        move |_, _| {
            let text = text.clone();
            async move { Ok(Some(text)) }
        },
        None,
    )
}

/// Hook handlers for tasks with `task`'s name; `handlers` is the task's hooks struct (TS `Partial<HooksOf<K>>`).
pub fn hook<I, S, R, H: Send + Sync + 'static>(
    task: &Task<I, S, R, H>,
    handlers: H,
) -> HookRegistration {
    HookRegistration {
        task: task.name().to_string(),
        handlers: Arc::new(handlers),
    }
}

/// Wrap the tool named like `tool` wherever the wrapping extension is selected.
pub fn wrap_tool(
    tool: &str,
    wrapper: impl Fn(ToolRegistration) -> Result<ToolRegistration> + Send + Sync + 'static,
) -> Wrap {
    Wrap::Tool {
        tool: tool.to_string(),
        wrap: Arc::new(wrapper),
    }
}

/// Wrap the section `key` wherever the wrapping extension is selected.
pub fn wrap_section(
    key: &str,
    wrapper: impl Fn(PromptSection) -> Result<PromptSection> + Send + Sync + 'static,
) -> Wrap {
    Wrap::Section {
        section: key.to_string(),
        wrap: Arc::new(wrapper),
    }
}
