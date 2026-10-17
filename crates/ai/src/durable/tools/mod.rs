//! Port of durable `src/tools/`: the `read`, `write`, `edit`, and `bash` coding tools over an
//! [`ExecutionEnv`](crate::durable::env::ExecutionEnv), and the [`CODING_TOOLS`] extension bundling them.
//!
//! Divergences from Pi:
//! - TypeBox schemas are JSON Schema literals and the `*ToolInput` types are serde structs; arguments that do not
//!   deserialize fail the call with a `TypeError`.
//! - `createBashTool(options?)` takes a [`BashToolOptions`] by value (`Default::default()` is TS `undefined`).
//! - `CodingTools` is the [`CODING_TOOLS`] static; clones of it are the same extension.
//! - JS numbers in messages (`offset`, `timeout`) print as JS would for finite values.

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod env;
pub mod file_mutation_queue;
pub mod image;
pub mod path_utils;
pub mod read;
pub mod write;

#[cfg(all(test, feature = "durable-local-env"))]
mod tests;

use std::sync::LazyLock;

use serde::de::DeserializeOwned;

use crate::chord::{Context, JsonValue};
use crate::durable::errors::{Error, Result};
use crate::durable::harness::define::define_extension;
use crate::durable::harness::types::{Extension, ExtensionDefinition};

pub use bash::{BashExecution, BashPrepare, BashToolInput, BashToolOptions, create_bash_tool};
pub use edit::{EditToolDetails, EditToolInput, create_edit_tool};
pub use read::{ReadToolInput, create_read_tool};
pub use write::{WriteToolInput, create_write_tool};

/// `read`, `write`, `edit`, and `bash`; nothing installs it automatically (`CodingTools`).
pub static CODING_TOOLS: LazyLock<Extension> = LazyLock::new(|| {
    define_extension(ExtensionDefinition {
        tools: vec![
            create_read_tool(),
            create_write_tool(),
            create_edit_tool(),
            create_bash_tool(BashToolOptions::default()),
        ],
        ..ExtensionDefinition::new("coding-tools")
    })
});

/// The call's arguments as the tool's input type.
fn parse_args<T: DeserializeOwned>(args: JsonValue) -> Result<T> {
    serde_json::from_value(args).map_err(|error| Error::type_error(error.to_string()))
}

/// `if (context.abortSignal?.aborted) throw new Error("Operation aborted")`.
fn throw_if_aborted(context: &Context) -> Result<()> {
    if context
        .abort_signal()
        .is_some_and(|signal| signal.aborted())
    {
        return Err(Error::message("Operation aborted"));
    }
    Ok(())
}

/// `String(number)` for the numbers tools print.
fn js_number(number: f64) -> String {
    if number.is_finite() && number.fract() == 0.0 && number.abs() < 1e21 {
        format!("{}", number as i64)
    } else if number.is_nan() {
        "NaN".into()
    } else {
        format!("{number}")
    }
}

/// `Array.prototype.slice` index conversion: truncate toward zero, count negatives from the end, clamp.
fn js_slice_index(index: f64, length: usize) -> usize {
    if index.is_nan() {
        return 0;
    }
    let index = index.trunc();
    if index < 0.0 {
        (length as f64 + index).max(0.0) as usize
    } else {
        index.min(length as f64) as usize
    }
}
