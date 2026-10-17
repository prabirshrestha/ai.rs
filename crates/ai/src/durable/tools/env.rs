//! Port of durable `src/tools/env.ts`.

use std::sync::Arc;

use crate::durable::env::ExecutionEnv;
use crate::durable::errors::{Error, Result};
use crate::durable::harness::tool::ToolExecutionApi;

/// The call's execution environment; a tool without one fails with an ordinary error result.
pub fn require_env(api: &ToolExecutionApi) -> Result<Arc<dyn ExecutionEnv>> {
    api.env()
        .ok_or_else(|| Error::message("No execution environment is configured"))
}
