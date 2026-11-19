//! Port of `agent/internal/lifecycle-state-validation.ts`.
//!
//! [rust] `type` and `specificationVersion` are literal marker types, so a
//! typed state always has the expected values; the checks run on the
//! [`HarnessV1LifecycleState`] entry point. A resume state has no
//! `pendingToolApprovals` / `pendingToolResults` fields, so serde drops
//! them instead of the TS rejecting them.

use crate::harness::errors::{HarnessError, HarnessResult};
use crate::harness::provider::{JsonValue, validate_types};
use crate::harness::spec::harness::HarnessV1;
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1LifecycleState, HarnessV1ResumeSessionState,
};

/// The lifecycle state kinds (`state.type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleStateType {
    ResumeSession,
    ContinueTurn,
}

impl LifecycleStateType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ResumeSession => "resume-session",
            Self::ContinueTurn => "continue-turn",
        }
    }
}

/// `validateLifecycleStateData({ harness, state, expectedType })`.
pub fn validate_lifecycle_state_data(
    harness: &dyn HarnessV1,
    state: &HarnessV1LifecycleState,
    expected_type: LifecycleStateType,
) -> HarnessResult<HarnessV1LifecycleState> {
    let actual = match state {
        HarnessV1LifecycleState::ResumeSession(_) => LifecycleStateType::ResumeSession,
        HarnessV1LifecycleState::ContinueTurn(_) => LifecycleStateType::ContinueTurn,
    };
    if actual != expected_type {
        return Err(HarnessError::new(format!(
            "Lifecycle state has unexpected type '{}'; expected '{}'.",
            actual.as_str(),
            expected_type.as_str()
        ))
        .into());
    }
    Ok(match state {
        HarnessV1LifecycleState::ResumeSession(s) => {
            HarnessV1LifecycleState::ResumeSession(validate_resume_session_state(harness, s)?)
        }
        HarnessV1LifecycleState::ContinueTurn(s) => {
            HarnessV1LifecycleState::ContinueTurn(validate_continue_turn_state(harness, s)?)
        }
    })
}

/// `validateLifecycleStateData` for a `resume-session` state.
pub fn validate_resume_session_state(
    harness: &dyn HarnessV1,
    state: &HarnessV1ResumeSessionState,
) -> HarnessResult<HarnessV1ResumeSessionState> {
    check_harness_id(harness, &state.harness_id)?;
    let data = validate_data(harness, &state.data)?;
    let continue_from = state
        .continue_from
        .as_ref()
        .map(|c| validate_continue_turn_state(harness, c))
        .transpose()?;
    Ok(HarnessV1ResumeSessionState {
        data,
        continue_from,
        ..state.clone()
    })
}

/// `validateLifecycleStateData` for a `continue-turn` state.
pub fn validate_continue_turn_state(
    harness: &dyn HarnessV1,
    state: &HarnessV1ContinueTurnState,
) -> HarnessResult<HarnessV1ContinueTurnState> {
    check_harness_id(harness, &state.harness_id)?;
    let data = validate_data(harness, &state.data)?;
    Ok(HarnessV1ContinueTurnState {
        data,
        ..state.clone()
    })
}

fn check_harness_id(harness: &dyn HarnessV1, harness_id: &str) -> HarnessResult<()> {
    if harness_id != harness.harness_id() {
        return Err(HarnessError::new(format!(
            "Lifecycle state was produced by harness '{harness_id}' but this agent uses '{}'.",
            harness.harness_id()
        ))
        .into());
    }
    Ok(())
}

fn validate_data(harness: &dyn HarnessV1, data: &JsonValue) -> HarnessResult<JsonValue> {
    let Some(schema) = harness.lifecycle_state_schema() else {
        return Ok(data.clone());
    };
    validate_types(data, schema, None).map_err(|error| {
        HarnessError::with_cause(
            format!(
                "Lifecycle state failed schema validation for harness '{}': {error}",
                harness.harness_id()
            ),
            error,
        )
        .into()
    })
}
