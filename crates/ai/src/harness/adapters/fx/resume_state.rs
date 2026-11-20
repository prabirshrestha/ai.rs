//! Port of `harness-fx/src/fx-resume-state.ts`: what an fx session's
//! [`HarnessResumeState`](crate::harness::HarnessResumeState) holds.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transport::FxProvider;
use crate::harness::error::HarnessError;

/// `FxSessionState`: the provider the session's credentials come from and
/// the libfx checkpoint of its conversation (base64).
///
/// [rust] `turn`, the replay records of a suspended turn, is kept as it was
/// read so state written by the TypeScript adapter survives a round trip;
/// ai.rs sessions do not suspend turns, so they never write or replay one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FxSessionState {
    pub provider: FxProvider,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<Value>,
}

impl FxSessionState {
    pub fn new(provider: FxProvider) -> Self {
        Self {
            provider,
            checkpoint: None,
            turn: None,
        }
    }
}

/// `FxResumeStateData`: the whole session, or the session storage entry the
/// session ends at.
#[derive(Debug, Clone, PartialEq)]
pub enum FxResumeStateData {
    Session(FxSessionState),
    Leaf(String),
}

impl FxResumeStateData {
    pub fn to_value(&self) -> Value {
        match self {
            Self::Session(session) => serde_json::json!({ "session": session }),
            Self::Leaf(leaf_id) => serde_json::json!({ "leafId": leaf_id }),
        }
    }
}

fn invalid(detail: impl std::fmt::Display) -> HarnessError {
    HarnessError::new(format!("Invalid fx lifecycle state: {detail}"))
}

/// `parseFxResumeStateData(data)` (`fxResumeStateSchema.parse`).
pub fn parse_fx_resume_state_data(data: &Value) -> Result<FxResumeStateData, HarnessError> {
    let object: &Map<String, Value> = data.as_object().ok_or_else(|| invalid("expected an object"))?;
    match (object.get("session"), object.get("leafId"), object.len()) {
        (Some(session), None, 1) => serde_json::from_value(session.clone())
            .map(FxResumeStateData::Session)
            .map_err(invalid),
        (None, Some(Value::String(leaf_id)), 1) if !leaf_id.is_empty() => {
            Ok(FxResumeStateData::Leaf(leaf_id.clone()))
        }
        _ => Err(invalid("expected { session } or { leafId }")),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn accepts_the_whole_session() {
        let turn = json!({ "id": "turn-1", "prompt": "go", "signature": "signature", "models": [], "tools": [], "emittedKeys": ["start"], "steers": [] });
        let data = json!({ "session": { "provider": "gateway", "checkpoint": "Y2hlY2twb2ludA==", "turn": turn } });
        let parsed = parse_fx_resume_state_data(&data).unwrap();
        assert_eq!(parsed.to_value(), data);
    }

    #[test]
    fn accepts_a_session_storage_leaf() {
        assert_eq!(
            parse_fx_resume_state_data(&json!({ "leafId": "entry-1" })).unwrap(),
            FxResumeStateData::Leaf("entry-1".into())
        );
    }

    #[test]
    fn rejects_invalid_data() {
        for data in [
            json!({ "session": { "provider": "gateway" }, "leafId": "entry-1" }),
            json!({}),
            json!({ "leafId": "" }),
            json!({ "session": {} }),
        ] {
            assert!(parse_fx_resume_state_data(&data).is_err(), "{data}");
        }
    }
}
