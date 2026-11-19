//! Port of `harness-authentication.ts` and
//! `harness-v1-credential-forwarding.ts`.

use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::harness::errors::HarnessResult;

/// `HarnessV1AuthenticationEnvironment`: authentication environment
/// variables. As an adapter's `auth`, it replaces the host environment for
/// authentication discovery.
pub type HarnessV1AuthenticationEnvironment = IndexMap<String, String>;

/// The default adapter choice set, `'direct'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HarnessV1DirectAuthentication {
    Direct,
}

/// `HarnessV1Authentication<ADAPTER_CHOICES>`:
/// `'auto' | 'ai-gateway' | ADAPTER_CHOICES | HarnessV1AuthenticationEnvironment`.
///
/// [rust] `C` is an adapter enum of its string choices; TS requires the
/// choice union to be non-empty and concrete, which an enum always is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessV1Authentication<C = HarnessV1DirectAuthentication> {
    Auto,
    AiGateway,
    Choice(C),
    Environment(HarnessV1AuthenticationEnvironment),
}

// No `C: Default` bound, unlike a derive.
#[allow(clippy::derivable_impls)]
impl<C> Default for HarnessV1Authentication<C> {
    fn default() -> Self {
        Self::Auto
    }
}

impl<C: Serialize> Serialize for HarnessV1Authentication<C> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => s.serialize_str("auto"),
            Self::AiGateway => s.serialize_str("ai-gateway"),
            Self::Choice(c) => c.serialize(s),
            Self::Environment(env) => env.serialize(s),
        }
    }
}

impl<'de, C: serde::de::DeserializeOwned> Deserialize<'de> for HarnessV1Authentication<C> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        match &value {
            serde_json::Value::String(s) if s == "auto" => Ok(Self::Auto),
            serde_json::Value::String(s) if s == "ai-gateway" => Ok(Self::AiGateway),
            serde_json::Value::Object(_) => serde_json::from_value(value)
                .map(Self::Environment)
                .map_err(serde::de::Error::custom),
            _ => serde_json::from_value(value)
                .map(Self::Choice)
                .map_err(serde::de::Error::custom),
        }
    }
}

/// Input of [`HarnessV1CredentialForwarding`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessV1CredentialForwardingOptions {
    /// The value the adapter would otherwise forward: a sandbox placeholder
    /// when credential brokering is available, the real credential otherwise.
    pub credential: String,
    /// The environment variable name used to expose the value.
    pub environment_variable_name: String,
}

/// `HarnessV1CredentialForwarding`: customizes a credential value right
/// before an adapter forwards it into a sandbox process.
pub type HarnessV1CredentialForwarding = Arc<
    dyn Fn(HarnessV1CredentialForwardingOptions) -> BoxFuture<'static, HarnessResult<String>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Runtime counterpart of harness-authentication.test-d.ts.
    #[test]
    fn authentication_serializes_as_the_ts_union() {
        type A = HarnessV1Authentication;
        for (value, expected) in [
            (json!("auto"), A::Auto),
            (json!("ai-gateway"), A::AiGateway),
            (
                json!("direct"),
                A::Choice(HarnessV1DirectAuthentication::Direct),
            ),
            (
                json!({ "ANTHROPIC_API_KEY": "k" }),
                A::Environment(IndexMap::from([("ANTHROPIC_API_KEY".into(), "k".into())])),
            ),
        ] {
            let parsed: A = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(serde_json::to_value(&parsed).unwrap(), value);
        }
        assert!(serde_json::from_value::<A>(json!("subscription")).is_err());
    }
}
