//! Port of `harness-v1-network-sandbox-session.ts`,
//! `harness-v1-sandbox-template.ts` and the sandbox session create/resume
//! option types.
//!
//! Deprecated members are not ported: `getPortUrl` (use
//! `get_port_endpoint`) and `HarnessV1SandboxProvider` (supply a sandbox
//! session to `HarnessAgent::create_session`).

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{HarnessError, HarnessResult};
use crate::harness::sandbox::SandboxSession;

/// Connection details for a sandbox-exposed port. Headers must be sent
/// when opening the connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1PortEndpoint {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
}

/// `protocol` of `getPortEndpoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessV1PortProtocol {
    Http,
    Https,
    Ws,
}

/// `HarnessV1NetworkPolicy`: outbound network policy.
///
/// `custom` must set at least one of `allowed_hosts` / `allowed_cidrs`
/// (TS encodes this as two union arms); [`HarnessV1NetworkPolicy::custom`]
/// enforces it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum HarnessV1NetworkPolicy {
    AllowAll,
    DenyAll,
    Custom {
        #[serde(rename = "allowedHosts", skip_serializing_if = "Option::is_none")]
        allowed_hosts: Option<Vec<String>>,
        #[serde(rename = "allowedCIDRs", skip_serializing_if = "Option::is_none")]
        allowed_cidrs: Option<Vec<String>>,
        #[serde(rename = "deniedCIDRs", skip_serializing_if = "Option::is_none")]
        denied_cidrs: Option<Vec<String>>,
    },
}

impl HarnessV1NetworkPolicy {
    /// Build a `custom` policy, refusing the deny-only shape TS rejects.
    pub fn custom(
        allowed_hosts: Option<Vec<String>>,
        allowed_cidrs: Option<Vec<String>>,
        denied_cidrs: Option<Vec<String>>,
    ) -> Result<Self, HarnessError> {
        if allowed_hosts.is_none() && allowed_cidrs.is_none() {
            return Err(HarnessError::new(
                "A custom network policy requires allowedHosts or allowedCIDRs.",
            ));
        }
        Ok(Self::Custom {
            allowed_hosts,
            allowed_cidrs,
            denied_cidrs,
        })
    }
}

/// `{ exact } | { startsWith } | { regex }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HarnessV1RequestTransformationMatcher {
    Exact(String),
    StartsWith(String),
    Regex(String),
}

/// `HarnessV1RequestTransformationKeyValueMatcher`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1RequestTransformationKeyValueMatcher {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<HarnessV1RequestTransformationMatcher>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<HarnessV1RequestTransformationMatcher>,
}

/// `HarnessV1RequestTransformation['match']`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1RequestTransformationMatch {
    pub host: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<HarnessV1RequestTransformationMatcher>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_string: Option<Vec<HarnessV1RequestTransformationKeyValueMatcher>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Vec<HarnessV1RequestTransformationKeyValueMatcher>>,
}

/// `HarnessV1RequestTransformation['transform']`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1RequestTransformationTransform {
    pub headers: IndexMap<String, String>,
}

/// `HarnessV1RequestTransformation`: outbound HTTPS request transformation
/// applied outside the sandbox security boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1RequestTransformation {
    pub r#match: HarnessV1RequestTransformationMatch,
    pub transform: HarnessV1RequestTransformationTransform,
}

/// `HarnessV1RequestTransformationSources<AuthenticationMode>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessV1RequestTransformationSources<A> {
    pub env: IndexMap<String, String>,
    pub sandbox_env: IndexMap<String, String>,
    pub auth: A,
}

fn unsupported(member: &str) -> HarnessResult<()> {
    Err(HarnessError::capability_unsupported(
        format!("This sandbox session does not support {member}."),
        None,
    )
    .into())
}

/// `HarnessV1NetworkSandboxSession`: a [`SandboxSession`] plus ports,
/// lifecycle and network-policy mutation.
///
/// [rust] Optional TS members become a `supports_*` probe plus a default
/// body that returns `HarnessCapabilityUnsupportedError`.
#[async_trait]
pub trait HarnessV1NetworkSandboxSession: SandboxSession {
    /// Sandbox session id.
    fn id(&self) -> String;

    /// The sandbox's default working directory, read from the live sandbox.
    fn default_working_directory(&self) -> String;

    /// Ports the sandbox exposes.
    fn ports(&self) -> Vec<u16>;

    /// Resolve the connection details for a sandbox-exposed port.
    async fn get_port_endpoint(
        &self,
        port: u16,
        protocol: Option<HarnessV1PortProtocol>,
    ) -> HarnessResult<HarnessV1PortEndpoint>;

    /// Stop the sandbox. Idempotent.
    async fn stop(&self) -> HarnessResult<()>;

    /// Stop, then destroy the sandbox's backing resources.
    async fn destroy(&self) -> HarnessResult<()>;

    /// Whether `setNetworkPolicy` is present.
    fn supports_set_network_policy(&self) -> bool {
        false
    }

    /// `setNetworkPolicy?`. Callers treat a missing implementation as a no-op.
    async fn set_network_policy(&self, _policy: HarnessV1NetworkPolicy) -> HarnessResult<()> {
        unsupported("setNetworkPolicy")
    }

    /// Whether `setRequestTransformations` is present.
    fn supports_set_request_transformations(&self) -> bool {
        false
    }

    /// `setRequestTransformations?`: replace the complete rule set.
    async fn set_request_transformations(
        &self,
        _transformations: Vec<HarnessV1RequestTransformation>,
    ) -> HarnessResult<()> {
        unsupported("setRequestTransformations")
    }

    /// Whether `addRequestTransformations` is present.
    fn supports_add_request_transformations(&self) -> bool {
        false
    }

    /// `addRequestTransformations?`: add rules without replacing others.
    async fn add_request_transformations(
        &self,
        _transformations: Vec<HarnessV1RequestTransformation>,
    ) -> HarnessResult<()> {
        unsupported("addRequestTransformations")
    }

    /// Whether `setPorts` is present.
    fn supports_set_ports(&self) -> bool {
        false
    }

    /// `setPorts?`: replace the exposed port set.
    async fn set_ports(
        &self,
        _ports: Vec<u16>,
        _abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<()> {
        unsupported("setPorts")
    }

    /// Reduced view over the same sandbox: file I/O and processes only.
    fn restricted(&self) -> Arc<dyn SandboxSession>;
}

/// `HarnessV1SandboxTemplate['prepare']`.
pub type HarnessV1SandboxTemplatePrepare = Arc<
    dyn Fn(
            Arc<dyn SandboxSession>,
            Option<CancellationToken>,
        ) -> BoxFuture<'static, HarnessResult<()>>
        + Send
        + Sync,
>;

/// `HarnessV1SandboxTemplate`.
#[derive(Clone)]
pub struct HarnessV1SandboxTemplate {
    pub identity: String,
    pub prepare: HarnessV1SandboxTemplatePrepare,
}

impl fmt::Debug for HarnessV1SandboxTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessV1SandboxTemplate")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

/// `HarnessV1SandboxSessionCreateOptions<P>`: provider options plus these.
#[derive(Debug, Clone, Default)]
pub struct HarnessV1SandboxSessionCreateOptions<P> {
    pub provider: P,
    /// Manually assign the id of the new sandbox. Never resumes.
    pub sandbox_id: Option<String>,
    pub template: Option<HarnessV1SandboxTemplate>,
    pub abort_signal: Option<CancellationToken>,
}

/// `HarnessV1SandboxSessionResumeOptions<P>`: provider options plus these.
#[derive(Debug, Clone, Default)]
pub struct HarnessV1SandboxSessionResumeOptions<P> {
    pub provider: P,
    /// Id of the existing sandbox. Never creates one.
    pub sandbox_id: String,
    pub abort_signal: Option<CancellationToken>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn network_policy_json() {
        assert_eq!(
            serde_json::to_value(HarnessV1NetworkPolicy::AllowAll).unwrap(),
            json!({ "mode": "allow-all" })
        );
        let custom = HarnessV1NetworkPolicy::custom(
            None,
            Some(vec!["10.0.0.0/8".into()]),
            Some(vec!["169.254.169.254/32".into()]),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&custom).unwrap(),
            json!({ "mode": "custom", "allowedCIDRs": ["10.0.0.0/8"], "deniedCIDRs": ["169.254.169.254/32"] })
        );
        assert!(HarnessV1NetworkPolicy::custom(None, None, Some(vec![])).is_err());
    }

    #[test]
    fn request_transformation_json() {
        let t = json!({
            "match": {
                "host": "api.anthropic.com",
                "path": { "startsWith": "/v1" },
                "method": ["POST"],
                "headers": [{ "key": { "exact": "x-api-key" } }]
            },
            "transform": { "headers": { "x-api-key": "secret" } }
        });
        let parsed: HarnessV1RequestTransformation = serde_json::from_value(t.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), t);
    }
}
