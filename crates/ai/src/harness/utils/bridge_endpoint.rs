//! Port of `utils/bridge-endpoint.ts`.
//!
//! [rust] `warnMintBridgeTokenIgnored` is not ported, because the deprecated
//! `mintBridgeToken` setting is not.

use crate::harness::errors::{HarnessError, HarnessResult};
use crate::harness::spec::{
    HarnessV1PortEndpoint, HarnessV1PortProtocol, HarnessV1SandboxSessionRef,
};

/// `validateBridgeSandboxSettings(...)`: validates the bridge connection
/// settings against the sandbox session before any bridge work starts.
///
/// Basic sandbox sessions cannot expose ports, so they need both an explicit
/// `port` and `portEndpoint` setting.
pub fn validate_bridge_sandbox_settings(
    sandbox_session: &HarnessV1SandboxSessionRef,
    port_override: Option<u16>,
    port_endpoint_override: Option<&HarnessV1PortEndpoint>,
) -> Result<(), HarnessError> {
    if sandbox_session.network().is_none() {
        if port_override.is_none() {
            return Err(HarnessError::capability_unsupported(
                "The harness requires an explicit `port` setting when using a basic sandbox session.",
                None,
            ));
        }
        if port_endpoint_override.is_none() {
            return Err(missing_port_endpoint_error());
        }
    }
    Ok(())
}

/// `resolveBridgePort(...)`: the port the bridge listens on inside the
/// sandbox: the `port` setting if present, otherwise the first port the
/// sandbox exposes.
pub fn resolve_bridge_port(
    sandbox_session: &HarnessV1SandboxSessionRef,
    port_override: Option<u16>,
) -> Result<u16, HarnessError> {
    if let Some(port) = port_override {
        return Ok(port);
    }
    if let Some(port) = sandbox_session
        .network()
        .and_then(|session| session.ports().first().copied())
    {
        return Ok(port);
    }
    Err(HarnessError::capability_unsupported(
        "The harness needs a TCP port exposed by the sandbox. Create the sandbox with `ports: [<port>]` or configure the harness `port` setting.",
        None,
    ))
}

/// `resolveBridgeEndpoint(...)`: the host-reachable WebSocket endpoint for a
/// bridge port.
///
/// A `portEndpoint` setting takes precedence and is used as-is regardless of
/// `port`, so it must point at the port the bridge actually listens on.
pub async fn resolve_bridge_endpoint(
    sandbox_session: &HarnessV1SandboxSessionRef,
    port: u16,
    port_endpoint_override: Option<&HarnessV1PortEndpoint>,
) -> HarnessResult<HarnessV1PortEndpoint> {
    if let Some(endpoint) = port_endpoint_override {
        return Ok(endpoint.clone());
    }
    if let Some(session) = sandbox_session.network() {
        return session
            .get_port_endpoint(port, Some(HarnessV1PortProtocol::Ws))
            .await;
    }
    Err(missing_port_endpoint_error().into())
}

fn missing_port_endpoint_error() -> HarnessError {
    HarnessError::capability_unsupported(
        "The harness requires an explicit `portEndpoint` setting when using a basic sandbox session.",
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sandbox::SandboxRunResult;
    use crate::harness::test_support::{MockNetworkSandbox, MockSandbox};
    use indexmap::IndexMap;
    use std::sync::Arc;

    fn network(ports: Vec<u16>) -> (HarnessV1SandboxSessionRef, Arc<MockNetworkSandbox>) {
        let session = Arc::new(MockNetworkSandbox::new(ports));
        (
            HarnessV1SandboxSessionRef::Network(session.clone()),
            session,
        )
    }

    fn basic() -> HarnessV1SandboxSessionRef {
        HarnessV1SandboxSessionRef::Basic(Arc::new(MockSandbox::with_result(
            SandboxRunResult::default(),
        )))
    }

    fn endpoint(url: &str) -> HarnessV1PortEndpoint {
        HarnessV1PortEndpoint {
            url: url.into(),
            headers: None,
        }
    }

    #[test]
    fn accepts_a_network_sandbox_session_without_settings() {
        assert!(validate_bridge_sandbox_settings(&network(vec![4000]).0, None, None).is_ok());
    }

    #[test]
    fn requires_a_port_setting_for_a_basic_sandbox_session() {
        let error =
            validate_bridge_sandbox_settings(&basic(), None, Some(&endpoint("wss://example")))
                .unwrap_err();
        assert!(matches!(error, HarnessError::CapabilityUnsupported { .. }));
        assert_eq!(
            error.message(),
            "The harness requires an explicit `port` setting when using a basic sandbox session."
        );
    }

    #[test]
    fn requires_a_port_endpoint_setting_for_a_basic_sandbox_session() {
        let error = validate_bridge_sandbox_settings(&basic(), Some(4000), None).unwrap_err();
        assert_eq!(
            error.message(),
            "The harness requires an explicit `portEndpoint` setting when using a basic sandbox session."
        );
    }

    #[test]
    fn prefers_the_port_setting() {
        assert_eq!(
            resolve_bridge_port(&network(vec![4000]).0, Some(5000)).unwrap(),
            5000
        );
    }

    #[test]
    fn falls_back_to_the_first_sandbox_port() {
        assert_eq!(
            resolve_bridge_port(&network(vec![4100, 4200]).0, None).unwrap(),
            4100
        );
    }

    #[test]
    fn throws_when_no_port_is_available() {
        let error = resolve_bridge_port(&network(vec![]).0, None).unwrap_err();
        assert!(matches!(error, HarnessError::CapabilityUnsupported { .. }));
        assert_eq!(
            error.message(),
            "The harness needs a TCP port exposed by the sandbox. Create the sandbox with `ports: [<port>]` or configure the harness `port` setting."
        );
    }

    #[tokio::test]
    async fn resolves_the_sandbox_endpoint_without_credentials_in_the_url() {
        let (session, mock) = network(vec![4000]);
        assert_eq!(
            resolve_bridge_endpoint(&session, 4321, None).await.unwrap(),
            HarnessV1PortEndpoint {
                url: "wss://sandbox.example/4321".into(),
                headers: Some(IndexMap::from([(
                    "x-access-token".to_string(),
                    "traffic-token".to_string()
                )])),
            }
        );
        assert_eq!(
            *mock.endpoint_calls.lock(),
            vec![(4321, Some(HarnessV1PortProtocol::Ws))]
        );
    }

    #[tokio::test]
    async fn prefers_the_port_endpoint_setting_as_is() {
        let (session, mock) = network(vec![4000]);
        assert_eq!(
            resolve_bridge_endpoint(
                &session,
                4321,
                Some(&endpoint("wss://custom.example/bridge?a=b"))
            )
            .await
            .unwrap(),
            endpoint("wss://custom.example/bridge?a=b")
        );
        assert!(mock.endpoint_calls.lock().is_empty());
    }

    #[tokio::test]
    async fn throws_for_a_basic_sandbox_session_without_a_port_endpoint_setting() {
        let error = resolve_bridge_endpoint(&basic(), 4321, None)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "The harness requires an explicit `portEndpoint` setting when using a basic sandbox session."
        );
    }
}
