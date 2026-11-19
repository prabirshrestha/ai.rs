//! Port of `utils/sandbox-credential-brokering.ts`.

use std::sync::LazyLock;

use base64::Engine as _;
use indexmap::{IndexMap, IndexSet};
use regex::Regex;
use ring::rand::SecureRandom as _;

use crate::harness::error::HarnessResult;
use crate::harness::{
    HarnessRequestTransformation, HarnessRequestTransformationKeyValueMatcher,
    HarnessRequestTransformationMatch, HarnessRequestTransformationMatcher,
    HarnessRequestTransformationTransform,
};

const SANDBOX_CREDENTIAL_PLACEHOLDER_PREFIX: &str = "aisdkhc_";

const CREDENTIAL_BROKERING_UNAVAILABLE_WARNING: &str = "The sandbox implementation does not support configuring request transformations, so credential brokering does not work. Falling back to less secure credential forwarding.";

/// `generateSandboxCredentialPlaceholder()`: `aisdkhc_` and 32 random bytes
/// in base64url.
pub fn generate_sandbox_credential_placeholder() -> String {
    let mut bytes = [0u8; 32];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .expect("the system random number generator failed");
    format!(
        "{SANDBOX_CREDENTIAL_PLACEHOLDER_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    )
}

/// `isSandboxCredentialPlaceholder(value)`.
pub fn is_sandbox_credential_placeholder(value: &str) -> bool {
    static PATTERN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^aisdkhc_[A-Za-z0-9_-]{43}$").unwrap());
    PATTERN.is_match(value)
}

/// `warnCredentialBrokeringUnavailable(options)`: warns when credential
/// brokering is unavailable, but only if real credentials remain among the
/// credentials forwarded into the sandbox.
///
/// [rust] `console.warn` becomes a line on stderr.
pub fn warn_credential_brokering_unavailable(
    environment: &IndexMap<String, String>,
    forwarded_environment: &IndexMap<String, String>,
    credential_environment_variables: &[String],
) {
    if let Some(warning) = credential_brokering_unavailable_warning(
        environment,
        forwarded_environment,
        credential_environment_variables,
    ) {
        eprintln!("{warning}");
    }
}

fn credential_brokering_unavailable_warning(
    environment: &IndexMap<String, String>,
    forwarded_environment: &IndexMap<String, String>,
    credential_environment_variables: &[String],
) -> Option<&'static str> {
    let names: IndexSet<&String> = credential_environment_variables.iter().collect();
    let credentials: Vec<&String> = names
        .iter()
        .filter_map(|name| environment.get(*name))
        .filter(|credential| !credential.is_empty())
        .collect();
    let forwarded_credentials: Vec<&String> = names
        .iter()
        .filter_map(|name| forwarded_environment.get(*name))
        .collect();
    let preserved = credentials.iter().any(|credential| {
        forwarded_credentials
            .iter()
            .any(|forwarded| forwarded.contains(credential.as_str()))
    });
    preserved.then_some(CREDENTIAL_BROKERING_UNAVAILABLE_WARNING)
}

/// `maskSandboxCredentials({ environment, credentialEnvironmentVariables })`:
/// replaces each present credential with its own variable name.
pub fn mask_sandbox_credentials(
    environment: &IndexMap<String, String>,
    credential_environment_variables: &[String],
) -> IndexMap<String, String> {
    let mut masked_environment = environment.clone();
    for name in credential_environment_variables {
        if let Some(value) = masked_environment.get_mut(name) {
            *value = name.clone();
        }
    }
    masked_environment
}

/// `createCredentialRequestTransformation({ matchUrl, matchHeaders,
/// transformHeaders })`.
///
/// Fails like `new URL(matchUrl)` when `match_url` is not a valid URL.
pub fn create_credential_request_transformation(
    match_url: &str,
    match_headers: &IndexMap<String, String>,
    transform_headers: &IndexMap<String, String>,
) -> HarnessResult<HarnessRequestTransformation> {
    let url = reqwest::Url::parse(match_url)?;
    let pathname = url.path().trim_end_matches('/');
    Ok(HarnessRequestTransformation {
        r#match: HarnessRequestTransformationMatch {
            host: url.host_str().unwrap_or_default().to_string(),
            path: (!pathname.is_empty())
                .then(|| HarnessRequestTransformationMatcher::StartsWith(pathname.to_string())),
            method: None,
            query_string: None,
            headers: Some(
                match_headers
                    .iter()
                    .map(|(key, value)| HarnessRequestTransformationKeyValueMatcher {
                        key: Some(HarnessRequestTransformationMatcher::Exact(key.clone())),
                        value: Some(HarnessRequestTransformationMatcher::Exact(value.clone())),
                    })
                    .collect(),
            ),
        },
        transform: HarnessRequestTransformationTransform {
            headers: transform_headers.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::utils::ai_gateway_auth::env;
    use serde_json::json;

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn transformation(
        match_url: &str,
        header: &str,
        sandbox: &str,
        real: &str,
    ) -> serde_json::Value {
        serde_json::to_value(
            create_credential_request_transformation(
                match_url,
                &env(&[(header, sandbox)]),
                &env(&[(header, real)]),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn does_not_warn_when_credential_forwarding_replaces_all_credentials() {
        assert_eq!(
            credential_brokering_unavailable_warning(
                &env(&[
                    ("API_KEY", "real-secret"),
                    ("SECOND_API_KEY", "second-real-secret")
                ]),
                &env(&[
                    ("API_KEY", "caller-managed-credential"),
                    ("SECOND_API_KEY", "second-caller-managed-credential"),
                ]),
                &names(&["API_KEY", "SECOND_API_KEY"]),
            ),
            None
        );
    }

    #[test]
    fn warns_when_credential_forwarding_preserves_a_credential() {
        assert_eq!(
            credential_brokering_unavailable_warning(
                &env(&[
                    ("API_KEY", "real-secret"),
                    ("SECOND_API_KEY", "second-real-secret")
                ]),
                &env(&[
                    ("API_KEY", "caller-managed-credential"),
                    ("SECOND_API_KEY", "second-real-secret"),
                ]),
                &names(&["API_KEY", "SECOND_API_KEY"]),
            ),
            Some(
                "The sandbox implementation does not support configuring request transformations, so credential brokering does not work. Falling back to less secure credential forwarding."
            )
        );
    }

    #[test]
    fn warns_when_a_forwarded_credential_contains_an_original_credential() {
        assert_eq!(
            credential_brokering_unavailable_warning(
                &env(&[("API_KEY", "real-secret")]),
                &env(&[("API_KEY", "wrapped-real-secret")]),
                &names(&["API_KEY"]),
            ),
            Some(CREDENTIAL_BROKERING_UNAVAILABLE_WARNING)
        );
    }

    #[test]
    fn does_not_warn_when_no_original_credentials_are_present() {
        assert_eq!(
            credential_brokering_unavailable_warning(&env(&[]), &env(&[]), &names(&["API_KEY"])),
            None
        );
    }

    #[test]
    fn mask_replaces_credential_values_with_same_name_placeholders() {
        let environment = env(&[
            ("API_KEY", "real-secret"),
            ("SECOND_API_KEY", "second-secret"),
            ("BASE_URL", "https://api.example.com/v1"),
        ]);
        assert_eq!(
            mask_sandbox_credentials(&environment, &names(&["API_KEY", "SECOND_API_KEY"])),
            env(&[
                ("API_KEY", "API_KEY"),
                ("SECOND_API_KEY", "SECOND_API_KEY"),
                ("BASE_URL", "https://api.example.com/v1"),
            ])
        );
        assert_eq!(environment["API_KEY"], "real-secret");
    }

    #[test]
    fn mask_does_not_add_an_absent_credential_variable() {
        assert_eq!(
            mask_sandbox_credentials(&env(&[("OTHER", "value")]), &names(&["API_KEY"])),
            env(&[("OTHER", "value")])
        );
    }

    #[test]
    fn placeholders_are_recognizable_random_values() {
        let first = generate_sandbox_credential_placeholder();
        let second = generate_sandbox_credential_placeholder();
        assert!(is_sandbox_credential_placeholder(&first));
        assert!(is_sandbox_credential_placeholder(&second));
        assert_ne!(first, second);
    }

    #[test]
    fn placeholders_reject_values_outside_the_complete_format() {
        assert!(!is_sandbox_credential_placeholder("real-secret"));
        assert!(!is_sandbox_credential_placeholder("aisdkhc_"));
        assert!(!is_sandbox_credential_placeholder(&format!(
            "prefix-{}",
            generate_sandbox_credential_placeholder()
        )));
    }

    fn expected(path: Option<&str>, header: &str, sandbox: &str, real: &str) -> serde_json::Value {
        let mut matcher = json!({
            "host": "api.example.com",
            "headers": [{ "key": { "exact": header }, "value": { "exact": sandbox } }],
        });
        if let Some(path) = path {
            matcher["path"] = json!({ "startsWith": path });
        }
        json!({ "match": matcher, "transform": { "headers": { header: real } } })
    }

    #[test]
    fn maps_an_https_base_url_to_a_request_transformation() {
        assert_eq!(
            transformation(
                "https://api.example.com/v1",
                "Authorization",
                "Bearer sandbox-secret",
                "Bearer real-secret"
            ),
            expected(
                Some("/v1"),
                "Authorization",
                "Bearer sandbox-secret",
                "Bearer real-secret"
            )
        );
    }

    #[test]
    fn omits_a_root_path_matcher() {
        assert_eq!(
            transformation(
                "https://api.example.com/",
                "x-api-key",
                "sandbox-secret",
                "real-secret"
            ),
            expected(None, "x-api-key", "sandbox-secret", "real-secret")
        );
    }

    #[test]
    fn preserves_the_protocol_independent_host_and_path_match() {
        assert_eq!(
            transformation(
                "http://api.example.com/v1",
                "Authorization",
                "Bearer sandbox-secret",
                "Bearer real-secret"
            ),
            expected(
                Some("/v1"),
                "Authorization",
                "Bearer sandbox-secret",
                "Bearer real-secret"
            )
        );
    }

    #[test]
    fn does_not_include_the_base_url_port_in_the_domain_matcher() {
        assert_eq!(
            transformation(
                "https://api.example.com:8443/v1",
                "Authorization",
                "Bearer sandbox-secret",
                "Bearer real-secret"
            ),
            expected(
                Some("/v1"),
                "Authorization",
                "Bearer sandbox-secret",
                "Bearer real-secret"
            )
        );
    }

    #[test]
    fn matches_explicit_sandbox_headers_exactly() {
        assert_eq!(
            transformation(
                "https://api.example.com/v1",
                "Authorization",
                "Bearer aisdkhc_placeholder",
                "Bearer real-secret"
            ),
            expected(
                Some("/v1"),
                "Authorization",
                "Bearer aisdkhc_placeholder",
                "Bearer real-secret"
            )
        );
    }
}
