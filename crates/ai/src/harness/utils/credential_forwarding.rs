//! Port of `utils/credential-forwarding.ts`.

use indexmap::{IndexMap, IndexSet};

use crate::harness::errors::HarnessResult;
use crate::harness::spec::{HarnessV1CredentialForwarding, HarnessV1CredentialForwardingOptions};
use crate::harness::utils::sandbox_credential_brokering::generate_sandbox_credential_placeholder;

/// `applyCredentialForwarding({ environment, credentialEnvironmentVariables,
/// credentialForwarding })`: passes each present credential through the
/// callback, once per distinct name.
pub async fn apply_credential_forwarding(
    environment: &IndexMap<String, String>,
    credential_environment_variables: &[String],
    credential_forwarding: Option<&HarnessV1CredentialForwarding>,
) -> HarnessResult<IndexMap<String, String>> {
    let mut forwarded_environment = environment.clone();
    let Some(credential_forwarding) = credential_forwarding else {
        return Ok(forwarded_environment);
    };

    let names: IndexSet<&String> = credential_environment_variables.iter().collect();
    for environment_variable_name in names {
        let Some(credential) = forwarded_environment
            .get(environment_variable_name)
            .cloned()
        else {
            continue;
        };
        let forwarded = credential_forwarding(HarnessV1CredentialForwardingOptions {
            credential,
            environment_variable_name: environment_variable_name.clone(),
        })
        .await?;
        forwarded_environment.insert(environment_variable_name.clone(), forwarded);
    }

    Ok(forwarded_environment)
}

/// `resolveSandboxCredentialEnvironment(...)`: the credential values a
/// sandbox sees. Saved values are reused; new names get a fresh placeholder,
/// passed through the callback when one is given.
pub async fn resolve_sandbox_credential_environment(
    environment: &IndexMap<String, String>,
    credential_environment_variables: &[String],
    credential_forwarding: Option<&HarnessV1CredentialForwarding>,
    previous_sandbox_credential_environment: Option<&IndexMap<String, String>>,
) -> HarnessResult<IndexMap<String, String>> {
    let mut sandbox_credential_environment = IndexMap::new();

    let names: IndexSet<&String> = credential_environment_variables.iter().collect();
    for environment_variable_name in names {
        if !environment.contains_key(environment_variable_name) {
            continue;
        }

        if let Some(previous) = previous_sandbox_credential_environment
            .and_then(|previous| previous.get(environment_variable_name))
        {
            sandbox_credential_environment
                .insert(environment_variable_name.clone(), previous.clone());
            continue;
        }

        let placeholder = generate_sandbox_credential_placeholder();
        let value = match credential_forwarding {
            None => placeholder,
            Some(credential_forwarding) => {
                credential_forwarding(HarnessV1CredentialForwardingOptions {
                    credential: placeholder,
                    environment_variable_name: environment_variable_name.clone(),
                })
                .await?
            }
        };
        sandbox_credential_environment.insert(environment_variable_name.clone(), value);
    }

    Ok(sandbox_credential_environment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::utils::ai_gateway_auth::env;
    use crate::harness::utils::is_sandbox_credential_placeholder;
    use crate::harness::utils::plain_error;
    use futures::FutureExt;
    use parking_lot::Mutex;
    use std::sync::Arc;

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    /// `vi.fn(callback)`: a forwarding callback that records its calls.
    fn recording(
        callback: impl Fn(&HarnessV1CredentialForwardingOptions) -> HarnessResult<String>
        + Send
        + Sync
        + 'static,
    ) -> (
        HarnessV1CredentialForwarding,
        Arc<Mutex<Vec<HarnessV1CredentialForwardingOptions>>>,
    ) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let forwarding: HarnessV1CredentialForwarding = Arc::new(move |options| {
            let result = callback(&options);
            recorded.lock().push(options);
            async move { result }.boxed()
        });
        (forwarding, calls)
    }

    fn options(credential: &str, name: &str) -> HarnessV1CredentialForwardingOptions {
        HarnessV1CredentialForwardingOptions {
            credential: credential.into(),
            environment_variable_name: name.into(),
        }
    }

    #[tokio::test]
    async fn apply_preserves_the_environment_when_no_callback_is_provided() {
        let environment = env(&[
            ("API_KEY", "real-secret"),
            ("BASE_URL", "https://api.example.com"),
        ]);
        assert_eq!(
            apply_credential_forwarding(&environment, &names(&["API_KEY"]), None)
                .await
                .unwrap(),
            environment
        );
    }

    #[tokio::test]
    async fn apply_forwards_each_present_credential_separately() {
        let (forwarding, calls) =
            recording(|options| Ok(format!("ephemeral-{}", options.environment_variable_name)));
        let result = apply_credential_forwarding(
            &env(&[
                ("API_KEY", "real-secret"),
                ("SECOND_API_KEY", "SECOND_API_KEY"),
                ("BASE_URL", "https://api.example.com"),
            ]),
            &names(&["API_KEY", "SECOND_API_KEY", "API_KEY", "MISSING_API_KEY"]),
            Some(&forwarding),
        )
        .await
        .unwrap();
        assert_eq!(
            result,
            env(&[
                ("API_KEY", "ephemeral-API_KEY"),
                ("SECOND_API_KEY", "ephemeral-SECOND_API_KEY"),
                ("BASE_URL", "https://api.example.com"),
            ])
        );
        assert_eq!(
            *calls.lock(),
            vec![
                options("real-secret", "API_KEY"),
                options("SECOND_API_KEY", "SECOND_API_KEY"),
            ]
        );
    }

    #[tokio::test]
    async fn apply_supports_callbacks_that_resolve_immediately() {
        let (forwarding, _) = recording(|options| Ok(format!("wrapped-{}", options.credential)));
        assert_eq!(
            apply_credential_forwarding(
                &env(&[("API_KEY", "real-secret")]),
                &names(&["API_KEY"]),
                Some(&forwarding),
            )
            .await
            .unwrap(),
            env(&[("API_KEY", "wrapped-real-secret")])
        );
    }

    #[tokio::test]
    async fn apply_propagates_callback_errors_without_using_the_original_credential() {
        let (forwarding, _) = recording(|_| Err(plain_error("credential forwarding failed")));
        let error = apply_credential_forwarding(
            &env(&[("API_KEY", "real-secret")]),
            &names(&["API_KEY"]),
            Some(&forwarding),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "credential forwarding failed");
    }

    #[tokio::test]
    async fn resolve_forwards_generated_placeholders_and_returns_the_exact_results() {
        let (forwarding, calls) =
            recording(|options| Ok(format!("wrapped-{}", options.credential)));
        let result = resolve_sandbox_credential_environment(
            &env(&[
                ("API_KEY", "real-secret"),
                ("SECOND_API_KEY", "second-real-secret"),
                ("BASE_URL", "https://api.example.com"),
            ]),
            &names(&["API_KEY", "SECOND_API_KEY"]),
            Some(&forwarding),
            None,
        )
        .await
        .unwrap();
        let calls = calls.lock();
        assert_eq!(calls.len(), 2);
        for call in calls.iter() {
            assert!(is_sandbox_credential_placeholder(&call.credential));
        }
        assert!(result["API_KEY"].starts_with("wrapped-aisdkhc_"));
        assert!(result["SECOND_API_KEY"].starts_with("wrapped-aisdkhc_"));
    }

    #[tokio::test]
    async fn resolve_does_not_add_absent_credential_variables() {
        assert_eq!(
            resolve_sandbox_credential_environment(
                &env(&[("BASE_URL", "https://api.example.com")]),
                &names(&["API_KEY"]),
                None,
                None,
            )
            .await
            .unwrap(),
            env(&[])
        );
    }

    #[tokio::test]
    async fn resolve_reuses_only_current_credentials_and_creates_placeholders_for_new_names() {
        let previous = env(&[
            ("API_KEY", "saved-placeholder"),
            ("REMOVED_API_KEY", "removed-placeholder"),
        ]);
        let (forwarding, calls) =
            recording(|options| Ok(format!("wrapped-{}", options.credential)));
        let result = resolve_sandbox_credential_environment(
            &env(&[
                ("API_KEY", "rotated-secret"),
                ("NEW_API_KEY", "new-secret"),
                ("BASE_URL", "https://api.example.com"),
            ]),
            &names(&["API_KEY", "NEW_API_KEY", "NEW_API_KEY", "REMOVED_API_KEY"]),
            Some(&forwarding),
            Some(&previous),
        )
        .await
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result["API_KEY"], "saved-placeholder");
        assert!(result["NEW_API_KEY"].starts_with("wrapped-aisdkhc_"));
        let calls = calls.lock();
        assert_eq!(calls.len(), 1);
        assert!(is_sandbox_credential_placeholder(&calls[0].credential));
        assert_eq!(calls[0].environment_variable_name, "NEW_API_KEY");
        assert_eq!(
            previous,
            env(&[
                ("API_KEY", "saved-placeholder"),
                ("REMOVED_API_KEY", "removed-placeholder"),
            ])
        );
    }

    #[tokio::test]
    async fn resolve_keeps_saved_credentials_with_custom_formats_without_forwarding_them_again() {
        let (forwarding, calls) = recording(|_| Ok("another-placeholder".into()));
        assert_eq!(
            resolve_sandbox_credential_environment(
                &env(&[("GITHUB_TOKEN", "rotated-secret")]),
                &names(&["GITHUB_TOKEN"]),
                Some(&forwarding),
                Some(&env(&[("GITHUB_TOKEN", "gho_saved-placeholder")])),
            )
            .await
            .unwrap(),
            env(&[("GITHUB_TOKEN", "gho_saved-placeholder")])
        );
        assert!(calls.lock().is_empty());
    }

    #[tokio::test]
    async fn resolve_generates_placeholders_when_older_lifecycle_state_has_no_saved_map() {
        let result = resolve_sandbox_credential_environment(
            &env(&[("API_KEY", "host-secret")]),
            &names(&["API_KEY"]),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(is_sandbox_credential_placeholder(&result["API_KEY"]));
        assert!(
            !serde_json::to_string(&result)
                .unwrap()
                .contains("host-secret")
        );
    }

    #[tokio::test]
    async fn resolve_propagates_forwarding_failures_before_returning_a_partial_environment() {
        let (forwarding, _) = recording(|_| Err(plain_error("forwarding failed")));
        let error = resolve_sandbox_credential_environment(
            &env(&[("API_KEY", "host-secret")]),
            &names(&["API_KEY"]),
            Some(&forwarding),
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "forwarding failed");
    }
}
