//! Port of `utils/native-subscription/should-resolve-native.ts`.

use indexmap::IndexMap;
use serde::Serialize;

use crate::harness::HarnessAuthentication;
use crate::harness::utils::ai_gateway_auth::get_ai_gateway_auth_from_env;

/// `shouldResolveNativeSubscription({ auth, env, hasDirectCredential })`:
/// whether an adapter should look up the native CLI subscription.
///
/// [rust] TS narrows `auth` to its string modes; an `Environment` auth is
/// treated like any non-`direct` mode.
pub fn should_resolve_native_subscription<C: Serialize>(
    auth: Option<&HarnessAuthentication<C>>,
    env: &IndexMap<String, String>,
    has_direct_credential: bool,
) -> bool {
    let is_direct = matches!(
        auth,
        Some(HarnessAuthentication::Choice(choice))
            if serde_json::to_value(choice).ok().as_ref().and_then(|v| v.as_str()) == Some("direct")
    );
    !matches!(auth, Some(HarnessAuthentication::AiGateway))
        && !has_direct_credential
        && (is_direct || get_ai_gateway_auth_from_env(env).api_key.is_none())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::HarnessDirectAuthentication;
    use crate::harness::utils::ai_gateway_auth::env;

    #[test]
    fn returns_the_expected_result_per_case() {
        type A = HarnessAuthentication;
        let direct = A::Choice(HarnessDirectAuthentication::Direct);
        type Case<'a> = (Option<A>, &'a [(&'a str, &'a str)], bool, bool);
        let cases: [Case; 7] = [
            (None, &[], false, true),
            (Some(A::Auto), &[], false, true),
            (
                Some(A::Auto),
                &[("AI_GATEWAY_API_KEY", "gateway-key")],
                false,
                false,
            ),
            (None, &[("VERCEL_OIDC_TOKEN", "oidc-token")], false, false),
            (
                Some(direct.clone()),
                &[("AI_GATEWAY_API_KEY", "gateway-key")],
                false,
                true,
            ),
            (Some(A::AiGateway), &[], false, false),
            (Some(direct), &[], true, false),
        ];
        for (auth, pairs, has_direct_credential, expected) in cases {
            assert_eq!(
                should_resolve_native_subscription(
                    auth.as_ref(),
                    &env(pairs),
                    has_direct_credential
                ),
                expected,
                "auth {auth:?}, env {pairs:?}, hasDirectCredential {has_direct_credential}"
            );
        }
    }
}
