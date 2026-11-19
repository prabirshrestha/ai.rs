//! Port of `utils/oauth-access-token.ts`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::harness::error::HarnessResult;
use crate::harness::utils::native_subscription::get_jwt_expires_at;
use crate::harness::utils::plain_error;

const DEFAULT_REFRESH_WINDOW_MS: f64 = 300_000.0;

/// `OAuthCredential`. Times are epoch milliseconds, as in TS.
#[derive(Debug, Clone, PartialEq)]
pub struct OAuthCredential {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: f64,
}

/// `RefreshOAuthAccessTokenResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct RefreshOAuthAccessTokenResult {
    pub access_token: String,
    pub expires_at: f64,
    /// Present only when the server rotated the refresh token.
    pub refresh_token: Option<String>,
}

/// `requestFormat`: `'json' | 'form'`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OAuthRequestFormat {
    Json,
    #[default]
    Form,
}

/// The request a [`OAuthFetch`] sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthFetchRequest {
    pub url: String,
    pub method: String,
    pub headers: IndexMap<String, String>,
    pub body: String,
}

/// The response a [`OAuthFetch`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthFetchResponse {
    pub status: u16,
    pub text: String,
}

/// The `fetch` option: `typeof globalThis.fetch`, reduced to what a token
/// refresh needs.
pub type OAuthFetch = Arc<
    dyn Fn(OAuthFetchRequest) -> BoxFuture<'static, HarnessResult<OAuthFetchResponse>>
        + Send
        + Sync,
>;

/// Options of [`refresh_oauth_access_token`].
#[derive(Clone, Default)]
pub struct RefreshOAuthAccessTokenOptions {
    pub token_url: String,
    pub client_id: String,
    pub refresh_token: String,
    pub request_format: OAuthRequestFormat,
    pub headers: Option<IndexMap<String, String>>,
    /// Defaults to a `reqwest` client.
    pub fetch: Option<OAuthFetch>,
}

impl RefreshOAuthAccessTokenOptions {
    pub fn new(
        token_url: impl Into<String>,
        client_id: impl Into<String>,
        refresh_token: impl Into<String>,
    ) -> Self {
        Self {
            token_url: token_url.into(),
            client_id: client_id.into(),
            refresh_token: refresh_token.into(),
            ..Self::default()
        }
    }
}

/// `Date.now()`.
pub(crate) fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as f64)
        .unwrap_or(0.0)
}

/// `isAccessTokenExpiringSoon({ expiresAt, now, refreshWindowMs })`: whether
/// the token expires within the window (five minutes by default).
pub fn is_access_token_expiring_soon(
    expires_at: f64,
    now: Option<f64>,
    refresh_window_ms: Option<f64>,
) -> bool {
    expires_at
        <= now.unwrap_or_else(now_ms) + refresh_window_ms.unwrap_or(DEFAULT_REFRESH_WINDOW_MS)
}

fn default_fetch() -> OAuthFetch {
    Arc::new(|request: OAuthFetchRequest| {
        async move {
            let client = reqwest::Client::new();
            let mut builder = client.request(
                reqwest::Method::from_bytes(request.method.as_bytes())?,
                &request.url,
            );
            for (name, value) in &request.headers {
                builder = builder.header(name, value);
            }
            let response = builder.body(request.body).send().await?;
            let status = response.status().as_u16();
            let text = response.text().await?;
            Ok(OAuthFetchResponse { status, text })
        }
        .boxed()
    })
}

#[derive(Serialize)]
struct RefreshGrant<'a> {
    grant_type: &'a str,
    client_id: &'a str,
    refresh_token: &'a str,
}

/// `refreshOAuthAccessToken(options)`: exchanges a refresh token for a new
/// access token. Errors never include token values.
pub async fn refresh_oauth_access_token(
    options: RefreshOAuthAccessTokenOptions,
) -> HarnessResult<RefreshOAuthAccessTokenResult> {
    let values = RefreshGrant {
        grant_type: "refresh_token",
        client_id: &options.client_id,
        refresh_token: &options.refresh_token,
    };
    let json = options.request_format == OAuthRequestFormat::Json;
    let mut headers = IndexMap::from([(
        "content-type".to_string(),
        if json {
            "application/json"
        } else {
            "application/x-www-form-urlencoded"
        }
        .to_string(),
    )]);
    headers.extend(options.headers.clone().unwrap_or_default());
    let body = if json {
        serde_json::to_string(&values)?
    } else {
        // `new URLSearchParams(values).toString()`.
        let mut url = reqwest::Url::parse("http://localhost/").expect("valid URL");
        url.query_pairs_mut().extend_pairs([
            ("grant_type", values.grant_type),
            ("client_id", values.client_id),
            ("refresh_token", values.refresh_token),
        ]);
        url.query().unwrap_or_default().to_string()
    };

    let fetch = options.fetch.clone().unwrap_or_else(default_fetch);
    let response = fetch(OAuthFetchRequest {
        url: options.token_url.clone(),
        method: "POST".to_string(),
        headers,
        body,
    })
    .await?;

    if !(200..=299).contains(&response.status) {
        return Err(plain_error(format!(
            "OAuth access token refresh failed with status {}.",
            response.status
        )));
    }

    let parsed = match serde_json::from_str::<Value>(&response.text) {
        Ok(Value::Object(parsed)) => parsed,
        _ => {
            return Err(plain_error(
                "OAuth access token refresh returned invalid JSON.",
            ));
        }
    };

    let access_token = match parsed.get("access_token") {
        Some(Value::String(token)) if !token.is_empty() => token.clone(),
        _ => {
            return Err(plain_error(
                "OAuth access token refresh response is missing access_token.",
            ));
        }
    };

    let expires_at = resolve_expires_at(&access_token, parsed.get("expires_in"))?;
    let refresh_token = match parsed.get("refresh_token") {
        None | Some(Value::Null) => None,
        Some(Value::String(token)) if !token.is_empty() => Some(token.clone()),
        Some(_) => {
            return Err(plain_error(
                "OAuth access token refresh response contains an invalid refresh_token.",
            ));
        }
    };

    Ok(RefreshOAuthAccessTokenResult {
        access_token,
        expires_at,
        refresh_token,
    })
}

fn resolve_expires_at(access_token: &str, expires_in: Option<&Value>) -> HarnessResult<f64> {
    if let Some(expires_in) = expires_in.and_then(Value::as_f64)
        && expires_in.is_finite()
        && expires_in >= 0.0
    {
        return Ok(now_ms() + expires_in * 1000.0);
    }

    if let Some(jwt_expires_at) = get_jwt_expires_at(access_token) {
        return Ok(jwt_expires_at);
    }

    Err(plain_error(
        "OAuth access token refresh response does not include a usable expiry.",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::utils::native_subscription::jwt::create_jwt;
    use parking_lot::Mutex;
    use serde_json::json;

    /// `vi.fn(async () => response)`.
    fn mock_fetch(
        status: u16,
        text: impl Into<String>,
    ) -> (OAuthFetch, Arc<Mutex<Vec<OAuthFetchRequest>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let text = text.into();
        let fetch: OAuthFetch = Arc::new(move |request| {
            recorded.lock().push(request);
            let response = OAuthFetchResponse {
                status,
                text: text.clone(),
            };
            async move { Ok(response) }.boxed()
        });
        (fetch, calls)
    }

    fn options(fetch: OAuthFetch, refresh_token: &str) -> RefreshOAuthAccessTokenOptions {
        RefreshOAuthAccessTokenOptions {
            fetch: Some(fetch),
            ..RefreshOAuthAccessTokenOptions::new(
                "https://example.com/token",
                "client",
                refresh_token,
            )
        }
    }

    #[test]
    fn uses_an_inclusive_five_minute_refresh_window() {
        assert!(is_access_token_expiring_soon(
            400_000.0,
            Some(100_000.0),
            None
        ));
        assert!(!is_access_token_expiring_soon(
            400_001.0,
            Some(100_000.0),
            None
        ));
    }

    #[test]
    fn supports_a_custom_refresh_window() {
        assert!(is_access_token_expiring_soon(
            102_000.0,
            Some(100_000.0),
            Some(2_000.0)
        ));
    }

    #[tokio::test]
    async fn sends_a_form_refresh_grant_and_preserves_an_omitted_refresh_token() {
        let (fetch, calls) = mock_fetch(
            200,
            json!({ "access_token": "access", "expires_in": 60 }).to_string(),
        );
        let before = now_ms();
        let result = refresh_oauth_access_token(options(fetch, "refresh"))
            .await
            .unwrap();
        let after = now_ms();
        assert_eq!(result.access_token, "access");
        assert_eq!(result.refresh_token, None);
        // [rust] TS freezes the clock at 1_000 and expects 61_000.
        assert!(result.expires_at >= before + 60_000.0 && result.expires_at <= after + 60_000.0);
        assert_eq!(
            *calls.lock(),
            vec![OAuthFetchRequest {
                url: "https://example.com/token".into(),
                method: "POST".into(),
                headers: IndexMap::from([(
                    "content-type".into(),
                    "application/x-www-form-urlencoded".into()
                )]),
                body: "grant_type=refresh_token&client_id=client&refresh_token=refresh".into(),
            }]
        );
    }

    #[tokio::test]
    async fn sends_json_and_returns_a_rotated_refresh_token() {
        let (fetch, calls) = mock_fetch(
            200,
            json!({ "access_token": "access", "refresh_token": "rotated", "expires_in": 60 })
                .to_string(),
        );
        let result = refresh_oauth_access_token(RefreshOAuthAccessTokenOptions {
            request_format: OAuthRequestFormat::Json,
            headers: Some(IndexMap::from([("x-client".into(), "test".into())])),
            ..options(fetch, "refresh")
        })
        .await
        .unwrap();
        assert_eq!(result.access_token, "access");
        assert_eq!(result.refresh_token.as_deref(), Some("rotated"));
        let calls = calls.lock();
        assert_eq!(
            calls[0].headers,
            IndexMap::from([
                ("content-type".to_string(), "application/json".to_string()),
                ("x-client".to_string(), "test".to_string()),
            ])
        );
        assert_eq!(
            calls[0].body,
            r#"{"grant_type":"refresh_token","client_id":"client","refresh_token":"refresh"}"#
        );
    }

    #[tokio::test]
    async fn derives_expiry_from_a_jwt_exp_claim() {
        let token = create_jwt(&json!({ "exp": 1234 }));
        let (fetch, _) = mock_fetch(200, json!({ "access_token": token }).to_string());
        assert_eq!(
            refresh_oauth_access_token(options(fetch, "refresh"))
                .await
                .unwrap(),
            RefreshOAuthAccessTokenResult {
                access_token: token,
                expires_at: 1_234_000.0,
                refresh_token: None,
            }
        );
    }

    #[tokio::test]
    async fn rejects_unsuccessful_and_malformed_responses_without_exposing_tokens() {
        let (fetch, _) = mock_fetch(401, "sensitive response");
        let error = refresh_oauth_access_token(options(fetch, "sensitive-refresh-token"))
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "OAuth access token refresh failed with status 401."
        );

        let (fetch, _) = mock_fetch(200, json!({ "access_token": "not-a-jwt" }).to_string());
        let error = refresh_oauth_access_token(options(fetch, "refresh"))
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "OAuth access token refresh response does not include a usable expiry."
        );
    }
}
