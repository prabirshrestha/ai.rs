//! Port of `oauth.ts` / `compat/extension-oauth-types.ts`: the legacy
//! callback-style OAuth surface, plus the pre-1.0 ai.rs conveniences built on
//! it (the [`OAuthLoginCallbacks::builder`] builder and a small OAuth
//! registry keyed by provider id).
//!
//! Pi 1.0 drives logins through `ProviderAuthInteraction`; this module adapts
//! legacy callbacks to that interaction. The registry has no Pi 1.0
//! counterpart (providers carry their `OAuthAuth` in `auth.oauth`); it is kept
//! so `get_oauth_provider("anthropic")` keeps working.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio_util::sync::CancellationToken;

use crate::auth::types::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthPromptKind, OAuthAuth, ProviderAuthInteraction,
};
use crate::{Error, Result};

pub use crate::auth::types::OAuthCredentials;

/// Legacy extension OAuth prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthPrompt {
    pub message: String,
    pub placeholder: Option<String>,
    pub allow_empty: bool,
}

/// Legacy extension OAuth authorization link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthAuthInfo {
    pub url: String,
    pub instructions: Option<String>,
}

/// Legacy extension OAuth device-code notification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthDeviceCodeInfo {
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: Option<u64>,
    pub expires_in_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSelectOption {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSelectPrompt {
    pub message: String,
    pub options: Vec<OAuthSelectOption>,
}

pub type OAuthPromptFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
pub type OAuthPromptCallback = Arc<dyn Fn(OAuthPrompt) -> OAuthPromptFuture + Send + Sync>;
pub type OAuthAuthCallback = Arc<dyn Fn(OAuthAuthInfo) + Send + Sync>;
pub type OAuthDeviceCodeCallback = Arc<dyn Fn(OAuthDeviceCodeInfo) + Send + Sync>;
pub type OAuthProgressCallback = Arc<dyn Fn(String) + Send + Sync>;
pub type OAuthManualCodeInputFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
pub type OAuthManualCodeInputCallback = Arc<dyn Fn() -> OAuthManualCodeInputFuture + Send + Sync>;
pub type OAuthSelectFuture = Pin<Box<dyn Future<Output = Result<Option<String>>> + Send>>;
pub type OAuthSelectCallback = Arc<dyn Fn(OAuthSelectPrompt) -> OAuthSelectFuture + Send + Sync>;

/// Callback surface retained for extension compatibility.
#[derive(Clone)]
pub struct OAuthLoginCallbacks {
    pub on_auth: Option<OAuthAuthCallback>,
    pub on_device_code: OAuthDeviceCodeCallback,
    pub on_prompt: OAuthPromptCallback,
    pub on_progress: Option<OAuthProgressCallback>,
    pub on_manual_code_input: Option<OAuthManualCodeInputCallback>,
    pub on_select: Option<OAuthSelectCallback>,
    pub signal: Option<CancellationToken>,
}

impl OAuthLoginCallbacks {
    pub fn builder() -> OAuthLoginCallbacksBuilder {
        OAuthLoginCallbacksBuilder::default()
    }
}

#[derive(Default)]
pub struct OAuthLoginCallbacksBuilder {
    on_auth: Option<OAuthAuthCallback>,
    on_device_code: Option<OAuthDeviceCodeCallback>,
    on_prompt: Option<OAuthPromptCallback>,
    on_progress: Option<OAuthProgressCallback>,
    on_manual_code_input: Option<OAuthManualCodeInputCallback>,
    on_select: Option<OAuthSelectCallback>,
    signal: Option<CancellationToken>,
}

impl OAuthLoginCallbacksBuilder {
    pub fn on_auth<F>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthAuthInfo) + Send + Sync + 'static,
    {
        self.on_auth = Some(Arc::new(callback));
        self
    }

    pub fn on_device_code<F>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthDeviceCodeInfo) + Send + Sync + 'static,
    {
        self.on_device_code = Some(Arc::new(callback));
        self
    }

    pub fn on_prompt<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthPrompt) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        self.on_prompt = Some(Arc::new(move |prompt| Box::pin(callback(prompt))));
        self
    }

    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        self.on_progress = Some(Arc::new(callback));
        self
    }

    pub fn on_manual_code_input<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        self.on_manual_code_input = Some(Arc::new(move || Box::pin(callback())));
        self
    }

    pub fn on_select<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthSelectPrompt) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        self.on_select = Some(Arc::new(move |prompt| Box::pin(callback(prompt))));
        self
    }

    pub fn signal(mut self, signal: CancellationToken) -> Self {
        self.signal = Some(signal);
        self
    }

    pub fn build(self) -> OAuthLoginCallbacks {
        OAuthLoginCallbacks {
            on_auth: self.on_auth,
            on_device_code: self.on_device_code.unwrap_or_else(|| Arc::new(|_| {})),
            on_prompt: self
                .on_prompt
                .unwrap_or_else(|| Arc::new(|_| Box::pin(async { Ok(String::new()) }))),
            on_progress: self.on_progress,
            on_manual_code_input: self.on_manual_code_input,
            on_select: self.on_select,
            signal: self.signal,
        }
    }
}

/// Adapts [`OAuthLoginCallbacks`] to the 1.0 [`AuthInteraction`].
struct CallbacksInteraction {
    callbacks: OAuthLoginCallbacks,
    /// Selected when no `on_select` callback is given.
    default_select: String,
}

async fn race_prompt(
    prompt: impl Future<Output = Result<String>>,
    signal: Option<CancellationToken>,
) -> Result<String> {
    match signal {
        Some(signal) => tokio::select! {
            _ = signal.cancelled() => Err(Error::message("Login cancelled")),
            result = prompt => result,
        },
        None => prompt.await,
    }
}

#[async_trait]
impl AuthInteraction for CallbacksInteraction {
    fn signal(&self) -> Option<CancellationToken> {
        self.callbacks.signal.clone()
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String> {
        let callbacks = &self.callbacks;
        match prompt.kind {
            AuthPromptKind::Text {
                message,
                placeholder,
            } => {
                let prompt_future = (callbacks.on_prompt)(OAuthPrompt {
                    message,
                    placeholder,
                    allow_empty: true,
                });
                race_prompt(prompt_future, prompt.signal).await
            }
            AuthPromptKind::Secret {
                message,
                placeholder,
            } => {
                let prompt_future = (callbacks.on_prompt)(OAuthPrompt {
                    message,
                    placeholder,
                    allow_empty: false,
                });
                race_prompt(prompt_future, prompt.signal).await
            }
            AuthPromptKind::ManualCode {
                message,
                placeholder,
            } => match &callbacks.on_manual_code_input {
                Some(manual) => race_prompt(manual(), prompt.signal).await,
                None => {
                    let prompt_future = (callbacks.on_prompt)(OAuthPrompt {
                        message,
                        placeholder,
                        allow_empty: false,
                    });
                    race_prompt(prompt_future, prompt.signal).await
                }
            },
            AuthPromptKind::Select { message, options } => {
                let Some(on_select) = &callbacks.on_select else {
                    return Ok(self.default_select.clone());
                };
                let selected = on_select(OAuthSelectPrompt {
                    message,
                    options: options
                        .into_iter()
                        .map(|option| OAuthSelectOption {
                            id: option.id,
                            label: option.label,
                        })
                        .collect(),
                })
                .await?;
                selected.ok_or_else(|| Error::message("Login cancelled"))
            }
        }
    }

    fn notify(&self, event: AuthEvent) {
        let callbacks = &self.callbacks;
        match event {
            AuthEvent::AuthUrl { url, instructions } => {
                if let Some(on_auth) = &callbacks.on_auth {
                    on_auth(OAuthAuthInfo { url, instructions });
                }
            }
            AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                interval_seconds,
                expires_in_seconds,
            } => (callbacks.on_device_code)(OAuthDeviceCodeInfo {
                user_code,
                verification_uri,
                interval_seconds,
                expires_in_seconds,
            }),
            AuthEvent::Progress { message } | AuthEvent::Info { message, .. } => {
                if let Some(on_progress) = &callbacks.on_progress {
                    on_progress(message);
                }
            }
        }
    }
}

/// A [`ProviderAuthInteraction`] driven by legacy callbacks.
pub(crate) fn callbacks_interaction(
    callbacks: OAuthLoginCallbacks,
    default_select: &str,
) -> ProviderAuthInteraction {
    let signal = callbacks.signal.clone().unwrap_or_default();
    ProviderAuthInteraction {
        interaction: Arc::new(CallbacksInteraction {
            callbacks,
            default_select: default_select.to_string(),
        }),
        signal,
    }
}

fn oauth_registry() -> &'static RwLock<HashMap<String, Arc<dyn OAuthAuth>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Arc<dyn OAuthAuth>>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

fn builtin_oauth_provider(id: &str) -> Option<Arc<dyn OAuthAuth>> {
    match id {
        "anthropic" => Some(super::anthropic::anthropic_oauth()),
        "github-copilot" => Some(super::github_copilot::github_copilot_oauth()),
        _ => None,
    }
}

/// The OAuth implementation registered for `id`, else the built-in one
/// (`anthropic`, `github-copilot`).
pub fn get_oauth_provider(id: &str) -> Option<Arc<dyn OAuthAuth>> {
    oauth_registry()
        .read()
        .get(id)
        .cloned()
        .or_else(|| builtin_oauth_provider(id))
}

/// Register (or replace) the OAuth implementation for a provider id.
pub fn register_oauth_provider(id: impl Into<String>, oauth: Arc<dyn OAuthAuth>) {
    oauth_registry().write().insert(id.into(), oauth);
}

/// Remove a registered implementation; built-ins come back.
pub fn unregister_oauth_provider(id: &str) {
    oauth_registry().write().remove(id);
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;
    use crate::auth::types::{AuthSelectOption, LoginOptions, ModelAuth, OAuthCredential};

    struct TestOAuth;

    #[async_trait]
    impl OAuthAuth for TestOAuth {
        fn name(&self) -> &str {
            "Test OAuth"
        }

        async fn login(
            &self,
            _interaction: ProviderAuthInteraction,
            _options: LoginOptions,
        ) -> Result<OAuthCredential> {
            Ok(OAuthCredential::default())
        }

        async fn refresh(
            &self,
            credential: OAuthCredential,
            _signal: CancellationToken,
        ) -> Result<OAuthCredential> {
            Ok(credential)
        }

        async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth> {
            Ok(ModelAuth {
                api_key: Some(credential.access.clone()),
                ..Default::default()
            })
        }
    }

    #[test]
    fn registry_serves_builtins_and_registered_overrides() {
        assert_eq!(
            get_oauth_provider("anthropic").unwrap().name(),
            "Anthropic (Claude Pro/Max)"
        );
        assert_eq!(
            get_oauth_provider("github-copilot").unwrap().name(),
            "GitHub Copilot"
        );
        assert!(get_oauth_provider("test-oauth-compat").is_none());
        register_oauth_provider("test-oauth-compat", Arc::new(TestOAuth));
        assert_eq!(
            get_oauth_provider("test-oauth-compat").unwrap().name(),
            "Test OAuth"
        );
        unregister_oauth_provider("test-oauth-compat");
        assert!(get_oauth_provider("test-oauth-compat").is_none());
    }

    #[tokio::test]
    async fn adapts_legacy_callbacks_to_the_auth_interaction() {
        let progress = Arc::new(Mutex::new(Vec::new()));
        let seen_progress = progress.clone();
        let devices = Arc::new(Mutex::new(Vec::new()));
        let seen_devices = devices.clone();
        let callbacks = OAuthLoginCallbacks::builder()
            .on_prompt(|prompt| async move {
                Ok(format!(
                    "{}:{}",
                    prompt.placeholder.unwrap_or_default(),
                    prompt.allow_empty
                ))
            })
            .on_manual_code_input(|| async { Ok("manual".to_string()) })
            .on_select(|prompt| async move { Ok(prompt.options.last().map(|o| o.id.clone())) })
            .on_progress(move |message| seen_progress.lock().push(message))
            .on_device_code(move |info| seen_devices.lock().push(info))
            .build();
        let interaction = callbacks_interaction(callbacks, "first");
        let text = interaction
            .prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Text {
                    message: "m".to_string(),
                    placeholder: Some("company.ghe.com".to_string()),
                },
            })
            .await
            .unwrap();
        assert_eq!(text, "company.ghe.com:true");
        let manual = interaction
            .prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::ManualCode {
                    message: "m".to_string(),
                    placeholder: None,
                },
            })
            .await
            .unwrap();
        assert_eq!(manual, "manual");
        let select = |options: Vec<&str>| AuthPrompt {
            signal: None,
            kind: AuthPromptKind::Select {
                message: "pick".to_string(),
                options: options
                    .into_iter()
                    .map(|id| AuthSelectOption {
                        id: id.to_string(),
                        label: id.to_string(),
                        description: None,
                    })
                    .collect(),
            },
        };
        assert_eq!(
            interaction
                .prompt(select(vec!["first", "second"]))
                .await
                .unwrap(),
            "second"
        );
        assert_eq!(
            interaction
                .prompt(select(vec![]))
                .await
                .unwrap_err()
                .to_string(),
            "Login cancelled"
        );
        interaction.notify(AuthEvent::Progress {
            message: "working".to_string(),
        });
        interaction.notify(AuthEvent::DeviceCode {
            user_code: "ABCD".to_string(),
            verification_uri: "https://github.com/login/device".to_string(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(900),
        });
        assert_eq!(*progress.lock(), vec!["working".to_string()]);
        assert_eq!(devices.lock()[0].user_code, "ABCD");

        let defaulted = callbacks_interaction(OAuthLoginCallbacks::builder().build(), "browser");
        assert_eq!(
            defaulted.prompt(select(vec!["browser"])).await.unwrap(),
            "browser"
        );
    }

    #[tokio::test]
    async fn manual_prompts_stop_when_their_signal_aborts() {
        let callbacks = OAuthLoginCallbacks::builder()
            .on_prompt(|_| std::future::pending())
            .build();
        let interaction = callbacks_interaction(callbacks, "first");
        let signal = CancellationToken::new();
        signal.cancel();
        let error = interaction
            .prompt(AuthPrompt {
                signal: Some(signal),
                kind: AuthPromptKind::ManualCode {
                    message: "m".to_string(),
                    placeholder: None,
                },
            })
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "Login cancelled");
    }
}
