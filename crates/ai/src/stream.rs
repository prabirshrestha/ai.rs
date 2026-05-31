use crate::AssistantMessageEventStream;
use crate::api_registry::get_api_provider;
use crate::env_api_keys::get_env_api_key;
use crate::providers::register_builtins::ensure_builtins_registered;
use crate::types::{AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{Error, Result};

fn has_explicit_api_key(api_key: &Option<String>) -> bool {
    api_key
        .as_deref()
        .is_some_and(|api_key| !api_key.trim().is_empty())
}

fn with_env_api_key(model: &Model, mut options: StreamOptions) -> StreamOptions {
    if !has_explicit_api_key(&options.api_key)
        && let Some(api_key) = get_env_api_key(&model.provider)
    {
        options.api_key = Some(api_key);
    }
    options
}

fn with_env_api_key_simple(model: &Model, mut options: SimpleStreamOptions) -> SimpleStreamOptions {
    options.stream = with_env_api_key(model, options.stream);
    options
}

pub fn stream(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessageEventStream> {
    ensure_builtins_registered();
    let provider =
        get_api_provider(&model.api).ok_or_else(|| Error::NoApiProvider(model.api.clone()))?;
    let options = with_env_api_key(&model, options.unwrap_or_default());
    (provider.stream)(model, context, options)
}

pub async fn complete(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessage> {
    let mut stream = stream(model, context, options)?;
    while futures::StreamExt::next(&mut stream).await.is_some() {}
    stream.result().await
}

pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessageEventStream> {
    ensure_builtins_registered();
    let provider =
        get_api_provider(&model.api).ok_or_else(|| Error::NoApiProvider(model.api.clone()))?;
    let options = with_env_api_key_simple(&model, options.unwrap_or_default());
    (provider.stream_simple)(model, context, options)
}

pub async fn complete_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessage> {
    let mut stream = stream_simple(model, context, options)?;
    while futures::StreamExt::next(&mut stream).await.is_some() {}
    stream.result().await
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::api_registry::{
        ApiProvider, register_api_provider, unregister_api_providers, wrap_stream,
        wrap_stream_simple,
    };
    use crate::types::{
        AssistantContent, AssistantMessageEvent, ModelCost, ModelInput, StopReason, TextContent,
        Usage,
    };

    use super::*;

    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    struct SavedEnv {
        key: &'static str,
        value: Option<String>,
    }

    impl SavedEnv {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                value: std::env::var(key).ok(),
            }
        }

        fn restore(self) {
            unsafe {
                if let Some(value) = self.value {
                    std::env::set_var(self.key, value);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    fn test_model(api: &str) -> Model {
        Model {
            id: "mock".to_string(),
            name: "mock".to_string(),
            api: api.to_string(),
            provider: "openai".to_string(),
            base_url: "https://example.invalid".to_string(),
            reasoning: false,
            input: vec![ModelInput::Text],
            cost: ModelCost::default(),
            context_window: 8192,
            max_tokens: 2048,
            ..Model::default()
        }
    }

    fn done_stream(model: &Model) -> AssistantMessageEventStream {
        let message = AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: "ok".to_string(),
                text_signature: None,
            })],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        };
        let reason = message.stop_reason;
        let (mut sender, stream) = AssistantMessageEventStream::channel();
        sender.push(AssistantMessageEvent::Done {
            reason,
            message: message.clone(),
        });
        stream
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_simple_injects_env_api_key_before_provider_dispatch() {
        let _guard = ENV_LOCK.lock().await;
        let openai = SavedEnv::capture("OPENAI_API_KEY");
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "env-openai-key");
        }

        let source_id = "stream-env-key-test";
        let observed_key = Arc::new(Mutex::new(None));
        register_api_provider(
            ApiProvider {
                api: "stream-env-key-test".to_string(),
                stream: wrap_stream("stream-env-key-test", |_model, _context, _options| {
                    panic!("stream should not be called")
                }),
                stream_simple: wrap_stream_simple("stream-env-key-test", {
                    let observed_key = Arc::clone(&observed_key);
                    move |model, _context, options| {
                        *observed_key.lock().expect("observed key lock poisoned") =
                            options.stream.api_key.clone();
                        Ok(done_stream(&model))
                    }
                }),
            },
            Some(source_id.to_string()),
        );

        let mut events = stream_simple(test_model("stream-env-key-test"), Context::default(), None)
            .expect("stream_simple should dispatch");
        while futures::StreamExt::next(&mut events).await.is_some() {}
        let _message = events.result().await.expect("stream result");

        assert_eq!(
            observed_key
                .lock()
                .expect("observed key lock poisoned")
                .as_deref(),
            Some("env-openai-key")
        );

        unregister_api_providers(source_id);
        openai.restore();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn explicit_api_key_takes_precedence_over_env_api_key() {
        let _guard = ENV_LOCK.lock().await;
        let openai = SavedEnv::capture("OPENAI_API_KEY");
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "env-openai-key");
        }

        let source_id = "stream-explicit-key-test";
        let observed_key = Arc::new(Mutex::new(None));
        register_api_provider(
            ApiProvider {
                api: "stream-explicit-key-test".to_string(),
                stream: wrap_stream("stream-explicit-key-test", |_model, _context, _options| {
                    panic!("stream should not be called")
                }),
                stream_simple: wrap_stream_simple("stream-explicit-key-test", {
                    let observed_key = Arc::clone(&observed_key);
                    move |model, _context, options| {
                        *observed_key.lock().expect("observed key lock poisoned") =
                            options.stream.api_key.clone();
                        Ok(done_stream(&model))
                    }
                }),
            },
            Some(source_id.to_string()),
        );

        let options = SimpleStreamOptions {
            stream: StreamOptions {
                api_key: Some("explicit-key".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut events = stream_simple(
            test_model("stream-explicit-key-test"),
            Context::default(),
            Some(options),
        )
        .expect("stream_simple should dispatch");
        while futures::StreamExt::next(&mut events).await.is_some() {}
        let _message = events.result().await.expect("stream result");

        assert_eq!(
            observed_key
                .lock()
                .expect("observed key lock poisoned")
                .as_deref(),
            Some("explicit-key")
        );

        unregister_api_providers(source_id);
        openai.restore();
    }

    #[test]
    fn stream_reports_unregistered_provider() {
        let model = test_model("missing-api-provider-test");
        let error = match stream(model, Context::default(), None) {
            Ok(_) => panic!("expected missing provider error"),
            Err(error) => error,
        };

        assert!(matches!(&error, Error::NoApiProvider(api) if api == "missing-api-provider-test"));
        assert_eq!(
            error.to_string(),
            "No API provider registered for api: missing-api-provider-test"
        );
    }
}
