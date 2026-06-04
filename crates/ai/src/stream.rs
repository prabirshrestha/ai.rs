use crate::AssistantEventStream;
use crate::env_api_keys::get_env_api_key;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, StreamOptions,
};
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
) -> Result<AssistantEventStream> {
    let api = model
        .language_api()
        .ok_or_else(|| Error::unsupported_capability(model.provider.clone(), "language models"))?;
    let options = with_env_api_key(&model, options.unwrap_or_default());
    api.stream(model, context, options)
}

pub async fn complete(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessage> {
    final_message_from_stream(stream(model, context, options)?).await
}

pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantEventStream> {
    let api = model
        .language_api()
        .ok_or_else(|| Error::unsupported_capability(model.provider.clone(), "language models"))?;
    let options = with_env_api_key_simple(&model, options.unwrap_or_default());
    api.stream_simple(model, context, options)
}

pub async fn complete_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessage> {
    final_message_from_stream(stream_simple(model, context, options)?).await
}

pub async fn final_message_from_stream(
    mut stream: AssistantEventStream,
) -> Result<AssistantMessage> {
    while let Some(event) = futures::StreamExt::next(&mut stream).await {
        match event? {
            AssistantMessageEvent::Done { message, .. } => return Ok(message),
            AssistantMessageEvent::Error { error, .. } => return Ok(error),
            _ => {}
        }
    }
    Err(Error::StreamClosed)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::provider::LanguageModelApi;
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

    #[derive(Clone)]
    struct TestLanguageModelApi {
        api: &'static str,
        observed_key: Arc<Mutex<Option<String>>>,
    }

    impl LanguageModelApi for TestLanguageModelApi {
        fn id(&self) -> &str {
            self.api
        }

        fn stream(
            &self,
            _model: Model,
            _context: Context,
            _options: StreamOptions,
        ) -> Result<AssistantEventStream> {
            panic!("stream should not be called")
        }

        fn stream_simple(
            &self,
            model: Model,
            _context: Context,
            options: SimpleStreamOptions,
        ) -> Result<AssistantEventStream> {
            *self
                .observed_key
                .lock()
                .expect("observed key lock poisoned") = options.stream.api_key.clone();
            Ok(done_stream(&model))
        }
    }

    fn test_model(api: &str, language_api: Option<Arc<dyn LanguageModelApi>>) -> Model {
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
            language_api,
            ..Model::default()
        }
    }

    fn done_stream(model: &Model) -> AssistantEventStream {
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
        let (mut sender, stream) = crate::create_assistant_message_event_stream();
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

        let observed_key = Arc::new(Mutex::new(None));
        let api = Arc::new(TestLanguageModelApi {
            api: "stream-env-key-test",
            observed_key: Arc::clone(&observed_key),
        });

        let events = stream_simple(
            test_model("stream-env-key-test", Some(api)),
            Context::default(),
            None,
        )
        .expect("stream_simple should dispatch");
        let _message = crate::stream::final_message_from_stream(events)
            .await
            .expect("stream result");

        assert_eq!(
            observed_key
                .lock()
                .expect("observed key lock poisoned")
                .as_deref(),
            Some("env-openai-key")
        );

        openai.restore();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn explicit_api_key_takes_precedence_over_env_api_key() {
        let _guard = ENV_LOCK.lock().await;
        let openai = SavedEnv::capture("OPENAI_API_KEY");
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "env-openai-key");
        }

        let observed_key = Arc::new(Mutex::new(None));
        let api = Arc::new(TestLanguageModelApi {
            api: "stream-explicit-key-test",
            observed_key: Arc::clone(&observed_key),
        });

        let options = SimpleStreamOptions {
            stream: StreamOptions {
                api_key: Some("explicit-key".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let events = stream_simple(
            test_model("stream-explicit-key-test", Some(api)),
            Context::default(),
            Some(options),
        )
        .expect("stream_simple should dispatch");
        let _message = crate::stream::final_message_from_stream(events)
            .await
            .expect("stream result");

        assert_eq!(
            observed_key
                .lock()
                .expect("observed key lock poisoned")
                .as_deref(),
            Some("explicit-key")
        );

        openai.restore();
    }

    #[test]
    fn stream_reports_model_without_language_api() {
        let model = test_model("missing-api-provider-test", None);
        let error = match stream(model, Context::default(), None) {
            Ok(_) => panic!("expected missing provider error"),
            Err(error) => error,
        };

        assert!(matches!(
            &error,
            Error::UnsupportedCapability {
                provider,
                capability: "language models",
            } if provider == "openai"
        ));
    }
}
